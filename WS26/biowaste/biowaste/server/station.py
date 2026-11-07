#!/usr/bin/env python3
"""
Bio Waste Tracker - STATION: scale (ESP32) + camera (phone) + type keys -> one record per disposal.

    scale settles  ->  camera picture, recognise what was added (vision/vision.py)
    type           ->  ESP32 button, or key 1/2/3 (keyboard here or the dashboard buttons),
                       pressed up to 20 s before or KEY_WAIT s after the disposal; else time rule (low confidence)
    record         ->  data/station_events.csv  (record_no, timestamp, weight_g, type, station_id,
                       confidence, source, food, food_conf, image_id, device_event_id)
    dashboard      ->  http://localhost:8000

Run
    python station.py --port COM8 --phone http://192.168.23.43:8080 --profile pitch --burst 1
    python station.py --fake-scale --phone http://...      no ESP32: type 'd 150' = a 150 g disposal
    python station.py --port COM8                          scale only, no camera
Keyboard (type + Enter here):  1 / 2 / 3 = type   r = new camera reference   d 150 = fake disposal   q = quit

Leaves server.py / dashboard.html untouched. Needs: pip install pyserial (and the vision packages for --phone).
"""
import argparse
import csv
import io
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EVENTS_CSV = os.path.join(DATA, "station_events.csv")
RAW_LOG = os.path.join(DATA, "station_raw.jsonl")
TYPES = ["preparation_waste", "spoiled_stock", "plate_leftovers"]
KEYS = {"1": "preparation_waste", "2": "spoiled_stock", "3": "plate_leftovers"}
FIELDS = ["record_no", "timestamp", "weight_g", "type", "station_id", "confidence", "source",
          "food", "food_conf", "image_id", "device_event_id"]
KEY_BEFORE_S = 20      # a type key pressed this long before the disposal still counts
KEY_WAIT_S = 6         # after the disposal, wait this long for a type key
CAPTURE_MAX_AGE_S = 60  # a camera picture taken at "weighed" is used for the disposal that follows

# Estimated purchase price per kg (wholesale-ish, EUR) - edit for your kitchen. Used for the money view.
PRICE_EUR_PER_KG = {"apple": 2.5, "orange": 2.8, "banana": 1.9, "pasta": 1.8, "bread": 4.0, "potato": 1.2,
                    "salad": 4.5, "vegetable_trimmings": 1.5, "rice": 2.2, "meat_fish": 14.0,
                    "green_grapes": 4.0, "cherry_tomato": 6.0,
                    "noodles": 2.0, "broccoli": 3.0}
PRICE_DEFAULT = 3.0
REPEAT_MIN = 2              # same food, same type, this many times today = "keeps coming back"
REPEAT_TIPS = {   # (tip, YouTube search) - plate leftovers are never reused (hygiene), so: change dish/portion
    "plate_leftovers": ("Guests keep leaving the {food}. Try a smaller portion or a different recipe.",
                        "{food} side dish ideas restaurant"),
    "preparation_waste": ("{food} trimmings keep piling up. Use the surplus instead of binning it.",
                          "zero waste {food} recipes trimmings"),
    "spoiled_stock": ("{food} keeps going off before it is used. Plan a dish that uses it up early.",
                      "recipes to use up {food}"),
}
SERVICE_DAYS_PER_WEEK = 6   # used to turn "today" into a weekly purchasing hint
ORDER_MIN_G = {"spoiled_stock": 80, "preparation_waste": 250}   # below this a single day is too little to act on
# kg CO2e per kg of food: production footprint, farm to retail (Poore & Nemecek 2018, via Our World in Data).
# pasta/bread = "Wheat & Rye"; trimmings = "Root vegetables"; salad = "Brassicas"; meat_fish = "Poultry meat" (assumption)
CO2E_PER_KG = {"apple": 0.43, "banana": 0.86, "orange": 0.39, "pasta": 1.57, "bread": 1.57, "rice": 4.45,
               "potato": 0.46, "vegetable_trimmings": 0.43, "salad": 0.51, "meat_fish": 9.87,
               "green_grapes": 1.53, "cherry_tomato": 2.09,   # "Berries & Grapes", "Tomatoes"
               "noodles": 1.57, "broccoli": 0.51}            # "Wheat & Rye", "Brassicas"
CO2E_DEFAULT = 1.5     # unknown food: rough mid value, shown as approx.
CO2E_SOURCE = "Poore & Nemecek (2018), via Our World in Data: production emissions per kg, farm to retail"
SERVICE_PERIODS = [("Morning prep", "00:00", "11:00"), ("Lunch", "11:00", "14:30"),
                   ("Afternoon", "14:30", "17:30"), ("Dinner", "17:30", "23:59")]


def co2_per_kg(food):
    parts = [f for f in str(food).split("+") if f in CO2E_PER_KG]
    return sum(CO2E_PER_KG[f] for f in parts) / len(parts) if parts else CO2E_DEFAULT
CROPS_DIR = os.path.join(HERE, "..", "vision", "crops")


def price_per_kg(food):
    parts = [f for f in str(food).split("+") if f in PRICE_EUR_PER_KG]
    return sum(PRICE_EUR_PER_KG[f] for f in parts) / len(parts) if parts else PRICE_DEFAULT


TIME_RULE = [("00:00", "11:30", "preparation_waste"), ("11:30", "14:30", "plate_leftovers"),
             ("14:30", "17:30", "preparation_waste"), ("17:30", "23:59", "plate_leftovers")]

ACTIONS = {
    "preparation_waste": ("Over-production and trimmings are the biggest loss.",
                          "Tomorrow: prep {food} in two smaller batches instead of one big one (second batch only "
                          "after the 12:15 demand check), and send usable trimmings to the stock pot."),
    "spoiled_stock": ("Food is spoiling in storage.",
                      "Tomorrow 7:00: FIFO check of the walk-in for {food}, date-label every opened container, "
                      "and order 20% less {food} next time."),
    "plate_leftovers": ("Guests leave the most food on their plates.",
                        "Tomorrow: serve 15% smaller {food} portions with free refills on request - guests who "
                        "want more ask, the rest stops landing in the bin."),
}

lock = threading.Lock()
S = {
    "events": [], "raw_tail": [],
    "live": {"g": 0.0, "stable": 1, "state": "idle", "pending": 0, "seen": 0},
    "counters": {"removal": 0, "ignored": 0, "unstable": 0, "overload": 0, "time_rule": 0, "no_camera": 0},
    "device": "not connected", "camera": "off",
    "key": None,            # (type, time) last type key pressed
    "waiting": None,        # {"weight_g", "since"} disposal waiting for a type key
    "camera_jpg": None,     # latest annotated camera frame (bytes)
    "last_items": [],
    "accepted": None,       # action text accepted for tomorrow
}
CFG = {"station": "bin-1", "serial": None}
cam_jobs = queue.Queue()
disposals = queue.Queue()
capture = {"t": 0, "items": None, "used": True, "busy": False, "done": threading.Event()}  # picture at "weighed"


def request_early_capture():
    """Scale says 'weighed' -> take the picture now (hands are out). One at a time."""
    if capture["busy"]:
        return
    capture.update(t=time.time(), items=None, used=False, busy=True, done=threading.Event())
    cam_jobs.put(("capture_early", None, None))


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def time_rule(dt):
    hm = dt.strftime("%H:%M")
    for a, b, t in TIME_RULE:
        if a <= hm < b:
            return t
    return "plate_leftovers"


def raw(rec):
    rec = {"ts": now_iso(), **rec}
    with open(RAW_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    with lock:
        S["raw_tail"] = (S["raw_tail"] + [rec])[-30:]


# ---------------- records ----------------
def load_existing():
    if not os.path.exists(EVENTS_CSV):
        return
    with open(EVENTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["weight_g"] = float(r["weight_g"])
        r["record_no"] = int(r["record_no"])
        r["food_conf"] = float(r["food_conf"] or 0)
    S["events"] = rows


def write_record(row):
    new = not os.path.exists(EVENTS_CSV)
    with open(EVENTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def take_key(since):
    """Return the type of a key pressed after `since` (and consume it)."""
    with lock:
        k = S["key"]
        if k and k[1] >= since:
            S["key"] = None
            return k[0]
    return None


def disposal_worker():
    """Turns scale disposals into records: camera result + type."""
    while True:
        d = disposals.get()
        t0 = d["t"]
        # 1. camera: use the picture taken when the scale said "weighed", else take one now
        items = None
        if CFG.get("phone"):
            if not capture["used"] and (capture["busy"] or t0 - capture["t"] < CAPTURE_MAX_AGE_S):
                capture["done"].wait(30)                  # picture already on its way - wait for it
                items, capture["used"] = capture["items"], True
            else:
                done = threading.Event()
                box = {}
                cam_jobs.put(("capture", done, box))
                done.wait(30)
                items = box.get("items")
        # 2. type: ESP32 button > key (before or after) > time rule
        wtype, source, conf = d.get("type"), d.get("source", "button"), "high"
        if wtype not in TYPES:
            wtype = take_key(t0 - KEY_BEFORE_S)
            source = "key"
            if wtype is None:
                with lock:
                    S["waiting"] = {"weight_g": d["weight_g"], "since": time.time(),
                                    "food": (items[0]["food"] if items else None)}
                deadline = time.time() + KEY_WAIT_S
                while wtype is None and time.time() < deadline:
                    time.sleep(0.1)
                    wtype = take_key(t0 - KEY_BEFORE_S)
                with lock:
                    S["waiting"] = None
            if wtype is None:
                wtype, source, conf = time_rule(datetime.now()), "time_rule", "low"
                with lock:
                    S["counters"]["time_rule"] += 1
        # 3. food
        foods = [r for r in (items or []) if r["food"] != "empty"]
        if foods:
            food = "+".join(r["food"] for r in foods)
            food_conf, image_id = foods[0]["food_conf"], foods[0].get("image_id", "")
        else:
            food, food_conf, image_id = ("no_camera" if not CFG.get("phone") else "not_seen"), 0.0, ""
            if CFG.get("phone"):
                with lock:
                    S["counters"]["no_camera"] += 1
        with lock:
            row = {
                "record_no": (S["events"][-1]["record_no"] + 1) if S["events"] else 1,
                "timestamp": now_iso(), "weight_g": round(float(d["weight_g"]), 1), "type": wtype,
                "station_id": CFG["station"], "confidence": conf, "source": source,
                "food": food, "food_conf": round(float(food_conf), 2), "image_id": image_id,
                "device_event_id": d.get("id", ""),
            }
            S["events"].append(row)
        write_record(row)
        print(f"[{row['timestamp']}] #{row['record_no']} {row['weight_g']:7.1f} g  {wtype:18s} "
              f"{food} {food_conf:.0%}  ({source})")


# ---------------- scale ----------------
def handle_device(msg):
    ev = msg.get("event")
    if ev == "weight":
        with lock:
            was = S["live"].get("pending", 0)
            S["live"] = {**msg, "seen": time.time()}
        if msg.get("pending") and not was and CFG.get("phone"):
            request_early_capture()                       # weighed -> picture now, hands are out
        return
    raw(msg)
    with lock:
        if ev in S["counters"]:
            S["counters"][ev] += 1
    if ev == "disposal":
        if CFG.get("phone") and capture["used"] and not capture["busy"]:
            request_early_capture()                       # firmware skipped "pending" (button pressed first)
        disposals.put({"t": time.time(), "weight_g": msg["weight_g"], "type": msg.get("type"),
                       "source": msg.get("source", "button"), "id": msg.get("id", "")})
    elif ev == "removal" and CFG.get("phone"):
        cam_jobs.put(("reference", None, None))          # tray emptied -> new camera reference


def serial_reader(port, baud):
    import serial  # pip install pyserial
    while True:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                CFG["serial"] = ser
                S["device"] = f"scale on {port}"
                print(f"scale connected on {port}")
                while True:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line.startswith("{"):
                        try:
                            handle_device(json.loads(line))
                        except (json.JSONDecodeError, KeyError):
                            pass
        except Exception as ex:
            CFG["serial"] = None
            S["device"] = f"scale reconnecting ({ex.__class__.__name__})"
            print(f"serial: {ex}; retrying in 2 s")
            time.sleep(2)


def send_to_scale(text):
    ser = CFG.get("serial")
    if ser:
        try:
            ser.write((text + "\n").encode())
        except Exception:
            pass


# ---------------- camera ----------------
def camera_worker(phone):
    import vision as V
    ref = V.snapshot_retry(phone)
    S["camera"] = f"phone {phone}"
    set_camera(V, ref, [])
    print("camera ready (reference taken)")
    last_live = 0
    while True:
        try:
            job, done, box = cam_jobs.get(timeout=1.0)
        except queue.Empty:
            if time.time() - last_live > 1.5:            # idle: keep the live view fresh
                try:
                    set_camera(V, V.snapshot(phone), S["last_items"] if time.time() - last_live < 10 else [])
                except Exception:
                    S["camera"] = "camera not reachable"
                last_live = time.time()
            continue
        try:
            if job == "reference":
                ref = V.snapshot_retry(phone)
                with lock:
                    S["last_items"] = []
                set_camera(V, ref, [])
                raw({"event": "camera_reference"})
                continue
            frames = [V.snapshot_retry(phone) for _ in range(V.BURST)]
            items = V.recognise_items(frames, ref)
            ref = frames[-1]                                # this picture is "before" for the next one
            with lock:
                S["last_items"] = [{k: r[k] for k in ("food", "food_conf", "box", "boxes") if k in r}
                                   for r in items]
            set_camera(V, frames[-1], items)
            last_live = time.time()
            raw({"event": "camera", "items": [(r["food"], r["food_conf"]) for r in items]})
            if job == "capture_early":
                capture.update(t=time.time(), items=items, busy=False)
                capture["done"].set()
            if box is not None:
                box["items"] = items
        except Exception as ex:
            print(f"camera error: {ex}")
            raw({"event": "camera_error", "msg": str(ex)})
            if job == "capture_early":
                capture.update(busy=False)
                capture["done"].set()
        finally:
            if done is not None:
                done.set()


def set_camera(V, frame, items):
    buf = io.BytesIO()
    img = V.annotate(frame, items or [])
    img.thumbnail((960, 960))
    img.save(buf, "JPEG", quality=75)
    with lock:
        S["camera_jpg"] = buf.getvalue()


def reset_data():
    """Start a fresh demo: move today's log files to an archive name (nothing is deleted), clear the dashboard."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = ""
    with lock:
        for path in (EVENTS_CSV, RAW_LOG):
            if os.path.exists(path):
                root, ext = os.path.splitext(path)
                os.replace(path, f"{root}_archive_{stamp}{ext}")
                if path == EVENTS_CSV:
                    archived = os.path.basename(f"{root}_archive_{stamp}{ext}")
        S["events"] = []
        S["raw_tail"] = []
        S["counters"] = {k: 0 for k in S["counters"]}
        S["accepted"] = None
        S["last_items"] = []
        S["key"] = None
    capture.update(used=True)
    if CFG.get("phone"):
        cam_jobs.put(("reference", None, None))
    print(f"demo data reset (archived as {archived or 'nothing to archive'})")
    return archived


# ---------------- keys ----------------
def press_key(k):
    t = KEYS.get(k)
    if not t:
        return False
    with lock:
        S["key"] = (t, time.time())
    send_to_scale(f"k{k}")       # firmware with key support finishes its pending disposal right away
    raw({"event": "key", "type": t})
    return True


def keyboard():
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd in KEYS:
            press_key(cmd)
            print(f"  type: {KEYS[cmd]}")
        elif cmd == "r":
            cam_jobs.put(("reference", None, None))
            print("  new camera reference")
        elif cmd.startswith("d "):
            try:
                g = float(cmd[2:])
                if CFG.get("phone"):
                    request_early_capture()
                disposals.put({"t": time.time(), "weight_g": g, "type": None, "source": "fake", "id": ""})
            except ValueError:
                print("  usage: d 150")
        elif cmd == "q":
            os._exit(0)


# ---------------- summary + HTTP ----------------
def summary(day=None):
    day = day or datetime.now().strftime("%Y-%m-%d")
    with lock:
        evs = [e for e in S["events"] if e["timestamp"].startswith(day)]
        live, counters = dict(S["live"]), dict(S["counters"])
        waiting = dict(S["waiting"]) if S["waiting"] else None
        key = S["key"]
        raw_tail = list(S["raw_tail"])
    totals = {t: 0.0 for t in TYPES}
    counts = {t: 0 for t in TYPES}
    foods = {t: {} for t in TYPES}
    hourly = {}
    for e in evs:
        t = e["type"] if e["type"] in TYPES else TYPES[0]
        totals[t] += e["weight_g"]
        counts[t] += 1
        for f in str(e["food"]).split("+"):
            if f and f not in ("not_seen", "no_camera", "unknown"):
                foods[t][f] = foods[t].get(f, 0.0) + e["weight_g"] / len(str(e["food"]).split("+"))
        h = int(e["timestamp"][11:13])
        hourly.setdefault(h, {x: 0.0 for x in TYPES})[t] += e["weight_g"]
    total = sum(totals.values())
    shares = {t: (100.0 * totals[t] / total if total else 0.0) for t in TYPES}
    biggest = max(TYPES, key=lambda t: totals[t]) if total else None
    top_food = max(foods[biggest], key=foods[biggest].get) if biggest and foods[biggest] else None
    insight = action = None
    if biggest:
        insight = ACTIONS[biggest][0]
        action = ACTIONS[biggest][1].format(food=top_food.replace("_", " ") if top_food else "the top item")
    live["connected"] = (time.time() - live.get("seen", 0)) < 3
    rank = {}
    cost_total = 0.0
    recent = []
    for e in evs:
        c = e["weight_g"] / 1000.0 * price_per_kg(e["food"])
        cost_total += c
        f = str(e["food"]) if e["food"] not in ("not_seen", "no_camera") else "not recognised"
        r = rank.setdefault(f, {"food": f, "g": 0.0, "eur": 0.0, "n": 0, "by_type": {x: 0.0 for x in TYPES}})
        r["g"] += e["weight_g"]; r["eur"] += c; r["n"] += 1
        r["by_type"][e["type"] if e["type"] in TYPES else TYPES[0]] += e["weight_g"]
    for e in evs[-12:]:
        recent.append({**e, "eur": round(e["weight_g"] / 1000.0 * price_per_kg(e["food"]), 2),
                       "img": f"/crop/{e['image_id']}" if e.get("image_id") else ""})
    # purchasing hints: spoiled stock = bought too much; prep waste = maybe bought/prepped too much
    orders = []
    per_food = {}
    for e in evs:
        if e["type"] in ORDER_MIN_G and e["food"] not in ("not_seen", "no_camera", "unknown"):
            for f in str(e["food"]).split("+"):
                d = per_food.setdefault(f, {"spoiled_stock": 0.0, "preparation_waste": 0.0})
                d[e["type"]] += e["weight_g"] / len(str(e["food"]).split("+"))
    for f, d in per_food.items():
        for t in ("spoiled_stock", "preparation_waste"):
            if d[t] >= ORDER_MIN_G[t]:
                week_kg = d[t] / 1000.0 * SERVICE_DAYS_PER_WEEK
                orders.append({"food": f, "type": t, "g_today": round(d[t]), "week_kg": round(week_kg, 1),
                               "week_eur": round(week_kg * price_per_kg(f), 2),
                               "strength": "order" if t == "spoiled_stock" else "check"})
                break   # one hint per food, spoiled first
    orders.sort(key=lambda o: (o["strength"] != "order", -o["week_eur"]))
    from urllib.parse import quote_plus
    reps = {}
    for e in evs:
        if e["type"] in TYPES and e["food"] not in ("not_seen", "no_camera", "unknown", "non_food"):
            for f in str(e["food"]).split("+"):
                r = reps.setdefault((f, e["type"]), {"n": 0, "g": 0.0})
                r["n"] += 1
                r["g"] += e["weight_g"]
    repeats = []
    for (f, t), r in reps.items():
        if r["n"] >= REPEAT_MIN:
            words = f.replace("_", " ")
            tip, query = REPEAT_TIPS[t]
            repeats.append({"food": f, "type": t, "n": r["n"], "g": round(r["g"]),
                            "tip": tip.format(food=words)[0].upper() + tip.format(food=words)[1:],
                            "link": "https://www.youtube.com/results?search_query=" + quote_plus(query.format(food=words))})
    repeats.sort(key=lambda x: -x["g"])
    # service periods (today)
    periods = []
    for name, a, b in SERVICE_PERIODS:
        pe = [e for e in evs if a <= e["timestamp"][11:16] < b]
        periods.append({"name": name, "from": a, "to": b, "g": round(sum(e["weight_g"] for e in pe), 1),
                        "by_type": {t: round(sum(e["weight_g"] for e in pe if e["type"] == t), 1) for t in TYPES}})
    # history: earlier days in the log -> yesterday, 7-day average, spoilage trend
    with lock:
        all_evs = list(S["events"])
    days = {}
    for e in all_evs:
        dd = e["timestamp"][:10]
        if dd >= day:
            continue
        x = days.setdefault(dd, {"g": 0.0, "by_type": {t: 0.0 for t in TYPES}})
        x["g"] += e["weight_g"]
        if e["type"] in TYPES:
            x["by_type"][e["type"]] += e["weight_g"]
    past = sorted(days)
    from datetime import date as _date, timedelta as _td
    yday = (_date.fromisoformat(day) - _td(days=1)).isoformat()
    last7 = [days[d_] for d_ in past[-7:]]
    history = {
        "days_with_data": len(past),
        "yesterday": days.get(yday),
        "avg7": ({"g": round(sum(x["g"] for x in last7) / len(last7), 1),
                  "by_type": {t: round(sum(x["by_type"][t] for x in last7) / len(last7), 1) for t in TYPES}}
                 if last7 else None),
        "spoiled_trend": [{"day": d_, "weekday": _date.fromisoformat(d_).strftime("%a"),
                           "g": round(days[d_]["by_type"]["spoiled_stock"], 1)} for d_ in past[-13:]]
                         + [{"day": day, "weekday": _date.fromisoformat(day).strftime("%a"),
                             "g": round(totals["spoiled_stock"], 1)}],
    }
    with lock:
        accepted = S["accepted"]
    return {
        "orders": orders[:4], "service_days": SERVICE_DAYS_PER_WEEK, "repeats": repeats[:4],
        "cost_eur": round(cost_total, 2), "co2_kg": round(sum(e["weight_g"] / 1000.0 * co2_per_kg(e["food"]) for e in evs), 2),
        "co2_source": CO2E_SOURCE,
        "cost_outlook": {"day": round(cost_total, 2), "week": round(cost_total * SERVICE_DAYS_PER_WEEK, 2),
                         "month": round(cost_total * SERVICE_DAYS_PER_WEEK * 52 / 12, 2)},
        "periods": periods, "history": history,
        "rank": sorted(({**r, "g": round(r["g"], 1), "eur": round(r["eur"], 2)} for r in rank.values()), key=lambda r: -r["g"])[:8],
        "accepted": accepted == action and action is not None,
        "day": day, "total_g": round(total, 1), "count": len(evs),
        "totals_g": {t: round(v, 1) for t, v in totals.items()}, "counts": counts,
        "shares_pct": {t: round(v, 1) for t, v in shares.items()},
        "foods_g": {t: dict(sorted(((f, round(g, 1)) for f, g in fs.items()), key=lambda x: -x[1])) for t, fs in foods.items()},
        "biggest": biggest, "top_food": top_food, "insight": insight, "action": action,
        "hourly": [{"hour": h, **{t: round(v, 1) for t, v in hourly[h].items()}} for h in sorted(hourly)],
        "recent": list(reversed(recent)),
        "counters": counters, "live": live, "device": S["device"], "camera": S["camera"],
        "waiting": waiting, "armed": key[0] if key and time.time() - key[1] < KEY_BEFORE_S else None,
        "raw_tail": list(reversed(raw_tail[-10:])),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "station.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            self._send(200, json.dumps(summary(parse_qs(u.query).get("day", [None])[0])).encode(), "application/json")
        elif u.path == "/camera.jpg":
            jpg = S["camera_jpg"]
            self._send(200, jpg, "image/jpeg") if jpg else self._send(404, b"no camera", "text/plain")
        elif u.path in ("/site", "/site/") or u.path.startswith("/site/"):
            rel = u.path[len("/site/"):] or "index.html"
            path = os.path.normpath(os.path.join(HERE, "site", rel))
            types = {".html": "text/html; charset=utf-8", ".css": "text/css", ".svg": "image/svg+xml",
                     ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
            ext = os.path.splitext(path)[1].lower()
            if path.startswith(os.path.join(HERE, "site")) and os.path.isfile(path) and ext in types:
                self._send(200, open(path, "rb").read(), types[ext])
            elif u.path in ("/site", "/site/"):
                self.send_response(302); self.send_header("Location", "/site/index.html"); self.end_headers()
            else:
                self._send(404, b"not found", "text/plain")
        elif u.path.startswith("/crop/"):
            name = os.path.basename(u.path[6:])
            path = os.path.join(CROPS_DIR, name)
            if name.lower().endswith(".jpg") and os.path.isfile(path):
                self._send(200, open(path, "rb").read(), "image/jpeg")
            else:
                self._send(404, b"no image", "text/plain")
        elif u.path == "/events.csv":
            self._send(200, open(EVENTS_CSV, "rb").read() if os.path.exists(EVENTS_CSV) else b"", "text/csv")
        elif u.path == "/raw.jsonl":
            self._send(200, open(RAW_LOG, "rb").read() if os.path.exists(RAW_LOG) else b"", "application/x-ndjson")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/key":
            ok = press_key(q.get("k", [""])[0])
            self._send(200 if ok else 400, b"ok" if ok else b"bad key", "text/plain")
        elif u.path == "/api/accept":
            act = None if q.get("undo") else summary()["action"]   # summary() takes the lock itself
            with lock:
                S["accepted"] = act
            self._send(200, b"ok", "text/plain")
        elif u.path == "/api/reset":
            name = reset_data()
            self._send(200, json.dumps({"archived": name}).encode(), "application/json")
        elif u.path == "/api/reference":
            cam_jobs.put(("reference", None, None))
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="ESP32 serial port, e.g. COM8")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fake-scale", action="store_true", help="no ESP32: type 'd 150' for a 150 g disposal")
    ap.add_argument("--phone", help="IP Webcam address, e.g. http://192.168.23.43:8080")
    ap.add_argument("--profile", help="restaurant profile in vision/profiles/, e.g. pitch")
    ap.add_argument("--model", help="CLIP model override")
    ap.add_argument("--burst", type=int, default=1)
    ap.add_argument("--station", default="bin-1")
    ap.add_argument("--http", type=int, default=8000)
    a = ap.parse_args()
    if not a.port and not a.fake_scale:
        ap.error("give --port COMx (ESP32) or --fake-scale")
    CFG["station"], CFG["phone"] = a.station, a.phone
    os.makedirs(DATA, exist_ok=True)
    load_existing()
    if a.phone:
        sys.path.insert(0, os.path.join(HERE, "..", "vision"))
        import vision as V
        if a.model:
            V.MODEL = a.model
        V.BURST = a.burst
        if a.profile:
            V.load_profile(a.profile)
        print("loading recognition model...")
        V.load_model()
        V.load_roi()
        threading.Thread(target=camera_worker, args=(a.phone,), daemon=True).start()
    if a.port:
        threading.Thread(target=serial_reader, args=(a.port, a.baud), daemon=True).start()
    else:
        S["device"] = "fake scale (type d 150)"
    threading.Thread(target=disposal_worker, daemon=True).start()
    threading.Thread(target=keyboard, daemon=True).start()
    print(f"dashboard: http://localhost:{a.http}   keys: 1/2/3 type, r new reference, d 150 fake disposal, q quit")
    ThreadingHTTPServer(("0.0.0.0", a.http), Handler).serve_forever()


if __name__ == "__main__":
    main()
