#!/usr/bin/env python3
"""
Bio Waste Tracker - laptop logger + live dashboard.

Reads JSON lines from the ESP32 over USB serial (works without Wi-Fi), timestamps
every disposal with the laptop clock, applies the time rule when nobody pressed a
type button, and writes:
    data/events.csv     one line per disposal (the D2 log)
    data/raw_log.jsonl  every device message incl. removals, ignored bumps, unstable,
                        overload - the honest log the judges asked for
With --camera (bin camera, see docs/camera-setup.md) also:
    data/vision.jsonl        one line per lid opening: which foods went in (bin_camera.py)
    data/vision_links.csv    record_no -> lid session (joins events.csv with vision.jsonl)
    data/vision/<session>/   before.jpg, after.jpg and piece_<k>.jpg cut-outs
Serves the dashboard at http://localhost:8000

Usage
    pip install pyserial
    python3 server.py --port /dev/ttyUSB0      (Windows: --port COM5, macOS: /dev/cu.usbserial-*)
    python3 server.py --simulate               (no hardware: fake disposals, for UI work)
    python3 server.py --port COM5 --camera 0   (+ bin camera: USB webcam 0, or an IP Webcam URL)
    python3 server.py --simulate --camera replay   (fake scale + lid, real recognition on test photos)
Only the standard library is needed for --simulate without --camera.
Manual lid for testing: http://localhost:8000/api/lid?state=open  ...  ?state=closed
"""
import argparse
import csv
import json
import os
import random
import threading
import time
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EVENTS_CSV = os.path.join(DATA, "events.csv")
RAW_LOG = os.path.join(DATA, "raw_log.jsonl")
TYPES = ["preparation_waste", "spoiled_stock", "plate_leftovers"]
CSV_FIELDS = ["record_no", "timestamp", "weight_g", "type", "station_id",
              "confidence", "source", "device_event_id"]

# Time rule used only when no button was pressed (confidence = low).
# Spoiled stock is never guessed: throwing out stock is a deliberate act -> needs the button.
TIME_RULE = [  # (from_hh:mm, to_hh:mm, type)
    ("00:00", "11:30", "preparation_waste"),
    ("11:30", "14:30", "plate_leftovers"),
    ("14:30", "17:30", "preparation_waste"),
    ("17:30", "23:59", "plate_leftovers"),
]

ACTIONS = {
    "preparation_waste": (
        "Over-production and trimmings are the biggest loss.",
        "Tomorrow: cook the top over-produced dish in two smaller batches (second batch only "
        "after 12:15 demand check) and send vegetable trimmings to the stock pot instead of the bin."),
    "spoiled_stock": (
        "Food is spoiling in storage.",
        "Tomorrow 7:00: FIFO check of the walk-in, date-label every opened container, and cut "
        "the next order of the most-binned item by 20%."),
    "plate_leftovers": (
        "Guests leave the most food on their plates.",
        "Tomorrow: reduce the side-dish portion by 15% and offer free refills on request - "
        "guests who want more ask, the rest stops landing in the bin."),
}

LINKS_CSV = os.path.join(DATA, "vision_links.csv")
vision = None   # bin_camera.LidSessions when started with --camera

lock = threading.Lock()
state = {
    "sessions": {},        # lid session id -> finished vision result
    "links": {},           # record_no -> lid session id
    "events": [],          # list of dicts (CSV rows)
    "live": {"g": 0.0, "stable": 1, "state": "idle", "pending": 0, "seen": 0},
    "counters": {"removal": 0, "ignored": 0, "unstable": 0, "overload": 0, "unassigned": 0, "lid_open": 0},
    "device": "not connected",
    "raw_tail": [],
}


def time_rule(dt):
    hm = dt.strftime("%H:%M")
    for a, b, t in TIME_RULE:
        if a <= hm < b:
            return t
    return "plate_leftovers"


def load_existing():
    if os.path.exists(LINKS_CSV):
        with open(LINKS_CSV, newline="") as f:
            state["links"] = {int(r["record_no"]): r["session"] for r in csv.DictReader(f)}
    from bin_camera import load_sessions  # stdlib only at import time
    state["sessions"] = {s["session"]: s for s in load_sessions(DATA)}
    if not os.path.exists(EVENTS_CSV):
        return
    with open(EVENTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            row["weight_g"] = float(row["weight_g"])
            row["record_no"] = int(row["record_no"])
            attach_foods(row)
            state["events"].append(row)


def attach_foods(row):
    """Give a disposal row the foods of its lid session (if the camera saw it)."""
    sid = state["links"].get(row["record_no"])
    row["vision_session"] = sid
    s = state["sessions"].get(sid) if sid else None
    row["foods"] = s["foods"] if s else []
    row["vision"] = "none" if not sid else ("done" if s else "pending")


def append_csv(row):
    new = not os.path.exists(EVENTS_CSV)
    with open(EVENTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def append_link(record_no, session):
    new = not os.path.exists(LINKS_CSV)
    with open(LINKS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["record_no", "session"])
        w.writerow([record_no, session])


def on_vision_result(rec):
    """Called by bin_camera when a lid session has been recognised."""
    with lock:
        state["sessions"][rec["session"]] = rec
        for e in state["events"]:
            if e.get("vision_session") == rec["session"]:
                attach_foods(e)


def handle_line(line, station_id):
    line = line.strip()
    if not line.startswith("{"):
        return
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return
    now = datetime.now()
    ev = msg.get("event")
    if ev == "lid" and vision is not None:
        vision.lid(msg.get("state"))       # starts / ends a camera session (photos in a thread)
    with lock:
        if ev == "weight":
            state["live"] = {**msg, "seen": time.time()}
            return
        rec = {"ts": now.isoformat(timespec="seconds"), **msg}
        with open(RAW_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        state["raw_tail"] = (state["raw_tail"] + [rec])[-30:]
        if ev in state["counters"]:
            state["counters"][ev] += 1
        if ev == "lid" and msg.get("state") == "open":
            state["counters"]["lid_open"] += 1
        if ev != "disposal":
            return
        typ, source, conf = msg.get("type"), msg.get("source", "button"), "high"
        if typ not in TYPES:
            typ, source, conf = time_rule(now), "time_rule", "low"
            state["counters"]["unassigned"] += 1
        row = {
            "record_no": (state["events"][-1]["record_no"] + 1) if state["events"] else 1,
            "timestamp": now.isoformat(timespec="seconds"),
            "weight_g": round(float(msg["weight_g"]), 1),
            "type": typ,
            "station_id": station_id,
            "confidence": conf,
            "source": source,
            "device_event_id": msg.get("id", ""),
        }
        sid = vision.session_for_disposal() if vision is not None else None
        if sid:
            state["links"][row["record_no"]] = sid
            append_link(row["record_no"], sid)
        attach_foods(row)
        state["events"].append(row)
        append_csv(row)
        print(f"[{row['timestamp']}] #{row['record_no']} {row['weight_g']:7.1f} g  {typ:18s} ({source})"
              f"{'  lid session ' + sid if sid else ''}")


def summary(day=None):
    day = day or datetime.now().strftime("%Y-%m-%d")
    with lock:
        evs = [e for e in state["events"] if e["timestamp"].startswith(day)]
        live = dict(state["live"])
        counters = dict(state["counters"])
        raw_tail = list(state["raw_tail"])
    totals = {t: 0.0 for t in TYPES}
    counts = {t: 0 for t in TYPES}
    hourly = {}
    for e in evs:
        totals[e["type"]] += e["weight_g"]
        counts[e["type"]] += 1
        h = int(e["timestamp"][11:13])
        hourly.setdefault(h, {t: 0.0 for t in TYPES})[e["type"]] += e["weight_g"]
    total = sum(totals.values())
    shares = {t: (100.0 * totals[t] / total if total else 0.0) for t in TYPES}
    biggest = max(TYPES, key=lambda t: totals[t]) if total else None
    live["connected"] = (time.time() - live.get("seen", 0)) < 3

    # which foods make up each stream (camera): a disposal's weight is split evenly over its foods
    foods_by_type = {}
    for t in TYPES:
        g, n, seen_g = {}, {}, 0.0
        for e in evs:
            if e["type"] != t or not e.get("foods"):
                continue
            seen_g += e["weight_g"]
            for f in e["foods"]:
                g[f] = g.get(f, 0.0) + e["weight_g"] / len(e["foods"])
                n[f] = n.get(f, 0) + 1
        foods_by_type[t] = {
            "camera_g": round(seen_g, 1),       # weight of the disposals the camera recognised
            "foods": [{"food": f, "g": round(v, 1), "n": n[f], "pct": round(100 * v / seen_g, 1)}
                      for f, v in sorted(g.items(), key=lambda kv: -kv[1])[:5]],
        }
    food_insight = None
    if biggest and foods_by_type[biggest]["foods"]:
        top = foods_by_type[biggest]["foods"][0]
        food_insight = f"Mostly {top['food']} ({top['pct']:.0f}% of the {biggest.replace('_', ' ')} the camera saw)."
    return {
        "day": day,
        "total_g": round(total, 1),
        "count": len(evs),
        "totals_g": {t: round(v, 1) for t, v in totals.items()},
        "counts": counts,
        "shares_pct": {t: round(v, 1) for t, v in shares.items()},
        "biggest": biggest,
        "insight": ACTIONS[biggest][0] if biggest else None,
        "action": ACTIONS[biggest][1] if biggest else None,
        "hourly": [{"hour": h, **{t: round(v, 1) for t, v in hourly[h].items()}} for h in sorted(hourly)],
        "recent": list(reversed(evs[-15:])),
        "low_confidence": sum(1 for e in evs if e["confidence"] == "low"),
        "counters": counters,
        "live": live,
        "device": state["device"],
        "raw_tail": list(reversed(raw_tail[-10:])),
        "foods_by_type": foods_by_type,
        "food_insight": food_insight,
        "vision": vision.state() if vision is not None else {"enabled": False},
    }


# ---------------- device readers ----------------
def serial_reader(port, baud, station_id):
    import serial  # pip install pyserial
    while True:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                state["device"] = f"serial {port}"
                print(f"connected to {port}")
                while True:
                    raw = ser.readline().decode("utf-8", errors="ignore")
                    if raw:
                        handle_line(raw, station_id)
        except Exception as ex:  # unplugged, permission, ... -> retry
            state["device"] = f"reconnecting ({ex.__class__.__name__})"
            print(f"serial error: {ex}; retrying in 2 s")
            time.sleep(2)


def simulator(station_id):
    """Fake device: a disposal every 4-9 s, live weight stream in between."""
    state["device"] = "SIMULATOR"
    tray, eid = 0.0, 0
    while True:
        for _ in range(random.randint(8, 18)):
            handle_line(json.dumps({"event": "weight", "g": round(tray + random.uniform(-1, 1), 1),
                                    "stable": 1, "state": "idle", "pending": 0}), station_id)
            time.sleep(0.5)
        r = random.random()
        if tray > 3500:
            tray = 0.0
            handle_line(json.dumps({"event": "removal", "delta_g": -3500}), station_id)
            continue
        if r < 0.08:
            handle_line(json.dumps({"event": "ignored", "delta_g": round(random.uniform(-6, 6), 1)}), station_id)
            continue
        eid += 1
        typ = random.choices(TYPES + ["unassigned"], weights=[40, 15, 35, 10])[0]
        g = {"preparation_waste": random.uniform(80, 900), "spoiled_stock": random.uniform(150, 1200),
             "plate_leftovers": random.uniform(40, 350), "unassigned": random.uniform(50, 400)}[typ]
        tray += g
        # like the real bin: lid opens, food goes in and is weighed, lid closes
        handle_line(json.dumps({"event": "lid", "state": "open"}), station_id)
        time.sleep(1.5)
        handle_line(json.dumps({"event": "disposal", "id": eid, "weight_g": round(g, 1), "type": typ,
                                "source": "button" if typ != "unassigned" else "none"}), station_id)
        time.sleep(1.0)
        handle_line(json.dumps({"event": "lid", "state": "closed"}), station_id)


# ---------------- HTTP ----------------
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
            with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            day = parse_qs(u.query).get("day", [None])[0]
            self._send(200, json.dumps(summary(day)).encode(), "application/json")
        elif u.path == "/events.csv":
            body = open(EVENTS_CSV, "rb").read() if os.path.exists(EVENTS_CSV) else b""
            self._send(200, body, "text/csv")
        elif u.path == "/raw_log.jsonl":
            body = open(RAW_LOG, "rb").read() if os.path.exists(RAW_LOG) else b""
            self._send(200, body, "application/x-ndjson")
        elif u.path == "/vision.jsonl":
            p = os.path.join(DATA, "vision.jsonl")
            self._send(200, open(p, "rb").read() if os.path.exists(p) else b"", "application/x-ndjson")
        elif u.path == "/api/lid":
            # manual lid for testing without the switch: /api/lid?state=open, then ?state=closed
            st = parse_qs(u.query).get("state", [""])[0]
            if st not in ("open", "closed"):
                return self._send(400, b"use ?state=open or ?state=closed", "text/plain")
            handle_line(json.dumps({"event": "lid", "state": st, "source": "http"}), self.server.station)
            self._send(200, json.dumps({"lid": st, "camera": vision is not None}).encode(), "application/json")
        elif u.path.startswith("/vision/"):
            # photos of a lid session: /vision/<session>/<before|after|piece_k>.jpg
            parts = u.path.split("/")
            if len(parts) != 4 or not all(c.isalnum() or c in "-_." for c in parts[2] + parts[3]) \
                    or ".." in parts[2] + parts[3] or not parts[3].endswith(".jpg"):
                return self._send(404, b"not found", "text/plain")
            p = os.path.join(DATA, "vision", parts[2], parts[3])
            if not os.path.exists(p):
                return self._send(404, b"not found", "text/plain")
            self._send(200, open(p, "rb").read(), "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="serial port of the ESP32")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--station", default="bin-1")
    ap.add_argument("--http", type=int, default=8000)
    ap.add_argument("--camera", help="bin camera: USB index (0), IP Webcam URL (http://...:8080) or 'replay'")
    ap.add_argument("--vision-model", choices=["so400m", "base"], help="default: so400m if the GPU has room")
    a = ap.parse_args()
    if not a.port and not a.simulate:
        ap.error("give --port <serial port> or --simulate")
    global DATA, EVENTS_CSV, RAW_LOG, LINKS_CSV, vision
    if a.simulate:  # never mix fake data into the real D2 log
        DATA = os.path.join(HERE, "data_sim")
        EVENTS_CSV, RAW_LOG = os.path.join(DATA, "events.csv"), os.path.join(DATA, "raw_log.jsonl")
        LINKS_CSV = os.path.join(DATA, "vision_links.csv")
    os.makedirs(DATA, exist_ok=True)
    load_existing()
    if a.camera:
        import bin_camera
        vision = bin_camera.LidSessions(bin_camera.open_camera(a.camera), DATA, on_vision_result,
                                        model=a.vision_model)
        print(f"bin camera: {vision.camera.name} (model loads in the background)")
    target = simulator if a.simulate else serial_reader
    args = (a.station,) if a.simulate else (a.port, a.baud, a.station)
    threading.Thread(target=target, args=args, daemon=True).start()
    print(f"dashboard: http://localhost:{a.http}")
    httpd = ThreadingHTTPServer(("0.0.0.0", a.http), Handler)
    httpd.station = a.station
    httpd.serve_forever()


if __name__ == "__main__":
    main()
