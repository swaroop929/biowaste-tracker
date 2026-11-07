#!/usr/bin/env python3
"""
tray_cam.py - Bio Waste Tracker: "what food got thrown away?"  (automatic before/after)

Phone is fixed ~30-50 cm above the weighing tray, looking straight down. It never moves.
The script watches the picture all the time:

    STILL  -> keep the latest picture as BEFORE
    MOVING -> hand comes in, food drops (we ignore these frames, the hand is in them)
    STILL again for STILL_S seconds -> this picture is AFTER
    diff(BEFORE, AFTER) -> crop of the new food -> Claude says what it is
    the weight record from the scale (events.csv or /weight) is attached
    AFTER becomes the next BEFORE

Based on food_cam.py (same classifier, prompt, classes, CSV, HTTP). New: automatic trigger,
.env support (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN), phone snapshot URL, weight matching.

Run (Windows):
  py -3.12 -m pip install opencv-python numpy
  py -3.12 tray_cam.py --camera sim --offline                       # test, no phone, no API
  py -3.12 tray_cam.py --camera http://PHONE_IP:8080/shot.jpg       # phone "IP Webcam" app
  py -3.12 tray_cam.py --camera 0                                   # DroidCam / Iriun / USB cam
  add  --events ..\\server\\data\\events.csv   to attach weights from the scale

Keys: SPACE = manual snap   b = new before-frame   d = show motion numbers   q = quit
Dashboard: http://localhost:8001/   (live camera, last snap before/after, totals, history)
HTTP: :8001/snap?weight_g=123  :8001/weight?weight_g=123  :8001/last  :8001/history  :8001/status  :8001/img/<file>

Output: captures/food_log.csv  + captures/snap_XXX_{before,after,crop,full}.jpg
"""
import argparse, base64, csv, json, os, sys, threading, time, urllib.request, urllib.error
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- settings ---
FOOD_CLASSES = [
    "bread", "fries_potatoes", "rice_pasta", "salad_vegetables",
    "meat_fish", "fruit_peels", "sauce_liquid", "dairy_cheese", "other",
]
DIFF_THRESHOLD = 35      # pixel change (0-255) counted as "changed"
MIN_AREA_FRAC = 0.002    # ignore changes smaller than 0.2% of the image
CROP_PAD = 40            # px padding around the changed region

# automatic trigger
MOTION_PIX = 10          # grey change of one pixel between two frames that counts as movement
MOTION_FRAC = 0.01       # >1 % of pixels moving = "MOVING"
STILL_S = 1.0            # quiet this long = still. Raise to 1.5-2 if the hand stays still over the tray
STILL_MIN_FRAMES = 4     # and at least this many NEW frames (protects against a frozen WiFi stream)
MATCH_WINDOW_S = 12      # weight record and photo belong together if this close in time

OUT = Path("captures")


# ------------------------------------------------------------------- .env ----
def load_env(path=".env"):
    """Tiny .env reader (no extra package). Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        p = Path(__file__).with_name(".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()
MODEL = os.environ.get("FOOD_CAM_MODEL", "claude-haiku-5-5")   # fast + cheap; claude-sonnet-5-5 if accuracy is low

PROMPT = f"""You are the vision sensor of a restaurant food-waste tracker.
Image 1 is the tray BEFORE, image 2 is AFTER something was thrown on it
(image 2 may be a crop of the changed region). Identify the food that was
just thrown away (what is new in image 2).

Answer ONLY with JSON, no other text:
{{"item": one of {FOOD_CLASSES},
  "food": "specific name, e.g. apple, banana peel, white bread",
  "confidence": number 0..1,
  "description": "max 8 words, e.g. 'two slices of white bread'"}}
If nothing new is visible or it is not food, use "other" with low confidence."""

# ------------------------------------------------------------------ vision ---
def jpg_b64(img, max_side=768):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def changed_region(before, after):
    """Return (x,y,w,h) of what changed between frames, or None."""
    g1 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (21, 21), 0).astype(np.int16)
    g2 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (21, 21), 0).astype(np.int16)
    # phone auto-exposure: remove a global brightness shift before comparing
    g2 = g2 - int(np.median(g2) - np.median(g1))
    dg = np.clip(np.abs(g2 - g1), 0, 255).astype(np.uint8)
    dc = cv2.absdiff(before, after).max(axis=2)          # colour difference (food on similar-brightness tray)
    diff = np.maximum(dg, cv2.GaussianBlur(dc, (21, 21), 0))
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_FRAC * mask.size
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(np.vstack(cnts))
    H, W = mask.shape
    x0, y0 = max(0, x - CROP_PAD), max(0, y - CROP_PAD)
    x1, y1 = min(W, x + w + CROP_PAD), min(H, y + h + CROP_PAD)
    return x0, y0, x1 - x0, y1 - y0


def classify(before, after_crop, offline=False):
    if offline:
        return {"item": "other", "food": "", "confidence": 0.0, "description": "offline mode"}
    # ANTHROPIC_AUTH_TOKEN -> Bearer (custom endpoint); ANTHROPIC_API_KEY -> x-api-key (api.anthropic.com)
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not token or "paste-your-key" in token:
        return {"item": "other", "food": "", "confidence": 0.0, "description": "no key in .env"}
    auth = ({"authorization": f"Bearer {token}"} if os.environ.get("ANTHROPIC_AUTH_TOKEN")
            else {"x-api-key": token})
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    body = {
        "model": MODEL,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpg_b64(before, 512)}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpg_b64(after_crop)}},
            {"type": "text", "text": PROMPT},
        ]}],
    }
    headers = {**auth, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    req = urllib.request.Request(base + "/v1/messages", data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = json.loads(r.read())["content"][0]["text"]
        text = text[text.find("{"): text.rfind("}") + 1]
        res = json.loads(text)
        if res.get("item") not in FOOD_CLASSES:
            res["item"] = "other"
        res["confidence"] = float(res.get("confidence", 0))
        res.setdefault("food", "")
        return res
    except urllib.error.HTTPError as e:
        return {"item": "other", "food": "", "confidence": 0.0, "description": f"API error {e.code}: {e.read()[:120]!r}"}
    except Exception as e:
        return {"item": "other", "food": "", "confidence": 0.0, "description": f"error: {e}"[:120]}

# ------------------------------------------------------------------ camera ---
class SimSource:
    """Fake tray: a 'hand' comes in, drops a coloured blob, leaves. For testing without a phone."""
    def __init__(self):
        self.t, self.items = 30, []
    def read(self):
        time.sleep(0.05)
        img = np.full((480, 640, 3), (200, 200, 205), np.uint8)
        cv2.rectangle(img, (60, 60), (580, 420), (170, 170, 175), -1)
        for c, p in self.items:
            cv2.circle(img, p, 35, c, -1)
        ph = self.t % 80
        if 20 <= ph < 40:
            k = ph - 20
            cv2.ellipse(img, (120 + 18 * k, 150 + 6 * k), (80, 45), 20, 0, 360, (90, 120, 170), -1)
        if ph == 40:
            col = [(40, 160, 60), (60, 140, 200), (30, 60, 200)][len(self.items) % 3]
            self.items.append((col, (140 + (110 * len(self.items)) % 400, 260)))
        self.t += 1
        return True, cv2.add(img, np.random.randint(0, 4, img.shape, dtype=np.uint8))

class SnapshotSource:
    """Polls http://PHONE:8080/shot.jpg - more robust over WiFi than the MJPEG stream."""
    def __init__(self, url):
        self.url = url
    def read(self):
        try:
            data = urllib.request.urlopen(self.url, timeout=3).read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            return img is not None, img
        except Exception:
            time.sleep(0.3)
            return False, None

class Camera:
    """Grabs frames in a background thread so we always have the latest one (+ a frame counter)."""
    def __init__(self, src):
        src = str(src)
        if src == "sim":
            self.cap = SimSource()
        elif src.lower().endswith((".jpg", ".jpeg")):
            self.cap = SnapshotSource(src)
        else:
            self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
            if not self.cap.isOpened():
                sys.exit(f"Cannot open camera {src}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame, self.fid, self.lock = None, 0, threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 10:
            time.sleep(0.05)
        if self.frame is None:
            sys.exit(f"Camera gives no frames - open {src} in the browser to check")

    def _loop(self):
        while True:
            ok, f = self.cap.read()
            if ok and f is not None:
                with self.lock:
                    self.frame, self.fid = f, self.fid + 1
            else:
                time.sleep(0.02)

    def get(self, with_id=False):
        with self.lock:
            return (self.frame.copy(), self.fid) if with_id else self.frame.copy()

# ----------------------------------------------------- weights from the scale ---
class Weights:
    """Weight records from events.csv (tail) or HTTP /weight. Matched to photos by arrival time."""
    def __init__(self, events_path):
        self.q, self.lock, self.path, self.pos, self.header = deque(maxlen=50), threading.Lock(), events_path, 0, None
        if events_path and os.path.exists(events_path):
            with open(events_path, encoding="utf-8") as f:
                self.header = f.readline().strip().split(",")
                f.read(); self.pos = f.tell()
        if events_path:
            threading.Thread(target=self._tail, daemon=True).start()
            print(f"following weights in {events_path}")

    def add(self, weight_g, record_no="", wtype=""):
        with self.lock:
            self.q.append(dict(t=time.time(), weight_g=str(weight_g), record_no=str(record_no), type=wtype))

    def _tail(self):
        while True:
            time.sleep(0.3)
            if not os.path.exists(self.path):
                continue
            with open(self.path, encoding="utf-8") as f:
                if self.header is None:
                    self.header = f.readline().strip().split(","); self.pos = f.tell()
                f.seek(self.pos)
                for line in f:
                    if line.strip():
                        r = dict(zip(self.header, line.strip().split(",")))
                        self.add(r.get("weight_g", ""), r.get("record_no", ""), r.get("type", ""))
                self.pos = f.tell()

    def take(self, t_photo, wait_s):
        """Return the weight record closest to the photo time; waits up to wait_s for it to arrive."""
        end = time.time() + wait_s
        while True:
            with self.lock:
                cands = [w for w in self.q if abs(w["t"] - t_photo) <= MATCH_WINDOW_S]
                if cands:
                    w = min(cands, key=lambda w: abs(w["t"] - t_photo))
                    self.q.remove(w)
                    return w
            if time.time() >= end:
                return None
            time.sleep(0.2)

# ------------------------------------------------------------------- core ----
class FoodCam:
    def __init__(self, cam, offline, weights):
        self.cam, self.offline, self.weights = cam, offline, weights
        self.before = cam.get()
        self.lock = threading.Lock()
        self.last = None          # (result dict, crop image)
        self.history = deque(maxlen=200)
        self.busy = False
        self.state, self.motion = "STILL", 0.0
        OUT.mkdir(exist_ok=True)
        self.log = OUT / "food_log.csv"
        if not self.log.exists():
            with open(self.log, "w", newline="") as f:
                csv.writer(f).writerow(["snap_no", "timestamp", "item", "food", "confidence", "description",
                                        "weight_g", "record_no", "type", "image", "trigger", "latency_s"])
        with open(self.log) as f:
            self.n = sum(1 for _ in f) - 1

    def rebase(self):
        self.before = self.cam.get()

    # ---- automatic trigger: still -> moving -> still ----
    def watch(self):
        prev, quiet_since, quiet_frames, last_fid, moving = None, time.time(), 0, -1, False
        while True:
            time.sleep(0.03)
            frame, fid = self.cam.get(with_id=True)
            if fid == last_fid:
                continue                                   # no new frame from the phone yet
            last_fid = fid
            small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            self.motion = 0.0 if prev is None else float(np.mean(cv2.absdiff(small, prev) > MOTION_PIX))
            prev = small
            now = time.time()
            if self.motion > MOTION_FRAC:
                quiet_since, quiet_frames = now, 0
                if not moving:
                    moving, self.state = True, "MOVING"
                continue
            quiet_frames += 1
            if moving and now - quiet_since >= STILL_S and quiet_frames >= STILL_MIN_FRAMES:
                moving, self.state = False, "STILL"
                if self.busy:
                    continue
                if changed_region(self.before, frame) is None:
                    print("still again - nothing new on the tray (hand passed / bump)")
                    self.before = frame
                else:
                    threading.Thread(target=self.snap, kwargs=dict(after=frame, trigger="auto"), daemon=True).start()

    def snap(self, weight_g="", after=None, trigger="manual"):
        with self.lock:                     # one snap at a time
            self.busy = True
            try:
                t_photo = time.time()
                if after is None:
                    time.sleep(0.6)
                    after = self.cam.get()
                before = self.before
                self.before = after         # new baseline = current tray content
                box = changed_region(before, after)
                crop = after if box is None else after[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                res = classify(before, crop, self.offline)
                dt = time.time() - t_photo
                w = None
                if not weight_g and self.weights:
                    w = self.weights.take(t_photo, wait_s=max(0, 4 - dt))   # scale is usually ~1-2 s behind
                    weight_g = w["weight_g"] if w else ""
                self.n += 1
                name = f"snap_{self.n:03d}"
                cv2.imwrite(str(OUT / f"{name}_before.jpg"), before)
                cv2.imwrite(str(OUT / f"{name}_after.jpg"), after)
                cv2.imwrite(str(OUT / f"{name}_crop.jpg"), crop)
                full = after.copy()
                if box:
                    x, y, bw, bh = box
                    cv2.rectangle(full, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                cv2.imwrite(str(OUT / f"{name}_full.jpg"), full)
                res.update(snap_no=self.n, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           weight_g=weight_g, record_no=w["record_no"] if w else "", type=w["type"] if w else "",
                           image=f"{name}_crop.jpg", full=f"{name}_full.jpg", before=f"{name}_before.jpg",
                           region_found=box is not None, trigger=trigger, latency_s=round(dt, 2))
                with open(self.log, "a", newline="") as f:
                    csv.writer(f).writerow([self.n, res["timestamp"], res["item"], res.get("food", ""),
                                            f'{res["confidence"]:.2f}', res.get("description", ""),
                                            weight_g, res["record_no"], res["type"], res["image"], trigger, res["latency_s"]])
                print(f"#{self.n} {res['item']:<18} {res['confidence']:.2f}  {res.get('food','')}  "
                      f"{res.get('description','')}  weight={weight_g or '-'}g  ({dt:.1f}s, {trigger})")
                self.last = (res, crop)
                self.history.appendleft(res)
                return res
            finally:
                self.busy = False

# ------------------------------------------------------------- http server ---
def start_http(fc, port):
    class H(BaseHTTPRequestHandler):
        def send_json(self, out):
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(b)

        def send_bytes(self, b, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/snap":
                return self.send_json(fc.snap(q.get("weight_g", [""])[0]))
            if u.path == "/weight":                          # scale server can push weights here
                fc.weights.add(q.get("weight_g", [""])[0], q.get("record_no", [""])[0], q.get("type", [""])[0])
                return self.send_json({"ok": True})
            if u.path == "/rebase":
                fc.rebase(); return self.send_json({"ok": True})
            if u.path == "/last":
                return self.send_json(fc.last[0] if fc.last else {})
            if u.path == "/history":
                return self.send_json(list(fc.history))
            if u.path == "/status":
                return self.send_json({"state": fc.state, "motion": round(fc.motion, 3), "busy": fc.busy, "snaps": fc.n,
                                       "offline": fc.offline})
            if u.path in ("/", "/dashboard"):
                return self.send_bytes((Path(__file__).with_name("tray_dashboard.html")).read_bytes(),
                                       "text/html; charset=utf-8")
            if u.path == "/live.jpg":
                return self.send_bytes(cv2.imencode(".jpg", fc.cam.get(), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes(),
                                       "image/jpeg")
            if u.path.startswith("/img/"):
                p = OUT / Path(u.path).name
                if p.exists():
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                    self.wfile.write(p.read_bytes()); return
            self.send_error(404)
        def log_message(self, *a): pass
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Dashboard: http://localhost:{port}/")
    print(f"HTTP: http://localhost:{port}/status  /last  /history  /snap?weight_g=123  /weight?weight_g=123")

# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0", help="0 | http://PHONE:8080/shot.jpg | http://PHONE:8080/video | sim")
    ap.add_argument("--events", default="", help="path to the scale's events.csv to attach weights")
    ap.add_argument("--http-port", type=int, default=8001)
    ap.add_argument("--offline", action="store_true", help="no API calls (test camera + crop)")
    ap.add_argument("--no-window", action="store_true", help="headless")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (testing)")
    a = ap.parse_args()

    fc = FoodCam(Camera(a.camera), a.offline, Weights(a.events))
    start_http(fc, a.http_port)
    threading.Thread(target=fc.watch, daemon=True).start()
    print(f"model={MODEL}  api={'OFFLINE' if a.offline else os.environ.get('ANTHROPIC_BASE_URL', 'api.anthropic.com')}")
    print("automatic: throw something on the tray, take the hand away, wait 1 s")
    print("SPACE = manual snap   b = new 'before' frame   d = motion numbers   q = quit")
    t_end = time.time() + a.seconds if a.seconds else None

    if a.no_window:
        while not t_end or time.time() < t_end: time.sleep(0.5)
        time.sleep(1); return

    debug = False
    while not t_end or time.time() < t_end:
        f = fc.cam.get()
        h, w = f.shape[:2]
        if fc.last:
            res, crop = fc.last
            th = min(200, h // 3)
            cw = min(int(crop.shape[1] * th / crop.shape[0]), w // 2)
            f[10:10 + th, w - cw - 10:w - 10] = cv2.resize(crop, (cw, th))
            txt = f"#{res['snap_no']} {res['item']} ({res['confidence']:.0%}) {res.get('weight_g') or ''}g"
            cv2.rectangle(f, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.putText(f, txt, (15, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(f, (res.get("food", "") + "  " + res.get("description", ""))[:60], (15, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        label = "ANALYSING..." if fc.busy else fc.state
        color = (0, 200, 255) if fc.busy else ((0, 0, 255) if fc.state == "MOVING" else (0, 180, 0))
        cv2.putText(f, label + (f"  motion {fc.motion:.3f}" if debug else ""), (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.imshow("tray_cam", f)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("b"):
            fc.rebase(); print("new before-frame")
        if k == ord("d"):
            debug = not debug
        if k == ord(" ") and not fc.busy:
            threading.Thread(target=fc.snap, daemon=True).start()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
