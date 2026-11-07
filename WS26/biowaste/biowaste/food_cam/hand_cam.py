#!/usr/bin/env python3
"""
hand_cam.py - "catch the food in the hand"

The camera watches the bin opening. When motion starts (a hand comes in with
food) it keeps a rolling buffer of frames. When motion stops (hand is gone)
it picks the sharpest frames from the throw + the before/after crop and sends
them to Claude in ONE request: "what food was thrown away?"

No button, no extra sensor: motion is the trigger.

Run:
  pip install opencv-python numpy
  export ANTHROPIC_API_KEY=sk-ant-...        (Windows: set ANTHROPIC_API_KEY=...)
  python3 hand_cam.py --camera 0
  python3 hand_cam.py --camera http://PHONE_IP:8080/video     # phone "IP Webcam"
  python3 hand_cam.py --camera 0 --offline                    # test motion/frames without API

Keys in the window:
  r = draw the bin-opening region (drag a box, ENTER)   - do this first!
  SPACE = force a snap   b = new "before" frame   d = debug view   q = quit

HTTP (for dashboard / server.py):
  GET :8001/last             latest result as JSON
  GET :8001/img/<file>       saved images
  GET :8001/snap?weight_g=   force a snap (e.g. from the scale)

Dashboard: http://localhost:8001/
Output: captures/throws.csv + captures/throw_XXX_*.jpg
"""
import argparse, base64, csv, json, os, sys, threading, time, collections
import urllib.request, urllib.error
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
MODEL = os.environ.get("FOOD_CAM_MODEL", "claude-sonnet-5-5")  # haiku-5-5 = faster/cheaper

MOTION_PIX = 25          # grey-level change per pixel counted as motion
MOTION_START = 0.01      # >1% of region moving -> throw started  (tune with d = debug view)
MOTION_STOP = 0.003      # <0.3% moving ...
QUIET_S = 0.8            # ... for this long -> throw finished
MIN_THROW_S = 0.3        # shorter motion = flicker, ignored
MAX_THROW_S = 8.0       # longer = someone standing there, force finish
BUFFER_S = 4.0           # rolling frame buffer
N_HAND_FRAMES = 3        # in-hand frames sent to Claude
COOLDOWN_S = 1.0         # ignore motion right after a throw
OUT = Path("captures")
LOG_COLS = ["throw_no", "timestamp", "food", "item", "confidence", "description",
            "seen_in_hand", "weight_g", "throw_s", "latency_s", "trigger", "image"]
HERE = Path(__file__).resolve().parent

# --- API settings: read food_cam/.env (never committed) then environment variables ---
def _load_env(path):
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_env(HERE / ".env")
API_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
if API_BASE.endswith("/v1"):
    API_BASE = API_BASE[:-3]
MODEL = os.environ.get("FOOD_CAM_MODEL", MODEL)


def _auth_headers():
    """ANTHROPIC_AUTH_TOKEN -> 'Authorization: Bearer' (custom endpoints); else ANTHROPIC_API_KEY -> x-api-key."""
    tok = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    key = os.environ.get("ANTHROPIC_API_KEY")
    return {"x-api-key": key} if key else None

PROMPT = f"""You are the vision sensor of a restaurant food-waste tracker.
The camera looks at the opening of a waste bin.

Image 1: the bin BEFORE.
Images 2..{1+N_HAND_FRAMES}: frames DURING the throw - a hand is holding the food
  that is being thrown away. These are the most important images.
Last image: the bin AFTER, cropped to the region that changed.

Identify the food that was thrown away. Ignore hands, gloves, packaging, the bin
itself and food already in the bin before.

Answer ONLY with JSON, no other text:
{{"food": "the specific food, English, lowercase, singular, 1-3 words, e.g. 'apple', 'banana peel', 'bread roll', 'french fries'",
  "item": category, one of {FOOD_CLASSES},
  "confidence": number 0..1 (how sure you are about "food"),
  "description": "max 8 words, e.g. 'half-eaten red apple'",
  "seen_in_hand": true/false}}
If no food is visible (e.g. only a hand, or packaging), use food "nothing", item "other", confidence < 0.3."""

# ----------------------------------------------------------------- helpers ---
def jpg_b64(img, max_side=768):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def changed_region(before, after, thr=35, pad=40, min_frac=0.002):
    g1 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    dc = cv2.GaussianBlur(cv2.absdiff(before, after).max(axis=2), (21, 21), 0)
    diff = np.maximum(cv2.absdiff(g1, g2), dc)
    _, m = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)),
                   np.ones((15, 15), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_frac * m.size]
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(np.vstack(cnts))
    H, W = m.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    return x0, y0, min(W, x + w + pad) - x0, min(H, y + h + pad) - y0


def pick_hand_frames(frames, n=N_HAND_FRAMES):
    """frames: list of (t, img, motion). Split the throw into n parts, take the
    sharpest frame with real motion from each part -> food at different moments."""
    moving = [f for f in frames if f[2] >= MOTION_STOP] or frames
    if len(moving) <= n:
        return [f[1] for f in moving]
    chunks = np.array_split(np.arange(len(moving)), n)
    out = []
    for c in chunks:
        best = max(c, key=lambda i: sharpness(moving[i][1]))
        out.append(moving[best][1])
    return out


def classify(before, hand_frames, after_crop, offline=False):
    if offline:
        return {"item": "other", "confidence": 0.0, "description": "offline mode", "seen_in_hand": False}
    auth = _auth_headers()
    if not auth:
        return {"item": "other", "confidence": 0.0, "description": "no API key: create food_cam/.env (see README)", "seen_in_hand": False}
    imgs = [before] + hand_frames + [after_crop]
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": jpg_b64(im, 512 if i == 0 else 768)}}
               for i, im in enumerate(imgs)]
    content.append({"type": "text", "text": PROMPT})
    body = {"model": MODEL, "max_tokens": 1024, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(
        f"{API_BASE}/v1/messages", data=json.dumps(body).encode(),
        headers={**auth, "anthropic-version": "2023-06-01", "content-type": "application/json",
                 "user-agent": os.environ.get("FOOD_CAM_USER_AGENT",
                     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) food-cam/1.0")})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            text = next(c["text"] for c in json.loads(r.read())["content"] if c.get("type") == "text")  # skip thinking blocks
        res = json.loads(text[text.find("{"): text.rfind("}") + 1])
        if res.get("item") not in FOOD_CLASSES:
            res["item"] = "other"
        res["confidence"] = float(res.get("confidence", 0))
        res["food"] = str(res.get("food") or res["item"]).strip().lower()[:40]
        return res
    except urllib.error.HTTPError as e:
        return {"item": "other", "confidence": 0.0, "description": f"API {e.code}: {e.read()[:100]!r}", "seen_in_hand": False}
    except Exception as e:
        return {"item": "other", "confidence": 0.0, "description": f"error: {e}"[:120], "seen_in_hand": False}


class Camera:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if not self.cap.isOpened():
            sys.exit(f"Cannot open camera {src}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame, self.seq, self.lock = None, 0, threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 10:
            time.sleep(0.05)
        if self.frame is None:
            sys.exit("Camera gives no frames")

    def _loop(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.seq = f, self.seq + 1
            else:
                time.sleep(0.02)

    def get(self):
        with self.lock:
            return self.frame.copy(), self.seq

# -------------------------------------------------------------------- core ---
class HandCam:
    IDLE, THROW = "IDLE", "THROW"

    def __init__(self, cam, offline, roi=None):
        self.cam, self.offline = cam, offline
        f, _ = cam.get()
        H, W = f.shape[:2]
        self.roi = roi or (0, 0, W, H)
        self.before = f
        self.prev_small = None
        self.buf = collections.deque()
        self.state, self.t_start, self.t_quiet, self.t_cool = self.IDLE, 0, None, 0
        self.motion = 0.0
        self.ignored = 0
        self.idle_hist = collections.deque(maxlen=150)   # ~5 s of idle motion values
        self.noise = 0.0
        self.th_start, self.th_stop = MOTION_START, MOTION_STOP
        self.busy, self.last = False, None
        self.mask_dbg = None
        OUT.mkdir(exist_ok=True)
        self.view_jpg = None      # latest annotated frame for the dashboard
        self.log = OUT / "throws.csv"
        if not self.log.exists():
            with open(self.log, "w", newline="") as fh:
                csv.writer(fh).writerow(LOG_COLS)
        with open(self.log, newline="") as fh:
            self.history = list(csv.DictReader(fh))
        self.n = len(self.history)

    def crop(self, f):
        x, y, w, h = self.roi
        return f[y:y + h, x:x + w]

    def motion_frac(self, f):
        # colour difference (max over B,G,R) - catches food with same brightness as background
        small = cv2.GaussianBlur(cv2.resize(self.crop(f), (320, 180)), (7, 7), 0)
        if self.prev_small is None:
            self.prev_small = small
            return 0.0
        d = cv2.absdiff(small, self.prev_small).max(axis=2)
        self.prev_small = small
        m = (d > MOTION_PIX).astype(np.uint8)
        self.mask_dbg = m * 255
        return float(m.mean())

    def step(self, f):
        """Call for every new frame. Runs the IDLE/THROW state machine."""
        now = time.time()
        self.motion = self.motion_frac(f)
        self.buf.append((now, f, self.motion))
        while self.buf and now - self.buf[0][0] > BUFFER_S:
            self.buf.popleft()
        if self.busy or now < self.t_cool:
            return
        if self.state == self.IDLE:
            # learn the background flicker of this camera and stay above it
            self.idle_hist.append(self.motion)
            if len(self.idle_hist) >= 30:
                self.noise = float(np.percentile(self.idle_hist, 90))
                self.th_stop = max(MOTION_STOP, self.noise * 1.5 + 0.002)
                self.th_start = max(MOTION_START, self.noise * 3 + 0.008, self.th_stop * 2)
            if self.motion > self.th_start:
                self.state, self.t_start, self.t_quiet = self.THROW, now, None
                # before = last quiet frame in buffer
                quiet = [b for b in self.buf if b[2] < self.th_stop and b[0] < now - 0.15]
                if quiet:
                    self.before = quiet[-1][1]
            elif self.motion < self.th_stop:
                self.before = f   # keep baseline fresh while idle (light changes)
        else:  # THROW
            if self.motion < self.th_stop:
                self.t_quiet = self.t_quiet or now
            else:
                self.t_quiet = None
            dur = now - self.t_start
            if dur > MAX_THROW_S:
                # motion never stopped: noise, camera shake or someone standing there -> no throw
                self.state = self.IDLE
                self.before = f
                self.ignored += 1
                self.idle_hist.clear()          # re-learn noise level
                print(f"   (ignored: motion for {dur:.0f}s without a stop - noise? raise --start/--pix or set region with r)")
                return
            if self.t_quiet and now - self.t_quiet > QUIET_S:
                self.state = self.IDLE
                if dur - QUIET_S < MIN_THROW_S:
                    self.ignored += 1
                    return  # flicker
                frames = [b for b in self.buf if self.t_start - 0.2 <= b[0] <= (self.t_quiet or now)]
                after = f
                threading.Thread(target=self.finish, args=(self.before, frames, after, dur, "motion"),
                                 daemon=True).start()

    def force(self, weight_g="", trigger="manual"):
        """Manual / HTTP snap: use the last BUFFER_S seconds."""
        f, _ = self.cam.get()
        frames = list(self.buf)
        return self.finish(self.before, frames, f, 0.0, trigger, weight_g)

    def finish(self, before, frames, after, dur, trigger, weight_g=""):
        self.busy = True
        try:
            t0 = time.time()
            hand = pick_hand_frames(frames) if frames else []
            hand = [self.crop(h) for h in hand]
            b, a = self.crop(before), self.crop(after)
            box = changed_region(b, a)
            a_crop = a if box is None else a[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            res = classify(b, hand, a_crop, self.offline)
            res.setdefault("food", res["item"] if not self.offline else "offline test")
            lat = time.time() - t0
            self.n += 1
            name = f"throw_{self.n:03d}"
            # collage for dashboard/pitch: hand frames + after crop
            tiles = [cv2.resize(im, (int(im.shape[1] * 240 / im.shape[0]), 240)) for im in hand + [a_crop]]
            collage = cv2.hconcat(tiles) if tiles else a_crop
            cv2.imwrite(str(OUT / f"{name}_collage.jpg"), collage)
            cv2.imwrite(str(OUT / f"{name}_after.jpg"), a_crop)
            for i, h in enumerate(hand):
                cv2.imwrite(str(OUT / f"{name}_hand{i+1}.jpg"), h)
            res.update(throw_no=self.n, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       weight_g=weight_g, throw_s=round(dur, 2), latency_s=round(lat, 2),
                       trigger=trigger, image=f"{name}_collage.jpg")
            row = {c: res.get(c, "") for c in LOG_COLS}
            row["confidence"] = f'{res["confidence"]:.2f}'
            with open(self.log, "a", newline="") as fh:
                csv.DictWriter(fh, LOG_COLS).writerow(row)
            self.history.append(row)
            print(f"#{self.n} {res['food']:<16} [{res['item']}] {res['confidence']:.2f} hand={res.get('seen_in_hand')}  "
                  f"{res.get('description','')}  (throw {dur:.1f}s, {len(hand)} frames, {lat:.1f}s)")
            self.last = (res, collage)
            self.before = after
            return res
        finally:
            self.busy = False
            self.t_cool = time.time() + COOLDOWN_S
            self.prev_small = None


def start_http(hc, port):
    class H(BaseHTTPRequestHandler):
        def _json(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            if u.path == "/snap":
                return self._json(hc.force(q.get("weight_g", [""])[0], "http"))
            if u.path == "/last":
                return self._json(hc.last[0] if hc.last else {})
            if u.path == "/status":
                return self._json({"state": hc.state, "motion": round(hc.motion, 4), "busy": hc.busy,
                                   "n": hc.n, "offline": hc.offline})
            if u.path == "/history":
                return self._json(hc.history[-200:])
            if u.path in ("/", "/dashboard"):
                b = (HERE / "dashboard.html").read_bytes()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers(); self.wfile.write(b); return
            if u.path == "/live.jpg":
                if hc.view_jpg is None:
                    return self.send_error(503)
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store"); self.end_headers()
                self.wfile.write(hc.view_jpg); return
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
    print(f"DASHBOARD: http://localhost:{port}/   (from phone/other laptop: http://<this-laptop-ip>:{port}/)")


def main():
    global MOTION_START, MOTION_STOP, MOTION_PIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0")
    ap.add_argument("--http-port", type=int, default=8001)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--roi", help="x,y,w,h of the bin opening (else press r)")
    ap.add_argument("--start", type=float, help=f"motion fraction that starts a throw (default {MOTION_START})")
    ap.add_argument("--stop", type=float, help=f"motion fraction counted as quiet (default {MOTION_STOP})")
    ap.add_argument("--pix", type=int, help=f"per-pixel change counted as motion (default {MOTION_PIX})")
    a = ap.parse_args()
    MOTION_START = a.start if a.start is not None else MOTION_START
    MOTION_STOP = a.stop if a.stop is not None else MOTION_STOP
    MOTION_PIX = a.pix if a.pix is not None else MOTION_PIX
    print(f"AI: {'OFF (--offline)' if a.offline else API_BASE + '  model ' + MODEL + ('  key OK' if _auth_headers() else '  NO KEY FOUND (food_cam/.env)')}")
    print(f"motion: start>{MOTION_START*100:.2f}%  stop<{MOTION_STOP*100:.2f}%  pixel>{MOTION_PIX}")

    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else None
    cam = Camera(a.camera)
    hc = HandCam(cam, a.offline, roi)
    start_http(hc, a.http_port)
    print("r = set bin region   SPACE = force snap   b = new before   d = debug   q = quit")
    if not roi:
        print(">>> No region set: press r in the camera window and draw a box around the bin opening! <<<")
    debug, last_seq, last_jpg = False, -1, 0.0

    while True:
        f, seq = cam.get()
        if seq != last_seq:
            hc.step(f)
            last_seq = seq
        v = f.copy()
        H, W = v.shape[:2]
        x, y, w, h = hc.roi
        col = (0, 0, 255) if hc.state == hc.THROW else (0, 255, 0)
        cv2.rectangle(v, (x, y), (x + w, y + h), col, 2)
        cv2.putText(v, f"{hc.state}  motion {hc.motion*100:.1f}%  (noise {hc.noise*100:.1f}%, start>{hc.th_start*100:.1f}%)", (x + 8, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        if hc.busy:
            cv2.putText(v, "ANALYSING...", (15, H - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
        if hc.last:
            res, col_img = hc.last
            th = 160
            cw = min(int(col_img.shape[1] * th / col_img.shape[0]), W - 20)
            v[10:10 + th, W - cw - 10:W - 10] = cv2.resize(col_img, (cw, th))
            cv2.rectangle(v, (0, H - 70), (W, H), (0, 0, 0), -1)
            cv2.putText(v, f"#{res['throw_no']} {res['food']}  {res['confidence']:.0%}  {res.get('weight_g') or ''}",
                        (15, H - 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(v, str(res.get("description", ""))[:70], (15, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if debug and hc.mask_dbg is not None:
            m = cv2.cvtColor(cv2.resize(hc.mask_dbg, (320, 180)), cv2.COLOR_GRAY2BGR)
            v[H - 260:H - 80, 10:330] = m
        if time.time() - last_jpg > 0.15:      # ~7 fps for the dashboard
            ok, jb = cv2.imencode(".jpg", cv2.resize(v, (W * 640 // max(W, 1), H * 640 // max(W, 1))),
                                  [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                hc.view_jpg = jb.tobytes()
            last_jpg = time.time()
        cv2.imshow("hand_cam", v)
        k = cv2.waitKey(15) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("d"):
            debug = not debug
        elif k == ord("b"):
            hc.before = f
        elif k == ord("r"):
            r = cv2.selectROI("hand_cam", f, showCrosshair=False)
            if r[2] > 20 and r[3] > 20:
                hc.roi = tuple(int(v) for v in r)
                hc.prev_small = None
                print(f"ROI set: --roi {','.join(map(str, hc.roi))}")
        elif k == ord(" ") and not hc.busy:
            threading.Thread(target=hc.force, daemon=True).start()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
