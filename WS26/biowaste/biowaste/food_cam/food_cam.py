#!/usr/bin/env python3
"""
food_cam.py - Bio Waste Tracker: "what food got thrown away?"

Camera looks down at the bin / tray. On a trigger it compares the new frame
with the last "before" frame, crops the region that changed (= the item that
was just thrown in) and asks a vision AI model what food it is.

Triggers:
  SPACE key in the preview window            (demo by hand)
  HTTP  GET http://localhost:8001/snap       (server.py / ESP32 can call this)
  optional: ?weight_g=212  -> stored with the result

Other keys:  b = re-take "before" frame   q = quit

Run:
  pip install opencv-python numpy
  export ANTHROPIC_API_KEY=sk-ant-...          (Windows: set ANTHROPIC_API_KEY=...)
  python3 food_cam.py --camera 0                          # USB / laptop webcam
  python3 food_cam.py --camera http://PHONE_IP:8080/video  # phone "IP Webcam" app
  python3 food_cam.py --camera 0 --offline                # no API: just test crop/diff

Output:
  captures/food_log.csv   (snap_no, timestamp, item, confidence, description, weight_g, image)
  captures/snap_XXX_crop.jpg + snap_XXX_full.jpg
"""
import argparse, base64, csv, json, os, sys, threading, time, urllib.request, urllib.error
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
MODEL = os.environ.get("FOOD_CAM_MODEL", "claude-haiku-5-5")   # fast + cheap; try claude-sonnet-5-5 if accuracy is low
DIFF_THRESHOLD = 35      # pixel change (0-255) counted as "changed"
MIN_AREA_FRAC = 0.002    # ignore changes smaller than 0.2% of the image
CROP_PAD = 40            # px padding around the changed region
SETTLE_S = 0.6           # wait after trigger so the item has landed
OUT = Path("captures")

PROMPT = f"""You are the vision sensor of a restaurant food-waste tracker.
Image 1 is the bin/tray BEFORE, image 2 is AFTER something was thrown in
(image 2 may be a crop of the changed region). Identify the food that was
just thrown away (what is new in image 2).

Answer ONLY with JSON, no other text:
{{"item": one of {FOOD_CLASSES},
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
    g1 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    # colour difference too (food on similar-brightness background)
    dc = cv2.absdiff(before, after).max(axis=2)
    diff = np.maximum(cv2.absdiff(g1, g2), cv2.GaussianBlur(dc, (21, 21), 0))
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
        return {"item": "other", "confidence": 0.0, "description": "offline mode"}
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"item": "other", "confidence": 0.0, "description": "no ANTHROPIC_API_KEY set"}
    body = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpg_b64(before, 512)}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpg_b64(after_crop)}},
            {"type": "text", "text": PROMPT},
        ]}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = next(c["text"] for c in json.loads(r.read())["content"] if c.get("type") == "text")  # skip thinking blocks
        text = text[text.find("{"): text.rfind("}") + 1]
        res = json.loads(text)
        if res.get("item") not in FOOD_CLASSES:
            res["item"] = "other"
        res["confidence"] = float(res.get("confidence", 0))
        return res
    except urllib.error.HTTPError as e:
        return {"item": "other", "confidence": 0.0, "description": f"API error {e.code}: {e.read()[:120]!r}"}
    except Exception as e:
        return {"item": "other", "confidence": 0.0, "description": f"error: {e}"[:120]}

# ------------------------------------------------------------------ camera ---
class Camera:
    """Grabs frames in a background thread so we always have the latest one."""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if not self.cap.isOpened():
            sys.exit(f"Cannot open camera {src}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame, self.lock = None, threading.Lock()
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
                    self.frame = f
            else:
                time.sleep(0.02)

    def get(self):
        with self.lock:
            return self.frame.copy()

# ------------------------------------------------------------------- core ----
class FoodCam:
    def __init__(self, cam, offline):
        self.cam, self.offline = cam, offline
        self.before = cam.get()
        self.lock = threading.Lock()
        self.last = None          # (result dict, crop image)
        self.busy = False
        OUT.mkdir(exist_ok=True)
        self.log = OUT / "food_log.csv"
        if not self.log.exists():
            with open(self.log, "w", newline="") as f:
                csv.writer(f).writerow(["snap_no", "timestamp", "item", "confidence",
                                        "description", "weight_g", "image", "latency_s"])
        with open(self.log) as f:
            self.n = sum(1 for _ in f) - 1

    def rebase(self):
        self.before = self.cam.get()

    def snap(self, weight_g=""):
        with self.lock:                     # one snap at a time
            self.busy = True
            try:
                time.sleep(SETTLE_S)
                t0 = time.time()
                after = self.cam.get()
                box = changed_region(self.before, after)
                crop = after if box is None else after[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                res = classify(self.before, crop, self.offline)
                dt = time.time() - t0
                self.n += 1
                name = f"snap_{self.n:03d}"
                cv2.imwrite(str(OUT / f"{name}_crop.jpg"), crop)
                full = after.copy()
                if box:
                    x, y, w, h = box
                    cv2.rectangle(full, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.imwrite(str(OUT / f"{name}_full.jpg"), full)
                res.update(snap_no=self.n, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           weight_g=weight_g, image=f"{name}_crop.jpg",
                           region_found=box is not None, latency_s=round(dt, 2))
                with open(self.log, "a", newline="") as f:
                    csv.writer(f).writerow([self.n, res["timestamp"], res["item"],
                                            f'{res["confidence"]:.2f}', res.get("description", ""),
                                            weight_g, res["image"], res["latency_s"]])
                print(f"#{self.n} {res['item']:<18} {res['confidence']:.2f}  "
                      f"{res.get('description','')}  ({dt:.1f}s, region={'yes' if box else 'NO'})")
                self.last = (res, crop)
                self.before = after          # new baseline = current bin content
                return res
            finally:
                self.busy = False

# ------------------------------------------------------------- http server ---
def start_http(fc, port):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/snap":
                out = fc.snap(q.get("weight_g", [""])[0])
            elif u.path == "/rebase":
                fc.rebase(); out = {"ok": True}
            elif u.path == "/last":
                out = fc.last[0] if fc.last else {}
            elif u.path.startswith("/img/"):
                p = OUT / Path(u.path).name
                if p.exists():
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                    self.wfile.write(p.read_bytes()); return
                self.send_error(404); return
            else:
                self.send_error(404); return
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(b)
        def log_message(self, *a): pass
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"HTTP trigger: http://localhost:{port}/snap?weight_g=123")

# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0", help="webcam index or stream URL")
    ap.add_argument("--http-port", type=int, default=8001)
    ap.add_argument("--offline", action="store_true", help="no API calls (test camera + crop)")
    ap.add_argument("--no-window", action="store_true", help="headless, HTTP trigger only")
    a = ap.parse_args()

    fc = FoodCam(Camera(a.camera), a.offline)
    start_http(fc, a.http_port)
    print("SPACE = snap   b = new 'before' frame   q = quit")

    if a.no_window:
        while True: time.sleep(1)

    while True:
        f = fc.cam.get()
        h, w = f.shape[:2]
        if fc.last:
            res, crop = fc.last
            th = 200
            cw = int(crop.shape[1] * th / crop.shape[0])
            cw = min(cw, w // 2)
            f[10:10 + th, w - cw - 10:w - 10] = cv2.resize(crop, (cw, th))
            txt = f"#{res['snap_no']} {res['item']} ({res['confidence']:.0%}) {res.get('weight_g') or ''}"
            cv2.rectangle(f, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.putText(f, txt, (15, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(f, res.get("description", "")[:60], (15, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if fc.busy:
            cv2.putText(f, "ANALYSING...", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
        cv2.imshow("food_cam", f)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("b"):
            fc.rebase(); print("new before-frame")
        if k == ord(" ") and not fc.busy:
            threading.Thread(target=fc.snap, daemon=True).start()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
