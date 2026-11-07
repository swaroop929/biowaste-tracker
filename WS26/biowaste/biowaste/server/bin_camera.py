"""
Bin camera pipeline: lid opens -> BEFORE photo, lid closes -> AFTER photo -> what food went in?

Used by server.py when it is started with --camera. One "lid session" = one opening of the lid:

    lid open    -> session starts, BEFORE photo right away (lamp just switched on by the ESP32)
    lid closed  -> wait SETTLE_S (food stops moving, lamp still on) -> AFTER photo
    worker      -> vision.recognise(before, after): cut out what is new, name each piece
    result      -> data/vision.jsonl (one line per session) + photos in data/vision/<session>/
                   + server links it to the disposal(s) the scale logged during the session

Camera sources (--camera):
    http://192.168.43.1:8080   phone with the IP Webcam app (uses /shot.jpg)
    0, 1, ...                  USB webcam (OpenCV device index)
    replay                     test pairs from tools/vision_bench/binpairs (no hardware needed)
"""
import io
import json
import os
import queue
import sys
import threading
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "firmware", "bin_tracker"))

SETTLE_S = 1.0      # after the lid closes, wait this long before the AFTER photo
LINK_S = 15         # a disposal up to this long after the lid closed still belongs to that session
MAX_OPEN_S = 60     # lid open longer than this -> take the AFTER photo anyway and close the session
CAMERA_WARMUP_S = 0.3  # lamp/auto-exposure needs a moment after the lid opens


# ------------------------------------------------------------------ cameras
class PhoneCamera:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.name = f"phone {self.url}"

    def snapshot(self, phase=None):
        from PIL import Image
        with urllib.request.urlopen(self.url + "/shot.jpg", timeout=3) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")


class UsbCamera:
    def __init__(self, index, width=1280, height=720):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"USB camera {index} not found")
        self.lock = threading.Lock()
        self.name = f"usb camera {index}"

    def snapshot(self, phase=None):
        from PIL import Image
        with self.lock:
            for _ in range(4):           # drop buffered frames: we want the scene NOW
                self.cap.grab()
            ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("USB camera returned no frame")
        return Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))


class ReplayCamera:
    """Plays the before/after test pairs (tools/vision_bench/binpairs) in order: the BEFORE photo of
    pair k at lid-open, its AFTER photo at lid-close. For working on server + dashboard without hardware."""

    def __init__(self, folder=None):
        from PIL import Image
        self.Image = Image
        self.folder = folder or os.path.join(HERE, "..", "tools", "vision_bench", "binpairs")
        meta = os.path.join(self.folder, "pairs.json")
        if not os.path.exists(meta):
            raise RuntimeError("no test pairs - run tools/vision_bench/make_testset.py first")
        self.pairs = json.load(open(meta))
        self.k = -1
        self.name = f"replay ({len(self.pairs)} test pairs)"

    def snapshot(self, phase):
        if phase == "before":
            self.k = (self.k + 1) % len(self.pairs)
        pid = self.pairs[max(self.k, 0)]["id"]
        return self.Image.open(os.path.join(self.folder, f"{pid}_{phase}.jpg")).convert("RGB")

    def truth(self):
        return self.pairs[max(self.k, 0)]["labels"]


def open_camera(spec):
    if spec == "replay":
        return ReplayCamera()
    if spec.startswith("http"):
        return PhoneCamera(spec)
    if spec.isdigit():
        return UsbCamera(int(spec))
    raise ValueError(f"unknown camera '{spec}' (use an IP Webcam URL, a USB index or 'replay')")


# ------------------------------------------------------------------ lid sessions
class LidSessions:
    def __init__(self, camera, data_dir, on_result, model=None):
        self.camera, self.on_result, self.model = camera, on_result, model
        self.dir = os.path.join(data_dir, "vision")
        self.log = os.path.join(data_dir, "vision.jsonl")
        os.makedirs(self.dir, exist_ok=True)
        self.lock = threading.Lock()
        self.current = None                 # open session dict
        self.recent = None                  # (session id, time.time() when closed)
        self.jobs = queue.Queue()
        self.status = "loading model"
        self.model_name = ""
        self.last = None                    # last finished session
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()

    # --- events from the ESP32 (or /api/lid, or the simulator)
    def lid(self, state):
        if state == "open":
            self._open()
        elif state == "closed":
            self._close("lid closed")

    def _open(self):
        with self.lock:
            if self.current:
                return                      # already open (bouncing switch)
            now = datetime.now()
            sid = now.strftime("%Y%m%d-%H%M%S")
            self.current = {"session": sid, "opened": now.isoformat(timespec="seconds"), "closed": None,
                            "before": None, "status": "open"}
            s = self.current
        threading.Thread(target=self._shoot, args=(s, "before", CAMERA_WARMUP_S), daemon=True).start()

    def _close(self, why):
        with self.lock:
            s, self.current = self.current, None
            if s:
                self.recent = (s["session"], time.time())
        if not s:
            return
        s["closed"] = datetime.now().isoformat(timespec="seconds")
        s["close_reason"] = why
        s["status"] = "processing"
        threading.Thread(target=self._shoot, args=(s, "after", SETTLE_S), daemon=True).start()

    def _shoot(self, s, phase, delay):
        time.sleep(delay)
        try:
            img = self.camera.snapshot(phase)
            os.makedirs(os.path.join(self.dir, s["session"]), exist_ok=True)
            img.save(os.path.join(self.dir, s["session"], f"{phase}.jpg"), quality=90)
            s[phase] = img
            if phase == "after" and isinstance(self.camera, ReplayCamera):
                s["truth"] = self.camera.truth()   # only in replay mode: the known answer
        except Exception as ex:             # camera unplugged, phone asleep, ...
            s.setdefault("errors", []).append(f"{phase}: {ex.__class__.__name__}: {ex}")
        if phase == "after":
            self.jobs.put(s)

    def _watchdog(self):
        while True:
            time.sleep(1)
            with self.lock:
                s = self.current
            if s and (datetime.now() - datetime.fromisoformat(s["opened"])).total_seconds() > MAX_OPEN_S:
                self._close("lid open too long")

    # --- recognition
    def _worker(self):
        import vision
        try:
            m = vision.load_model(self.model)
            self.model_name = m["name"]
            self.status = "ready"
        except Exception as ex:
            self.status = f"model failed: {ex.__class__.__name__}: {ex}"
            print("vision:", self.status)
        while True:
            s = self.jobs.get()
            self.status = "processing"
            rec = {"session": s["session"], "opened": s["opened"], "closed": s["closed"],
                   "close_reason": s.get("close_reason"), "foods": [], "food_conf": {}, "parts": [],
                   "errors": s.get("errors", [])}
            if s.get("before") is not None and s.get("after") is not None and not self.status.startswith("model"):
                t0 = time.perf_counter()
                try:
                    r = vision.recognise(s["before"], s["after"],
                                         save_dir=os.path.join(self.dir, s["session"]), tag="piece")
                    rec.update({k: r[k] for k in ("foods", "food_conf", "model")})
                    rec["parts"] = [{"box": p["box"], "foods": p["foods"], "image": f"piece_{k}.jpg"}
                                    for k, p in enumerate(r["parts"])]
                except Exception as ex:
                    rec["errors"].append(f"recognise: {ex.__class__.__name__}: {ex}")
                rec["ms"] = round(1000 * (time.perf_counter() - t0))
            if "truth" in s:
                rec["truth"] = s["truth"]
            with open(self.log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.last = rec
            self.status = "ready"
            print(f"[vision] {rec['session']}: {', '.join(rec['foods']) or 'nothing recognised'}"
                  f"{'  ERR ' + '; '.join(rec['errors']) if rec['errors'] else ''}")
            self.on_result(rec)

    def session_for_disposal(self):
        """Which lid session a disposal the scale just logged belongs to (None = no lid opening)."""
        with self.lock:
            if self.current:
                return self.current["session"]
            if self.recent and time.time() - self.recent[1] <= LINK_S:
                return self.recent[0]
        return None

    def state(self):
        with self.lock:
            cur = self.current
        return {"enabled": True, "camera": self.camera.name, "model": self.model_name, "status": self.status,
                "lid": "open" if cur else "closed", "last": self.last}


def load_sessions(data_dir):
    """All finished sessions from earlier runs (to restore the food labels after a restart)."""
    p = os.path.join(data_dir, "vision.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out
