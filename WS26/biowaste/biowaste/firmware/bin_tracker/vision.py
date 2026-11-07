#!/usr/bin/env python3
"""
Food recognition from the phone camera (IP Webcam app) - runs locally on the laptop, no cloud.

Recognising the food does NOT give the waste type (origin) - buttons/scale do that.
The camera adds a "food" field so the insight can name the item: "62 % of prep waste is pasta".

Setup
    pip install torch transformers pillow numpy
    Phone: IP Webcam app -> Start server. Phone looks straight DOWN at the tray only (no faces).
    Laptop + phone on the same network (safest: laptop joins the phone's hotspot).

Modes
    python vision.py --image apple.jpg                 classify one photo (no phone needed)
    python vision.py --phone http://192.168.43.1:8080  KEY-PRESS mode (default): throw something in,
                                                        press Enter -> it names what was added
    python vision.py --phone ... --watch               automatic mode: detects each new item by motion
    python vision.py --phone ... --teach apple         save 5-10 reference crops of cut apple; afterwards
                                                        recognition compares every crop with them too

Output: vision/vision_log.csv (timestamp, food, food_conf, top3, image_id, changed_pct)
Privacy: only the cropped changed region is saved (vision/crops/), never the full frame.
"""
import argparse
import csv
import io
import os
import time
import urllib.request
from datetime import datetime

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CROPS = os.path.join(HERE, "crops")
LOG = os.path.join(HERE, "vision_log.csv")

# label -> list of descriptions (averaged = "prompt ensemble", more robust than one sentence).
# Edit freely: add a label, or add descriptions to fix confusions. Fewer labels = fewer mix-ups,
# so comment out what your kitchen never throws away.
FOODS = {
    "pasta": ["a photo of dry uncooked pasta", "a photo of penne or fusilli pasta",
              "a photo of cooked spaghetti", "a photo of cooked pasta with sauce"],
    "rice": ["a photo of cooked white rice", "a photo of uncooked rice grains",
             "a photo of a pile of rice"],
    "bread": ["a photo of bread slices", "a photo of a bread roll", "a photo of bread crust",
              "a photo of a piece of baguette"],
    "apple": ["a photo of an apple", "a photo of apple pieces", "a photo of an apple core",
              "a photo of apple slices with white flesh and red skin", "a photo of apple wedges",
              "a photo of half an apple cut open", "a photo of browned apple pieces"],
    "orange": ["a photo of an orange", "a photo of orange peel", "a photo of orange slices"],
    "banana": ["a photo of a banana", "a photo of a banana peel"],
    "potato": ["a photo of potatoes", "a photo of potato peels", "a photo of french fries",
               "a photo of mashed potatoes"],
    "salad": ["a photo of lettuce leaves", "a photo of green salad", "a photo of cucumber slices"],
    "vegetable_trimmings": ["a photo of vegetable peels and trimmings", "a photo of carrot peels",
                            "a photo of onion skins", "a photo of tomato pieces"],
    "meat_fish": ["a photo of meat leftovers", "a photo of chicken bones", "a photo of a piece of fish"],
    "eggshells": ["a photo of eggshells"],
    "coffee_grounds": ["a photo of used coffee grounds"],
    "non_food": ["a photo of a paper napkin", "a photo of plastic packaging", "a photo of a plastic cup",
                 "a photo of aluminium foil"],
    "empty": ["a photo of an empty white tray", "a photo of an empty white plastic bag",
              "a photo of an empty white surface"],
}
MIN_CONF = 0.35        # below -> "unknown" (honest beats wrong). More labels -> lower scores, so 0.35

# watch-mode tuning
PIX_THRESH = 25        # colour difference (0-255, strongest channel) that counts as "changed pixel"
MOTION_PCT = 0.6       # % changed pixels frame-to-frame = something is moving (hand)
CHANGE_PCT = 0.3       # % changed pixels vs reference = something new on the tray
STABLE_FRAMES = 4      # frames without motion before we look (~1-1.5 s)
REBASE_S = 20          # idle this long -> refresh reference (slow light changes)
BURST = 3              # frames averaged per recognition

# Teach-by-example: reference crops in vision/refs/<label>/*.jpg (made with --teach <label>,
# or by moving wrongly recognised crops from vision/crops/ into the right refs folder).
REFS = os.path.join(HERE, "refs")
REF_SIM = 0.80         # cosine similarity a crop needs to a reference before refs get a vote
REF_WEIGHT = 0.6       # share of the reference vote in the final score (rest = text descriptions)
REF_TEMP = 50.0        # sharpness of the reference vote

# Model: "openai/clip-vit-base-patch32" (600 MB, ~0.2 s) or "openai/clip-vit-large-patch14"
# (1.7 GB, ~1 s per image on CPU, clearly better on hard items like cut fruit). Override with --model.
MODEL = "openai/clip-vit-large-patch14"

_model = _proc = None
_refs = {}             # label -> tensor [n, dim] of normalised image embeddings


def load_model():
    global _model, _proc
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor  # first run downloads ~600 MB
        print(f"model: {MODEL}")
        _model = CLIPModel.from_pretrained(MODEL).eval()
        _proc = CLIPProcessor.from_pretrained(MODEL)
        load_refs()
    return _model, _proc


def load_refs():
    import torch
    _refs.clear()
    if not os.path.isdir(REFS):
        return
    for label in sorted(os.listdir(REFS)):
        folder = os.path.join(REFS, label)
        if label not in FOODS or not os.path.isdir(folder):
            continue
        embs = []
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                _, emb = _forward(Image.open(os.path.join(folder, fn)).convert("RGB"))
                embs.append(emb)
        if embs:
            _refs[label] = torch.stack(embs)
    if _refs:
        print("reference photos: " + ", ".join(f"{l} {len(e)}" for l, e in _refs.items()))


# Direct connection to the phone - ignore Windows/university proxy settings
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def snapshot(phone_url):
    with _opener.open(phone_url.rstrip("/") + "/shot.jpg", timeout=5) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def _small(img, w=320):
    h = int(img.height * w / img.width)
    return np.asarray(img.convert("RGB").resize((w, h)), dtype=np.int16)


ROI_FILE = os.path.join(HERE, "roi.json")
ROI = None             # (x0, y0, x1, y1) as fractions of the frame; only this area is analysed


def load_roi():
    global ROI
    if os.path.exists(ROI_FILE):
        import json
        ROI = tuple(json.load(open(ROI_FILE)))
        print(f"area of interest: {ROI}")


def save_roi(roi):
    import json
    global ROI
    ROI = roi
    if roi is None:
        if os.path.exists(ROI_FILE):
            os.remove(ROI_FILE)
    else:
        json.dump([round(v, 3) for v in roi], open(ROI_FILE, "w"))


def _roi_mask(h, w):
    m = np.zeros((h, w), bool)
    if ROI is None:
        m[:] = True
    else:
        x0, y0, x1, y1 = ROI
        m[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = True
    return m


def _changed_mask(a, b):
    """Per-pixel change inside the area of interest.
    - strongest colour channel: pale items on white (cut apple, bread, rice) differ in colour more than brightness
    - the median shift is subtracted first: the phone's auto-exposure / white balance changing between
      shots moves ALL pixels a bit - that is not a new item."""
    A = _small(a)
    B = _small(b.resize(a.size) if a.size != b.size else b)
    d = B - A
    roi = _roi_mask(*d.shape[:2])
    shift = np.median(d[roi], axis=0)
    return (np.abs(d - shift).max(axis=2) > PIX_THRESH) & roi


def auto_roi(frame, pad=0.06):
    """Find the white paper / bag (largest bright, colourless area) -> area of interest."""
    A = _small(frame)
    grey, sat = A.mean(axis=2), A.max(axis=2) - A.min(axis=2)
    lab, blobs = _label((grey > 140) & (sat < 45))
    if not blobs:
        return None
    x0, y0, x1, y1, area = max(blobs, key=lambda b: b[4])
    h, w = grey.shape
    if area < 0.03 * h * w:
        return None
    return (float(max(0.0, x0 / w - pad)), float(max(0.0, y0 / h - pad)),
            float(min(1.0, (x1 + 1) / w + pad)), float(min(1.0, (y1 + 1) / h + pad)))


def diff_pct(a, b):
    """% of pixels that changed between two frames."""
    return 100.0 * _changed_mask(a, b).mean()


def changed_region(before, after, pad=20):
    """Bounding box (full-res coords) of what changed. None if nothing changed."""
    a = _small(before)
    mask = _changed_mask(before, after)
    if 100.0 * mask.mean() < CHANGE_PCT:
        return None
    ys, xs = np.where(mask)
    x0, x1 = np.percentile(xs, [2, 98])
    y0, y1 = np.percentile(ys, [2, 98])
    sx, sy = after.width / a.shape[1], after.height / a.shape[0]
    return (max(0, int(x0 * sx) - pad), max(0, int(y0 * sy) - pad),
            min(after.width, int(x1 * sx) + pad), min(after.height, int(y1 * sy) + pad))


MIN_BLOB_PCT = 0.08    # blobs smaller than this % of the image are noise / pile settling
MERGE_PX = 4           # pixels (at 320 px width) to grow blobs so pieces of one item join up


def _shift_or(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy or dx:
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def _label(mask):
    """Connected components (4-neighbour) -> (label image, list of (x0, y0, x1, y1, area)); label k+1 = blob k."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    out = []
    for y, x in zip(*np.nonzero(mask)):
        if lab[y, x]:
            continue
        k = len(out) + 1
        stack = [(y, x)]
        lab[y, x] = k
        x0 = x1 = x
        y0 = y1 = y
        area = 0
        while stack:
            cy, cx = stack.pop()
            area += 1
            x0, x1, y0, y1 = min(x0, cx), max(x1, cx), min(y0, cy), max(y1, cy)
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = k
                    stack.append((ny, nx))
        out.append((x0, y0, x1, y1, area))
    return lab, out


def changed_items(before, after, pad=12):
    """One (box, mask) per new thing, largest first. box = full-res crop box,
    mask = full-res 'L' image (255 = belongs to this item) used to white out everything else."""
    from PIL import ImageFilter
    raw = _changed_mask(before, after)
    clean = ~_shift_or(~raw, 1)                 # erode: drop single-pixel speckle
    grown = _shift_or(clean, MERGE_PX)          # dilate: join pieces of one item
    h, w = raw.shape
    min_area = MIN_BLOB_PCT / 100.0 * h * w
    sx, sy = after.width / w, after.height / h
    lab, blobs = _label(grown)
    items = []
    for k, (x0, y0, x1, y1, area) in enumerate(blobs, start=1):
        real = int((clean & (lab == k)).sum())  # area before growing
        if real < min_area:
            continue
        box = (max(0, int(x0 * sx) - pad), max(0, int(y0 * sy) - pad),
               min(after.width, int((x1 + 1) * sx) + pad), min(after.height, int((y1 + 1) * sy) + pad))
        m = Image.fromarray(((lab == k) * 255).astype(np.uint8)).resize(after.size, Image.NEAREST)
        m = m.filter(ImageFilter.MaxFilter(9))  # a little margin around the item
        items.append((real, box, m))
    items.sort(key=lambda t: -t[0])
    return [(box, m) for _, box, m in items]


def cutout(frame, box, mask):
    """Crop to box and paint everything that is not this item white (neighbours can't leak in)."""
    crop = frame.crop(box)
    return Image.composite(crop, Image.new("RGB", crop.size, (255, 255, 255)), mask.crop(box))


def _forward(img):
    """One CLIP pass: per-label text scores (averaged over descriptions) + normalised image embedding."""
    import torch
    model, proc = _model, _proc
    labels = list(FOODS)
    prompts, owner = [], []
    for i, l in enumerate(labels):
        descs = FOODS[l] if isinstance(FOODS[l], list) else [FOODS[l]]
        prompts += descs
        owner += [i] * len(descs)
    inputs = proc(text=prompts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits_per_image[0]                          # one score per description
    owner_t = torch.tensor(owner)
    per_label = torch.stack([logits[owner_t == i].mean() for i in range(len(labels))])
    return per_label, out.image_embeds[0]


def class_probs(img):
    """Text descriptions vote; if reference photos exist and one is similar enough, they vote too."""
    import torch
    load_model()
    labels = list(FOODS)
    per_label, emb = _forward(img)
    probs = dict(zip(labels, per_label.softmax(dim=0).tolist()))
    if _refs:
        sims = {l: float((e @ emb).max()) for l, e in _refs.items()}
        if max(sims.values()) >= REF_SIM:
            ls = list(sims)
            rp = torch.tensor([sims[l] for l in ls]).mul(REF_TEMP).softmax(dim=0).tolist()
            ref_probs = dict(zip(ls, rp))
            probs = {l: (1 - REF_WEIGHT) * probs[l] + REF_WEIGHT * ref_probs.get(l, 0.0) for l in labels}
    return probs


def decide(probs):
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    best, conf = ranked[0]
    if best != "empty" and conf < MIN_CONF:
        best = "unknown"
    return best, conf, ranked[:3]


def recognise(frames, reference, save=True):
    """frames: 1..n images of the same scene (burst). Crops to what changed vs reference."""
    box = changed_region(reference, frames[-1])
    crops = [f.crop(box) if box else f for f in frames]
    probs = [class_probs(c) for c in crops]
    avg = {k: sum(p[k] for p in probs) / len(probs) for k in FOODS}
    food, conf, top3 = decide(avg)
    image_id = ""
    if save:
        os.makedirs(CROPS, exist_ok=True)
        image_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
        crops[-1].save(os.path.join(CROPS, image_id), quality=80)
    return {"food": food, "food_conf": round(conf, 2), "top3": top3, "box": box, "image_id": image_id}


def recognise_items(frames, reference, save=True):
    """One result per new item. Same-food blobs (scattered pasta) are merged into one item."""
    parts = changed_items(reference, frames[-1])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    found = []
    for box, mask in parts:
        crops = [cutout(f, box, mask) for f in frames]
        probs = [class_probs(c) for c in crops]
        avg = {k: sum(p[k] for p in probs) / len(probs) for k in FOODS}
        food, conf, top3 = decide(avg)
        found.append({"food": food, "food_conf": round(conf, 2), "top3": top3, "box": box, "crop": crops[-1]})
    merged = {}
    for r in found:
        m = merged.get(r["food"])
        if m is None:
            merged[r["food"]] = dict(r, boxes=[r["box"]])
        else:
            m["boxes"].append(r["box"])
            if r["food_conf"] > m["food_conf"]:
                m["food_conf"], m["crop"] = r["food_conf"], r["crop"]
            b = m["box"]
            m["box"] = (min(b[0], r["box"][0]), min(b[1], r["box"][1]), max(b[2], r["box"][2]), max(b[3], r["box"][3]))
    items = list(merged.values())
    if save:
        os.makedirs(CROPS, exist_ok=True)
        for i, r in enumerate(items):
            r["image_id"] = "" if r["food"] == "empty" else f"{stamp}_{i}.jpg"
            if r["image_id"]:
                r["crop"].save(os.path.join(CROPS, r["image_id"]), quality=90)   # the cut-out the model saw
    return items


COLORS = {"empty": (150, 150, 150), "unknown": (200, 120, 0), "non_food": (220, 40, 40)}


def annotate(frame, items):
    from PIL import ImageDraw
    img = frame.copy()
    d = ImageDraw.Draw(img)
    if ROI is not None:
        W, H = img.size
        d.rectangle((ROI[0] * W, ROI[1] * H, ROI[2] * W, ROI[3] * H), outline=(60, 120, 230), width=2)
    for r in items:
        c = COLORS.get(r["food"], (20, 160, 90))
        for b in r.get("boxes", [r["box"]]):
            d.rectangle(b, outline=c, width=3)
        x0, y0 = r["box"][0], max(0, r["box"][1] - 18)
        text = f"{r['food']} {r['food_conf']:.0%}"
        d.rectangle((x0, y0, x0 + 8 * len(text) + 8, y0 + 16), fill=c)
        d.text((x0 + 4, y0 + 2), text, fill=(255, 255, 255))
    return img


def log(r, changed):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "food", "food_conf", "top3", "image_id", "changed_pct"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), r["food"], r["food_conf"],
                    "; ".join(f"{l}:{p:.2f}" for l, p in r["top3"]), r["image_id"], round(changed, 2)])


def show(r, ms):
    top = ", ".join(f"{l} {p:.0%}" for l, p in r["top3"])
    print(f"  -> {r['food'].upper():20s} {r['food_conf']:.0%}   [{top}]   {ms:.0f} ms   {r['image_id']}")


def teach_save(label, frame, ref):
    parts = changed_items(ref, frame)
    if not parts:
        print("  (nothing new to save)")
        return
    folder = os.path.join(REFS, label)
    os.makedirs(folder, exist_ok=True)
    fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
    cutout(frame, *parts[0]).save(os.path.join(folder, fn), quality=90)
    n = len([x for x in os.listdir(folder) if x.endswith(".jpg")])
    print(f"  saved reference #{n} for '{label}'  ({fn}) - next one, or Ctrl+C when you have 5-10")


_view = {"jpg": b"", "events": []}


def set_view(frame, items=None, note=None):
    buf = io.BytesIO()
    annotate(frame, items or []).save(buf, "JPEG", quality=80)
    _view["jpg"] = buf.getvalue()
    if note:
        _view["events"] = ([{"t": datetime.now().strftime("%H:%M:%S"), "msg": note}] + _view["events"])[:15]


VIEW_HTML = b"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Bin camera</title>
<style>body{font:15px system-ui;margin:16px;background:#f3f2ef;color:#111}img{max-width:100%;border-radius:10px}
.wrap{display:flex;gap:16px;flex-wrap:wrap}.ev{min-width:260px}li{margin:4px 0;font-variant-numeric:tabular-nums}</style>
</head><body><h2 style="margin:0 0 10px">What the bin camera sees</h2><div class="wrap">
<img id="cam" src="/latest.jpg"><div class="ev"><b>Detected</b><ul id="ev"></ul></div></div>
<script>setInterval(async()=>{document.getElementById('cam').src='/latest.jpg?'+Date.now();
const e=await (await fetch('/events')).json();document.getElementById('ev').innerHTML=e.map(x=>`<li>${x.t} &nbsp;${x.msg}</li>`).join('')},1000)</script>
</body></html>"""


def start_viewer(port):
    import json
    import threading
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            body, ctype = {"/": (VIEW_HTML, "text/html"), "/latest.jpg": (_view["jpg"], "image/jpeg"),
                           "/events": (json.dumps(_view["events"]).encode(), "application/json")}.get(path, (b"", ""))
            self.send_response(200 if ctype else 404)
            self.send_header("Content-Type", ctype or "text/plain")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever, daemon=True).start()
    print(f"live view: http://localhost:{port}")


def watch(phone, teach=None):
    ref = snapshot_retry(phone)
    print("Watching the tray. Put items on it one at a time, hands out of view afterwards. Ctrl+C to stop.")
    prev, still, moving, idle_since = ref, 0, False, time.time()
    while True:
        try:
            f = snapshot(phone)
        except Exception as ex:
            print(f"  camera error: {ex} - retrying")
            time.sleep(1)
            continue
        motion = diff_pct(prev, f) > MOTION_PCT
        prev = f
        if not moving and not _view["jpg"] or motion:
            set_view(f)
        if motion:
            if not moving:
                print("  motion...")
            moving, still = True, 0
            continue
        if moving:
            still += 1
            if still < STABLE_FRAMES:
                continue
            moving = False
            changed = diff_pct(ref, f)
            if changed < CHANGE_PCT:
                print("  (nothing new on the tray)")
            elif teach:
                teach_save(teach, f, ref)
            else:
                t = time.time()
                frames = [f] + [snapshot(phone) for _ in range(BURST - 1)]
                items = recognise_items(frames, ref)
                ms = 1000 * (time.time() - t)
                if not items:
                    print("  (change too small / only noise)")
                for r in items:
                    if r["food"] == "empty":
                        print("  tray cleared")
                    else:
                        show(r, ms)
                    log(r, changed)
                f = frames[-1]
                set_view(f, items, ", ".join(f"{r['food']} {r['food_conf']:.0%}" for r in items) or None)
            ref, idle_since = f, time.time()
        elif time.time() - idle_since > REBASE_S:
            ref, idle_since = f, time.time()      # follow slow lighting changes
        time.sleep(0.15)


def snapshot_retry(phone):
    while True:
        try:
            return snapshot(phone)
        except Exception as ex:
            print(f"  can't reach camera at {phone}/shot.jpg ({ex}). Check it in a browser. Retrying...")
            time.sleep(2)


def manual(phone):
    print("""
KEY-PRESS mode - one picture per disposal, compared with the previous picture.
  Enter              throw something in, hand out of view, then press Enter -> what was added
  r                  new reference (tray emptied / bag changed / camera moved)
  roi                find the white paper and only look there (ignores walls + reflections) - do this first
  roi off            look at the whole picture again
  c apple            last result was wrong, it was 'apple' -> its cut-out becomes a reference photo
  c 2 apple          same, for the 2nd item of the last result
  q                  quit
vision/last.jpg (and the live view) shows each detected item with a box; blue = area of interest.""")
    ref = snapshot_retry(phone)
    set_view(ref)
    annotate(ref, []).save(os.path.join(HERE, "last.jpg"), quality=80)
    print("Reference taken. Ready." + ("" if ROI else "  Tip: type 'roi' with an empty tray first."))
    last_items = []
    n = 0
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        if cmd == "r":
            ref = snapshot_retry(phone)
            set_view(ref, note="new reference")
            print("  new reference taken")
            continue
        if cmd == "roi off":
            save_roi(None)
            print("  looking at the whole picture")
            continue
        if cmd == "roi":
            f = snapshot_retry(phone)
            roi = auto_roi(f)
            if roi is None:
                print("  couldn't find the white paper - is the tray empty and lit?")
            else:
                save_roi(roi)
                annotate(f, []).save(os.path.join(HERE, "last.jpg"), quality=80)
                set_view(f, note="area of interest set")
                print(f"  area of interest set {tuple(round(v, 2) for v in roi)} - check the blue box in last.jpg")
                ref = f
            continue
        if cmd.startswith("c "):
            parts = cmd.split()
            idx = int(parts[1]) - 1 if len(parts) == 3 and parts[1].isdigit() else 0
            label = parts[-1]
            if label not in FOODS:
                print(f"  unknown label. Labels: {', '.join(FOODS)}")
            elif not 0 <= idx < len(last_items):
                print("  nothing to correct")
            else:
                folder = os.path.join(REFS, label)
                os.makedirs(folder, exist_ok=True)
                last_items[idx]["crop"].save(os.path.join(folder, last_items[idx]["image_id"] or "corr.jpg"), quality=90)
                load_refs()
                print(f"  saved as reference for '{label}' - used from the next item on")
            continue
        if cmd:
            print("  unknown command (Enter / r / roi / c <label> / q)")
            continue
        frames = [snapshot_retry(phone) for _ in range(BURST)]
        changed = diff_pct(ref, frames[-1])
        if changed < CHANGE_PCT:
            print(f"  nothing new compared with the last picture ({changed:.1f}% changed)")
            set_view(frames[-1])
            continue
        t = time.time()
        items = recognise_items(frames, ref)
        ms = 1000 * (time.time() - t)
        annotate(frames[-1], items).save(os.path.join(HERE, "last.jpg"), quality=80)
        food_items = [r for r in items if r["food"] != "empty"]
        if not items:
            print("  change too small / only noise")
        elif not food_items:
            print("  tray looks emptied -> new reference")
        else:
            n += 1
            for i, r in enumerate(food_items):
                print(f"  #{n}{'abcdefgh'[i] if len(food_items) > 1 else ''}", end="")
                show(r, ms)
        for r in items:
            log(r, changed)
        set_view(frames[-1], items, ", ".join(f"{r['food']} {r['food_conf']:.0%}" for r in food_items) or None)
        last_items = food_items
        ref = frames[-1]          # this picture is the "before" for the next disposal


def main():
    global MODEL, BURST
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone", help="IP Webcam address, e.g. http://192.168.43.1:8080")
    ap.add_argument("--image", help="classify a single image file")
    ap.add_argument("--watch", action="store_true", help="automatic: detect each new item by motion")
    ap.add_argument("--model", help="e.g. openai/clip-vit-base-patch32 (fast) or openai/clip-vit-large-patch14 (accurate)")
    ap.add_argument("--burst", type=int, help=f"frames averaged per item (default {BURST}); 1 = fastest")
    ap.add_argument("--view-port", type=int, default=8001, help="live camera view in the browser (0 = off)")
    ap.add_argument("--teach", metavar="LABEL", help="save reference photos for LABEL, e.g. --teach apple")
    a = ap.parse_args()
    if a.model:
        MODEL = a.model
    if a.burst:
        BURST = a.burst
    print("loading model (first run downloads ~600 MB)...")
    load_model()
    load_roi()
    if a.image:
        img = Image.open(a.image).convert("RGB")
        t = time.time()
        food, conf, top3 = decide(class_probs(img))
        show({"food": food, "food_conf": conf, "top3": top3, "image_id": ""}, 1000 * (time.time() - t))
    elif a.phone and a.view_port:
        start_viewer(a.view_port)
    if a.image:
        pass
    elif a.phone and a.teach:
        if a.teach not in FOODS:
            ap.error(f"unknown label '{a.teach}'. Labels: {', '.join(FOODS)}")
        print(f"TEACH mode for '{a.teach}': put one piece at a time on the tray, hand away, vary position and angle.")
        watch(a.phone, teach=a.teach)
    elif a.phone:
        (watch if a.watch else manual)(a.phone)
    else:
        ap.error("give --phone <url> or --image <file>")


if __name__ == "__main__":
    main()
