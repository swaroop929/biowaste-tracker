#!/usr/bin/env python3
"""
Small before/after test set for the bin camera: "what food was just thrown in?"  ->  binpairs/

The camera sits inside the bin and looks down on the waste. Lid opens -> BEFORE photo,
food goes in -> AFTER photo. Real before/after photos from inside a food-waste bin do not exist
online, so each pair is built from real pieces:
  * BEFORE = a real photo looking down on food waste in a bin / caddy (backgrounds/, 4 photos)
  * AFTER  = the same photo with 1-2 real food cut-outs dropped on top (items: FoodSeg103 masks
             and onion skins from a photo), with a drop shadow, plus what really happens between
             the two photos: the bin gets bumped (a few pixels shift, ~1 deg turn), the light
             changes a little, sensor noise. Half the pairs are dark with a warm LED cast,
             like the inside of a closed bin with a small lamp.
Ground truth per pair: mask of the new food (pixel exact) and its labels (from vision.FOODS).

    python make_testset.py
"""
import io
import json
import os
import random
import sys
import urllib.request

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "binpairs")
W = 800  # working width of the test photos

# real photos looking down on food waste (name -> url); crop = part of the photo that is bin content
BACKGROUNDS = {
    "caddy_scraps": ("https://images.pexels.com/photos/14824327/pexels-photo-14824327.jpeg?auto=compress&w=1200",
                     (0.0, 0.0, 0.75, 1.0)),
    "carrot_bowl": ("https://images.pexels.com/photos/5740584/pexels-photo-5740584.jpeg?auto=compress&w=1200",
                    (0.22, 0.12, 0.80, 0.92)),
    "peel_pile": ("https://images.unsplash.com/photo-1614272527834-e95f090a34c5?w=1200&q=85",
                  (0.0, 0.0, 1.0, 1.0)),
    "bin_scraps": ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Biodegradable_waste.jpg/960px-Biodegradable_waste.jpg",
                   (0.0, 0.0, 1.0, 1.0)),
}
ONION_SKINS = "https://images.unsplash.com/photo-1760445529166-1184fc7bb808?w=1200&q=85"

FOODSEG_TO_FOODS = {  # FoodSeg103 class -> label in vision.FOODS (only foods vision.py knows)
    "bread": "bread", "rice": "rice", "pasta": "pasta", "noodles": "pasta", "potato": "potatoes",
    "apple": "apple", "lettuce": "lettuce leaves", "cucumber": "cucumber", "tomato": "tomato",
    "steak": "cooked meat", "pork": "cooked meat", "chicken duck": "cooked meat", "sausage": "cooked meat",
    "fish": "fish", "cheese butter": "cheese", "broccoli": "vegetable trimmings",
}


def fetch(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "biowaste-tracker test set (student project)"})
        open(path, "wb").write(urllib.request.urlopen(req, timeout=60).read())
    return Image.open(path).convert("RGB")


def load_backgrounds():
    d = os.path.join(OUT, "backgrounds")
    os.makedirs(d, exist_ok=True)
    out = {}
    for name, (url, (x0, y0, x1, y1)) in BACKGROUNDS.items():
        im = fetch(url, os.path.join(d, f"{name}.jpg"))
        w, h = im.size
        im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        out[name] = im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)
    return out


def onion_skin_cutouts():
    """Onion skins on a plain beige table -> cut out by colour distance to the table."""
    im = np.asarray(fetch(ONION_SKINS, os.path.join(OUT, "items", "onion_skins.jpg")))
    h, w = im.shape[:2]
    im = im[:, : int(w * 0.62)]                      # leave out the red tin on the right
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)
    table = np.median(lab.reshape(-1, 3), axis=0)
    m = (np.linalg.norm(lab - table, axis=2) > 22).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(m)
    cuts = []
    for k in np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:6] + 1:
        x, y, ww, hh, area = stats[k]
        if area < 0.004 * m.size:
            continue
        alpha = cv2.GaussianBlur((comp == k).astype(np.float32), (5, 5), 0)
        rgba = np.dstack([im, (alpha * 255).astype(np.uint8)])[y:y + hh, x:x + ww]
        cuts.append(("onion skin", "onion skin", Image.fromarray(rgba, "RGBA")))
    return cuts


def foodseg_cutouts(n=36, seed=1):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    id2label = {int(k): v.strip() for k, v in json.load(open(hf_hub_download(
        "EduardoPacheco/FoodSeg103", "id2label.json", repo_type="dataset"))).items()}
    t = pq.read_table(hf_hub_download("EduardoPacheco/FoodSeg103", "data/validation-00000-of-00001.parquet",
                                      repo_type="dataset"))
    idx = list(range(t.num_rows))
    random.Random(seed).shuffle(idx)
    cuts, per = [], {}
    for i in idx:
        mask = np.asarray(Image.open(io.BytesIO(t.column("label")[i].as_py()["bytes"])))
        mask = mask[..., 0] if mask.ndim == 3 else mask
        img = None
        for c in np.unique(mask):
            name = id2label.get(int(c), "")
            if name not in FOODSEG_TO_FOODS or per.get(name, 0) >= 3:
                continue
            nl, comp, stats, _ = cv2.connectedComponentsWithStats((mask == c).astype(np.uint8))
            if nl < 2:
                continue
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, w, h, area = stats[k]
            if not 0.05 <= area / mask.size <= 0.45:
                continue
            if img is None:
                img = np.asarray(Image.open(io.BytesIO(t.column("image")[i].as_py()["bytes"])).convert("RGB"))
            alpha = cv2.GaussianBlur((comp == k).astype(np.float32), (5, 5), 0)
            rgba = np.dstack([img, (alpha * 255).astype(np.uint8)])[y:y + h, x:x + w]
            cuts.append((FOODSEG_TO_FOODS[name], name, Image.fromarray(rgba, "RGBA")))
            per[name] = per.get(name, 0) + 1
        if len(cuts) >= n:
            break
    return cuts


def drop(base, cut, rnd):
    """Drop an RGBA cut-out somewhere on the waste, with a soft contact shadow."""
    Wb, Hb = base.size
    c = cut.copy()
    s = rnd.uniform(0.22, 0.38) * Wb / max(c.size)
    c = c.resize((max(8, int(c.width * s)), max(8, int(c.height * s))), Image.LANCZOS)
    c = c.rotate(rnd.uniform(0, 360), expand=True, resample=Image.BICUBIC)
    px = int(rnd.uniform(0.05 * Wb, max(0.05 * Wb + 1, 0.95 * Wb - c.width)))
    py = int(rnd.uniform(0.05 * Hb, max(0.05 * Hb + 1, 0.95 * Hb - c.height)))
    out = base.convert("RGBA")
    sh = Image.new("RGBA", c.size, (0, 0, 0, 0))
    sh.putalpha(c.getchannel("A").point(lambda v: int(v * 0.5)))
    out.alpha_composite(sh, (min(Wb - 1, px + 7), min(Hb - 1, py + 9)))
    out.alpha_composite(c, (px, py))
    m = np.zeros((Hb, Wb), np.uint8)
    a = np.asarray(c.getchannel("A")) > 127
    hh, ww = min(a.shape[0], Hb - py), min(a.shape[1], Wb - px)
    m[py:py + hh, px:px + ww] |= a[:hh, :ww]
    return out.convert("RGB"), m


def bin_light(img, strength):
    """Inside a bin: darker, warm small-LED cast, light falls off to the edges."""
    a = np.asarray(img).astype(np.float32) / 255
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h * 0.45) / (h / 2))
    vign = np.clip(1.0 - 0.55 * strength * r ** 1.6, 0.25, 1)[..., None]
    a = a ** (1 + 0.4 * strength) * (1 - 0.45 * strength) * vign * np.array([1.08, 0.97, 0.80])
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def second_shot(img, rnd):
    """The AFTER photo: bin bumped (shift + small turn), light a bit different, sensor noise."""
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rnd.uniform(-1.2, 1.2), 1.0)
    M[:, 2] += [rnd.uniform(-0.012, 0.012) * w, rnd.uniform(-0.012, 0.012) * h]
    a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    a = a * rnd.uniform(0.88, 1.12) + np.random.default_rng(rnd.randint(0, 9999)).normal(0, 2.5, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), M


def main(n=10, seed=11):
    rnd = random.Random(seed)
    bgs = load_backgrounds()
    cuts = foodseg_cutouts() + onion_skin_cutouts()
    print(f"{len(bgs)} backgrounds, {len(cuts)} food cut-outs: " +
          ", ".join(sorted({c[0] for c in cuts})))
    os.makedirs(OUT, exist_ok=True)
    names = list(bgs)
    meta = []
    for k in range(n):
        bg_name = names[k % len(names)]
        dark = k % 2 == 1
        items = rnd.sample(cuts, 2 if k % 3 == 0 else 1)
        before = bgs[bg_name]
        after, mask = before, np.zeros((before.height, before.width), np.uint8)
        for c in items:
            after, m = drop(after, c[2], rnd)
            mask = np.maximum(mask, m)
        if dark:
            s = rnd.uniform(0.6, 1.0)
            before, after = bin_light(before, s), bin_light(after, s)
        after, M = second_shot(after, rnd)
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
        pid = f"pair{k:02d}_{bg_name}{'_dark' if dark else ''}"
        before.save(os.path.join(OUT, f"{pid}_before.jpg"), quality=92)
        after.save(os.path.join(OUT, f"{pid}_after.jpg"), quality=92)
        Image.fromarray(mask * 255).save(os.path.join(OUT, f"{pid}_mask.png"))
        labels = sorted({c[0] for c in items})
        meta.append({"id": pid, "labels": labels, "items": [c[1] for c in items],
                     "background": bg_name, "dark": dark})
        print(f"{pid}: {labels}")
    json.dump(meta, open(os.path.join(OUT, "pairs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
