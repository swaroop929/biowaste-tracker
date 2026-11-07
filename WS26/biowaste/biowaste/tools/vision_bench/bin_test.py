#!/usr/bin/env python3
"""
Test "what was thrown into the bin?" on binpairs/ (make_testset.py):
  1. change detection: does the cut-out find the new food?  (pixel IoU against the true mask)
  2. classification: are the right foods named?  Compared inputs:
       full       the whole AFTER photo (no change detection)
       old_box    crop from the old grey-level difference (the previous vision.py)
       new_box    crop around everything the new change detection found
       new_parts  each new piece cut out separately, old waste greyed out  <- vision.recognise()
       true_cut   cut-out from the TRUE mask (upper limit of the classifier itself)

    python bin_test.py [--model so400m|base]
Writes results/binpairs_report.md and results/binpairs_sheet.jpg.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "firmware", "bin_tracker"))
import vision  # noqa: E402

PAIRS = os.path.join(HERE, "binpairs")
RES = os.path.join(HERE, "results")


def old_box(before, after, thresh=35, pad=20):
    """The previous vision.py: grey-level difference > 35, box around the 2-98 % of changed pixels."""
    a = np.asarray(before.convert("L").resize(after.size), dtype=np.int16)
    b = np.asarray(after.convert("L"), dtype=np.int16)
    m = np.abs(a - b) > thresh
    if m.mean() < 0.002:
        return None, m
    ys, xs = np.where(m)
    x0, x1 = np.percentile(xs, [2, 98]).astype(int)
    y0, y1 = np.percentile(ys, [2, 98]).astype(int)
    w, h = after.size
    return (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)), m


def box_mask(box, shape):
    m = np.zeros(shape, bool)
    if box:
        m[box[1]:box[3], box[0]:box[2]] = True
    return m


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def f1(pred, true):
    p, t = set(pred), set(true)
    return 2 * len(p & t) / (len(p) + len(t)) if (p or t) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="so400m", choices=list(vision.MODELS))
    a = ap.parse_args()
    vision.load_model(a.model)
    meta = json.load(open(os.path.join(PAIRS, "pairs.json")))
    rows, tiles = [], []
    for m in meta:
        before = Image.open(os.path.join(PAIRS, f"{m['id']}_before.jpg")).convert("RGB")
        after = Image.open(os.path.join(PAIRS, f"{m['id']}_after.jpg")).convert("RGB")
        gt = np.asarray(Image.open(os.path.join(PAIRS, f"{m['id']}_mask.png"))) > 127
        gt_box = (lambda ys, xs: (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))(*np.nonzero(gt))

        ob, _ = old_box(before, after)
        t0 = time.perf_counter()
        nm, parts = vision.change_mask(before, after)
        t_change = time.perf_counter() - t0
        nb = vision.changed_region(before, after)

        t0 = time.perf_counter()
        rec = vision.recognise(before, after)
        t_rec = time.perf_counter() - t0
        preds = {
            "full": vision.classify(after)[0],
            "old_box": vision.classify(after.crop(ob))[0] if ob else [],
            "new_box": vision.classify(after.crop(nb))[0] if nb else [],
            "new_parts": rec["foods"],
            "true_cut": vision.classify(vision.cutout(after, gt, gt_box))[0],
        }
        r = {"id": m["id"], "truth": m["labels"], "dark": m["dark"],
             "iou_mask_new": iou(nm, gt), "iou_box_old": iou(box_mask(ob, gt.shape), box_mask(gt_box, gt.shape)),
             "iou_box_new": iou(box_mask(nb, gt.shape), box_mask(gt_box, gt.shape)),
             "parts": len(parts), "ms_change": 1000 * t_change, "ms_total": 1000 * t_rec,
             **{f"f1_{k}": f1(v, m["labels"]) for k, v in preds.items()}, "pred": preds}
        rows.append(r)
        print(f"{m['id']:28s} truth={m['labels']}  mask IoU {r['iou_mask_new']:.2f} (old box {r['iou_box_old']:.2f})"
              f"  -> {rec['foods']}  [{r['ms_total']:.0f} ms]")

        # sheet row: before | after | detection (green hit, red missed, blue extra) | first cut-out | text
        th = 200
        def small(im):
            im = im.copy(); im.thumbnail((th * 4 // 3, th)); return im
        ov = np.asarray(after).copy()
        ov[nm & gt] = (0.4 * ov[nm & gt] + [0, 150, 0]).astype(np.uint8)
        ov[~nm & gt] = (0.4 * ov[~nm & gt] + [150, 0, 0]).astype(np.uint8)
        ov[nm & ~gt] = (0.4 * ov[nm & ~gt] + [0, 0, 150]).astype(np.uint8)
        cut = vision.cutout(after, nm, parts[0]) if parts else Image.new("RGB", (10, 10), "grey")
        tiles.append((m, r, [small(before), small(after), small(Image.fromarray(ov)), small(cut)]))

    # ---------- report
    k = ["full", "old_box", "new_box", "new_parts", "true_cut"]
    mean = lambda key, rs=rows: float(np.mean([r[key] for r in rs]))  # noqa: E731
    lines = ["# Bin camera test: what was thrown in?", "",
             f"{len(rows)} before/after pairs from `binpairs/` (real photos looking down on food waste, real food "
             "cut-outs dropped in, bin bumped and light changed between the two photos, half of them dark with "
             f"a warm LED). Classifier: SigLIP 2 {a.model}. See `make_testset.py`.", "",
             "## Finding the new food", "",
             "| | IoU with the true new food |", "|---|---|",
             f"| Old: grey-level difference, one box | {mean('iou_box_old'):.2f} (box vs true box) |",
             f"| New: change detection, one box | {mean('iou_box_new'):.2f} (box vs true box) |",
             f"| New: change detection, pixel mask | **{mean('iou_mask_new'):.2f}** (mask vs true mask) |", "",
             f"Change detection takes {mean('ms_change'):.0f} ms per pair on the CPU; the whole recognition "
             f"(detection + classifying each piece) {mean('ms_total'):.0f} ms.", "",
             "## Naming the food (F1, 100 = exactly the right foods)", "",
             "| Input to the classifier | all | bright | dark |", "|---|---|---|---|"]
    names = {"full": "Whole after photo", "old_box": "Old crop (grey-level box)", "new_box": "New crop (one box)",
             "new_parts": "**New: each piece cut out, old waste greyed** (vision.py now)",
             "true_cut": "Cut-out from the true mask (classifier limit)"}
    br, dk = [r for r in rows if not r["dark"]], [r for r in rows if r["dark"]]
    for key in k:
        lines.append(f"| {names[key]} | {100 * mean('f1_' + key):.0f} | {100 * mean('f1_' + key, br):.0f} | "
                     f"{100 * mean('f1_' + key, dk):.0f} |")
    lines += ["", "## Per pair", "", "| Pair | Truth | Found (vision.py now) | Mask IoU | Whole photo said |",
              "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {', '.join(r['truth'])} | {', '.join(r['pred']['new_parts']) or 'unknown'} | "
                     f"{r['iou_mask_new']:.2f} | {', '.join(r['pred']['full'][:4]) or 'unknown'} |")
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "binpairs_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[lines.index("## Finding the new food"):]))

    tw = 270
    S = Image.new("RGB", (4 * tw + 330, len(tiles) * 215), "white")
    d = ImageDraw.Draw(S)
    for i, (m, r, ims) in enumerate(tiles):
        for j, im in enumerate(ims):
            S.paste(im, (j * tw, i * 215 + 8))
        d.text((4 * tw + 6, i * 215 + 10), f"{m['id']}\ntruth: {', '.join(m['labels'])}\n"
               f"found: {', '.join(r['pred']['new_parts']) or 'unknown'}\nmask IoU {r['iou_mask_new']:.2f}\n\n"
               "green = found, red = missed,\nblue = wrongly marked", fill="black")
    S.save(os.path.join(RES, "binpairs_sheet.jpg"), quality=88)
    print("written results/binpairs_report.md, results/binpairs_sheet.jpg")


if __name__ == "__main__":
    main()
