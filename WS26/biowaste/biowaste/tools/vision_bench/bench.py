#!/usr/bin/env python3
"""
Food-waste recognition benchmark: which model names the foods on the tray best,
with several foods per image, bad light and bad angles?

  python bench.py run                      # all models on plates + foodseg (+ own/ if it has photos)
  python bench.py run --models siglip2-b16 owlv2 --datasets own
  python bench.py report                   # rebuild results/REPORT.md from saved raw results
  python bench.py conditions               # results/conditions.jpg: what each simulated condition looks like

Raw scores are saved per model in results/raw/, so a crashed or interrupted run resumes where it stopped
(use --force to recompute). See README.md for the method.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_data  # noqa: E402
import degrade  # noqa: E402

RAW = os.path.join(HERE, "results", "raw")
REPORT = os.path.join(HERE, "results", "REPORT.md")
BAD = ["dim", "warm_light", "motion_blur", "tilted", "glare", "worst"]


# ======================================================================= run
def conditions_for(ds, degrade_own):
    return list(degrade.CONDITIONS) if (ds.synthetic_conditions or degrade_own) else ["clean"]


def save_truth(ds):
    os.makedirs(os.path.join(RAW, ds.name), exist_ok=True)
    truth = {"dataset": ds.name, "description": ds.description, "vocab": ds.vocab,
             "ids": [s.id for s in ds.samples], "labels": [s.labels for s in ds.samples],
             "groups": [s.group for s in ds.samples],
             "lighting": [getattr(s, "lighting", "") for s in ds.samples]}
    json.dump(truth, open(os.path.join(RAW, ds.name, "_truth.json"), "w"), indent=0)


def sync(device):
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def mem(device):
    """RAM of this process and peak VRAM since the model was loaded, for the progress lines."""
    import psutil
    import torch
    s = f"RAM {psutil.Process().memory_info().rss / 1e9:.1f} GB"
    if device.startswith("cuda"):
        s += f", VRAM peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    return s


def cap_vram(device, margin_gb=0.4):
    """On Windows the driver silently spills a full GPU into system RAM (very slow, looks like a RAM leak).
    Cap PyTorch at the VRAM that is actually free, so an overflow fails loudly instead."""
    import torch
    if not device.startswith("cuda"):
        return
    free, total = torch.cuda.mem_get_info()
    frac = max(0.1, min(0.95, (free - margin_gb * 1e9) / total))
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"VRAM: {free / 1e9:.1f} of {total / 1e9:.1f} GB free, PyTorch capped at {frac * total / 1e9:.1f} GB")


def infer(model, images, prompts):
    if model.kind == "scores":
        return model.scores(images, prompts).tolist()
    return model.predict(images, prompts)


def run_model(model, ds, base_images, conds, device):
    prompts = [ds.prompt_name(l) for l in ds.vocab]
    res = {"model": model.name, "hf_id": model.hf_id, "note": model.note, "kind": model.kind,
           "vocab": ds.vocab, "ids": [s.id for s in ds.samples], "conditions": {}}
    t_inf, n_inf = 0.0, 0
    for cond in conds:
        out = []
        if hasattr(model, "raw"):
            model.raw = []
        for b in range(0, len(ds.samples), model.batch):
            chunk = ds.samples[b:b + model.batch]
            imgs = [degrade.apply(base_images[s.id], cond, s.id) for s in chunk]
            sync(device)
            t0 = time.perf_counter()
            out += infer(model, imgs, prompts)
            sync(device)
            if b > 0 or cond != conds[0]:          # first batch = warm-up, not timed
                t_inf += time.perf_counter() - t0
                n_inf += len(chunk)
        res["conditions"][cond] = [[round(float(x), 5) for x in r] for r in out] if model.kind == "scores" else out
        if hasattr(model, "raw"):
            res.setdefault("raw_answers", {})[cond] = list(model.raw)
        print(f"    {cond:12s} done   {mem(device)}", flush=True)
    res["ms_per_image_" + ("gpu" if device.startswith("cuda") else "cpu")] = round(1000 * t_inf / max(n_inf, 1), 1)
    if device.startswith("cuda"):
        import torch
        res["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return res


def cpu_latency(model, ds, base_images, n):
    """Latency on CPU only (the demo laptop may have no GPU). Takes the median of n single-image calls."""
    import torch
    model.to("cpu")
    if model.kind == "scores" or model.name.startswith("clip"):
        model.model.float()
    prompts = [ds.prompt_name(l) for l in ds.vocab]
    ts = []
    with torch.no_grad():
        for s in ds.samples[: n + 1]:
            t0 = time.perf_counter()
            infer(model, [base_images[s.id]], prompts)
            ts.append(time.perf_counter() - t0)
    return round(1000 * float(np.median(ts[1:])), 1)


def cmd_run(a):
    import torch
    from models import MODELS
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(MODELS) if a.models == ["all"] else a.models
    datasets = []
    for dn in a.datasets:
        ds = {"plates": lambda: bench_data.load_plates(per_meal=a.per_meal),
              "foodseg": lambda: bench_data.load_foodseg(n=a.foodseg_n),
              "own": bench_data.load_own}[dn]()
        if ds is None:
            print(f"[{dn}] no data - skipped (see README: own photos)")
            continue
        print(f"[{dn}] {len(ds.samples)} images, {len(ds.vocab)} labels - {ds.description}")
        save_truth(ds)
        datasets.append((ds, {s.id: s.image() for s in ds.samples}))
    meta = {"device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
            "cpu": platform.processor(), "torch": torch.__version__}
    json.dump(meta, open(os.path.join(RAW, "_machine.json"), "w"))
    cap_vram(device)

    for name in names:
        todo = [(ds, imgs) for ds, imgs in datasets
                if a.force or not os.path.exists(os.path.join(RAW, ds.name, f"{name}.json"))]
        if not todo:
            print(f"== {name}: already done (use --force to redo)")
            continue
        print(f"== {name}: loading", flush=True)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model = MODELS[name]().load().to(device)
        for ds, imgs in todo:
            print(f"  [{ds.name}]", flush=True)
            res = run_model(model, ds, imgs, conditions_for(ds, a.degrade_own), device)
            if not a.no_cpu:
                dtype = model.model.dtype
                res["ms_per_image_cpu"] = cpu_latency(model, ds, imgs, 2 if model.kind == "set" and
                                                      model.batch == 1 else 6)
                model.model.to(dtype)
                model.to(device)
            json.dump(res, open(os.path.join(RAW, ds.name, f"{name}.json"), "w"))
        del model
        import gc
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        cmd_report(a, quiet=True)          # partial results are readable while the next model runs
        print(f"== {name}: finished, REPORT.md updated", flush=True)
    cmd_report(a)


# ======================================================================= evaluate
def sample_metrics(pred_sets, true_sets):
    tp = sum(len(p & t) for p, t in zip(pred_sets, true_sets))
    np_ = sum(len(p) for p in pred_sets)
    nt = sum(len(t) for t in true_sets)
    f1 = np.mean([2 * len(p & t) / (len(p) + len(t)) if (p or t) else 1.0 for p, t in zip(pred_sets, true_sets)])
    return {"f1": float(f1), "precision": tp / np_ if np_ else 0.0, "recall": tp / nt if nt else 0.0,
            "exact": float(np.mean([p == t for p, t in zip(pred_sets, true_sets)])),
            "unknown": float(np.mean([not p for p in pred_sets]))}


def best_threshold(S, Y):
    trues = [set(np.nonzero(y)[0]) for y in Y]
    cands = np.unique(np.quantile(S.ravel(), np.linspace(0.5, 0.9995, 250)))
    best, bt = -1, cands[-1]
    for t in cands:
        f = sample_metrics([set(np.nonzero(s >= t)[0]) for s in S], trues)["f1"]
        if f > best:
            best, bt = f, t
    return bt


def evaluate(truth, res):
    from sklearn.metrics import average_precision_score
    vocab = truth["vocab"]
    Y = np.array([[l in labs for l in vocab] for labs in truth["labels"]], dtype=int)
    trues = [set(np.nonzero(y)[0]) for y in Y]
    folds = np.array([bench_data.fold_of(g) for g in truth["groups"]])
    out = {}
    if res["kind"] == "scores":
        clean = np.array(res["conditions"]["clean"])
        thr = {f: best_threshold(clean[folds != f], Y[folds != f]) if (folds != f).any() else
               best_threshold(clean, Y) for f in (0, 1)}
    for cond, raw in res["conditions"].items():
        m = {}
        if res["kind"] == "scores":
            S = np.array(raw)
            aps = {vocab[l]: average_precision_score(Y[:, l], S[:, l]) for l in range(len(vocab))
                   if 0 < Y[:, l].sum() < len(Y)}
            m["map"] = float(np.mean(list(aps.values()))) if aps else float("nan")
            m["per_label_ap"] = aps
            m["top1"] = float(np.mean([Y[i, S[i].argmax()] for i in range(len(S))]))
            m["top3_recall"] = float(np.mean([len(set(np.argsort(-S[i])[:3]) & trues[i]) / min(3, len(trues[i]))
                                              for i in range(len(S))]))
            preds = [set(np.nonzero(S[i] >= thr[folds[i]])[0]) for i in range(len(S))]
        else:
            preds = [set(p) for p in raw]
            m["top1"] = float(np.mean([bool(p) and min(p) in t for p, t in zip(preds, trues)])) \
                if res["model"].endswith("rule)") else float("nan")
        m.update(sample_metrics(preds, trues))
        if any(truth.get("lighting", [])):
            m["by_lighting"] = {}
            for tag in sorted(set(truth["lighting"])):
                idx = [i for i, t in enumerate(truth["lighting"]) if t == tag]
                m["by_lighting"][tag] = {"n": len(idx), **sample_metrics([preds[i] for i in idx],
                                                                         [trues[i] for i in idx])}
        out[cond] = m
    return out


# ======================================================================= report
def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.0f}" if pct else f"{x:.0f}"


def cmd_report(a=None, quiet=False):
    lines = []
    machine = json.load(open(os.path.join(RAW, "_machine.json"))) if os.path.exists(
        os.path.join(RAW, "_machine.json")) else {}
    lines += ["# Food-waste recognition benchmark", "",
              f"Generated {time.strftime('%Y-%m-%d %H:%M')} on {machine.get('device', '?')} "
              f"(CPU: {machine.get('cpu', '?')}). Method: see README.md.", "",
              "**How to read it.** *F1* = how well the predicted set of foods matches the true set per image "
              "(100 = every food named, nothing extra). *mAP* = ranking quality, independent of any threshold. "
              "*Top-1* = the single best guess is one of the foods present. *Unknown* = model said nothing. "
              "Bad conditions are simulated (see `conditions.jpg`); *bad avg* is the mean over all six.", ""]
    order = ["clip-b32-visionpy", "clip-b32", "clip-l14", "siglip2-b16", "siglip2-so400m", "owlv2", "qwen3vl-2b"]
    for dn in ["own", "plates", "foodseg"]:
        d = os.path.join(RAW, dn)
        if not os.path.exists(os.path.join(d, "_truth.json")):
            continue
        truth = json.load(open(os.path.join(d, "_truth.json")))
        files = sorted((f[:-5] for f in os.listdir(d) if f.endswith(".json") and not f.startswith("_")),
                       key=lambda n: order.index(n) if n in order else 99)
        if not files:
            continue
        results = {n: json.load(open(os.path.join(d, f"{n}.json"))) for n in files}
        evals = {n: evaluate(truth, r) for n, r in results.items()}
        nl = np.mean([len(l) for l in truth["labels"]])
        lines += [f"## Dataset `{dn}`", "", f"{truth['description']}. {len(truth['ids'])} images, "
                  f"{len(truth['vocab'])} possible foods, on average {nl:.1f} foods per image.", ""]
        conds = list(next(iter(evals.values())).keys())
        has_bad = all(c in conds for c in BAD)
        hdr = "| Model | F1 clean | " + ("F1 bad avg | F1 worst | " if has_bad else "") + \
              "mAP clean | " + ("mAP worst | " if has_bad else "") + "Top-1 | Unknown | ms/img GPU | ms/img CPU | Note |"
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for n, ev in evals.items():
            r = results[n]
            c = ev["clean"]
            row = f"| {r['model']} | **{fmt(c['f1'])}** | "
            if has_bad:
                row += f"{fmt(np.mean([ev[b]['f1'] for b in BAD]))} | {fmt(ev['worst']['f1'])} | "
            row += f"{fmt(c.get('map'))} | "
            if has_bad:
                row += f"{fmt(ev['worst'].get('map'))} | "
            row += (f"{fmt(c.get('top1'))} | {fmt(c['unknown'])} | {fmt(r.get('ms_per_image_gpu'), False)} | "
                    f"{fmt(r.get('ms_per_image_cpu'), False)} | {r['note']} |")
            lines.append(row)
        lines.append("")
        if len(conds) > 1:
            lines += ["F1 per condition:", "", "| Model | " + " | ".join(conds) + " |",
                      "|---|" + "---|" * len(conds)]
            for n, ev in evals.items():
                lines.append(f"| {results[n]['model']} | " + " | ".join(fmt(ev[c]['f1']) for c in conds) + " |")
            lines.append("")
        if "by_lighting" in evals[files[0]]["clean"]:
            tags = list(evals[files[0]]["clean"]["by_lighting"])
            lines += ["F1 by lighting (as captured):", "", "| Model | " + " | ".join(
                f"{t} (n={evals[files[0]]['clean']['by_lighting'][t]['n']})" for t in tags) + " |",
                "|---|" + "---|" * len(tags)]
            for n, ev in evals.items():
                lines.append(f"| {results[n]['model']} | " + " | ".join(
                    fmt(ev["clean"]["by_lighting"][t]["f1"]) for t in tags) + " |")
            lines.append("")
        scored = [n for n in files if results[n]["kind"] == "scores"]
        if scored:
            best = max(scored, key=lambda n: evals[n]["clean"]["map"])
            aps = sorted(evals[best]["clean"]["per_label_ap"].items(), key=lambda kv: -kv[1])
            if len(aps) > 16:
                aps = aps[:8] + [("…", None)] + aps[-8:]
            lines += [f"Easiest and hardest foods for {results[best]['model']} (average precision, clean):", "",
                      ", ".join(f"{k} {fmt(v)}" if v is not None else k for k, v in aps), ""]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if not quiet:
        print("\n".join(lines))
        print(f"\nwritten {REPORT}")


# ======================================================================= conditions preview
def cmd_conditions(a):
    from PIL import Image, ImageDraw
    ds = bench_data.load_plates(per_meal=2)
    s = ds.samples[3]
    base = s.image()
    base.thumbnail((360, 360))
    tiles = [(c, degrade.apply(base, c, s.id)) for c in degrade.CONDITIONS]
    w, h = base.size
    sheet = Image.new("RGB", (w * 4, (h + 26) * 2), "white")
    for k, (c, im) in enumerate(tiles):
        x, y = (k % 4) * w, (k // 4) * (h + 26)
        sheet.paste(im, (x, y + 26))
        ImageDraw.Draw(sheet).text((x + 6, y + 6), c, fill="black")
    ImageDraw.Draw(sheet).text((3 * w + 6, h + 26 + 32), "true labels:\n" + "\n".join(s.labels), fill="black")
    out = os.path.join(HERE, "results", "conditions.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out, quality=90)
    print("written", out)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=["all"])
    r.add_argument("--datasets", nargs="+", default=["own", "plates", "foodseg"])
    r.add_argument("--device", default=None)
    r.add_argument("--per-meal", type=int, default=8, help="plates: photos per meal")
    r.add_argument("--foodseg-n", type=int, default=160)
    r.add_argument("--degrade-own", action="store_true", help="also apply simulated conditions to own photos")
    r.add_argument("--no-cpu", action="store_true", help="skip CPU latency measurement")
    r.add_argument("--force", action="store_true")
    sub.add_parser("report")
    sub.add_parser("conditions")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report, "conditions": cmd_conditions}[a.cmd](a)


if __name__ == "__main__":
    main()
