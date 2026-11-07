# Food-waste vision benchmark

Which model should `firmware/bin_tracker/vision.py` use to name the food on the tray, when a tip often
contains **several foods**, the **light is bad** and the **camera is not straight above**?
Results: [`results/REPORT.md`](results/REPORT.md) covers all 7 models on 399 images each.
[`results/REPORT_full.md`](results/REPORT_full.md) covers 6 models on 2016 images each, without Qwen
because it is too slow at that size. What the simulated conditions look like:
[`results/conditions.jpg`](results/conditions.jpg).

## Verdict (2026-11-06, RTX 3070)

| | Real plate leftovers F1 | Mixed dishes F1 | Worst conditions | GPU time / photo |
|---|---|---|---|---|
| Today's vision.py rule | 13–14 | 13 | 4 | 13 ms |
| **SigLIP 2 base** | 47–49 | 40–42 | 22–24 (plates) | **19 ms** |
| SigLIP 2 so400m | 47–53 | 35–36 | 25–31 (plates) | 38 ms |
| OWLv2 | 47 | 49–56 | 8–10 on plates (fails under motion blur) | 290 ms |
| **Qwen3-VL-2B** | **55** | **54** | **25 plates / 46 dishes** | 2.5–2.8 s |

The ranges span the small run and the full run.

* **Drop the current rule.** It can name only one food, so it says "unknown" on about 60 % of mixed plates.
* **SigLIP 2 base is the live default.** It scores about 3.5× today's rule, takes 19 ms on a GPU and
  0.4 s on a plain CPU, and is a small change in `vision.py`.
* **Qwen3-VL-2B is the most accurate and the most robust**, especially in bad conditions on mixed dishes.
  It needs a GPU and about 3 s per photo, which is fine if recognition runs after the tip and nobody waits
  for it. It over-lists, though: on the plates it names 6.9 foods on average when 3.4 are there.
* **Skip OWLv2.** It is slow and collapses under motion blur. CLIP-L/14 is no better than SigLIP 2 base.
* Even the best model names only about half the foods correctly. Use the food names as a hint for the
  insight ("mostly pasta"), keep the buttons as the source of the waste type, and fix the light first.
* The small run has only 32 plate photos and 25 dishes, so ±5–10 points is noise. The final call belongs
  to `own` photos from the real tray.

**Deployed:** the demo laptop has an RTX 3050 with 4 GB, which is too small for Qwen (it needs over 3.5 GB).
`vision.py` therefore uses **SigLIP 2 so400m** in fp16 (2.3 GB peak VRAM, ~60 ms per photo on a 3070).
It falls back to **base** automatically if less than 3 GB of VRAM is free, and runs base on the CPU when
there is no GPU. Its score thresholds (so400m 0.05, base 0.02) are the best-F1 values on `plates`. Re-check
them once `own` photos exist.

## Bin camera test: what was just thrown in?

The camera sits inside the bin. The lid opens and a BEFORE photo is taken, then food goes in and an
AFTER photo is taken. `vision.recognise(before, after)` finds what is new, cuts out each new piece
with the old waste greyed out, and classifies each piece.

```bash
python make_testset.py   # builds binpairs/ (10 pairs, downloads 5 photos)
python bin_test.py       # -> results/binpairs_report.md + results/binpairs_sheet.jpg
```

No real before/after photos from inside a food-waste bin exist online, so the pairs are built from real
parts. The BEFORE photo is one of 4 real photos looking down on food waste in a bin or caddy. The AFTER
photo is the same photo with 1–2 real food cut-outs dropped on top (FoodSeg103 masks and onion skins).
Between the two photos the bin gets bumped, the light changes and sensor noise is added. Half the pairs
are dark with a warm lamp. The true mask of the new food is known exactly.

| | Old vision.py (grey-level box) | Now |
|---|---|---|
| Finds the new food (IoU with the true food) | 0.17 | **0.79** (pixel mask) |
| Names the foods, lit bin (F1) | 6 | **93** |
| Names the foods, dark bin (F1) | 24 | 27 |
| Time per disposal (RTX 3070 + CPU) | | ~370 ms |

Classifying the whole after photo scores F1 5 in a lit bin: the model describes the old waste, not
the new item. In a dark bin everything fails, even with the true mask. Correcting colour and brightness
afterwards made it worse (27 → 13), so **put a small white LED in the bin**. Mount the camera on the bin
body, not on the lid.

## Run it

```bash
pip install torch transformers pillow numpy opencv-python-headless scikit-learn pyarrow huggingface_hub psutil
python bench.py run                         # all models, all datasets (first run downloads ~14 GB of models)
python bench.py run --models siglip2-b16    # one model
python bench.py report                      # rebuild REPORT.md from saved scores
python bench.py conditions                  # preview image of the simulated conditions
```

A full run takes about 2 hours on an RTX 3070 (8 GB), most of it Qwen3-VL. The quick version takes
about 25 minutes: `python bench.py run --datasets plates foodseg --per-meal 2 --foodseg-n 25 --no-cpu`
(399 images per model). Runs are resumable: finished models are skipped
(`--force` recomputes). Each model is also timed on the CPU, because the demo laptop may not have a GPU
(`--no-cpu` skips that). PyTorch is capped at the VRAM that is actually free. On Windows a full GPU
otherwise spills silently into system RAM, which looks like a RAM leak and is very slow.

## Datasets

| Name | What | Why it is in here |
|---|---|---|
| `own` | **Your tray photos** from `capture.py` | The only test that matches the real setup. Use it for the final decision. |
| `plates` | [AI-ServicesBB/food-waste-dataset](https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset) (CC-BY-4.0): real plate leftovers from a care-home kitchen, 16 meals × 8 photos from different angles. Label = every ingredient with ≥ 5 g weighed as returned. | Real food waste, several foods per image, angles vary. |
| `foodseg` | [FoodSeg103](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103) validation (Apache-2.0), 160 random images, 102 ingredients. Label = ingredient covering ≥ 2 % of the image. | The hardest "many foods, large vocabulary" test. These are dishes, not waste. |

There is no public dataset of kitchen bin scraps (peels, trimmings) that can be downloaded. The closest
one, from arXiv 2401.15175, has 19 classes but was never released. That is why `own` matters.

### Your own photos

```bash
python capture.py --phone http://192.168.23.223:8080
```

For each item you take a before photo and an after photo. You then type the foods (numbers or names from
`own/vocab.txt`) and the lighting (`good`/`dim`/`warm`/`mixed`/`glare`). The crop that is saved is exactly
what `vision.py` would classify. Without the phone, put photos into `own/images/` and write
`own/labels.csv` by hand:

```
file,labels,lighting,notes
images/0001.jpg,pasta;sauce,dim,lights off
```

Your own photos are not degraded artificially, because their conditions are already real. The report
splits them by lighting tag. `--degrade-own` adds the simulated conditions on top.

## Simulated conditions (plates, foodseg)

`clean` · `dim` (under-exposed + sensor noise) · `warm_light` (tungsten cast) · `motion_blur` ·
`tilted` (perspective ~30–40° + rotation) · `glare` (reflections) · `worst` (tilted + warm + dim +
blur + JPEG 30). They are deterministic, so every model sees identical pixels.

## Models

| Key | Model | Type |
|---|---|---|
| `clip-b32-visionpy` | CLIP ViT-B/32 with **today's vision.py rule** (softmax, top-1, ≥ 0.45) | one label or "unknown" |
| `clip-b32` | CLIP ViT-B/32, multi-label (thresholded scores) | scores |
| `clip-l14` | CLIP ViT-L/14 | scores |
| `siglip2-b16` | SigLIP 2 base, 384 px | scores (sigmoid, per label) |
| `siglip2-so400m` | SigLIP 2 so400m, 384 px | scores |
| `owlv2` | OWLv2 base (open-vocabulary detector; image score = best box) | scores |
| `qwen3vl-2b` | Qwen3-VL-2B-Instruct, asked for a JSON list of foods | set |

Every scoring model gets the same three prompt templates (`a photo of {}.`, `a photo of {}, food waste.`,
`leftover {} on a tray.`), averaged.

## Metrics

* **F1**: per image, how well the predicted set of foods matches the true set, averaged. This is the
  headline number. Scoring models need a threshold to turn scores into a set. It is chosen honestly with
  a 2-fold split by meal: tuned on the clean photos of one half, applied to the other half under every
  condition. This is the same as calibrating once at setup, after which the light changes.
* **mAP**: ranking quality per food, with no threshold. It shows a model's potential if the threshold
  were tuned well.
* **Top-1**: whether the single best guess is one of the foods present. This is what vision.py reports today.
* **Unknown**: share of images where the model named nothing.
* **ms/img**: inference time on the GPU (batched) and the CPU (one image at a time).

## Caveats

* `plates` has only 16 meals, so per-food numbers rest on 1–4 meals each. Read those as hints only.
* Labels come from weighed leftovers. A few grams of sauce smeared on the plate counts as "visible" but
  is hard to see. This lowers everyone's scores equally.
* FoodSeg103 is dishes, not waste. Use it to rank the models, not to estimate kitchen accuracy.
