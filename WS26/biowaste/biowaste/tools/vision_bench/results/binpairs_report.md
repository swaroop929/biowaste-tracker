# Bin camera test: what was thrown in?

10 before/after pairs from `binpairs/` (real photos looking down on food waste, real food cut-outs dropped in, bin bumped and light changed between the two photos, half of them dark with a warm LED). Classifier: SigLIP 2 so400m. See `make_testset.py`.

## Finding the new food

| | IoU with the true new food |
|---|---|
| Old: grey-level difference, one box | 0.17 (box vs true box) |
| New: change detection, one box | 0.70 (box vs true box) |
| New: change detection, pixel mask | **0.79** (mask vs true mask) |

Change detection takes 327 ms per pair on the CPU; the whole recognition (detection + classifying each piece) 367 ms.

## Naming the food (F1, 100 = exactly the right foods)

| Input to the classifier | all | bright | dark |
|---|---|---|---|
| Whole after photo | 12 | 5 | 19 |
| Old crop (grey-level box) | 15 | 6 | 24 |
| New crop (one box) | 19 | 8 | 31 |
| **New: each piece cut out, old waste greyed** (vision.py now) | 60 | 93 | 27 |
| Cut-out from the true mask (classifier limit) | 53 | 80 | 27 |

## Per pair

| Pair | Truth | Found (vision.py now) | Mask IoU | Whole photo said |
|---|---|---|---|---|
| pair00_caddy_scraps | cooked meat, tomato | tomato, cooked meat | 0.81 | vegetable trimmings, salad, carrot peels, potato peels |
| pair01_carrot_bowl_dark | onion skin | unknown | 0.87 | carrot peels, vegetable trimmings, salad, onion skin |
| pair02_peel_pile | onion skin | onion skin | 0.80 | vegetable trimmings, orange peel, carrot peels, salad |
| pair03_bin_scraps_dark | fish, rice | rice | 0.73 | vegetable trimmings, rice, salad, cucumber |
| pair04_caddy_scraps | potatoes | potatoes | 0.74 | vegetable trimmings, salad, carrot peels, potato peels |
| pair05_carrot_bowl_dark | potatoes | rice | 0.84 | carrot peels, vegetable trimmings, rice |
| pair06_peel_pile | rice, tomato | tomato | 0.74 | vegetable trimmings, tomato, salad, orange peel |
| pair07_bin_scraps_dark | potatoes | potatoes, onion skin | 0.85 | vegetable trimmings, salad, potato peels, onion skin |
| pair08_caddy_scraps | apple | apple | 0.85 | vegetable trimmings, salad, carrot peels |
| pair09_carrot_bowl_dark | cooked meat, cucumber | chicken bones | 0.65 | carrot peels, vegetable trimmings, cooked meat, chicken bones |
