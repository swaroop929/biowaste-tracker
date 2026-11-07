# Food-waste recognition benchmark

Generated 2026-11-06 23:38 on NVIDIA GeForce RTX 3070 (CPU: Intel64 Family 6 Model 167 Stepping 1, GenuineIntel). Method: see README.md.

**How to read it.** *F1* = how well the predicted set of foods matches the true set per image (100 = every food named, nothing extra). *mAP* = ranking quality, independent of any threshold. *Top-1* = the single best guess is one of the foods present. *Unknown* = model said nothing. Bad conditions are simulated (see `conditions.jpg`); *bad avg* is the mean over all six.

## Dataset `plates`

Real plate leftovers, 16 meals x up to 2 photos (different angles), label = every ingredient with >= 5 g returned. 32 images, 23 possible foods, on average 3.4 foods per image.

| Model | F1 clean | F1 bad avg | F1 worst | mAP clean | mAP worst | Top-1 | Unknown | ms/img GPU | ms/img CPU | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 (vision.py rule) | **14** | 10 | 4 | – | – | 22 | 59 | 13 | – | current code: one food per image, softmax >= 0.45 |
| clip-b32 | **32** | 30 | 23 | 49 | 31 | 44 | 6 | 14 | – | baseline, 0.6 GB |
| clip-l14 | **42** | 36 | 29 | 54 | 41 | 72 | 6 | 29 | – | bigger CLIP, 1.7 GB |
| siglip2-b16 | **47** | 39 | 24 | 64 | 41 | 81 | 6 | 19 | – | 1.5 GB |
| siglip2-so400m | **47** | 40 | 25 | 71 | 53 | 84 | 9 | 38 | – | 4.5 GB |
| owlv2 | **47** | 30 | 10 | 67 | 36 | 72 | 6 | 287 | – | detector, 0.6 GB |
| qwen3vl-2b | **55** | 43 | 25 | – | – | – | 0 | 2844 | – | VLM, 4.3 GB |

F1 per condition:

| Model | clean | dim | warm_light | motion_blur | tilted | glare | worst |
|---|---|---|---|---|---|---|---|
| clip-b32 (vision.py rule) | 14 | 10 | 13 | 13 | 10 | 11 | 4 |
| clip-b32 | 32 | 32 | 29 | 35 | 27 | 32 | 23 |
| clip-l14 | 42 | 37 | 42 | 36 | 36 | 40 | 29 |
| siglip2-b16 | 47 | 38 | 45 | 41 | 41 | 47 | 24 |
| siglip2-so400m | 47 | 46 | 47 | 42 | 39 | 44 | 25 |
| owlv2 | 47 | 34 | 44 | 13 | 37 | 43 | 10 |
| qwen3vl-2b | 55 | 42 | 54 | 38 | 47 | 54 | 25 |

Easiest and hardest foods for siglip2-so400m (average precision, clean):

breaded schnitzel 100, green beans 100, red cabbage 100, rice 100, zucchini 100, carrots 97, peas 93, brown gravy 89, …, potato pancake 61, mashed potatoes 59, roast meat 56, boiled potatoes 44, mushrooms 37, breaded fish 27, savoy cabbage 27, tomato sauce 27

## Dataset `foodseg`

FoodSeg103 validation, 25 random mixed dishes, 103-ingredient vocabulary, label = ingredient covering >= 2% of the image. 25 images, 102 possible foods, on average 3.5 foods per image.

| Model | F1 clean | F1 bad avg | F1 worst | mAP clean | mAP worst | Top-1 | Unknown | ms/img GPU | ms/img CPU | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 (vision.py rule) | **13** | 11 | 4 | – | – | 24 | 44 | 8 | – | current code: one food per image, softmax >= 0.45 |
| clip-b32 | **35** | 31 | 20 | 59 | 40 | 56 | 0 | 7 | – | baseline, 0.6 GB |
| clip-l14 | **35** | 34 | 27 | 69 | 59 | 64 | 8 | 22 | – | bigger CLIP, 1.7 GB |
| siglip2-b16 | **42** | 32 | 14 | 75 | 67 | 68 | 0 | 12 | – | 1.5 GB |
| siglip2-so400m | **36** | 31 | 12 | 64 | 66 | 52 | 16 | 33 | – | 4.5 GB |
| owlv2 | **56** | 47 | 26 | 79 | 48 | 84 | 0 | 278 | – | detector, 0.6 GB |
| qwen3vl-2b | **54** | 55 | 46 | – | – | – | 0 | 2448 | – | VLM, 4.3 GB |

F1 per condition:

| Model | clean | dim | warm_light | motion_blur | tilted | glare | worst |
|---|---|---|---|---|---|---|---|
| clip-b32 (vision.py rule) | 13 | 10 | 10 | 15 | 14 | 10 | 4 |
| clip-b32 | 35 | 35 | 37 | 33 | 28 | 33 | 20 |
| clip-l14 | 35 | 33 | 41 | 37 | 31 | 37 | 27 |
| siglip2-b16 | 42 | 25 | 46 | 33 | 32 | 43 | 14 |
| siglip2-so400m | 36 | 32 | 38 | 38 | 30 | 35 | 12 |
| owlv2 | 56 | 41 | 53 | 51 | 55 | 55 | 26 |
| qwen3vl-2b | 54 | 60 | 57 | 55 | 58 | 54 | 46 |

Easiest and hardest foods for owlv2 (average precision, clean):

asparagus 100, bread 100, broccoli 100, cabbage 100, cake 100, carrot 100, corn 100, cucumber 100, …, pasta 50, onion 45, pie 39, hanamaki baozi 36, pork 33, rape 17, spring onion 17, fried meat 14

