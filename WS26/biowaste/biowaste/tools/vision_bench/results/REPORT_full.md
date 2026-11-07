# Food-waste recognition benchmark - full run (2016 images per model, 6 models; Qwen3-VL not run)

Generated 2026-11-06 23:15 on NVIDIA GeForce RTX 3070 (CPU: Intel64 Family 6 Model 167 Stepping 1, GenuineIntel). Method: see README.md.

**How to read it.** *F1* = how well the predicted set of foods matches the true set per image (100 = every food named, nothing extra). *mAP* = ranking quality, independent of any threshold. *Top-1* = the single best guess is one of the foods present. *Unknown* = model said nothing. Bad conditions are simulated (see `conditions.jpg`); *bad avg* is the mean over all six.

## Dataset `plates`

Real plate leftovers, 16 meals x up to 8 photos (different angles), label = every ingredient with >= 5 g returned. 128 images, 23 possible foods, on average 3.4 foods per image.

| Model | F1 clean | F1 bad avg | F1 worst | mAP clean | mAP worst | Top-1 | Unknown | ms/img GPU | ms/img CPU | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 (vision.py rule) | **13** | 11 | 4 | – | – | 23 | 60 | 14 | 58 | current code: one food per image, softmax >= 0.45 |
| clip-b32 | **32** | 28 | 14 | 43 | 27 | 48 | 5 | 15 | 60 | baseline, 0.6 GB |
| clip-l14 | **35** | 32 | 28 | 54 | 35 | 66 | 7 | 29 | 585 | bigger CLIP, 1.7 GB |
| siglip2-b16 | **49** | 39 | 22 | 57 | 33 | 70 | 0 | 20 | 410 | 1.5 GB |
| siglip2-so400m | **53** | 44 | 31 | 69 | 41 | 82 | 1 | 39 | 2263 | 4.5 GB |
| owlv2 | **47** | 30 | 8 | 55 | 31 | 66 | 5 | 299 | 5578 | detector, 0.6 GB |

F1 per condition:

| Model | clean | dim | warm_light | motion_blur | tilted | glare | worst |
|---|---|---|---|---|---|---|---|
| clip-b32 (vision.py rule) | 13 | 11 | 13 | 13 | 10 | 12 | 4 |
| clip-b32 | 32 | 32 | 32 | 36 | 22 | 33 | 14 |
| clip-l14 | 35 | 34 | 34 | 30 | 34 | 34 | 28 |
| siglip2-b16 | 49 | 40 | 43 | 38 | 40 | 47 | 22 |
| siglip2-so400m | 53 | 45 | 52 | 42 | 42 | 51 | 31 |
| owlv2 | 47 | 36 | 45 | 10 | 39 | 44 | 8 |

Easiest and hardest foods for siglip2-so400m (average precision, clean):

green beans 100, red cabbage 100, carrots 91, rice 90, brown gravy 86, peas 84, sausage 78, zucchini 78, …, mushrooms 66, lentils 62, potato pancake 61, boiled potatoes 47, bread dumpling 43, tomato sauce 39, breaded fish 27, savoy cabbage 19

## Dataset `foodseg`

FoodSeg103 validation, 160 random mixed dishes, 103-ingredient vocabulary, label = ingredient covering >= 2% of the image. 160 images, 102 possible foods, on average 3.2 foods per image.

| Model | F1 clean | F1 bad avg | F1 worst | mAP clean | mAP worst | Top-1 | Unknown | ms/img GPU | ms/img CPU | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 (vision.py rule) | **13** | 11 | 7 | – | – | 24 | 49 | 8 | 53 | current code: one food per image, softmax >= 0.45 |
| clip-b32 | **30** | 26 | 21 | 39 | 28 | 42 | 4 | 8 | 48 | baseline, 0.6 GB |
| clip-l14 | **32** | 27 | 19 | 47 | 33 | 50 | 6 | 24 | 582 | bigger CLIP, 1.7 GB |
| siglip2-b16 | **40** | 32 | 17 | 54 | 38 | 60 | 4 | 13 | 413 | 1.5 GB |
| siglip2-so400m | **35** | 30 | 11 | 48 | 40 | 53 | 16 | 34 | 2351 | 4.5 GB |
| owlv2 | **49** | 42 | 28 | 60 | 37 | 72 | 1 | 275 | 6111 | detector, 0.6 GB |

F1 per condition:

| Model | clean | dim | warm_light | motion_blur | tilted | glare | worst |
|---|---|---|---|---|---|---|---|
| clip-b32 (vision.py rule) | 13 | 12 | 11 | 11 | 10 | 12 | 7 |
| clip-b32 | 30 | 26 | 29 | 27 | 26 | 29 | 21 |
| clip-l14 | 32 | 22 | 35 | 27 | 30 | 30 | 19 |
| siglip2-b16 | 40 | 29 | 41 | 33 | 31 | 39 | 17 |
| siglip2-so400m | 35 | 29 | 37 | 34 | 30 | 36 | 11 |
| owlv2 | 49 | 38 | 48 | 44 | 48 | 49 | 28 |

Easiest and hardest foods for owlv2 (average precision, clean):

 cabbage 100, banana 100, chocolate 100, grape 100, kiwi 100, melon 100, noodles 100, peanut 100, …, seaweed 11,  hanamaki baozi 8, soy 6, green beans 5, cheese butter 4, fried meat 4, lamb 3, mango 2

