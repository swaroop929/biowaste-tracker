"""
Benchmark datasets. Every image can show several foods (multi-label).

  plates   Real plate leftovers from a German care-home kitchen (AI-ServicesBB/food-waste-dataset,
           CC-BY-4.0). Each meal is photographed 12-40 times from different angles; the label is every
           ingredient that came back uneaten (>= MIN_RETURN_G grams, weighed per ingredient).
  foodseg  FoodSeg103 validation split (Apache-2.0): mixed dishes with pixel masks for 103 ingredients.
           Not waste, but the hardest "many foods in one image" test there is.
  own      Your own tray photos (see capture.py / README): own/labels.csv + own/images/.
"""
import csv
import hashlib
import io
import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SIDE = 1024          # every image is downscaled to this before degradation (models resize anyway)
MIN_RETURN_G = 5         # plates: an ingredient counts as "visible waste" from this many grams back
MIN_AREA = 0.02          # foodseg: an ingredient counts when it covers >= 2 % of the image


@dataclass
class Sample:
    id: str
    group: str               # samples of one group (one meal) always land in the same fold
    labels: list
    _bytes: bytes = field(repr=False, default=b"")
    path: str = ""

    def image(self):
        img = Image.open(io.BytesIO(self._bytes) if self._bytes else self.path).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        return img


@dataclass
class Dataset:
    name: str
    vocab: list              # label names (also what the models are asked about)
    samples: list
    description: str
    synthetic_conditions: bool = True   # own photos are already "real conditions"
    prompt_names: dict = field(default_factory=dict)  # label -> wording used in prompts

    def prompt_name(self, label):
        return self.prompt_names.get(label, label)


def fold_of(group):
    """Deterministic 2-fold split by group (meal), used for threshold calibration."""
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 2


# ---------------------------------------------------------------- plates
PLATE_MAP = {  # German article -> English label (several articles merge into one label)
    "Kartoffelpüree": "mashed potatoes", "Kartoffeln": "boiled potatoes", "Kartoffelwürfel": "boiled potatoes",
    "Reibekuchen": "potato pancake", "Grüne Bohnen": "green beans", "Erbsen": "peas", "Möhre": "carrots",
    "Zucchini": "zucchini", "Paprika": "bell pepper", "Rotkohl": "red cabbage", "Wirsing": "savoy cabbage",
    "Sauerkraut": "sauerkraut", "Blumenkohl": "cauliflower", "Pilze": "mushrooms", "Zwiebel": "onion",
    "Eisbergsalat": "lettuce", "Krautsalat": "coleslaw", "Linsen": "lentils", "Linseneintopf": "lentils",
    "Reis": "rice", "Eierspätzle": "spaetzle noodles", "Semmelknödel": "bread dumpling",
    "Apfelmus": "apple sauce", "Hähnchen": "chicken", "Hähnchenstreifen": "chicken",
    "Rinderbraten": "roast meat", "Schweinenackenbraten": "roast meat", "Schnitzel": "breaded schnitzel",
    "Rostbratwurst": "sausage", "Schinken Mettwurst": "sausage", "Fleischbällchen gebrüht": "meatballs",
    "Seelachs": "fish fillet", "Paniertes Fischfilet": "breaded fish",
    "Bratenjus": "brown gravy", "Braune Sauce": "brown gravy", "Malzbier-Senf-Sauce": "brown gravy",
    "Helle Sauce": "white sauce", "Tomaten-Curry-Sauce": "tomato sauce",
    "Sahne": "cream", "Pflanzencreme": "cream", "Dressing Portion": "salad dressing",
}
PLATE_FILES = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001-64762f3000d1dde2.parquet"]


def load_plates(per_meal=8, seed=0):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    samples, vocab = [], set(PLATE_MAP.values())
    for fn in PLATE_FILES:
        t = pq.read_table(hf_hub_download("AI-ServicesBB/food-waste-dataset", fn, repo_type="dataset"),
                          columns=["bonid", "image", "Artikel", "Menge_Rückläufer"])
        meals = {}
        for i, (bon, arts, grams) in enumerate(zip(t.column("bonid").to_pylist(), t.column("Artikel").to_pylist(),
                                                   t.column("Menge_Rückläufer").to_pylist())):
            meals.setdefault(bon, {"rows": [], "arts": arts, "grams": grams})["rows"].append(i)
        imgs = t.column("image")
        for bon, m in meals.items():
            labels = sorted({PLATE_MAP[a.strip()] for a, g in zip(m["arts"], m["grams"])
                             if g is not None and g >= MIN_RETURN_G})
            if not labels:
                continue
            rows = m["rows"]
            pick = [rows[round(k * (len(rows) - 1) / max(per_meal - 1, 1))] for k in range(min(per_meal, len(rows)))]
            for r in sorted(set(pick)):
                samples.append(Sample(id=f"plate_{bon}_{r}", group=str(bon), labels=labels,
                                      _bytes=imgs[r].as_py()["bytes"]))
    used = sorted({l for s in samples for l in s.labels})
    return Dataset("plates", used, samples,
                   f"Real plate leftovers, {len({s.group for s in samples})} meals x up to {per_meal} photos "
                   f"(different angles), label = every ingredient with >= {MIN_RETURN_G} g returned")


# ---------------------------------------------------------------- foodseg
FOODSEG_PROMPT = {  # clearer wording for some FoodSeg103 class names
    "chicken duck": "chicken or duck", "cheese butter": "cheese or butter", "cilantro mint": "cilantro or mint",
    "hamburg": "hamburger", "hanamaki baozi": "steamed buns", "rape": "bok choy greens",
    "wonton dumplings": "dumplings", "French beans": "french beans", "fried meat": "fried meat",
}


def load_foodseg(n=160, seed=0):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    id2label = json.load(open(hf_hub_download("EduardoPacheco/FoodSeg103", "id2label.json", repo_type="dataset")))
    id2label = {int(k): v.strip() for k, v in id2label.items()}  # two names have a leading space
    t = pq.read_table(hf_hub_download("EduardoPacheco/FoodSeg103", "data/validation-00000-of-00001.parquet",
                                      repo_type="dataset"))
    idx = list(range(t.num_rows))
    random.Random(seed).shuffle(idx)
    samples = []
    for i in idx:
        mask = np.asarray(Image.open(io.BytesIO(t.column("label")[i].as_py()["bytes"])))
        if mask.ndim == 3:
            mask = mask[..., 0]
        counts = np.bincount(mask.ravel(), minlength=len(id2label))
        share = counts / mask.size
        labels = sorted({id2label[c] for c in np.nonzero(share >= MIN_AREA)[0]
                         if c != 0 and id2label[c] not in ("background", "other ingredients")})
        if not labels:
            continue
        sid = str(t.column("id")[i].as_py())
        samples.append(Sample(id=f"foodseg_{sid}", group=sid, labels=labels,
                              _bytes=t.column("image")[i].as_py()["bytes"]))
        if len(samples) >= n:
            break
    vocab = sorted(v for k, v in id2label.items() if k != 0 and v not in ("background", "other ingredients"))
    return Dataset("foodseg", vocab, samples,
                   f"FoodSeg103 validation, {len(samples)} random mixed dishes, 103-ingredient vocabulary, "
                   f"label = ingredient covering >= {MIN_AREA:.0%} of the image", prompt_names=FOODSEG_PROMPT)


# ---------------------------------------------------------------- own photos
OWN_DIR = os.path.join(HERE, "own")


def load_own(folder=OWN_DIR):
    """own/labels.csv: file,labels,lighting,notes   (labels separated by ';', file relative to own/)
    own/vocab.txt (optional): one label per line - the foods the models are asked about."""
    csv_path = os.path.join(folder, "labels.csv")
    if not os.path.exists(csv_path):
        return None
    samples = []
    for row in csv.DictReader(open(csv_path, encoding="utf-8")):
        labels = sorted({l.strip() for l in row["labels"].split(";") if l.strip()})
        path = os.path.join(folder, row["file"])
        if not labels or not os.path.exists(path):
            continue
        s = Sample(id=f"own_{row['file']}", group=row.get("group") or row["file"], labels=labels, path=path)
        s.lighting = (row.get("lighting") or "unknown").strip()
        samples.append(s)
    if not samples:
        return None
    vocab = set(l for s in samples for l in s.labels)
    vp = os.path.join(folder, "vocab.txt")
    if os.path.exists(vp):
        vocab |= {l.strip() for l in open(vp, encoding="utf-8") if l.strip() and not l.startswith("#")}
    # use the exact prompt wording of the live system, so the benchmark tests what vision.py runs
    try:
        import sys
        sys.path.insert(0, os.path.join(HERE, "..", "..", "firmware", "bin_tracker"))
        from vision import FOODS
    except ImportError:
        FOODS = {}
    return Dataset("own", sorted(vocab), samples, f"Own tray photos, {len(samples)} images",
                   synthetic_conditions=False, prompt_names=dict(FOODS))


LOADERS = {"plates": load_plates, "foodseg": load_foodseg, "own": load_own}
