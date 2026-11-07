"""
Models under test. Two kinds:
  "scores"  one score per food label per image (CLIP, SigLIP 2, OWLv2) -> ranked, thresholded for a set
  "set"     directly answers which foods are visible (vision-language model, or the current vision.py rule)
"""
import json
import re

import numpy as np
import torch

TEMPLATES = ["a photo of {}.", "a photo of {}, food waste.", "leftover {} on a tray."]


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


class Base:
    kind = "scores"
    batch = 16

    def __init__(self, name, hf_id, note=""):
        self.name, self.hf_id, self.note = name, hf_id, note
        self.device = "cpu"
        self._text_cache = {}

    def to(self, device):
        self.device = device
        self.model.to(device)
        self._text_cache = {}
        return self


# ---------------------------------------------------------------- CLIP
class Clip(Base):
    def __init__(self, name, hf_id, templates=TEMPLATES, note=""):
        super().__init__(name, hf_id, note)
        self.templates = templates

    def load(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(self.hf_id).eval()
        self.proc = CLIPProcessor.from_pretrained(self.hf_id)
        return self

    @torch.no_grad()
    def text(self, prompts_per_label):
        key = tuple(prompts_per_label)
        if key not in self._text_cache:
            flat = [t.format(p) for p in prompts_per_label for t in self.templates]
            inp = self.proc(text=flat, return_tensors="pt", padding=True).to(self.device)
            e = _norm(self.model.get_text_features(**inp).float())
            e = _norm(e.view(len(prompts_per_label), len(self.templates), -1).mean(1))
            self._text_cache[key] = e
        return self._text_cache[key]

    @torch.no_grad()
    def scores(self, images, prompts_per_label):
        t = self.text(prompts_per_label)
        inp = self.proc(images=images, return_tensors="pt").to(self.device)
        i = _norm(self.model.get_image_features(**inp).float())
        return (i @ t.T * self.model.logit_scale.exp().float()).cpu().numpy()


class VisionPyRule(Clip):
    """Exactly what vision.py does today: CLIP B/32, softmax over labels, top-1 only if >= 0.45."""
    kind = "set"

    def __init__(self):
        super().__init__("clip-b32 (vision.py rule)", "openai/clip-vit-base-patch32",
                         templates=["a photo of {}"], note="current code: one food per image, softmax >= 0.45")

    def predict(self, images, prompts_per_label):
        s = self.scores(images, prompts_per_label)
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return [[int(p[k].argmax())] if p[k].max() >= 0.45 else [] for k in range(len(images))]


# ---------------------------------------------------------------- SigLIP 2
class Siglip(Base):
    def load(self):
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(self.hf_id, dtype=torch.float16).eval()
        self.proc = AutoProcessor.from_pretrained(self.hf_id)
        return self

    @torch.no_grad()
    def text(self, prompts_per_label):
        key = tuple(prompts_per_label)
        if key not in self._text_cache:
            flat = [t.format(p).lower() for p in prompts_per_label for t in TEMPLATES]
            inp = self.proc(text=flat, return_tensors="pt", padding="max_length", max_length=64).to(self.device)
            e = _norm(self.model.get_text_features(**inp).float())
            self._text_cache[key] = _norm(e.view(len(prompts_per_label), len(TEMPLATES), -1).mean(1))
        return self._text_cache[key]

    @torch.no_grad()
    def scores(self, images, prompts_per_label):
        t = self.text(prompts_per_label)
        inp = self.proc(images=images, return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].to(self.model.dtype)
        i = _norm(self.model.get_image_features(**inp).float())
        logits = i @ t.T * self.model.logit_scale.exp().float() + self.model.logit_bias.float()
        return torch.sigmoid(logits).cpu().numpy()


# ---------------------------------------------------------------- OWLv2 (open-vocabulary detector)
class Owl(Base):
    # OWLv2 looks at 960x960 = 3600 patches; its attention maps need ~0.3 GB per layer per image in fp16,
    # so one image at a time keeps it well inside 8 GB of VRAM.
    batch = 1

    def load(self):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.model = Owlv2ForObjectDetection.from_pretrained(self.hf_id, dtype=torch.float16).eval()
        self.proc = Owlv2Processor.from_pretrained(self.hf_id)
        return self

    @torch.no_grad()
    def scores(self, images, prompts_per_label):
        queries = [f"a photo of {p}" for p in prompts_per_label]
        inp = self.proc(text=[queries] * len(images), images=images, return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].to(self.model.dtype)
        out = self.model(**inp)
        # image-level score for a food = its best box anywhere in the image
        return torch.sigmoid(out.logits).max(dim=1).values.float().cpu().numpy()


# ---------------------------------------------------------------- Qwen3-VL (vision-language model)
class Vlm(Base):
    kind = "set"
    batch = 1
    max_side = 768

    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(self.hf_id, dtype=torch.bfloat16).eval()
        self.proc = AutoProcessor.from_pretrained(self.hf_id)
        self.raw = []  # keep the raw answers for inspection
        return self

    @torch.no_grad()
    def predict(self, images, prompts_per_label):
        out = []
        for img in images:
            img = img.copy()
            img.thumbnail((self.max_side, self.max_side))
            prompt = ("This photo shows food waste (leftovers or kitchen scraps), possibly in bad light or "
                      "at an angle. Which of these foods are visible? Answer only with a JSON list on one "
                      "line, names copied exactly from this list, or [] if none:\n" + json.dumps(prompts_per_label))
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                return_dict=True, return_tensors="pt").to(self.device)
            gen = self.model.generate(**inp, max_new_tokens=96, do_sample=False)
            txt = self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            self.raw.append(txt)
            out.append(parse_labels(txt, prompts_per_label))
        return out


def parse_labels(txt, names):
    low = [n.lower() for n in names]
    m = re.search(r"\[.*?\]", txt, re.S)
    items = []
    if m:
        try:
            items = [str(x) for x in json.loads(m.group(0))]
        except json.JSONDecodeError:
            items = re.findall(r'"([^"]+)"', m.group(0))
    else:
        items = [x for x in re.split(r"[,\n]", txt)]
    hits = set()
    for it in items:
        it = it.strip().strip('"').lower()
        if it in low:
            hits.add(low.index(it))
        else:  # tolerate plural / small wording differences
            for k, n in enumerate(low):
                if it and (it.rstrip("s") == n.rstrip("s") or (len(it) > 3 and it in n) or (len(n) > 3 and n in it)):
                    hits.add(k)
                    break
    return sorted(hits)


MODELS = {
    "clip-b32-visionpy": lambda: VisionPyRule(),
    "clip-b32": lambda: Clip("clip-b32", "openai/clip-vit-base-patch32", note="baseline, 0.6 GB"),
    "clip-l14": lambda: Clip("clip-l14", "openai/clip-vit-large-patch14", note="bigger CLIP, 1.7 GB"),
    "siglip2-b16": lambda: Siglip("siglip2-b16", "google/siglip2-base-patch16-384", note="1.5 GB"),
    "siglip2-so400m": lambda: Siglip("siglip2-so400m", "google/siglip2-so400m-patch14-384", note="4.5 GB"),
    "owlv2": lambda: Owl("owlv2", "google/owlv2-base-patch16-ensemble", note="detector, 0.6 GB"),
    "qwen3vl-2b": lambda: Vlm("qwen3vl-2b", "Qwen/Qwen3-VL-2B-Instruct", note="VLM, 4.3 GB"),
}
