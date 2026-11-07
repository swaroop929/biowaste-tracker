"""
Simulated kitchen conditions. Deterministic per (image, condition), so every model sees the same pixels.

  clean        the original photo
  dim          under-exposed + sensor noise (dark corner, no extra light)
  warm_light   yellow tungsten / warm LED cast, slightly darker
  motion_blur  tray or phone moving while the photo is taken
  tilted       camera not straight above: perspective tilt + rotation
  glare        bright reflections from ceiling lights on wet food / steel tray
  worst        tilted + warm + dim + motion blur + heavy JPEG (cheap camera stream)
"""
import hashlib
import io

import cv2
import numpy as np
from PIL import Image


def _f(img):
    return np.asarray(img).astype(np.float32) / 255.0


def _img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def dim(img, rng, gain=(0.30, 0.40), noise=0.03):
    a = _f(img) ** 1.5 * rng.uniform(*gain)
    a = a + rng.normal(0, noise, a.shape) + rng.normal(0, noise / 2, a.shape[:2] + (1,))  # colour + luma noise
    return _img(a)


def warm(img, rng):
    a = _f(img) * np.array([1.10, 0.93, 0.60], dtype=np.float32) * rng.uniform(0.75, 0.9)
    return _img(a)


def motion_blur(img, rng, length=(0.015, 0.025)):
    a = np.asarray(img)
    k = max(5, int(a.shape[1] * rng.uniform(*length)) | 1)
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kern = cv2.warpAffine(kern, rot, (k, k))
    kern /= kern.sum()
    return Image.fromarray(cv2.filter2D(a, -1, kern))


def tilted(img, rng):
    a = np.asarray(img)
    h, w = a.shape[:2]
    s = rng.uniform(0.18, 0.28)  # far edge shrinks -> camera looks at the tray at ~30-40 deg
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * s, h * s * 0.6], [w * (1 - s), h * s * 0.6], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, dst)
    # fill uncovered corners with the photo's own edge colour (tray / table), not mirrored copies
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(edge, axis=0))
    out = cv2.warpPerspective(a, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-25, 25), 1.0)
    return Image.fromarray(cv2.warpAffine(out, rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=fill))


def glare(img, rng):
    a = _f(img)
    h, w = a.shape[:2]
    spot = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(1, 3)):
        c = (int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h))
        ax = (int(rng.uniform(0.08, 0.18) * w), int(rng.uniform(0.04, 0.10) * h))
        cv2.ellipse(spot, c, ax, rng.uniform(0, 180), 0, 360, 1.0, -1)
    spot = cv2.GaussianBlur(spot, (0, 0), w * 0.03)[..., None]
    return _img(a * (1 - 0.85 * spot) + 0.85 * spot)


def jpeg(img, rng, q=30):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


CONDITIONS = {
    "clean": [],
    "dim": [dim],
    "warm_light": [warm],
    "motion_blur": [motion_blur],
    "tilted": [tilted],
    "glare": [glare],
    # milder steps combined: a cheap phone stream in a badly lit corner, still readable for a person
    "worst": [tilted, warm, lambda im, r: dim(im, r, gain=(0.55, 0.65), noise=0.025),
              lambda im, r: motion_blur(im, r, length=(0.008, 0.012)), jpeg],
}


def apply(img, condition, sample_id):
    seed = int(hashlib.md5(f"{sample_id}|{condition}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    for fn in CONDITIONS[condition]:
        img = fn(img, rng)
    return img
