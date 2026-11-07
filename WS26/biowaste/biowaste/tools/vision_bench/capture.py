#!/usr/bin/env python3
"""
Collect your own benchmark photos with the real setup (phone on the IP Webcam app above the tray).

    python capture.py --phone http://192.168.23.223:8080

Per item: Enter -> BEFORE photo, put the food on the tray, Enter -> AFTER photo, then type what is on it.
Saves exactly what vision.py would classify (the changed region of the tray) to own/images/ and appends
a line to own/labels.csv. Then run:  python bench.py run --datasets own

Tips for a useful test set (aim for 40+ items):
  * mix foods in one tip (pasta + sauce, peels + bread), not only single items
  * shoot the bad cases on purpose: lights off / one lamp, phone tilted, steam, wet and greasy food
  * set the lighting tag honestly - the report splits the results by it
Privacy: the phone must see only the tray. Only the cropped tray region is stored.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "firmware", "bin_tracker"))
from vision import changed_region, snapshot  # noqa: E402  (same crop logic as the live system)

OWN = os.path.join(HERE, "own")
LIGHTING = ["good", "dim", "warm", "mixed", "glare"]


def load_vocab():
    p = os.path.join(OWN, "vocab.txt")
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.startswith("#")]


def ask_labels(vocab):
    while True:
        raw = input("Foods on it (numbers or names, separated by ',' - '?' shows the list): ").strip()
        if raw == "?":
            print("  " + "   ".join(f"{k}:{v}" for k, v in enumerate(vocab)))
            continue
        out, bad = [], []
        for tok in (t.strip() for t in raw.split(",") if t.strip()):
            if tok.isdigit() and int(tok) < len(vocab):
                out.append(vocab[int(tok)])
            elif tok.lower() in vocab:
                out.append(tok.lower())
            else:
                bad.append(tok)
        if bad:
            add = input(f"  not in vocab.txt: {bad} - add them? [y/N] ").strip().lower() == "y"
            if not add:
                continue
            with open(os.path.join(OWN, "vocab.txt"), "a", encoding="utf-8") as f:
                for b in bad:
                    f.write(b.lower() + "\n")
                    vocab.append(b.lower())
            out += [b.lower() for b in bad]
        if out:
            return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone", required=True, help="IP Webcam address, e.g. http://192.168.23.223:8080")
    a = ap.parse_args()
    img_dir = os.path.join(OWN, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(OWN, "labels.csv")
    new = not os.path.exists(csv_path)
    n = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    vocab = load_vocab()
    lighting = "good"
    print(f"{n} photos so far. Ctrl+C to stop.")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "labels", "lighting", "notes"])
        while True:
            input("\nTray as it is now -> Enter for BEFORE photo")
            before = snapshot(a.phone)
            input("Food on the tray -> Enter for AFTER photo")
            after = snapshot(a.phone)
            box = changed_region(before, after)
            if box is None:
                print("  nothing changed between the photos - try again")
                continue
            crop = after.crop(box)
            labels = ask_labels(vocab)
            t = input(f"Lighting {LIGHTING} [Enter = {lighting}]: ").strip().lower()
            lighting = t if t in LIGHTING else lighting
            notes = input("Notes (optional): ").strip()
            n += 1
            name = f"images/{n:04d}.jpg"
            crop.save(os.path.join(OWN, name), quality=90)
            w.writerow([name, ";".join(labels), lighting, notes])
            f.flush()
            print(f"  saved {name}: {', '.join(labels)} ({lighting})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
