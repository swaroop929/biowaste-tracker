# Bin camera – setup and handover

The camera tells us **which food** went into the bin. The scale and the buttons still decide the
**weight** and the **waste type**; the camera only adds the food name, for example "62 % of the
preparation waste is pasta".

Everything in software is built and tested **without hardware**. This guide covers what is left to do
once the parts are here, and how to check each step.

## How it works

```
 lid opens ──► ESP32 sends {"event":"lid","state":"open"}, turns the bin lamp on
               laptop takes the BEFORE photo (0.3 s later)
 food goes in ─► scale logs the disposal (weight, type button) as usual
 lid closes ─► ESP32 sends {"event":"lid","state":"closed"}, lamp stays on 4 s
               laptop waits 1 s, takes the AFTER photo
 laptop ─────► vision.recognise(before, after)
                 1. what is new? the photos are aligned, the light evened out, colour and
                    texture compared → a mask of each new piece
                 2. each piece is cut out with the old waste greyed out
                 3. SigLIP 2 names each piece (several foods per disposal are fine)
               → data/vision.jsonl, photos in data/vision/<session>/
               → linked to the disposal(s) the scale logged while the lid was open
               → dashboard: "Foods" column, "mostly …" per waste type, bin camera card
```

| Part | File |
|---|---|
| Lid switch, lamp, lid events | `firmware/bin_tracker/bin_tracker.ino` |
| Change detection + classification | `firmware/bin_tracker/vision.py` (`recognise`, `change_mask`, `cutout`, `classify`) |
| Cameras, lid sessions, background worker | `server/bin_camera.py` |
| Linking to disposals, API, files | `server/server.py` |
| Dashboard | `server/dashboard.html` |
| Tests and model choice | `tools/vision_bench/` (`bin_test.py`, `README.md`) |

## Parts

* **Camera.** A USB webcam (720p is enough; fixed focus or autofocus) or an old phone running the
  **IP Webcam** app. Get a wide-angle one if the bin is small: it must see the whole opening from
  about 20–40 cm.
* **Lid switch.** A reed switch plus a magnet, or a micro switch that the closed lid presses.
* **Lamp.** A short white LED strip (5 V or 12 V) plus a logic-level MOSFET (IRLZ44N or similar),
  a 220 Ω gate resistor and a 10 kΩ pull-down. *This matters most:* in the dark test pairs recognition
  drops from F1 93 to 27.

## Wiring (ESP32)

| What | ESP32 pin | Notes |
|---|---|---|
| Lid switch | GPIO33 ↔ GND | internal pull-up. Reed switch closed (magnet near) = lid shut = LOW |
| Lamp MOSFET gate | GPIO32 via 220 Ω | 10 kΩ gate→GND. LED strip + to supply, − to drain, source to GND |
| (unchanged) HX711, buttons, status LED | see the main README | |

Then in `bin_tracker.ino`:

```c
#define USE_LID_SWITCH 1          // was 0: without a switch the pin would read "open" forever
const int LID_OPEN_LEVEL = HIGH;  // flip to LOW if your switch is closed when the lid is OPEN
```

Flash it, open the serial monitor (115200) and lift the lid. You should see
`{"event":"lid","state":"open",...}` and then `closed`. The lamp turns on with the lid and off 4 s
after it closes. Serial commands for testing:

* `o` / `x`: pretend the lid opened / closed (works without a switch).
* `L1` / `L0`: lamp forced on / back to automatic (for aiming the camera).

## Mounting the camera

* **Fix it to the bin body** (under the rim or on the wall), looking down into the bin. **Never mount
  it on the lid**: a lid camera moves between the two photos, and nothing can be compared.
* The lid must not show up in the picture when it is open. Otherwise the open lid itself counts
  as "new".
* Put the lamp next to the camera, pointing the same way. A light opposite the camera causes glare.
* Use the full bin opening. The new piece should cover at least ~1 % of the image; smaller pieces
  are ignored (`MIN_PART`).
* Steady light and a fixed position matter more than resolution.

## Software

```bash
pip install pyserial "transformers>=4.49" pillow numpy opencv-python
pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU; plain `pip install torch` = CPU
```

The first start downloads the model: SigLIP 2 so400m (4.5 GB) on a GPU with ≥ 3 GB free VRAM,
otherwise SigLIP 2 base (1.5 GB). Do this once on good Wi-Fi before the event. On the RTX 3050
(4 GB) it should pick so400m (2.3 GB peak). Check `nvidia-smi` first.

## Running it

```bash
cd server
python server.py --port COM5 --camera 0                         # scale + USB webcam 0
python server.py --port COM5 --camera http://192.168.23.223:8080  # scale + phone (IP Webcam)
python server.py --port COM5 --camera 0 --vision-model base     # force the small model
```

Open http://localhost:8000. The bin camera card at the bottom shows camera, model, lid state and the
last before/after photos with the cut-out pieces.

### Testing step by step (no need to wait for all the hardware)

1. **No hardware at all:** `python server.py --simulate --camera replay`. The simulated scale opens and
   closes the lid around every disposal, and the replay camera plays the 10 test pairs from
   `tools/vision_bench/binpairs`. The dashboard shows what was found *and* the true answer. Results go
   to `data_sim/`, never into the real log.
2. **Camera only, no lid switch:** start with `--port ... --camera 0`, then open
   `http://localhost:8000/api/lid?state=open`, drop food in, then open `...?state=closed`. Or type `o`
   and `x` in a serial terminal, *or* the server's serial port; only one program can hold it at a time.
3. **Camera alone, without the server:** `python firmware/bin_tracker/vision.py --phone http://<ip>:8080`
   (Enter for the before photo, Enter for the after photo).
4. **Everything:** set `USE_LID_SWITCH 1`, flash, start the server with `--camera`, and throw things in.

## Files it writes (`server/data/`)

| File | Content |
|---|---|
| `events.csv` | The D2 log, **unchanged** (one row per disposal) |
| `vision_links.csv` | `record_no,session`: which disposal happened during which lid opening |
| `vision.jsonl` | One line per lid opening: `session, opened, closed, close_reason, foods, food_conf, parts[{box, foods, image}], model, ms, errors` |
| `vision/<session>/` | `before.jpg`, `after.jpg`, `piece_<k>.jpg` (what the classifier saw) |

To get the foods per disposal, join `events.csv` → `vision_links.csv` → `vision.jsonl`. The server does
this on start, so restarts keep the foods. A lid opening with several disposals gives all of them the
same foods. A disposal without a lid opening (lid propped open for more than 60 s, camera off) has no
foods; the dashboard leaves the field empty.

The saved photos are your own test data. Label them in `tools/vision_bench/own/` and run the benchmark
on the real bin (see `tools/vision_bench/README.md`).

## Knobs

| Constant | File | Default | What it does |
|---|---|---|---|
| `SETTLE_S` | `bin_camera.py` | 1.0 s | Wait after the lid closes before the AFTER photo |
| `CAMERA_WARMUP_S` | `bin_camera.py` | 0.3 s | Wait after the lid opens (lamp, exposure) before the BEFORE photo |
| `MAX_OPEN_S` | `bin_camera.py` | 60 s | Lid open longer → AFTER photo anyway |
| `LINK_S` | `bin_camera.py` | 15 s | A disposal up to this long after the lid closed still belongs to it |
| `LAMP_AFTER_MS` | `bin_tracker.ino` | 4000 | Lamp stays on after the lid closes (must be > `SETTLE_S`) |
| `MIN_PART` | `vision.py` | 0.4 % | Smallest new piece |
| `DIFF_FLOOR` | `vision.py` | 14 | Colour difference that never counts as new. Raise it if light flicker creates false pieces |
| `MAX_PARTS`, `PIECE_REL`, `PIECE_MAX` | `vision.py` | 3, 0.5, 2 | Pieces per disposal; labels per piece |
| `MODELS[...]` threshold | `vision.py` | so400m 0.05, base 0.02 | Minimum score for a food |
| `FOODS` | `vision.py` | 26 foods | What the camera can name. Add what the kitchen really throws away (keep `tools/vision_bench/own/vocab.txt` the same) |

## Test results so far (no real bin yet)

These come from `tools/vision_bench/bin_test.py` on 10 pairs built from real food-waste photos with
real food dropped in (see `binpairs/`):

* New food found: 0.79 pixel IoU (the old grey-level box scored 0.17).
* Right foods named: **F1 93 in a lit bin, 27 in a dark bin**. Classifying the whole photo instead
  scores 5, because the model describes the old waste.
* About 0.4–0.7 s per disposal on an RTX 3070. The photos are taken while the lid is open; recognition
  runs in the background.

## Known limits / to do

* **Not tested on the real bin.** The test pairs are composites. First job with hardware: 30+ real
  throws, then check the bin camera card and `data/vision/`.
* **Dark bin = poor results.** Fix it with the lamp. Correcting the photos afterwards made it worse.
* **Hands in the BEFORE photo.** If the photo at lid-open already shows a hand or the food, the hand
  ends up in "before" and the food may be missed. If that happens, raise `CAMERA_WARMUP_S` only a
  little, or use the previous AFTER photo as the next BEFORE photo (`bin_camera.LidSessions._open`).
* **Waste that moves.** New food that pushes old waste around makes the moved old waste count as "new"
  too. Pieces are classified separately, so this usually adds an extra label rather than hiding the
  new one.
* **Food names are a hint, not the waste type.** The type still comes from the buttons and the time rule.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Card says "model failed" | No torch/transformers, or no internet on the first start. Read the server console |
| Every photo takes seconds, RAM grows | GPU full and Windows spills into RAM. Close other GPU programs or use `--vision-model base` |
| "nothing recognised" all the time | Too dark, camera moved, or pieces too small. Look at `data/vision/<session>/` |
| Old waste marked as new everywhere | Camera or bin moved a lot, or flickering light. Fix the mount, raise `DIFF_FLOOR` |
| Foods column empty | No lid event: check `lid opened` in the honest log and `USE_LID_SWITCH` |
| `USB camera 0 not found` | Try `--camera 1`, or close other apps that use the webcam |
