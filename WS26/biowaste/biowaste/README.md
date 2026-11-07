# Bio Waste Tracker – team plan and code (WebMeister360, CIC WS26)

Team: **Sai** (hardware/firmware), **Sunil** (mechanics, test campaign, accuracy), **Phil** (software/dashboard), **Claude** (code, docs, pitch support).
Deadline: **Saturday 13:30 submission**, 15:00 pitch.

## Concept in one line
A small **weighing tray (≤ 4.5 kg) on the 5 kg load cell in front of the bin**. Cooks tip onto the tray, the
ESP32 records each tip as a step in weight, and **3 big buttons** (wet-glove proof) set the type. When the
tray is full, the cook tips it into the real bin (logged as "removal", never as a disposal).
No button press → the laptop applies a **time rule** and marks the record `confidence=low`. Spoiled stock is never guessed.

## Wiring
| HX711 | ESP32 | | Buttons (to GND, internal pull-up) | ESP32 |
|---|---|---|---|---|
| VCC | 3V3 | | Preparation waste | GPIO25 |
| GND | GND | | Spoiled stock | GPIO26 |
| DT / DOUT | GPIO16 | | Plate leftovers | GPIO27 |
| SCK | GPIO4 | | Status LED | GPIO2 (on-board) |
| | | | Lid switch (bin camera) | GPIO33 |
| | | | Bin lamp MOSFET gate (bin camera) | GPIO32 |

Load cell → HX711: red E+, black E-, white A-, green A+ (if weight goes negative, swap A+/A-).

LED: **fast blink** = weighed, press a type · **short flash** = saved · **solid** = tray near 4.5 kg, empty it.

## Run it
1. Arduino IDE: board "ESP32 Dev Module", install library **HX711 by Bogdan Necula**, flash `firmware/bin_tracker/bin_tracker.ino`.
2. Serial monitor 115200, empty tray: `t` (tare). Put a known weight (use the kitchen scale to weigh e.g. an apple): `c<grams>`, e.g. `c212`. Calibration is saved in flash (survives power cuts).
3. Close the serial monitor, then on the laptop: `pip install pyserial` and
   `python3 server/server.py --port /dev/ttyUSB0` (Windows `COM5`, macOS `/dev/cu.usbserial-*`).
4. Open http://localhost:8000. Without hardware: `python3 server/server.py --simulate` (writes to `data_sim/`, never into the real log).
5. Bin camera (which food went in): add `--camera 0` (USB webcam) or `--camera http://<phone-ip>:8080`
   (IP Webcam app) to the server command. Lid opens → before photo; lid closes → after photo. Only the new
   food is cut out and named with SigLIP 2, and the dashboard shows it per disposal. Without hardware:
   `python3 server/server.py --simulate --camera replay`.
   **Wiring, mounting, testing and knobs: [docs/camera-setup.md](docs/camera-setup.md).**

Outputs: `server/data/events.csv` (D2: record_no, timestamp, weight_g, type, station_id, confidence, source),
`server/data/raw_log.jsonl` (every removal, ignored bump, unstable reading, overload, lid event – the honest log)
and, with the camera, `vision.jsonl` + `vision_links.csv` + photos in `vision/` (see the camera doc).

## Disposal detection (firmware)
10 Hz → median of 5 → 1.2 s window. IDLE → weight leaves baseline by >12 g → MOVING → window flat (range ≤ 6 g)
→ delta = stable − baseline: ≥ +10 g disposal · ≤ −10 g removal · else bump (ignored, logged). Slow drift tracking while idle.
Type: button up to 15 s **before** the tip, or within 8 s **after** it settles. All thresholds are constants at the top of the sketch – tune them tonight.

## Test protocol (Sunil – this is our differentiator)
Before each tip: weigh the item on the reference kitchen scale and write a line in `tools/ground_truth.csv`
(`test_no,true_weight_g,true_type,record_no,notes`). Fill `record_no` from the dashboard; leave empty if nothing appeared.
Then `python3 tools/accuracy.py tools/ground_truth.csv server/data/events.csv` → detection rate, weight error, type accuracy, confusion matrix.
Run a realistic "kitchen day" of 30+ tips including edge cases: two tips 3 s apart, forgotten button, bump the tray,
lean on it, empty the tray, wet/greasy item, very small item (< 20 g). Keep every failure in the log.

## Timeline
| Time | Sai | Sunil | Phil |
|---|---|---|---|
| 20:00–21:30 | Wire HX711 + buttons, flash, calibrate | Build tray/insert + button box (cardboard), reference scale ready | Run `--simulate`, adapt dashboard |
| 21:30–23:30 | Tune thresholds with real tips | First 10 tips with ground truth, find failures | Integrate serial, test on Sai's laptop |
| 23:30–01:30 | Fix what the tests found | Edge-case tests | D4 decisions + D5 scalability draft |
| 01:30–06:00 | Sleep in shifts – one person keeps the setup running | | |
| 06:00–09:00 | Freeze firmware, final calibration | **Full test day: 30+ tips → D2 log + accuracy report** | D3 insight from real data |
| 09:00–11:30 | D6 risks + mitigation | Photos/video of setup (backup if demo fails) | Pitch deck |
| 11:30–13:00 | Rehearse pitch 3× with 3 live disposals | | |
| 13:30 | **Submit** | | |

## Deliverable owners
D1 prototype – Sai · D2 log – Sunil · D3 insight – Phil (dashboard computes it) · D4 decisions – Sai ·
D5 scalability & cost – Phil + Claude · D6 risks – Sunil + Sai · D7 dashboard – Phil · Pitch – all.
