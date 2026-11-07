# food_cam – tray camera, automatic before/after (Sai)

Phone fixed 30–50 cm above the weighing tray, looking straight down. It never moves.

```
STILL   -> latest picture is BEFORE
MOVING  -> hand + falling food (these frames are ignored)
STILL again for 1 s -> AFTER
diff(BEFORE, AFTER) -> crop of the new food -> Claude names it
weight record from the scale (events.csv or /weight) is attached, AFTER becomes the next BEFORE
```

## Run (Windows)
```powershell
py -3.12 -m pip install opencv-python numpy
copy .env.example .env      # fill in ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN
py -3.12 tray_cam.py --camera sim --offline                    # test: no phone, no API
py -3.12 tray_cam.py --camera http://PHONE_IP:8080/shot.jpg    # Android "IP Webcam" app
py -3.12 tray_cam.py --camera 0                                # DroidCam / Iriun / USB
#   add --events ..\server\data\events.csv to attach weights
```
Dashboard: open http://localhost:8001/ (live camera, last snap with before/after, totals, history,
buttons for "New before" and "Snap now"). From another device on the same network: http://PC_IP:8001/

Keys: SPACE manual snap · b new before-frame · d motion numbers · q quit
HTTP :8001 → `/status` `/last` `/history` `/snap?weight_g=` `/weight?weight_g=&record_no=&type=` `/img/<file>`

Output: `captures/food_log.csv` + `captures/snap_XXX_{before,after,crop,full}.jpg`

Tuning (top of `tray_cam.py`): `STILL_S` (raise to 1.5–2 if the hand ends up in AFTER),
`DIFF_THRESHOLD`, `MIN_AREA_FRAC`, `MOTION_FRAC`. Model: `FOOD_CAM_MODEL` in `.env` (default claude-haiku-5-5).

Tested with `--camera sim`: 3 throws in a row, hand never in BEFORE/AFTER, only the new item boxed,
weights from a test events.csv matched to the right photo.
