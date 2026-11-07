/*
 * Bio Waste Tracker - ESP32 firmware (WebMeister360 challenge, CIC WS26)
 *
 * Hardware
 *   HX711 DOUT -> GPIO16, HX711 SCK -> GPIO4, HX711 VCC -> 3V3, GND -> GND
 *   Button PREP  (preparation_waste) -> GPIO25 to GND
 *   Button SPOIL (spoiled_stock)     -> GPIO26 to GND
 *   Button PLATE (plate_leftovers)   -> GPIO27 to GND
 *   Status LED -> GPIO2 (on-board LED on most dev boards)
 *
 * Library: "HX711 Arduino Library" by Bogdan Necula (bogde) - Library Manager.
 *
 * What it does
 *   The load cell weighs a small WEIGHING TRAY / insert (never the whole bin).
 *   Every 100 ms a reading is taken and median-filtered. A disposal is a step
 *   change in weight that settles to a new stable value:
 *       baseline --(weight leaves the noise band)--> MOVING
 *       MOVING   --(stable for STABLE_MS)--> delta = stable - baseline
 *         delta >= +MIN_DISPOSAL_G  -> disposal event (one record)
 *         delta <= -MIN_DISPOSAL_G  -> removal (tray emptied) - NOT a disposal
 *         otherwise                  -> bump / noise, ignored but logged
 *   Type: press a type button up to PRESELECT_MS before tipping, or within
 *   TYPE_WINDOW_MS after the weight settles. No press -> "unassigned"; the
 *   laptop applies the time rule and marks confidence low.
 *
 * Output: one JSON object per line on Serial (115200 baud).
 *   {"event":"disposal","id":7,"weight_g":312.4,"type":"plate_leftovers","source":"button","ms":123456}
 *   {"event":"weight","g":1204.1,"stable":1,"state":"idle"}          (every 500 ms, live view)
 *   {"event":"removal"|"ignored"|"unstable"|"overload"|"info"|..., ...}
 *
 * Serial commands (send with newline):
 *   t        tare (tray must be empty)
 *   c500     calibrate with a known 500 g weight on the tray (any grams value)
 *   f420.5   set a known calibration factor (from loadcell_check: "calibrated, factor ...")
 *   i        print calibration info
 *   z        reset event counter
 *   k1/k2/k3 type key from the laptop (same as pressing button 1/2/3)
 */

#include <Arduino.h>
#include <HX711.h>
#include <Preferences.h>

// ---------------- Pins ----------------
const int PIN_HX_DOUT   = 16;
const int PIN_HX_SCK    = 4;
const int PIN_BTN_PREP  = 25;
const int PIN_BTN_SPOIL = 26;
const int PIN_BTN_PLATE = 27;
const int PIN_LED       = 2;

// ---------------- Tunables (tune tonight with real tests) ----------------
const uint32_t SAMPLE_MS        = 100;    // 10 Hz
const int      STABLE_SAMPLES   = 12;     // 1.2 s window must be flat
const float    STABLE_BAND_G    = 6.0;    // max-min inside window to count as stable
const float    TRIGGER_G        = 12.0;   // leave baseline by this much -> MOVING
const float    MIN_DISPOSAL_G   = 10.0;   // smaller settled steps are noise
const uint32_t TYPE_WINDOW_MS   = 8000;   // wait for button after settling
const uint32_t PRESELECT_MS     = 15000;  // button pressed before tipping still counts
const uint32_t UNSTABLE_MS      = 20000;  // leaning on the tray etc.
const float    CAPACITY_G       = 4500.0; // 5 kg cell incl. tray - warn before overload
const uint32_t STREAM_MS        = 500;
const float    DRIFT_ALPHA      = 0.02;   // slow zero tracking while idle

// ---------------- State ----------------
HX711 scale;
Preferences prefs;

float calFactor = 420.0;   // overwritten by calibration
long  hxOffset  = 0;

const int MED_N = 5;
float medBuf[MED_N];
int   medIdx = 0, medCount = 0;

float win[STABLE_SAMPLES];
int   winIdx = 0, winCount = 0;

enum State { IDLE, MOVING };
State state = IDLE;
float baseline = 0;
uint32_t movingSince = 0;
bool unstableReported = false;

uint32_t eventId = 0;

// pending disposal waiting for a type button
bool     pendingActive = false;
uint32_t pendingId = 0;
float    pendingWeight = 0;
uint32_t pendingSince = 0;

// pre-selected type (button pressed before tipping)
int      preType = -1;
uint32_t preAt = 0;

const char* TYPE_NAMES[] = {"preparation_waste", "spoiled_stock", "plate_leftovers"};
const int   BTN_PINS[]   = {PIN_BTN_PREP, PIN_BTN_SPOIL, PIN_BTN_PLATE};
bool        btnLast[3]   = {true, true, true};
uint32_t    btnChangedAt[3] = {0, 0, 0};

uint32_t lastSample = 0, lastStream = 0;
float filtered = 0;
bool overloadReported = false;

// LED feedback
uint32_t ledUntil = 0;
int ledPattern = 0; // 0 off, 1 waiting-for-type fast blink, 2 confirm, 3 overload

// ---------------- Helpers ----------------
float median5(float* a, int n) {
  float t[MED_N];
  for (int i = 0; i < n; i++) t[i] = a[i];
  for (int i = 1; i < n; i++) {            // insertion sort, n <= 5
    float k = t[i]; int j = i - 1;
    while (j >= 0 && t[j] > k) { t[j + 1] = t[j]; j--; }
    t[j + 1] = k;
  }
  return t[n / 2];
}

void winPush(float v) {
  win[winIdx] = v;
  winIdx = (winIdx + 1) % STABLE_SAMPLES;
  if (winCount < STABLE_SAMPLES) winCount++;
}

void winReset() { winIdx = 0; winCount = 0; }

bool winStable(float* meanOut) {
  if (winCount < STABLE_SAMPLES) return false;
  float mn = win[0], mx = win[0], sum = 0;
  for (int i = 0; i < STABLE_SAMPLES; i++) {
    mn = min(mn, win[i]); mx = max(mx, win[i]); sum += win[i];
  }
  if (meanOut) *meanOut = sum / STABLE_SAMPLES;
  return (mx - mn) <= STABLE_BAND_G;
}

void emitDisposal(uint32_t id, float grams, const char* type, const char* source) {
  Serial.printf("{\"event\":\"disposal\",\"id\":%lu,\"weight_g\":%.1f,\"type\":\"%s\",\"source\":\"%s\",\"ms\":%lu}\n",
                (unsigned long)id, grams, type, source, (unsigned long)millis());
}

void finalizePending(int type, const char* source) {
  if (!pendingActive) return;
  emitDisposal(pendingId, pendingWeight, type >= 0 ? TYPE_NAMES[type] : "unassigned", source);
  pendingActive = false;
  ledPattern = 2; ledUntil = millis() + 600;
}

void saveCal() {
  prefs.begin("bwt", false);
  prefs.putFloat("cal", calFactor);
  prefs.putLong("off", hxOffset);
  prefs.end();
}

void onButton(int t) {
  uint32_t now = millis();
  if (pendingActive) {
    finalizePending(t, "button");          // type the disposal that just settled
  } else {
    preType = t; preAt = now;              // type the next disposal
    Serial.printf("{\"event\":\"preselect\",\"type\":\"%s\",\"ms\":%lu}\n", TYPE_NAMES[t], (unsigned long)now);
    ledPattern = 2; ledUntil = now + 300;
  }
}

void handleSerial() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd == "t") {
    scale.tare(20);
    hxOffset = scale.get_offset();
    saveCal();
    baseline = 0; winReset(); medCount = 0; state = IDLE;
    Serial.println("{\"event\":\"info\",\"msg\":\"tared\"}");
  } else if (cmd.startsWith("c")) {
    float grams = cmd.substring(1).toFloat();
    if (grams <= 0) { Serial.println("{\"event\":\"info\",\"msg\":\"usage: c<grams>\"}"); return; }
    scale.set_scale(1.0);
    float raw = scale.get_units(20);       // (raw - offset) with scale 1
    calFactor = raw / grams;
    scale.set_scale(calFactor);
    saveCal();
    baseline = scale.get_units(10); winReset(); medCount = 0; state = IDLE;
    Serial.printf("{\"event\":\"info\",\"msg\":\"calibrated\",\"factor\":%.4f}\n", calFactor);
  } else if (cmd.startsWith("f")) {
    float f = cmd.substring(1).toFloat();   // factor from loadcell_check / calibrate (counts per gram)
    if (f == 0) { Serial.println("{\"event\":\"info\",\"msg\":\"usage: f<factor>\"}"); return; }
    calFactor = f;
    scale.set_scale(calFactor);
    saveCal();
    baseline = scale.get_units(10); winReset(); medCount = 0; state = IDLE;
    Serial.printf("{\"event\":\"info\",\"msg\":\"factor set\",\"factor\":%.4f}\n", calFactor);
  } else if (cmd == "i") {
    Serial.printf("{\"event\":\"info\",\"factor\":%.4f,\"offset\":%ld,\"baseline\":%.1f}\n", calFactor, hxOffset, baseline);
  } else if (cmd == "k1" || cmd == "k2" || cmd == "k3") {
    onButton(cmd.charAt(1) - '1');         // type key from the laptop/dashboard = virtual button press
  } else if (cmd == "z") {
    eventId = 0;
    Serial.println("{\"event\":\"info\",\"msg\":\"counter reset\"}");
  }
}

void updateLed() {
  uint32_t now = millis();
  bool on = false;
  if (filtered > CAPACITY_G) on = true;                        // solid: empty the tray
  else if (pendingActive) on = (now / 150) % 2;                 // fast blink: press type
  else if (ledPattern == 2 && now < ledUntil) on = true;        // short flash: confirmed
  digitalWrite(PIN_LED, on ? HIGH : LOW);
}

// ---------------- Setup / loop ----------------
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  for (int i = 0; i < 3; i++) pinMode(BTN_PINS[i], INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);

  scale.begin(PIN_HX_DOUT, PIN_HX_SCK);

  // Restore calibration so a power cut mid-shift does not zero a full tray.
  prefs.begin("bwt", true);
  bool haveCal = prefs.isKey("cal");
  calFactor = prefs.getFloat("cal", calFactor);
  hxOffset  = prefs.getLong("off", 0);
  prefs.end();

  scale.set_scale(calFactor);
  if (haveCal) {
    scale.set_offset(hxOffset);
  } else {
    scale.tare(20);
    hxOffset = scale.get_offset();
    saveCal();
  }
  baseline = scale.get_units(10);
  Serial.printf("{\"event\":\"info\",\"msg\":\"boot\",\"factor\":%.4f,\"restored\":%d,\"baseline\":%.1f}\n",
                calFactor, haveCal ? 1 : 0, baseline);
}

void loop() {
  uint32_t now = millis();
  handleSerial();

  // Buttons: active low, 40 ms debounce, act on press edge
  for (int i = 0; i < 3; i++) {
    bool v = digitalRead(BTN_PINS[i]);
    if (v != btnLast[i] && now - btnChangedAt[i] > 40) {
      btnChangedAt[i] = now;
      btnLast[i] = v;
      if (!v) onButton(i);
    }
  }

  // Pending disposal timed out -> unassigned (laptop applies time rule)
  if (pendingActive && now - pendingSince > TYPE_WINDOW_MS) finalizePending(-1, "none");

  if (now - lastSample >= SAMPLE_MS && scale.is_ready()) {
    lastSample = now;
    float g = scale.get_units(1);
    medBuf[medIdx] = g; medIdx = (medIdx + 1) % MED_N;
    if (medCount < MED_N) medCount++;
    filtered = median5(medBuf, medCount);
    winPush(filtered);

    float mean;
    bool stable = winStable(&mean);

    if (filtered > CAPACITY_G && !overloadReported) {
      Serial.printf("{\"event\":\"overload\",\"g\":%.1f}\n", filtered);
      overloadReported = true;
    } else if (filtered < CAPACITY_G - 200) {
      overloadReported = false;
    }

    if (state == IDLE) {
      if (fabs(filtered - baseline) > TRIGGER_G) {
        state = MOVING; movingSince = now; unstableReported = false;
        // A new tip starts: close any disposal still waiting for its type
        if (pendingActive) finalizePending(-1, "none");
      } else if (stable && fabs(mean - baseline) < STABLE_BAND_G) {
        baseline += DRIFT_ALPHA * (mean - baseline);   // slow drift compensation
      }
    } else { // MOVING
      if (stable) {
        float delta = mean - baseline;
        if (delta >= MIN_DISPOSAL_G) {
          eventId++;
          // pre-selection counts if pressed during this tip, or at most PRESELECT_MS before it started
          bool usePre = preType >= 0 &&
                        ((int32_t)(preAt - movingSince) >= 0 || (movingSince - preAt) <= PRESELECT_MS);
          if (usePre) {
            emitDisposal(eventId, delta, TYPE_NAMES[preType], "button_pre");
            ledPattern = 2; ledUntil = now + 600;
          } else {
            pendingActive = true; pendingId = eventId; pendingWeight = delta; pendingSince = now;
          }
          preType = -1;
        } else if (delta <= -MIN_DISPOSAL_G) {
          Serial.printf("{\"event\":\"removal\",\"delta_g\":%.1f,\"g\":%.1f,\"ms\":%lu}\n", delta, mean, (unsigned long)now);
        } else {
          Serial.printf("{\"event\":\"ignored\",\"delta_g\":%.1f,\"ms\":%lu}\n", delta, (unsigned long)now);
        }
        baseline = mean;
        state = IDLE;
      } else if (now - movingSince > UNSTABLE_MS && !unstableReported) {
        Serial.printf("{\"event\":\"unstable\",\"g\":%.1f,\"ms\":%lu}\n", filtered, (unsigned long)now);
        unstableReported = true;
      }
    }
  }

  if (now - lastStream >= STREAM_MS) {
    lastStream = now;
    float m; bool st = winStable(&m);
    Serial.printf("{\"event\":\"weight\",\"g\":%.1f,\"stable\":%d,\"state\":\"%s\",\"pending\":%d}\n",
                  filtered, st ? 1 : 0, state == IDLE ? "idle" : "moving", pendingActive ? 1 : 0);
  }

  updateLed();
}
