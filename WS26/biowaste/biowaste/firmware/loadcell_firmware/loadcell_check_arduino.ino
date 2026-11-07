/*
 * Load cell validation sketch - ESP32 WROOM-32 + HX711 (same pins as bin_tracker.ino)
 *   HX711 DOUT -> GPIO16, SCK -> GPIO4, VCC -> 3V3, GND -> GND
 *   Load cell -> HX711: red E+, black E-, white A-, green A+
 * Library: "HX711 Arduino Library" by Bogdan Necula (Library Manager).
 * Board: ESP32 Dev Module.
 *
 * Serial Monitor 115200 baud, line ending "Newline". Commands:
 *   t          tare (platform empty)
 *   c<grams>   calibrate with a known weight on the platform, e.g. c500
 *   f<factor>  set a calibration factor printed earlier (skips c after a reset)
 *   m          measure: 20 readings -> mean, std dev, min, max
 *   s          live stream on/off
 *   i          print factor and offset
 */
#include <Arduino.h>
#include <HX711.h>

const int PIN_HX_DOUT = 16;
const int PIN_HX_SCK  = 4;
const int MEASURE_N   = 20;

HX711 scale;
bool streaming = true;
uint32_t lastPrint = 0;

bool readOne(float &grams, long &raw) {
  if (!scale.wait_ready_timeout(500)) return false;
  raw = scale.read();
  grams = (raw - scale.get_offset()) / scale.get_scale();
  return true;
}

void measure() {
  float v[MEASURE_N];
  long raw;
  for (int i = 0; i < MEASURE_N; i++) {
    if (!readOne(v[i], raw)) { Serial.println("HX711 not responding - check wiring"); return; }
  }
  float sum = 0, mn = v[0], mx = v[0];
  for (int i = 0; i < MEASURE_N; i++) { sum += v[i]; mn = min(mn, v[i]); mx = max(mx, v[i]); }
  float mean = sum / MEASURE_N, var = 0;
  for (int i = 0; i < MEASURE_N; i++) var += (v[i] - mean) * (v[i] - mean);
  Serial.printf("MEASURE mean %.1f g | std %.2f g | min %.1f | max %.1f | n=%d\n",
                mean, sqrt(var / (MEASURE_N - 1)), mn, mx, MEASURE_N);
}

void handleSerial() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd == "t") {
    scale.tare(20);
    Serial.printf("tared, offset %ld\n", scale.get_offset());
  } else if (cmd.startsWith("c")) {
    float grams = cmd.substring(1).toFloat();
    if (grams <= 0) { Serial.println("usage: c<grams>, e.g. c500 (tare first)"); return; }
    float counts = scale.get_value(20);   // average raw minus offset
    if (fabs(counts) < 100) { Serial.println("no signal change - weight on? wiring ok?"); return; }
    scale.set_scale(counts / grams);
    Serial.printf("calibrated, factor %.4f (write it down, set later with f%.4f)\n",
                  scale.get_scale(), scale.get_scale());
  } else if (cmd.startsWith("f")) {
    float f = cmd.substring(1).toFloat();
    if (f == 0) { Serial.println("usage: f<factor>"); return; }
    scale.set_scale(f);
    Serial.printf("factor set to %.4f\n", f);
  } else if (cmd == "m") {
    measure();
  } else if (cmd == "s") {
    streaming = !streaming;
    Serial.printf("stream %s\n", streaming ? "on" : "off");
  } else if (cmd == "i") {
    Serial.printf("factor %.4f, offset %ld\n", scale.get_scale(), scale.get_offset());
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  scale.begin(PIN_HX_DOUT, PIN_HX_SCK);
  if (!scale.wait_ready_timeout(2000)) {
    Serial.println("HX711 not found - check DOUT/SCK/VCC/GND");
  } else {
    scale.tare(20);
    Serial.println("ready (tared at boot). Uncalibrated: weights are raw counts until you run c<grams>.");
  }
}

void loop() {
  handleSerial();
  if (streaming && millis() - lastPrint >= 500) {
    lastPrint = millis();
    float g; long raw;
    if (readOne(g, raw)) Serial.printf("weight %8.1f g   raw %ld\n", g, raw);
    else Serial.println("HX711 not ready");
  }
}
