/*
 * HX711 calibration sketch - Bio Waste Tracker
 * Same wiring as bin_tracker: DT -> GPIO16, SCK -> GPIO4, VCC -> 3V3.
 * Saves factor + zero offset to flash ("bwt" namespace) - bin_tracker reads them at boot.
 *
 * Serial Monitor: 115200 baud, line ending "Newline". Follow the prompts.
 *   Enter     next step
 *   t         re-zero (tare) with the empty plate
 *   c         re-run calibration
 *   w<grams>  change the reference weight, e.g. w157
 */
#include <Arduino.h>
#include <HX711.h>
#include <Preferences.h>

const int PIN_HX_DOUT = 16;
const int PIN_HX_SCK  = 4;

// Reference: 150 g yoghurt cup. The label is the net content - cup + lid add ~5-8 g.
// Best: KNOWN_G = what a kitchen scale says. Otherwise 157 is a good guess for a full 150 g cup.
float KNOWN_G = 150.0;

HX711 scale;
Preferences prefs;
float factor = 1.0;
long offset = 0;

String waitLine() {
  while (!Serial.available()) delay(10);
  String s = Serial.readStringUntil('\n');
  s.trim();
  return s;
}

void save() {
  prefs.begin("bwt", false);
  prefs.putFloat("cal", factor);
  prefs.putLong("off", offset);
  prefs.end();
}

void tare() {
  Serial.println("\n>> Empty plate only (no cup). Press Enter to zero...");
  waitLine();
  scale.tare(30);
  offset = scale.get_offset();
  Serial.printf("   zero offset = %ld\n", offset);
}

void calibrate() {
  Serial.printf("\n>> Put the reference (%.1f g) in the CENTRE of the plate. Press Enter...\n", KNOWN_G);
  waitLine();
  delay(1500);                         // let it settle
  scale.set_scale(1.0);
  float raw = scale.get_units(30);     // average of 30 readings minus zero
  if (fabs(raw) < 100) {
    Serial.println("   !! almost no change - is the cup on the plate? load cell wired? Try again (c).");
    return;
  }
  factor = raw / KNOWN_G;              // negative factor = A+/A- swapped, still works
  scale.set_scale(factor);
  save();
  Serial.printf("   factor = %.4f  (saved)\n", factor);
  Serial.println("   Now: take the cup off -> ~0 g, put it back -> ~reference. Try other positions on the plate too.");
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  delay(500);
  scale.begin(PIN_HX_DOUT, PIN_HX_SCK);
  Serial.println("\n=== HX711 calibration ===");
  Serial.printf("Reference weight: %.1f g (change with w<grams>)\n", KNOWN_G);
  tare();
  calibrate();
}

void loop() {
  if (Serial.available()) {
    String s = Serial.readStringUntil('\n');
    s.trim();
    if (s == "t") { tare(); scale.set_scale(factor); save(); Serial.println("   zero saved"); }
    else if (s == "c") calibrate();
    else if (s.startsWith("w")) { KNOWN_G = s.substring(1).toFloat(); Serial.printf("   reference = %.1f g, now send c\n", KNOWN_G); }
  }
  static uint32_t last = 0;
  if (millis() - last > 500 && scale.is_ready()) {
    last = millis();
    Serial.printf("%8.1f g\n", scale.get_units(5));
  }
}
