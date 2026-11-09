#include <cmath>
#include <cstdio>
#include <random>

#include "BeatDetector.h"
#include "Spo2Estimator.h"

// Builds a signal whose ratio of ratios is (110 - target) / 25, then checks the estimate.
static int runCase(float targetSpo2, float bpm) {
  BeatDetector beats(100.0f);
  Spo2Estimator spo2;
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0f, 15.0f);
  const float ratio = (110.0f - targetSpo2) / 25.0f;
  const float irDc = 120000.0f;
  const float redDc = 90000.0f;
  const float irAc = 900.0f;
  const float redAc = ratio * (irAc / irDc) * redDc;
  for (int i = 0; i < 100 * 20; i++) {
    const float phase = std::fmod(i / 100.0f * bpm / 60.0f, 1.0f);
    const float pulse = phase < 0.15f ? phase / 0.15f : std::exp(-(phase - 0.15f) * 4.0f);
    const float ir = irDc - irAc * pulse + noise(rng);
    const float red = redDc - redAc * pulse + noise(rng);
    const bool beat = beats.addSample(ir);
    spo2.addSample(red, ir, beat);
  }
  const float measured = spo2.percent();
  const bool pass = std::fabs(measured - targetSpo2) <= 2.0f;
  std::printf("%s target=%.0f%% measured=%.1f%% at %.0f bpm\n", pass ? "PASS" : "FAIL", targetSpo2,
              measured, bpm);
  return pass ? 0 : 1;
}

static int noContactStaysUnavailable() {
  Spo2Estimator spo2;
  for (int i = 0; i < 2000; i++) {
    spo2.addSample(1000.0f, 1000.0f, i % 80 == 0);  // Flat signal: no pulse.
  }
  const bool pass = spo2.percent() == 0.0f;
  std::printf("%s flat signal stays unavailable\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int main() {
  int failures = 0;
  failures += runCase(98, 70);
  failures += runCase(95, 90);
  failures += runCase(90, 110);
  failures += noContactStaysUnavailable();
  return failures;
}
