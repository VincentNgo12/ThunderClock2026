// File: test_beat_detector.cpp
// Purpose: Host check of BeatDetector on synthetic PPG. Run from firmware/:
//   g++ -std=c++17 -Iinclude src/BeatDetector.cpp test/native_beat/test_beat_detector.cpp -o /tmp/beat && /tmp/beat
// Contributors: Suvir Duggal
#include <cmath>
#include <cstdio>
#include <random>

#include "BeatDetector.h"

static int runCase(float trueBpm, float noise, float drift) {
  BeatDetector detector(100.0f);
  std::mt19937 rng(42);
  std::normal_distribution<float> gaussian(0.0f, noise);
  const float beatHz = trueBpm / 60.0f;
  for (int i = 0; i < 100 * 20; i++) {  // 20 s at 100 Hz.
    const float t = i / 100.0f;
    const float phase = std::fmod(t * beatHz, 1.0f);
    // Sharp systolic dip with a slower recovery, like a real IR PPG.
    const float pulse = phase < 0.15f ? phase / 0.15f : std::exp(-(phase - 0.15f) * 4.0f);
    const float ir = 120000.0f - 600.0f * pulse + drift * t + gaussian(rng);
    detector.addSample(ir);
  }
  const float measured = detector.bpm();
  const bool pass = std::fabs(measured - trueBpm) <= 3.0f;
  std::printf("%s true=%.0f measured=%.1f noise=%.0f drift=%.0f/s\n", pass ? "PASS" : "FAIL",
              trueBpm, measured, noise, drift);
  return pass ? 0 : 1;
}

int main() {
  int failures = 0;
  failures += runCase(60, 20, 0);
  failures += runCase(75, 60, 0);
  failures += runCase(110, 60, 200);
  failures += runCase(150, 40, -300);
  failures += runCase(45, 30, 100);
  return failures;
}
