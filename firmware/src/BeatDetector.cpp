/**
 * File: BeatDetector.cpp
 * Purpose: Baseline removal, smoothing and adaptive-threshold peak detection for PPG.
 * Contributors:
 * - Suvir Duggal
 */
#include "BeatDetector.h"

#include <cmath>

namespace {
constexpr float kMinIntervalMs = 300.0f;   // 200 bpm upper bound, also the refractory time.
constexpr float kMaxIntervalMs = 1500.0f;  // 40 bpm lower bound.
constexpr float kMaxIntervalChange = 0.30f;  // Reject beats >30% off the running average.
constexpr float kBaselineAlpha = 0.02f;  // Slow tracker of the skin/DC level.
constexpr float kSmoothAlpha = 0.25f;  // Removes sample-to-sample noise.
constexpr float kEnvelopeDecay = 0.995f;  // Lets the threshold follow weaker pulses.
constexpr float kThresholdFraction = 0.4f;
}  // namespace

BeatDetector::BeatDetector(float sampleRateHz) : samplePeriodMs_(1000.0f / sampleRateHz) {}

void BeatDetector::reset() {
  const float period = samplePeriodMs_;
  *this = BeatDetector(1000.0f / period);
}

bool BeatDetector::addSample(float rawIr) {
  sampleClockMs_ += static_cast<uint32_t>(samplePeriodMs_);
  if (!primed_) {
    baseline_ = rawIr;
    primed_ = true;
  }
  baseline_ += kBaselineAlpha * (rawIr - baseline_);
  const float pulse = baseline_ - rawIr;  // Inverted, so a beat is a positive peak.
  smoothed_ += kSmoothAlpha * (pulse - smoothed_);

  envelope_ = std::fmax(envelope_ * kEnvelopeDecay, std::fabs(smoothed_));
  const float threshold = envelope_ * kThresholdFraction;

  bool beat = false;
  const bool wasRising = rising_;
  rising_ = smoothed_ > previous_;
  // A peak is the first falling sample after rising above the threshold.
  if (wasRising && !rising_ && previous_ > threshold) {
    const uint32_t peakMs = sampleClockMs_ - static_cast<uint32_t>(samplePeriodMs_);
    if (!haveLastBeat_) {
      lastBeatMs_ = peakMs;
      haveLastBeat_ = true;
    } else {
      const float intervalMs = static_cast<float>(peakMs - lastBeatMs_);
      if (intervalMs >= kMinIntervalMs) {
        lastBeatMs_ = peakMs;
        bool plausible = intervalMs <= kMaxIntervalMs;
        if (plausible && intervalTotal_ >= 2) {
          float sum = 0;
          const size_t count = intervalTotal_ < kIntervalCount ? intervalTotal_ : kIntervalCount;
          for (size_t i = 0; i < count; i++) {
            sum += intervalsMs_[i];
          }
          const float average = sum / count;
          plausible = std::fabs(intervalMs - average) <= kMaxIntervalChange * average;
        }
        if (plausible) {
          intervalsMs_[intervalWrite_] = intervalMs;
          intervalWrite_ = (intervalWrite_ + 1) % kIntervalCount;
          intervalTotal_++;
          acceptedBeats_++;
          beat = true;
        } else if (intervalTotal_ < 2) {
          // Early on, an outlier is more likely our history than this beat; restart.
          intervalTotal_ = 0;
          intervalWrite_ = 0;
          acceptedBeats_ = 0;
        }
      }
    }
  }
  previous_ = smoothed_;
  return beat;
}

float BeatDetector::bpm() const {
  if (acceptedBeats_ < kBeatsBeforeValid) {
    return 0;
  }
  const size_t count = intervalTotal_ < kIntervalCount ? intervalTotal_ : kIntervalCount;
  float sum = 0;
  for (size_t i = 0; i < count; i++) {
    sum += intervalsMs_[i];
  }
  return 60000.0f / (sum / count);
}

uint32_t BeatDetector::msSinceLastBeat() const {
  return haveLastBeat_ ? sampleClockMs_ - lastBeatMs_ : sampleClockMs_;
}
