/**
 * File: Spo2Estimator.cpp
 * Purpose: Per-beat ratio of ratios with plausibility checks and averaging.
 * Contributors:
 * - Suvir Duggal
 */
#include "Spo2Estimator.h"

namespace {
constexpr float kMinRatio = 0.4f;  // About 100% after clamping.
constexpr float kMaxRatio = 1.6f;  // About 70%; anything beyond is likely artifact.
constexpr float kMinPerfusion = 0.0005f;  // AC/DC below this is too weak to trust.
constexpr size_t kMinWindowSamples = 25;  // 250 ms at 100 Hz.
}  // namespace

void Spo2Estimator::reset() {
  *this = Spo2Estimator();
}

void Spo2Estimator::addSample(float red, float ir, bool beatEnded) {
  if (!windowOpen_) {
    redMin_ = redMax_ = red;
    irMin_ = irMax_ = ir;
    redSum_ = irSum_ = 0;
    windowSamples_ = 0;
    windowOpen_ = true;
  }
  redMin_ = red < redMin_ ? red : redMin_;
  redMax_ = red > redMax_ ? red : redMax_;
  irMin_ = ir < irMin_ ? ir : irMin_;
  irMax_ = ir > irMax_ ? ir : irMax_;
  redSum_ += red;
  irSum_ += ir;
  windowSamples_++;

  if (!beatEnded) {
    return;
  }
  windowOpen_ = false;  // The next sample starts a new beat window.
  if (windowSamples_ < kMinWindowSamples) {
    return;
  }
  const float redDc = static_cast<float>(redSum_ / windowSamples_);
  const float irDc = static_cast<float>(irSum_ / windowSamples_);
  if (redDc <= 0 || irDc <= 0) {
    return;
  }
  const float redPerfusion = (redMax_ - redMin_) / redDc;
  const float irPerfusion = (irMax_ - irMin_) / irDc;
  if (redPerfusion < kMinPerfusion || irPerfusion < kMinPerfusion) {
    return;
  }
  const float ratio = redPerfusion / irPerfusion;
  if (ratio < kMinRatio || ratio > kMaxRatio) {
    return;
  }
  float estimate = 110.0f - 25.0f * ratio;
  estimate = estimate > 100.0f ? 100.0f : estimate;
  history_[historyWrite_] = estimate;
  historyWrite_ = (historyWrite_ + 1) % kHistory;
  accepted_++;
}

float Spo2Estimator::percent() const {
  if (accepted_ < kBeatsBeforeValid) {
    return 0;
  }
  const size_t count = accepted_ < kHistory ? accepted_ : kHistory;
  float sum = 0;
  for (size_t i = 0; i < count; i++) {
    sum += history_[i];
  }
  return sum / count;
}
