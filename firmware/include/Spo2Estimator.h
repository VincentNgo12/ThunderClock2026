#pragma once
#include <cstddef>

// Uncalibrated estimate for a prototype indicator, not a medical measurement.
// Each beat window gives R = (AC_red / DC_red) / (AC_ir / DC_ir), and the generic
// linear approximation SpO2 = 110 - 25 R. Only trust it while the wearer is still.
class Spo2Estimator {
 public:
  void reset();
  // Add one red/IR sample. Pass beatEnded = true on the sample where the beat
  // detector accepted a beat; that closes the current window.
  void addSample(float red, float ir, bool beatEnded);
  // Average of recent plausible beats in percent, or 0 when not yet trustworthy.
  float percent() const;

 private:
  static constexpr size_t kHistory = 5;
  static constexpr size_t kBeatsBeforeValid = 4;

  bool windowOpen_ = false;
  float redMin_ = 0;
  float redMax_ = 0;
  float irMin_ = 0;
  float irMax_ = 0;
  double redSum_ = 0;
  double irSum_ = 0;
  size_t windowSamples_ = 0;
  float history_[kHistory] = {};
  size_t historyWrite_ = 0;
  size_t accepted_ = 0;
};
