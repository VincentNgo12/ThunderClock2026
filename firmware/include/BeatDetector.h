#pragma once
#include <cstddef>
#include <cstdint>

// Feed raw IR samples at a fixed rate. Blood volume rises at each beat, which
// lowers the IR reading, so the detector looks for peaks in the inverted signal.
class BeatDetector {
 public:
  explicit BeatDetector(float sampleRateHz = 100.0f);
  void reset();
  // Returns true when this sample completes an accepted beat.
  bool addSample(float rawIr);
  // Average of recent accepted beat intervals, or 0 when not yet trustworthy.
  float bpm() const;
  // Milliseconds of sample time since the last accepted beat.
  uint32_t msSinceLastBeat() const;

 private:
  static constexpr size_t kIntervalCount = 5;
  static constexpr size_t kBeatsBeforeValid = 4;

  float samplePeriodMs_;
  uint32_t sampleClockMs_ = 0;
  float baseline_ = 0;
  float smoothed_ = 0;
  float previous_ = 0;
  float envelope_ = 0;
  bool rising_ = false;
  bool primed_ = false;
  uint32_t lastBeatMs_ = 0;
  bool haveLastBeat_ = false;
  float intervalsMs_[kIntervalCount] = {};
  size_t intervalWrite_ = 0;
  size_t intervalTotal_ = 0;
  size_t acceptedBeats_ = 0;
};
