/**
 * File: OrientationProcessor.cpp
 * Purpose: Four x-io Fusion AHRS filters (6-axis, no magnetometer), one per strip slot.
 * Contributors:
 * - Unassigned
 */
#include "OrientationProcessor.h"

#include <cmath>

#include "SystemConfig.h"

namespace {
constexpr float kRadToDeg = 57.29578f;
constexpr float kStandardGravity = 9.80665f;  // m/s² per g.
constexpr float kUnitNormTolerance = 0.001f;
constexpr uint32_t kMaxSampleGapMs = SystemConfig::kMaxSampleGapUs / 1000;

FusionVector toFusionVector(const fw::Vec3& value, float scale) {
  FusionVector result;
  result.axis.x = value.x * scale;
  result.axis.y = value.y * scale;
  result.axis.z = value.z * scale;
  return result;
}

// Fusion's NWU body frame (+X forward, +Y left, +Z up) to the protocol frame
// (+X right, +Y up, +Z posterior). See protocol/ORIENTATION.md.
fw::Quaternion toProtocolFrame(const FusionQuaternion& nwu) {
  fw::Quaternion result;
  result.w = nwu.element.w;
  result.x = -nwu.element.y;
  result.y = nwu.element.z;
  result.z = -nwu.element.x;
  return result;
}

bool isUsable(const fw::Quaternion& q) {
  if (!std::isfinite(q.w) || !std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z)) {
    return false;
  }
  const float norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  return std::fabs(norm - 1.0f) <= kUnitNormTolerance;
}
}  // namespace

void OrientationProcessor::begin() {
  FusionAhrsSettings ahrsSettings = fusionAhrsDefaultSettings;
  ahrsSettings.sampleRate = SystemConfig::kImuRateHz;
  ahrsSettings.convention = FusionConventionNwu;
  ahrsSettings.gain = 0.5f;
  ahrsSettings.gyroscopeRange = 1000.0f;       // deg/s, matches the ImuManager gyro range.
  ahrsSettings.accelerationRejection = 10.0f;  // deg; ignore accel during sharp movement.
  ahrsSettings.magneticRejection = 0.0f;       // Magnetometer is not used.
  ahrsSettings.rejectionTimeout = 5.0f;        // s

  // Re-estimates gyro offset whenever the sensor stays below 3 deg/s for 3 s.
  FusionBiasSettings biasSettings = fusionBiasDefaultSettings;
  biasSettings.sampleRate = SystemConfig::kImuRateHz;
  biasSettings.stationaryThreshold = 3.0f;
  biasSettings.stationaryPeriod = 3.0f;

  for (size_t slot = 0; slot < kSlotCount; ++slot) {
    FusionAhrsInitialise(&ahrs[slot]);
    FusionAhrsSetSettings(&ahrs[slot], &ahrsSettings);
    FusionBiasInitialise(&bias[slot]);
    FusionBiasSetSettings(&bias[slot], &biasSettings);
    hasPreviousSample[slot] = false;
  }
}

fw::Result OrientationProcessor::update(const fw::StripData& input,
                                        fw::StripDataProcessed& output) {
  output.time = input.time;
  size_t validCount = 0;

  for (size_t slot = 0; slot < kSlotCount; ++slot) {
    output.data[slot] = fw::OrientationData{{1.0f, 0.0f, 0.0f, 0.0f}, false};

    const fw::IMUData& sample = input.data[slot];
    if (!sample.valid) {
      continue;  // Filter state is kept; a long gap restarts it on the next valid sample.
    }

    // A repeated timestamp is a strip already fused; report it again without re-integrating.
    bool alreadyFused = false;
    if (hasPreviousSample[slot]) {
      const uint32_t gapMs = input.time - lastSampleTimeMs[slot];
      if (gapMs == 0) {
        alreadyFused = true;
      } else if (gapMs > kMaxSampleGapMs) {
        // Too long without a sample means the old orientation can't be trusted.
        restartSlot(slot);
        FusionAhrsSetSamplePeriod(&ahrs[slot], 1.0f / SystemConfig::kImuRateHz);
      } else {
        FusionAhrsSetSamplePeriod(&ahrs[slot], gapMs / 1000.0f);
      }
    } else {
      FusionAhrsSetSamplePeriod(&ahrs[slot], 1.0f / SystemConfig::kImuRateHz);
    }

    if (!alreadyFused) {
      lastSampleTimeMs[slot] = input.time;
      hasPreviousSample[slot] = true;

      // Fusion expects deg/s and g. Raw accel keeps gravity.
      // TODO(FW-MOUNTING): apply a per-slot FusionRemap once board mounting on the strip is known.
      const FusionVector gyroDps = toFusionVector(sample.gyro, kRadToDeg);
      const FusionVector accelG = toFusionVector(sample.accel, 1.0f / kStandardGravity);

      const FusionVector correctedGyro = FusionBiasUpdate(&bias[slot], gyroDps);
      FusionAhrsUpdateNoMagnetometer(&ahrs[slot], correctedGyro, accelG);
    }

    const fw::Quaternion rotation = toProtocolFrame(FusionAhrsGetQuaternion(&ahrs[slot]));
    const bool settled = !FusionAhrsGetFlags(&ahrs[slot]).startup;
    if (settled && isUsable(rotation)) {
      output.data[slot].rotation = rotation;
      output.data[slot].valid = true;
      validCount++;
    }
  }

  if (validCount == 0) {
    return fw::Result::NotReady;
  }
  return fw::Result::Ok;
}

// Restarts one slot's filter, including its ~3 s startup phase. Gyro offset is kept.
void OrientationProcessor::restartSlot(size_t slot) {
  FusionAhrsRestart(&ahrs[slot]);
  hasPreviousSample[slot] = false;
}
