/**
 * File: TelemetryManager.h
 * Purpose: Bounded raw and quaternion encoder interfaces for the telemetry task.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include "FirmwareTypes.h"

// namespace telemetry {
// // Design targets. Verify against generated bounds when schemas are integrated.
// constexpr size_t kRawMaxBytes = 150;
// constexpr size_t kOrientationMaxBytes = 153;
// 
// // outputLength is zero until encoding succeeds. The caller owns the buffer.
// fw::Result encodeRaw(const fw::TelemetrySnapshot& snapshot, fw::TimeUs now,
//                      uint8_t* buffer, size_t capacity, size_t& outputLength);
// fw::Result encodeOrientations(const fw::TelemetrySnapshot& snapshot, fw::TimeUs now,
//                               uint32_t sequence, uint8_t* buffer, size_t capacity,
//                               size_t& outputLength);
// }  // namespace telemetry
