/**
 * File: AudioMonitor.cpp
 * Purpose: Unimplemented microphone acquisition and relative-noise processing.
 * Contributors:
 * - Unassigned
 */
#include "AudioMonitor.h"

fw::Result AudioMonitor::begin() {
  // TODO(FW-AUDIO): configure I2S and verify sample alignment.
  return fw::Result::NotImplemented;
}

fw::Result AudioMonitor::read(fw::AudioSummary& output) {
  output = fw::AudioSummary{};
  // TODO(FW-AUDIO): summarize a bounded window with timing and overflow/clipping diagnostics.
  return fw::Result::NotImplemented;
}
