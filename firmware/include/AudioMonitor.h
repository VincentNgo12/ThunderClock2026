/**
 * File: AudioMonitor.h
 * Purpose: Separate INMP441 I2S boundary for the future audio task.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include "FirmwareTypes.h"

class AudioMonitor {
 public:
  fw::Result begin();  // Future I2S setup using confirmed microphone pins.
  fw::Result read(fw::AudioSummary& output);  // Future bounded relative-noise window.
};
