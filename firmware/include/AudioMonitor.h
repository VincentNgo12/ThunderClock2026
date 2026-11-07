/**
 * File: AudioMonitor.h
 * Purpose: Separate INMP441 I2S boundary for the future audio task.
 * Contributors:
 * - Unassigned
 */

#pragma once

#include <ESP_I2S.h>
#include "FirmwareTypes.h"

class AudioMonitor {
public:
    AudioMonitor(int sckPin, int wsPin, int sdPin)
        : sckPin_(sckPin), wsPin_(wsPin), sdPin_(sdPin) {}

    fw::Result begin();
    fw::Result read(fw::AudioData& output);

private:
    static constexpr uint32_t kSampleRate = 16000; 
    static constexpr size_t kFramesPerRead = 512; // 512 / 1600 = 32 ms of data 
    static constexpr uint32_t kTimeoutMs = 100;

    fw::Result readFrames();

    const int sckPin_;
    const int wsPin_;
    const int sdPin_;
    I2SClass i2s_;
    bool ready_ = false;
    int32_t frames_[kFramesPerRead][2];  // Left, right
};