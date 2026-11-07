/**
 * File: AudioMonitor.cpp
 * Purpose: Unimplemented microphone acquisition and relative-noise processing.
 * Contributors:
 * - Unassigned
 */

#include "AudioMonitor.h"

#include <cmath>

fw::Result AudioMonitor::begin() {
    if (ready_) {
        return fw::Result::Ok;
    }

    // Pin order is SCK, WS, data out, data in.
    i2s_.setPins(sckPin_, wsPin_, -1, sdPin_);

    if (!i2s_.begin(I2S_MODE_STD, kSampleRate,
                    I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
        i2s_.end();
        return fw::Result::IoFailure;
    }

    // Discard the first 4096 frames while the mic starts up.
    for (size_t discarded = 0; discarded < 4096;
         discarded += kFramesPerRead) {
        const fw::Result result = readFrames();
        if (result != fw::Result::Ok) {
            i2s_.end();
            return result;
        }
    }

    ready_ = true;
    return fw::Result::Ok;
}

fw::Result AudioMonitor::read(fw::AudioData& output) {
    if (!ready_) {
        return fw::Result::IoFailure;
    }

    const fw::Result result = readFrames();
    if (result != fw::Result::Ok) {
        return result;
    }

    double sum = 0.0;
    double sumSquares = 0.0;
    for (const auto& frame : frames_) {
        const int32_t sample = frame[0] >> 8;  // Keep the signed upper 24 bits.
        const double normalized = sample / 8388608.0;  // 2^23
        sum += normalized;
        sumSquares += normalized * normalized;
    }

    // Remove the DC offset before computing the signal level.
    const double mean = sum / kFramesPerRead;
    const double meanSquare = sumSquares / kFramesPerRead - mean * mean;
    const double power = std::fmax(meanSquare, 1e-12);

    // 10 * log10(mean square) equals 20 * log10(RMS).
    output.relativeDb = static_cast<float>(10.0 * std::log10(power));
    return fw::Result::Ok;
}

// fills frames_ with raw i2s aduio data
fw::Result AudioMonitor::readFrames() {
    size_t bytesRead = 0;
    const esp_err_t error = i2s_channel_read(
        i2s_.rxChan(), frames_, sizeof(frames_), &bytesRead, kTimeoutMs);

    if (error == ESP_ERR_TIMEOUT) {
        return fw::Result::Timeout;
    }

    if (error != ESP_OK || bytesRead != sizeof(frames_)) {
        return fw::Result::IoFailure;
    }

    return fw::Result::Ok;
}