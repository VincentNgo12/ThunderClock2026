#pragma once
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#include "FirmwareTypes.h"

// Take snapshotMutex for short copies or updates, then unlock before I/O or math.
// extern fw::WearableSnapshot sharedSnapshot;
// extern SemaphoreHandle_t snapshotMutex;
