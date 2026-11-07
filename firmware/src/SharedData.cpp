/**
 * File: SharedData.cpp
 * Purpose: Snapshot storage whose mutex is created in setup before tasks start.
 * Contributors:
 * - Unassigned
 */
#include "SharedData.h"

fw::WearableSnapshot sharedSnapshot{};  // Every reading starts unavailable.
SemaphoreHandle_t snapshotMutex = nullptr;
