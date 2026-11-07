/**
 * File: FirmwareTasks.h
 * Purpose: Optional acquisition, telemetry and audio task entry points.
 * Contributors:
 * - Unassigned
 */
#pragma once

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "SystemConfig.h"
#include "AudioMonitor.h"

// global sensor obj
extern AudioMonitor audioMonitor;

// Bodies stay empty until their corresponding work is implemented.
void acquisitionTask(void* argument);
void IMUTask(void* argument);
void telemetryTask(void* argument);
