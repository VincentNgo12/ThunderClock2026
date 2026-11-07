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
#include <freertos/queue.h>

#include "SystemConfig.h"
#include "AudioMonitor.h"

// global sensor obj
extern AudioMonitor audioMonitor;

// fw::StripData items from IMUTask. Created by the queue owner before IMUTask starts.
//extern QueueHandle_t imuQueue;

// Bodies stay empty until their corresponding work is implemented.
void acquisitionTask(void* argument);
void IMUTask(void* argument);
void telemetryTask(void* argument);
