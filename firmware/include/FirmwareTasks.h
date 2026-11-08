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
#include "ImuManager.h"
#include "BLEManager.h"

#include "telemetry.pb.h"
#include <pb_encode.h>

// global mutexes and queues
extern SemaphoreHandle_t i2cMutex;
extern QueueHandle_t imuToTelemetryQueue;
extern QueueHandle_t imuToSnapshotQueue;
extern QueueHandle_t snapshotToTelemetryQueue;

// global sensor obj
extern AudioMonitor audioMonitor;
extern ImuManager imuManager;

// fw::StripData items from IMUTask. Created by the queue owner before IMUTask starts.
//extern QueueHandle_t imuQueue;

// Bodies stay empty until their corresponding work is implemented.
void acquisitionTask(void* argument);
void IMUTask(void* argument);
void telemetryTask(void* argument);
