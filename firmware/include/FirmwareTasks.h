#pragma once

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <math.h>

#include "SystemConfig.h"
#include "AudioMonitor.h"
#include "EnvironmentManager.h"
#include "HeartRateMonitor.h"
#include "ImuManager.h"
#include "BLEManager.h"

#include "telemetry.pb.h"
#include "orientation.pb.h"
#include <pb_encode.h>

// global mutexes and queues
extern SemaphoreHandle_t i2cMutex;
extern QueueHandle_t imuToSnapshotQueue;

// global sensor obj
extern AudioMonitor audioMonitor;
extern ImuManager imuManager;
extern EnvironmentManager environmentManager;
extern HeartRateMonitor heartRateMonitor;

// rtos tasks
void acquisitionTask(void* argument);
void IMUTask(void* argument);
