#include "FirmwareTasks.h"
#include "ImuManager.h"
#include "OrientationProcessor.h"

// debug functions
void printSnapshot(const fw::Snapshot& snapshot); 
void printRawStrip(const fw::StripData& stripData);
void printStripOrientation(const fw::StripDataProcessed& stripData);

// protobuf helper functions 
void makeIMUFieldsAvailable(telemetry_ImuStripData& message); 
void makeSnapshotFieldsAvailable(telemetry_orientation_v1_OrientationPacket& packet);
void fillOrientationPacket(const fw::Snapshot& snapshot, telemetry_orientation_v1_OrientationPacket& packet); 
void fillIMUMessage(telemetry_ImuStripData &message, fw::StripData &stripData);


void acquisitionTask(void*) {
  TickType_t lastWake = xTaskGetTickCount();
  TickType_t lastAudioTime = lastWake;
  TickType_t lastHeartTime = lastWake;

  const TickType_t imuPeriod = pdMS_TO_TICKS(20);      // 0.02 s, 50 Hz
  const TickType_t audioPeriod = pdMS_TO_TICKS(100);   // 0.1s
  const TickType_t heartPeriod = pdMS_TO_TICKS(SystemConfig::kSlowSensorPollMs);  // Drain FIFO every 50 ms.

  // initial data
  fw::Snapshot snapshot{};
  snapshot.envData = {NAN, NAN, NAN, NAN, NAN};
  snapshot.heartData = {NAN, NAN};
  snapshot.audioData = {NAN};
  fw::StripData stripData{};
  OrientationProcessor orientationProcessor;
  orientationProcessor.begin();

  // protobuff settings
  telemetry_orientation_v1_OrientationPacket packet =
    telemetry_orientation_v1_OrientationPacket_init_zero;
  makeSnapshotFieldsAvailable(packet);
  packet.version = 2;
  uint32_t sequence = 0;
  uint8_t buffer[514]; 

  // uint32_t t = 0;      // DEBUG q to yaw
  
  while (true) {
    const TickType_t now = xTaskGetTickCount();

    // env160 + aht21 + max6050 read w mutex safety
    // the manager schedules conversions internally. poll each pass without
    // waiting for the AHT21 conversion while holding the shared bus.
    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(10)) == pdTRUE) {

      if (SystemConfig::kInitEnvironment) {
        fw::EnvData environment;
        if (environmentManager.read(environment) == fw::Result::Ok) {
          // Keep each last-known value between reads, including partial reads.
          if (std::isfinite(environment.temperature)) {
            snapshot.envData.temperature = environment.temperature;
          }
          if (std::isfinite(environment.humidity)) {
            snapshot.envData.humidity = environment.humidity;
          }
          if (std::isfinite(environment.air_quality_index)) {
            snapshot.envData.air_quality_index = environment.air_quality_index;
          }
          if (std::isfinite(environment.eco2_ppm)) {
            snapshot.envData.eco2_ppm = environment.eco2_ppm;
          }
          if (std::isfinite(environment.tvoc_ppb)) {
            snapshot.envData.tvoc_ppb = environment.tvoc_ppb;
          }
        }
      }

      // Keep the last successful readings in every snapshot, as with environment.
      if (now - lastHeartTime >= heartPeriod && SystemConfig::kInitHeartRate) {
        lastHeartTime = now;
        fw::HeartData heart;
        if (heartRateMonitor.read(heart) == fw::Result::Ok) {
          if (std::isfinite(heart.bpm)) {
            snapshot.heartData.bpm = heart.bpm;
          }
          if (std::isfinite(heart.oxi)) {
            snapshot.heartData.oxi = heart.oxi;
          }
        }
      }

      xSemaphoreGive(i2cMutex);
    } 

    // audio read
    if (now - lastAudioTime >= audioPeriod && SystemConfig::kInitAudio) {
      lastAudioTime = now;
      audioMonitor.read(snapshot.audioData);
    } 

    // imu read from queue 20 (hz)
    if (xQueueReceive(imuToSnapshotQueue, &stripData, portMAX_DELAY) == pdTRUE) {

      orientationProcessor.update(stripData, snapshot.stripData); // raw imu to processed

      // leave this here for testing q to roll,pitch,yaw    // DEBUG q to yaw
      // if (t >= 20) {                                     // DEBUG q to yaw
      //   printStripOrientation(snapshot.stripData);       // DEBUG q to yaw
      //   t = 0;// DEBUG q to yaw                          // DEBUG q to yaw
      // }                                                  // DEBUG q to yaw
      // t++;                                               // DEBUG q to yaw

      // protobuff encoding
      packet.sequence = sequence++;
      fillOrientationPacket(snapshot, packet);
      pb_ostream_t stream = pb_ostream_from_buffer(buffer, sizeof(buffer));
      if (pb_encode(&stream, telemetry_orientation_v1_OrientationPacket_fields, &packet)) {
        // Serial.println("succesful snapshot encoding");;  // DEBUG proto
      }
      size_t length = stream.bytes_written;  
      // Serial.printf("Encoded %zu bytes\n", length);      // DEBUG proto 

      // send data through ble
      pCharSnapshot->setValue(buffer, length);
      pCharSnapshot->notify();
    }

    // printSnapshot(snapshot);   // DEBUG full snapshot

    vTaskDelayUntil(&lastWake, imuPeriod);
  }
}


void IMUTask(void*) {
  TickType_t lastWake = xTaskGetTickCount();
  TickType_t lastImuToSnapshotTime = lastWake;

  const TickType_t imuToTelemetryPeriod = pdMS_TO_TICKS(10); // target: 100Hz 
  const TickType_t imuToSnapshotPeriod = pdMS_TO_TICKS(50);  // target: 20Hz 
  
  fw::StripData stripData{};

  // protobuf settings 
  telemetry_ImuStripData message = telemetry_ImuStripData_init_zero; 
  message.data_count = 4; 
  makeIMUFieldsAvailable(message);
  uint8_t buffer[300];   

  while (true) {
    const TickType_t now = xTaskGetTickCount();

    if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
      imuManager.read(stripData);
      xSemaphoreGive(i2cMutex);
    }
    
    // encode data
    fillIMUMessage(message, stripData);
    pb_ostream_t stream = pb_ostream_from_buffer(buffer, sizeof(buffer));
    if (pb_encode(&stream, telemetry_ImuStripData_fields, &message)) {
      // Serial.println("succesful imu encoding");;     // DEBUG proto
    }
    size_t length = stream.bytes_written; 
    // Serial.printf("Encoded %zu bytes\n", length);    // DEBUG proto 

    // send data ble
    pCharIMU->setValue(buffer, length);
    pCharIMU->notify();

    if (now - lastImuToSnapshotTime >= imuToSnapshotPeriod) {
      xQueueOverwrite(imuToSnapshotQueue, &stripData);    
      lastImuToSnapshotTime = now;
    }

    // printRawStrip(stripData); // DEBUG raw imu data

    vTaskDelayUntil(&lastWake, imuToTelemetryPeriod);
  }
}

// -------------------------------------------
// protobuff helper funcitons implementations 
// -------------------------------------------

// set fields to available
void makeIMUFieldsAvailable(telemetry_ImuStripData& message) {
  for (int i = 0; i < 4; ++i) {
    message.data[i].has_accel = true;
    message.data[i].has_gyro = true;
  } // set has fields of protobut to true so that it gets encoded
}
// set fields to available
void makeSnapshotFieldsAvailable(telemetry_orientation_v1_OrientationPacket& packet) {
    packet.has_version = true;
    packet.has_sequence = true;
    packet.has_sample_time_ms = true;
    // Optional readings get their presence flags when the snapshot is filled.

    packet.orientations_count = 4;
    for (auto& orientation : packet.orientations) {
        orientation.has_sensor_id = true;
        orientation.has_valid = true;
        orientation.valid = true;
        orientation.has_rotation = true;
    }
}
// fill IMU message 
void fillIMUMessage(telemetry_ImuStripData &message, fw::StripData &stripData) {
  message.time = stripData.time;
  for (int i = 0; i < 4; ++i) {
    message.data[i].accel.x = stripData.data[i].accel.x;
    message.data[i].accel.y = stripData.data[i].accel.y;
    message.data[i].accel.z = stripData.data[i].accel.z;

    message.data[i].gyro.x = stripData.data[i].gyro.x;
    message.data[i].gyro.y = stripData.data[i].gyro.y;
    message.data[i].gyro.z = stripData.data[i].gyro.z;
  }
}
// fill orientation packet
void fillOrientationPacket(const fw::Snapshot& snapshot,
  telemetry_orientation_v1_OrientationPacket& packet) {

  packet.version = 2;
  packet.sample_time_ms = snapshot.stripData.time;

  for (pb_size_t i = 0; i < packet.orientations_count; ++i) {
    const auto& imu = snapshot.stripData.data[i];
    auto& orientation = packet.orientations[i];

    orientation.sensor_id = i + 1;
    orientation.valid = imu.valid;
    orientation.has_rotation = imu.valid;
    orientation.rotation = {};

    if (imu.valid) {
      const auto& q = imu.rotation;
      orientation.rotation = {q.w, q.x, q.y, q.z};
    }
  }

  const auto& env = snapshot.envData;
  auto& environment = packet.environment;
  environment = {};
  // Snapshot values persist between sensor reads, so these flags stay on
  // after each field's first usable reading.
  environment.has_temperature_c = std::isfinite(env.temperature);
  environment.has_humidity_percent = std::isfinite(env.humidity);
  environment.has_air_quality_index = std::isfinite(env.air_quality_index);
  environment.has_eco2_ppm = std::isfinite(env.eco2_ppm);
  environment.has_tvoc_ppb = std::isfinite(env.tvoc_ppb);
  environment.has_noise_relative_db = std::isfinite(snapshot.audioData.relativeDb);

  if (environment.has_temperature_c) {
    environment.temperature_c = env.temperature;
  }
  if (environment.has_humidity_percent) {
    environment.humidity_percent = env.humidity;
  }
  // Never cast an unavailable (NaN) reading to an integer or send it over BLE.
  if (environment.has_air_quality_index) {
    environment.air_quality_index = static_cast<uint32_t>(env.air_quality_index);
  }
  if (environment.has_eco2_ppm) {
    environment.eco2_ppm = static_cast<uint32_t>(env.eco2_ppm);
  }
  if (environment.has_tvoc_ppb) {
    environment.tvoc_ppb = static_cast<uint32_t>(env.tvoc_ppb);
  }
  if (environment.has_noise_relative_db) {
    environment.noise_relative_db = snapshot.audioData.relativeDb;
  }
  packet.has_environment = environment.has_temperature_c || environment.has_humidity_percent ||
    environment.has_air_quality_index || environment.has_eco2_ppm ||
    environment.has_tvoc_ppb || environment.has_noise_relative_db;

  packet.has_heart_rate_bpm = std::isfinite(snapshot.heartData.bpm) && snapshot.heartData.bpm > 0;
  packet.has_spo2_percent = std::isfinite(snapshot.heartData.oxi);
  packet.heart_rate_bpm = packet.has_heart_rate_bpm ? snapshot.heartData.bpm : 0;
  packet.spo2_percent = packet.has_spo2_percent ? snapshot.heartData.oxi : 0;
}

// -------------------------------------------
// debug funcitons implementations 
// -------------------------------------------

// print snapshot debug 
void printSnapshot(const fw::Snapshot& snapshot) {
  const auto& strip = snapshot.stripData;
  const auto& env = snapshot.envData;
  const auto& heart = snapshot.heartData;
  const auto& audio = snapshot.audioData;

  Serial.println("\n=== Snapshot ===");

  Serial.printf("IMU timestamp: %lu\n", static_cast<unsigned long>(strip.time));
  for (size_t i = 0; i < sizeof(strip.data) / sizeof(strip.data[0]); ++i) {
    if (!strip.data[i].valid) {
      Serial.printf("  IMU[%u]  INVALID\n", static_cast<unsigned>(i));
      continue;
    }
    const auto& q = strip.data[i].rotation;
    Serial.printf("  IMU[%u]  w=% .4f  x=% .4f  y=% .4f  z=% .4f\n",
      static_cast<unsigned>(i), q.w, q.x, q.y, q.z);
  }

  Serial.println("Environment");
  Serial.printf("  Temperature : %7.2f C\n", env.temperature);
  Serial.printf("  Humidity    : %7.2f %%\n", env.humidity);
  Serial.printf("  Air quality : %7.0f AQI\n", env.air_quality_index);
  Serial.printf("  eCO2        : %7.0f ppm\n", env.eco2_ppm);
  Serial.printf("  TVOC        : %7.0f ppb\n", env.tvoc_ppb);

  Serial.println("Heart");
  Serial.printf("  Heart rate  : %7.1f BPM\n", heart.bpm);
  Serial.printf("  SpO2        : %7.1f %%\n", heart.oxi);

  Serial.println("Audio");
  Serial.printf("  Loudness    : %7.2f dB relative\n", audio.relativeDb);
}
// print raw imu strip debug function
void printRawStrip(const fw::StripData& rawStrip) {
  Serial.printf("\n=== Raw IMU data | timestamp: %lu ===\n",
    static_cast<unsigned long>(rawStrip.time));

  unsigned index = 0;
  for (const auto& imu : rawStrip.data) {
    Serial.printf("  IMU[%u]\n", index++);
    Serial.printf("    Accel  x=% .4f  y=% .4f  z=% .4f\n",
      imu.accel.x, imu.accel.y, imu.accel.z);
    Serial.printf("    Gyro   x=% .4f  y=% .4f  z=% .4f\n",
      imu.gyro.x, imu.gyro.y, imu.gyro.z);
  }
}
// print q as roll pitch and yaw debug function
void printStripOrientation(const fw::StripDataProcessed& stripData) {
  constexpr float radToDeg = 57.2957795f;

  for (unsigned i = 0; i < 4; ++i) {
    const auto& orientation = stripData.data[i];

    if (!orientation.valid) {
      Serial.printf("IMU %u: orientation unavailable\n", i);
      continue;
    }

    const auto& q = orientation.rotation;
    const float normSq = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;

    if (normSq <= 0.0f || !isfinite(normSq)) {
      Serial.printf("IMU %u: invalid quaternion\n", i);
      continue;
    }

    const float roll = atan2f(
      2.0f * (q.w*q.x + q.y*q.z),
      normSq - 2.0f * (q.x*q.x + q.y*q.y));

    const float sinPitch = 2.0f * (q.w*q.y - q.z*q.x) / normSq;
    const float pitch = asinf(fminf(1.0f, fmaxf(-1.0f, sinPitch)));

    const float yaw = atan2f(
      2.0f * (q.w*q.z + q.x*q.y),
      normSq - 2.0f * (q.y*q.y + q.z*q.z));

    Serial.printf("IMU %u (deg): roll=%.2f, pitch=%.2f, yaw=%.2f\n",
      i, roll * radToDeg, pitch * radToDeg, yaw * radToDeg);
  }
}
