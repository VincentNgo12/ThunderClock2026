/**
 * File: ImuManager.cpp
 * Purpose: Starts and reads two ICM-20948 and two MPU-6050 sensors through the mux.
 * Contributors: Pedro
 */
#include "ImuManager.h"

<<<<<<< HEAD
// init each of the 4 imus sequentially, return Ok when at least one starts
fw::Result ImuManager::begin() {
  wire_.setClock(400000);  // 400 kHz so four sequential reads fit inside the 10 ms period.
  wire_.setTimeOut(10);    // ms, so a stuck sensor cannot stall acquisition.
=======
>>>>>>> origin/main

  size_t startedCount = 0;
  for (uint8_t i = 0; i < kSlotCount; ++i) {
    bool started = false;
    if (setChannel(i) == fw::Result::Ok) {
      // ±8 g keeps fall impacts from clipping. ±1000 deg/s covers fast trunk rotation.
      // The ~10 Hz low-pass suits posture; the backend can filter further.
      if (isIcm[i]) {
        started = icmSensors[i].begin_I2C(ICM20948_I2CADDR_DEFAULT, &wire_);  // 0x69
        if (started) {
          icmSensors[i].setAccelRange(ICM20948_ACCEL_RANGE_8_G);
          icmSensors[i].setGyroRange(ICM20948_GYRO_RANGE_1000_DPS);
          icmSensors[i].enableAccelDLPF(true, ICM20X_ACCEL_FREQ_11_5_HZ);
          icmSensors[i].enableGyrolDLPF(true, ICM20X_GYRO_FREQ_11_6_HZ);
          icmSensors[i].setAccelRateDivisor(10);  // 1125 Hz / (1 + 10) ≈ 102 Hz. Library default is ~54 Hz.
          icmSensors[i].setGyroRateDivisor(10);   // 1100 Hz / (1 + 10) = 100 Hz.
        }
      } else {
        started = mpuSensors[i].begin(MPU6050_I2CADDR_DEFAULT, &wire_);  // 0x68
        if (started) {
          mpuSensors[i].setAccelerometerRange(MPU6050_RANGE_8_G);
          mpuSensors[i].setGyroRange(MPU6050_RANGE_1000_DEG);
          mpuSensors[i].setFilterBandwidth(MPU6050_BAND_10_HZ);
        }
      }
    }

    slotAvailable[i] = started;
    const char* kindName = isIcm[i] ? "ICM-20948" : "MPU-6050";
    if (started) {
      Serial.printf("Found IMU %u (%s)\n", static_cast<unsigned>(i + 1), kindName);
      startedCount++;
    } else {
      Serial.printf("Failed to init IMU %u (%s)\n", static_cast<unsigned>(i + 1), kindName);
    }
  }

<<<<<<< HEAD
  if (startedCount == 0) {
    return fw::Result::Fail;
  }
  return fw::Result::Ok;
}

// read data from the 4 imus sequentially, return Ok when at least one slot is valid
fw::Result ImuManager::read(fw::StripData& output) {
  sensors_event_t a, g, t;
  size_t validCount = 0;
  for (uint8_t i = 0; i < kSlotCount; ++i) {
    output.data[i] = fw::IMUData{};  // Zeroed and invalid until a read succeeds.
    if (!slotAvailable[i] || setChannel(i) != fw::Result::Ok) {
      continue;
    }z

    // getEvent does not report bus errors, so confirm the sensor answers first.
    wire_.beginTransmission(isIcm[i] ? ICM20948_I2CADDR_DEFAULT : MPU6050_I2CADDR_DEFAULT);
    if (wire_.endTransmission() != 0) {
      continue;
    }

    if (isIcm[i]) {
      icmSensors[i].getEvent(&a, &g, &t);
    } else {
      mpuSensors[i].getEvent(&a, &g, &t);
    }
    output.data[i].accel.x = a.acceleration.x;
    output.data[i].accel.y = a.acceleration.y;
    output.data[i].accel.z = a.acceleration.z;
    output.data[i].gyro.x = g.gyro.x;
    output.data[i].gyro.y = g.gyro.y;
    output.data[i].gyro.z = g.gyro.z;
    output.data[i].valid = true;
    validCount++;
  }
  output.time = millis();

  if (validCount == 0) {
    return fw::Result::Fail;
  }
  return fw::Result::Ok;
}

bool ImuManager::available(size_t slot) const {
  if (slot >= kSlotCount) {
    return false;
  }
  return slotAvailable[slot];
}

// set mux channel, helper function
fw::Result ImuManager::setChannel(uint8_t channel) {
  wire_.beginTransmission(muxAddress);
  wire_.write(1 << channel);
  if (wire_.endTransmission() != 0) {
    return fw::Result::IoFailure;
  }
  return fw::Result::Ok;
}
=======
 #include "ImuManager.h"
 // init each of the 4 imus sequentially, reutrn Ok
 fw::Result ImuManager::begin() {
 
   // init imuSensors 
   for (uint8_t i = 0; i < 4; ++i) {
     setChannel(i);
     if (!imuSensors[i].begin(MPU6050_I2CADDR_DEFAULT, &wire_)) {
       Serial.print("Failed to init IMU chip #"); // logs failed to print chip
       Serial.println(i + 1);
       return fw::Result::Fail;
     }
     Serial.print("Found IMU chip #");
     Serial.println(i + 1);
     
     // set configurations
     imuSensors[i].setAccelerometerRange(MPU6050_RANGE_2_G); // ±2 g accelerometer, posture movements are gentle, more sensativity
     imuSensors[i].setGyroRange(MPU6050_RANGE_250_DEG);      // posture changes are relatively slow, so the smallest gyro range is appropriate.
     imuSensors[i].setFilterBandwidth(MPU6050_BAND_10_HZ);   // posture is a low-frequency signal. This removes a lot of vibration/noise 
   }
   return fw::Result::Ok;
 }
 
 // read data from the 4 imus sequentially, return Ok
 fw::Result ImuManager::read(fw::StripData& output) {
   sensors_event_t a, g, t;
   for (uint8_t i = 0; i < 4; ++i) {
     setChannel(i);
     imuSensors[i].getEvent(&a, &g, &t);
     output.data[i].accel.x = a.acceleration.x;
     output.data[i].accel.y = a.acceleration.y;
     output.data[i].accel.z = a.acceleration.z;
     output.data[i].gyro.x = g.gyro.x;
     output.data[i].gyro.y = g.gyro.y;
     output.data[i].gyro.z = g.gyro.z;
   }
   output.time = millis();
   return fw::Result::Ok;
 }
 
 // set mux channel, helper function
 fw::Result ImuManager::setChannel(uint8_t channel) {
   wire_.beginTransmission(muxAddress);
   wire_.write(1 << channel);
   if (wire_.endTransmission() != 0) {
     Serial.println("Change mux channel failed");
     while (1) delay(1000);
   }
   return fw::Result::Ok;
 }
 
>>>>>>> origin/main
