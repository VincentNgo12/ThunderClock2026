/**
 * File: ImuManager.cpp
 * Purpose: Raw acquisition step before OrientationProcessor; no sensor driver yet.
 * Contributors: Pedro
 */


 // status : working 


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
 fw::Result ImuManager::read(fw::ImuStripData& output) {
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
 