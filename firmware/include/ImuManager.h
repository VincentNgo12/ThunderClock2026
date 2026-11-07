/**
 * File: ImuManager.h
 * Purpose: Raw MPU6050 reads for four regions, called by acquisition.
 * Contributors:
 * - Unassigned
 */

 // status : working 

 #pragma once
 #include <Arduino.h>
 #include <Wire.h>
 #include "FirmwareTypes.h"
 #include <Adafruit_MPU6050.h>
 
 
 class ImuManager {
   public:
     explicit ImuManager(TwoWire& wire) : wire_(wire) {}
     fw::Result begin();  // Future per-sensor setup and identity checks.
     // Only Ok permits using output; each real sample carries its original acquisition time.
     fw::Result read(fw::StripData& output);
     // Routes the shared I2C bus to one mux channel. Public so the heart-rate and
     // environment sensors on channels 4 and 5 share this one mux helper.
     fw::Result setChannel(uint8_t channel);
 
   private:
     Adafruit_MPU6050 imuSensors[4];
     uint8_t muxAddress = 0x70;
     TwoWire& wire_;
 };
 
