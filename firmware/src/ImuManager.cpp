#include "ImuManager.h"

#include "SystemConfig.h"

// The range enums below are fixed at ±8 g and ±1000 deg/s; read scaling uses SystemConfig.
static_assert(SystemConfig::kImuAccelRangeG == 8.0f, "Update the *_8_G range calls in begin()");
static_assert(SystemConfig::kImuGyroRangeDps == 1000.0f,
              "Update the *_1000_* range calls in begin()");

namespace {
constexpr uint8_t kIcmBankSelectRegister = 0x7F;
constexpr uint8_t kIcmAccelXHighRegister = 0x2D;  // Bank 0: accel XYZ then gyro XYZ, big-endian.
constexpr size_t kIcmAccelGyroBytes = 12;
constexpr float kStandardGravity = 9.80665f;  // m/s² per g.
constexpr float kAccelMs2PerCount = SystemConfig::kImuAccelRangeG / 32768.0f * kStandardGravity;
constexpr float kGyroRadPerCount = SystemConfig::kImuGyroRangeDps / 32768.0f * PI / 180.0f;

int16_t bigEndianInt16(const uint8_t* bytes) {
  return static_cast<int16_t>((bytes[0] << 8) | bytes[1]);
}
}  // namespace

// init each of the 4 imus sequentially, return Ok when at least one starts
fw::Result ImuManager::begin() {
  // Bus clock comes from SystemConfig::kI2cClockHz in main.cpp.
  wire_.setTimeOut(10);  // ms, so a stuck sensor cannot stall acquisition.

  size_t startedCount = 0;
  for (uint8_t i = 0; i < kSlotCount; ++i) {
    bool started = false;
    if (setChannel(SystemConfig::kImuMuxChannels[i]) == fw::Result::Ok) {
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
          // The magnetometer is not used. begin_I2C starts it, so stop it here.
          icmSensors[i].setMagDataRate(AK09916_MAG_DATARATE_SHUTDOWN);
          // read() expects register bank 0.
          wire_.beginTransmission(ICM20948_I2CADDR_DEFAULT);
          wire_.write(kIcmBankSelectRegister);
          wire_.write(0x00);
          wire_.endTransmission();
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
      Serial.printf("Found IMU %u (%s) on mux channel %u\n", static_cast<unsigned>(i + 1), kindName,
                    static_cast<unsigned>(SystemConfig::kImuMuxChannels[i]));
      startedCount++;
    } else {
      Serial.printf("Failed to init IMU %u (%s) on mux channel %u\n", static_cast<unsigned>(i + 1), kindName,
                    static_cast<unsigned>(SystemConfig::kImuMuxChannels[i]));
    }
  }

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
    if (!slotAvailable[i] || setChannel(SystemConfig::kImuMuxChannels[i]) != fw::Result::Ok) {
      continue;
    }

    if (isIcm[i]) {
      // Reads only accel and gyro (12 bytes); getEvent() would also read the magnetometer.
      if (!readIcm(output.data[i])) {
        continue;
      }
    } else {
      // getEvent does not report bus errors, so confirm the sensor answers first.
      wire_.beginTransmission(MPU6050_I2CADDR_DEFAULT);
      if (wire_.endTransmission() != 0) {
        continue;
      }
      mpuSensors[i].getEvent(&a, &g, &t);
      output.data[i].accel.x = a.acceleration.x;
      output.data[i].accel.y = a.acceleration.y;
      output.data[i].accel.z = a.acceleration.z;
      output.data[i].gyro.x = g.gyro.x;
      output.data[i].gyro.y = g.gyro.y;
      output.data[i].gyro.z = g.gyro.z;
    }
    output.data[i].valid = true;
    validCount++;
  }
  output.time = millis();
  output.timeUs = micros();

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

// Reads accel (m/s²) and gyro (rad/s) from the ICM on the selected channel.
// False when the transfer fails; output is left unchanged then.
bool ImuManager::readIcm(fw::IMUData& output) {
  wire_.beginTransmission(ICM20948_I2CADDR_DEFAULT);
  wire_.write(kIcmAccelXHighRegister);
  if (wire_.endTransmission(false) != 0) {
    return false;
  }
  if (wire_.requestFrom(static_cast<uint8_t>(ICM20948_I2CADDR_DEFAULT),
                        static_cast<uint8_t>(kIcmAccelGyroBytes)) != kIcmAccelGyroBytes) {
    return false;
  }
  uint8_t buffer[kIcmAccelGyroBytes];
  for (size_t index = 0; index < kIcmAccelGyroBytes; ++index) {
    buffer[index] = static_cast<uint8_t>(wire_.read());
  }

  output.accel.x = bigEndianInt16(&buffer[0]) * kAccelMs2PerCount;
  output.accel.y = bigEndianInt16(&buffer[2]) * kAccelMs2PerCount;
  output.accel.z = bigEndianInt16(&buffer[4]) * kAccelMs2PerCount;
  output.gyro.x = bigEndianInt16(&buffer[6]) * kGyroRadPerCount;
  output.gyro.y = bigEndianInt16(&buffer[8]) * kGyroRadPerCount;
  output.gyro.z = bigEndianInt16(&buffer[10]) * kGyroRadPerCount;
  return true;
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
