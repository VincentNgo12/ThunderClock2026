#include <Arduino.h>
#include <cassert>
#include <cmath>
#include <cstdio>

#include "EnvironmentManager.h"
#include "SystemConfig.h"

static void assertUnavailable(const fw::EnvData& data) {
  assert(std::isnan(data.temperature));
  assert(std::isnan(data.humidity));
  assert(std::isnan(data.air_quality_index));
  assert(std::isnan(data.eco2_ppm));
  assert(std::isnan(data.tvoc_ppb));
}

static void testConversionAndCompensation() {
  testMillis = 0;
  TwoWire wire;
  EnvironmentManager manager(wire);
  assert(manager.begin() == fw::Result::Ok);
  const uint32_t started = testMillis;
  const uint32_t startupDelay = testDelayedMs;
  fw::EnvData data;

  assert(manager.read(data) == fw::Result::Ok);
  assert(std::isnan(data.temperature));
  assert(data.air_quality_index == 2);
  assert(data.eco2_ppm == 630);
  assert(data.tvoc_ppb == 0);  // A real zero remains available.
  assert(wire.conversions == 1);
  const unsigned firstTransactions = wire.transactions;

  testMillis = started + SystemConfig::kAht21ConversionMs - 1;
  assert(manager.read(data) == fw::Result::Ok);
  assert(std::isnan(data.temperature));
  assert(wire.transactions == firstTransactions);

  wire.selectedChannel = 1;  // Another sensor used the mux between calls.
  testMillis++;
  assert(manager.read(data) == fw::Result::Ok);
  assert(data.temperature == 25);
  assert(data.humidity == 50);
  assert(wire.selectedChannel == 0x20);
  const std::vector<uint8_t> expectedCompensation{0x89, 0x4A, 0x00, 0x64};
  assert(wire.compensation == expectedCompensation);

  testMillis = started + SystemConfig::kEnvironmentPeriodMs - 1;
  assert(manager.read(data) == fw::Result::Ok);
  assert(wire.conversions == 1);
  testMillis++;
  assert(manager.read(data) == fw::Result::Ok);
  assert(wire.conversions == 2);
  assert(testDelayedMs == startupDelay);  // read() never calls delay().
}

static void testPartialSensorsAndWarmup() {
  testMillis = 1000;
  TwoWire wire;
  wire.ensStatus = 0x06;  // ENS160 warm-up, even with new data.
  wire.ensData = {0, 0, 0, 0, 0};  // Forward zeros too; backend handles their meaning.
  EnvironmentManager manager(wire);
  assert(manager.begin() == fw::Result::Ok);
  const uint32_t started = testMillis;
  fw::EnvData data;
  assert(manager.read(data) == fw::Result::Ok);
  assert(std::isnan(data.temperature));
  assert(data.air_quality_index == 0);
  assert(data.eco2_ppm == 0);
  assert(data.tvoc_ppb == 0);
  testMillis += SystemConfig::kAht21ConversionMs;
  assert(manager.read(data) == fw::Result::Ok);
  assert(data.temperature == 25);
  assert(data.air_quality_index == 0);

  wire.ensData = {2, 0, 0, 0x76, 0x02};
  wire.ensStatus = 0x02;
  testMillis = started + SystemConfig::kEnvironmentPeriodMs;
  assert(manager.read(data) == fw::Result::Ok);
  assert(data.air_quality_index == 2);
  wire.ensStatus = 0x04;  // Without a new sample, keep the previous reading.
  testMillis += SystemConfig::kEnvironmentPeriodMs;
  assert(manager.read(data) == fw::Result::Ok);
  assert(data.air_quality_index == 2);

  // Each nonzero ENS160 validity flag must still allow new values through.
  for (uint8_t validity = 1; validity <= 3; ++validity) {
    wire.ensStatus = 0x02 | (validity << 2);
    wire.ensData = {validity, validity, 0, validity, 0};
    testMillis += SystemConfig::kEnvironmentPeriodMs;
    assert(manager.read(data) == fw::Result::Ok);
    assert(data.air_quality_index == validity);
    assert(data.tvoc_ppb == validity);
    assert(data.eco2_ppm == validity);
  }

  TwoWire ahtOnly;
  ahtOnly.ensPresent = false;
  EnvironmentManager temperature(ahtOnly);
  assert(temperature.begin() == fw::Result::IoFailure);
  assert(temperature.read(data) == fw::Result::NotReady);
  testMillis += SystemConfig::kAht21ConversionMs;
  assert(temperature.read(data) == fw::Result::Ok);
  assert(data.humidity == 50);
  assert(std::isnan(data.eco2_ppm));

  TwoWire ensOnly;
  ensOnly.ahtPresent = false;
  EnvironmentManager air(ensOnly);
  assert(air.begin() == fw::Result::IoFailure);
  assert(air.read(data) == fw::Result::Ok);
  assert(std::isnan(data.temperature));
  assert(data.eco2_ppm == 630);
}

static void testFailuresAndRetry() {
  testMillis = 10000;
  TwoWire wire;
  wire.ensPresent = false;
  EnvironmentManager manager(wire);
  fw::EnvData data{1, 2, 3, 4, 5};
  assert(manager.read(data) == fw::Result::NotReady);
  assertUnavailable(data);
  assert(manager.begin() == fw::Result::IoFailure);
  const uint32_t started = testMillis;
  assert(manager.read(data) == fw::Result::NotReady);
  wire.ahtFrame[0] |= 0x80;
  testMillis += SystemConfig::kAht21ConversionMs;
  assert(manager.read(data) == fw::Result::NotReady);
  assert(wire.conversions == 1);
  testMillis = started + SystemConfig::kEnvironmentPeriodMs;
  assert(manager.read(data) == fw::Result::Timeout);
  assert(wire.conversions == 2);  // A stuck busy bit cannot stop future conversions.

  wire.ahtFrame[0] = 0x08;
  wire.ahtFrame[6] ^= 1;
  testMillis += SystemConfig::kAht21ConversionMs;
  assert(manager.read(data) == fw::Result::Fail);
  assertUnavailable(data);
  testMillis = started + 2 * SystemConfig::kEnvironmentPeriodMs;
  assert(manager.read(data) == fw::Result::NotReady);
  wire.ahtFrame[6] ^= 1;
  testMillis += SystemConfig::kAht21ConversionMs;
  assert(manager.read(data) == fw::Result::Ok);

  wire.muxPresent = false;
  testMillis = started + 3 * SystemConfig::kEnvironmentPeriodMs;
  assert(manager.read(data) == fw::Result::IoFailure);
  assertUnavailable(data);
  assert(manager.begin() == fw::Result::IoFailure);
  assert(manager.read(data) == fw::Result::NotReady);
  assertUnavailable(data);  // Reinitializing cannot expose the previous session's data.

  wire.muxPresent = true;
  wire.ahtPresent = false;
  assert(manager.begin() == fw::Result::IoFailure);
}

static void testExpiryAndClockWrap() {
  testMillis = UINT32_MAX - 200;
  TwoWire wire;
  EnvironmentManager manager(wire);
  assert(manager.begin() == fw::Result::Ok);
  const uint32_t started = testMillis;
  fw::EnvData data;
  assert(manager.read(data) == fw::Result::Ok);
  testMillis += SystemConfig::kAht21ConversionMs;
  assert(manager.read(data) == fw::Result::Ok);
  const uint32_t temperatureTime = testMillis;
  wire.ahtReadFails = true;
  wire.ensStatus = 0;  // No new air-quality sample; its age must keep increasing.

  testMillis = started + SystemConfig::kEnvironmentPeriodMs;  // Wraps millis().
  assert(manager.read(data) == fw::Result::Ok);
  assert(wire.conversions == 2);
  testMillis = started + 5001;
  assert(manager.read(data) == fw::Result::Ok);
  assert(std::isnan(data.air_quality_index));
  assert(data.temperature == 25);  // The later AHT21 reading is still fresh.

  testMillis = temperatureTime + 5001;
  assert(manager.read(data) == fw::Result::IoFailure);
  assertUnavailable(data);
}

int main() {
  testConversionAndCompensation();
  testPartialSensorsAndWarmup();
  testFailuresAndRetry();
  testExpiryAndClockWrap();
  std::puts("PASS EnvironmentManager: timing, mux, compensation, partial sensors, expiry and failures");
}
