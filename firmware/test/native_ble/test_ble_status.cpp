#include <cassert>
#include <cstdio>
#include <string>
#include "BLEManager.h"
#include "SystemConfig.h"

static void assertColor(uint8_t red, uint8_t green, uint8_t blue) {
  bleUpdateStatus();
  assert(ledPin == SystemConfig::kStatusLedPin);
  assert(ledRed == red && ledGreen == green && ledBlue == blue);
}

int main(int argc, char** argv) {
  const std::string scenario = argc > 1 ? argv[1] : "normal";
  const uint8_t brightness = SystemConfig::kStatusLedBrightness;
  bleStatusBegin();
  assert(outputPin == SystemConfig::kStatusLedPowerPin);
  assert(poweredPin == SystemConfig::kStatusLedPowerPin);
  assertColor(0, 0, 0);

  if (scenario == "init_failure" || scenario == "advertising_failure") {
    NimBLEDevice::initSucceeds = scenario != "init_failure";
    testAdvertising.startSucceeds = scenario != "advertising_failure";
    assert(bleBegin() == fw::Result::IoFailure);
    assertColor(brightness, 0, 0);
  } else {
    assert(bleBegin() == fw::Result::Ok);
    assert(pCharIMU != nullptr && pCharSnapshot != nullptr);
    assertColor(0, 0, brightness);
    testMillis = SystemConfig::kBleBlinkPeriodMs - 1;
    assertColor(0, 0, brightness);
    testMillis++;
    assertColor(0, 0, 0);
    testMillis *= 2;
    assertColor(0, 0, brightness);

    NimBLEConnInfo connection;
    testServer.callbacks->onConnect(&testServer, connection);
    assertColor(0, brightness, 0);
    testMillis += SystemConfig::kBleBlinkPeriodMs;
    assertColor(0, brightness, 0);

    testAdvertising.startSucceeds = scenario != "restart_failure";
    testServer.callbacks->onDisconnect(&testServer, connection, 0);
    assert(testAdvertising.starts == 2);
    if (scenario == "restart_failure") {
      assertColor(brightness, 0, 0);
    } else {
      assertColor(0, 0, 0);
      testMillis += SystemConfig::kBleBlinkPeriodMs;
      assertColor(0, 0, brightness);
    }

    bleReportError();
    assertColor(brightness, 0, 0);
    testServer.callbacks->onConnect(&testServer, connection);
    assertColor(brightness, 0, 0);
    testServer.callbacks->onDisconnect(&testServer, connection, 0);
    assertColor(brightness, 0, 0);
  }
  std::printf("PASS BLE status: %s\n", scenario.c_str());
}
