#include "BLEManager.h"

#include <atomic>
#include "SystemConfig.h"

namespace {
enum class BleState { Starting, Advertising, Connected, Error };
std::atomic<BleState> state{BleState::Starting};

void setState(BleState next) {
    BleState current = state.load();
    // A callback must not clear a system error reported by setup().
    if (current != BleState::Error) {
        state.compare_exchange_strong(current, next);
    }
}

fw::Result bleFailure() {
    bleReportError();
    return fw::Result::IoFailure;
}
}  // namespace

const char* const DEVICE_NAME = "THUNDER CLOCK";
const char* const SERVICE_UUID = "AAAA";
const char* const CHARACTERISTIC_UUID_IMU = "00008888-0000-1000-8000-00805f9b34fb";
const char* const CHARACTERISTIC_UUID_Snapshot = "00008889-0000-1000-8000-00805f9b34fb";

NimBLECharacteristic *pCharIMU = nullptr;
NimBLECharacteristic *pCharSnapshot = nullptr;

void bleStatusBegin() {
    pinMode(SystemConfig::kStatusLedPowerPin, OUTPUT);
    digitalWrite(SystemConfig::kStatusLedPowerPin, HIGH);
    rgbLedWrite(SystemConfig::kStatusLedPin, 0, 0, 0);
}

void bleUpdateStatus() {
    const uint8_t brightness = SystemConfig::kStatusLedBrightness;
    switch (state.load()) {
        case BleState::Starting:
            rgbLedWrite(SystemConfig::kStatusLedPin, 0, 0, 0);
            break;
        case BleState::Advertising: {
            const bool blueOn = (millis() / SystemConfig::kBleBlinkPeriodMs) % 2 == 0;
            rgbLedWrite(SystemConfig::kStatusLedPin, 0, 0, blueOn ? brightness : 0);
            break;
        }
        case BleState::Connected:
            rgbLedWrite(SystemConfig::kStatusLedPin, 0, brightness, 0);
            break;
        case BleState::Error:
            rgbLedWrite(SystemConfig::kStatusLedPin, brightness, 0, 0);
            break;
    }
}

void bleReportError() {
    state.store(BleState::Error);
}


// object storing callbacks
class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo) override;
    void onDisconnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo, int reason) override;
    void onMTUChange(uint16_t MTU, NimBLEConnInfo& connInfo) override;
};

fw::Result bleBegin() {

    if (!NimBLEDevice::init(DEVICE_NAME) || !NimBLEDevice::setMTU(517)) {
        return bleFailure();
    }
    NimBLEServer *pServer = NimBLEDevice::createServer(); // create GATT server
    if (pServer == nullptr) {
        return bleFailure();
    }
    pServer->setCallbacks(new ServerCallbacks());         // set callbacks

    // describe gatt structure
    NimBLEService *pService = pServer->createService(SERVICE_UUID);
    if (pService == nullptr) {
        return bleFailure();
    }
    pCharIMU = pService->createCharacteristic(CHARACTERISTIC_UUID_IMU,
        NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY
    ); // bit mask for characteristic permissions
    pCharSnapshot = pService->createCharacteristic(CHARACTERISTIC_UUID_Snapshot,
        NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY
    ); // bit mask for characteristic permissions

    if (pCharIMU == nullptr || pCharSnapshot == nullptr) {
        return bleFailure();
    }

    // advertising as GAP peripheral and GATT server
    NimBLEAdvertising *pAdvertising = NimBLEDevice::getAdvertising();
    if (!pAdvertising->addServiceUUID(SERVICE_UUID) || !pAdvertising->setName(DEVICE_NAME)) {
        return bleFailure();
    }
    setState(BleState::Advertising);
    if (!pAdvertising->start()) {
        return bleFailure();
    }

    return fw::Result::Ok;
}

// ----------------------
// implementing callbacks 
// ----------------------

void ServerCallbacks::onConnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo) { 
    setState(BleState::Connected);
    Serial.println("Connected");
}

void ServerCallbacks::onDisconnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo, int reason) {
    setState(BleState::Advertising);
    Serial.println("Disconnected");
    if (!NimBLEDevice::getAdvertising()->start()) {
        bleReportError();
    }
}

void ServerCallbacks::onMTUChange(uint16_t MTU, NimBLEConnInfo& connInfo) {
    Serial.print("Negotiated MTU: ");
    Serial.println(MTU); // print MTU to serial
}
