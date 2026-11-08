/** * File: BLEManager.cpp
 * Purpose: Unimplemented BLE setup and packet publication.
 * Contributors:
 * - Unassigned
 */
#include "BLEManager.h"

const char* const DEVICE_NAME = "THUNDER CLOCK";
const char* const SERVICE_UUID = "OIIO";
const char* const CHARACTERISTIC_UUID_IMU = "7777";
const char* const CHARACTERISTIC_UUID_Snapshot = "8888";

NimBLECharacteristic *pCharIMU = nullptr;
NimBLECharacteristic *pCharSnapshot = nullptr;
bool deviceConnected = false;


// object storing callbacks
class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo) override;
    void onDisconnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo, int reason) override;
    void onMTUChange(uint16_t MTU, NimBLEConnInfo& connInfo) override;
};

fw::Result bleBegin() {

    NimBLEDevice::init(DEVICE_NAME); // initialize stack and set device name
    NimBLEDevice::setMTU(517);       // set MTU
    NimBLEServer *pServer = NimBLEDevice::createServer(); // create GATT server
    pServer->setCallbacks(new ServerCallbacks());         // set callbacks

    // describe gatt structure
    NimBLEService *pService = pServer->createService(SERVICE_UUID);
    pCharIMU = pService->createCharacteristic(CHARACTERISTIC_UUID_IMU,
        NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY
    ); // bit mask for characteristic permissions
    pCharSnapshot = pService->createCharacteristic(CHARACTERISTIC_UUID_Snapshot,
        NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY
    ); // bit mask for characteristic permissions

    // advertising as GAP peripheral and GATT server
    NimBLEAdvertising *pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID); 
    pAdvertising->setName(DEVICE_NAME);
    pAdvertising->start();  // start advertisement

    return fw::Result::Ok;
}

// ----------------------
// implementing callbacks 
// ----------------------

void ServerCallbacks::onConnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo) { 
    deviceConnected = true;
    Serial.println("Connected");
}

void ServerCallbacks::onDisconnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo, int reason) {
    deviceConnected = false;
    Serial.println("Disconnected");
    NimBLEDevice::getAdvertising()->start(); // on disconnect advertise for connection again
}

void ServerCallbacks::onMTUChange(uint16_t MTU, NimBLEConnInfo& connInfo) {
    Serial.print("Negotiated MTU: ");
    Serial.println(MTU); // print MTU to serial
}
