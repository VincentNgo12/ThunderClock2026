/**
 * File: NimBLEDevice.h
 * Purpose: Minimal BLE callbacks and controllable startup results for host checks.
 * Contributors: Unassigned
 */
#pragma once
#include <cstdint>

struct NimBLEConnInfo {};
class NimBLEServer;
class NimBLEServerCallbacks {
 public:
  virtual ~NimBLEServerCallbacks() = default;
  virtual void onConnect(NimBLEServer*, NimBLEConnInfo&) {}
  virtual void onDisconnect(NimBLEServer*, NimBLEConnInfo&, int) {}
  virtual void onMTUChange(uint16_t, NimBLEConnInfo&) {}
};
namespace NIMBLE_PROPERTY {
constexpr int READ = 1;
constexpr int NOTIFY = 2;
}
class NimBLECharacteristic {};
class NimBLEService {
 public:
  NimBLECharacteristic* createCharacteristic(const char*, int) { return &characteristic; }
 private:
  NimBLECharacteristic characteristic;
};
class NimBLEServer {
 public:
  NimBLEServerCallbacks* callbacks = nullptr;
  void setCallbacks(NimBLEServerCallbacks* value) { callbacks = value; }
  NimBLEService* createService(const char*) { return &service; }
 private:
  NimBLEService service;
};
class NimBLEAdvertising {
 public:
  bool startSucceeds = true;
  unsigned starts = 0;
  bool addServiceUUID(const char*) { return true; }
  bool setName(const char*) { return true; }
  bool start() {
    starts++;
    return startSucceeds;
  }
};
inline NimBLEServer testServer;
inline NimBLEAdvertising testAdvertising;
class NimBLEDevice {
 public:
  inline static bool initSucceeds = true;
  static bool init(const char*) { return initSucceeds; }
  static bool setMTU(uint16_t) { return true; }
  static NimBLEServer* createServer() { return &testServer; }
  static NimBLEAdvertising* getAdvertising() { return &testAdvertising; }
};
