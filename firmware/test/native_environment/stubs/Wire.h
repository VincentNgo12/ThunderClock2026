#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

class TwoWire {
 public:
  bool muxPresent = true;
  bool ahtPresent = true;
  bool ensPresent = true;
  bool ahtReadFails = false;
  uint8_t ensStatus = 0x02;
  std::vector<uint8_t> ensData{2, 0, 0, 0x76, 0x02};  // AQI 2, TVOC 0, eCO2 630.
  uint8_t selectedChannel = 0;
  unsigned conversions = 0;
  unsigned transactions = 0;
  // 50% RH and 25 C, with a valid CRC. Tests also corrupt it or mark it busy.
  std::vector<uint8_t> ahtFrame{0x08, 0x80, 0x00, 0x06, 0x00, 0x00, 0xA6};
  std::vector<uint8_t> compensation;

  void beginTransmission(uint8_t address) {
    address_ = address;
    command_.clear();
  }
  size_t write(uint8_t value) {
    command_.push_back(value);
    return 1;
  }
  size_t write(const uint8_t* data, size_t count) {
    command_.insert(command_.end(), data, data + count);
    return count;
  }
  uint8_t endTransmission(bool = true) {
    transactions++;
    if (address_ == 0x70) {
      if (!muxPresent) {
        return 2;
      }
      selectedChannel = command_.at(0);
      return 0;
    }
    if (selectedChannel != 0x20) {
      return 2;
    }
    if (address_ == 0x38 && ahtPresent) {
      if (command_.at(0) == 0xAC) {
        conversions++;
      }
      return 0;
    }
    if (address_ == 0x53 && ensPresent) {
      ensRegister_ = command_.at(0);
      if (ensRegister_ == 0x13) {
        compensation.assign(command_.begin() + 1, command_.end());
      }
      return 0;
    }
    return 2;
  }
  uint8_t requestFrom(uint8_t address, uint8_t count) {
    transactions++;
    response_.clear();
    readIndex_ = 0;
    if (selectedChannel != 0x20) {
      return 0;
    }
    if (address == 0x38 && ahtPresent && !ahtReadFails) {
      response_ = count == 1 ? std::vector<uint8_t>{0x08} : ahtFrame;
    } else if (address == 0x53 && ensPresent) {
      if (ensRegister_ == 0x00) {
        response_ = {0x60, 0x01};
      } else if (ensRegister_ == 0x20) {
        response_ = {ensStatus};
      } else if (ensRegister_ == 0x21) {
        response_ = ensData;
      }
    }
    return static_cast<uint8_t>(response_.size());
  }
  int read() { return response_.at(readIndex_++); }

 private:
  uint8_t address_ = 0;
  uint8_t ensRegister_ = 0;
  size_t readIndex_ = 0;
  std::vector<uint8_t> command_;
  std::vector<uint8_t> response_;
};
