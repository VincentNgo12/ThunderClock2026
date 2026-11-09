# BLE status host checks

Compile the real BLE manager with a controllable clock, LED observations and
BLE callbacks. Run each scenario in a fresh process to check startup, blue
blinking, solid green, reconnection and latched red errors:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter -Itest/native_ble/stubs -Iinclude src/BLEManager.cpp test/native_ble/test_ble_status.cpp -o /tmp/thunderclock-ble-test
/tmp/thunderclock-ble-test
/tmp/thunderclock-ble-test init_failure
/tmp/thunderclock-ble-test advertising_failure
/tmp/thunderclock-ble-test restart_failure
```

Physical acceptance: boot on battery, check blue blinking while advertising,
connect from the backend for solid green, then disconnect to restore blue
blinking. A detected sensor, I2C, resource, task or BLE startup failure should
show solid red until reboot. These checks do not detect arbitrary memory
corruption, hangs or loss of power.
