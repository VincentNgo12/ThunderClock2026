# EnvironmentManager host checks

These checks compile the real driver with a controllable clock and a small I2C
bus double. They cover conversion scheduling without delays, mux selection,
partial sensors, ENS160 warm-up, temperature/humidity compensation, CRC and bus
failures, stuck conversions, retained-reading expiry and timestamp wraparound.
They do not establish physical sensor acceptance.

From `firmware/`:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -Itest/native_environment/stubs -Iinclude src/EnvironmentManager.cpp test/native_environment/test_environment_manager.cpp -o /tmp/thunderclock-environment-test
/tmp/thunderclock-environment-test
```

Application usage after `Wire.begin()` is `environmentManager.begin()`, then
`environmentManager.read(environmentData)` regularly from one task while holding
the shared I2C mutex. Both return `fw::Result`. `read()` returns `Ok` whenever at
least one fresh reading is available; missing fields remain `NaN`. It schedules
one measurement cycle per `kEnvironmentPeriodMs` and checks completed AHT21
conversions on later calls, without a conversion wait inside the mutex.

`begin()` requires both sensors to initialize. ENS160 samples pass through during
warm-up, initial startup and invalid-quality states, including zero values; the
existing packet has no environment-validity field. Measurement quality is left
to the backend.
