# ThunderClock backend

Follow the root AGENTS.md. Start with the standalone BLE diagnostic CLI described
in [DIAGNOSTICS.md](DIAGNOSTICS.md).

- Keep BLE receipt, packet decoding and application telemetry separate. Label
  simulated data explicitly.
- Every handwritten backend Python file starts with File, a concise contextual
  Purpose and Contributors using verified human names or Unassigned. Include
  tests and package files. Omit dates and preserve license notices. Exclude
  generated, vendor, third-party and virtual-environment files.
- Use brief comments for resource ownership, async lifecycle and failure behavior.
  Keep the implementation approachable and add abstractions only for concrete needs.
- Plan one device stream shared by the CLI or optional API. WebSocket clients
  must share the backend reader. Only one process may own a wearable connection.
- Keep BLE callbacks bounded to copying bytes and original local receipt times.
  Decode, validate, print and record outside the callbacks.
- Firmware owns sensor calibration and fusion. Backend work starts with explicit
  decoding and diagnostics. Preserve raw acceleration including gravity.
- Keep unavailable measurements explicit. Four valid orientations alone cannot
  establish posture, fall state, fatigue or overall risk.
- Route OpenBCI Cyton through its official dongle and a separate laptop acquisition
  path. Add BrainFlow only when that integration is implemented.
