# ThunderClock frontend — spine sensor test page

A minimal page for checking the live sensor → backend → browser link. It shows only
the vertebral column (sacrum to T1) on a white background, driven by the four spine
IMUs from `WS /ws/telemetry`.

```sh
# 1. Backend (from backend/): simulated, or hardware with DEVICE_ID / DEVICE_FORMAT set
TELEMETRY_SOURCE=simulated python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2. Frontend (from frontend/)
npm install
npm run dev        # http://localhost:5173
```

Point it at another backend with `?ws=ws://HOST:8000/ws/telemetry` or `VITE_TELEMETRY_URL`.

- Sensor pods sit at the sacrum (pelvis), L3 (lumbar), T8 (thoracic) and T2 (upper
  thoracic). Change `SENSOR_ANCHORS` in `src/lib/posture3d/spine.ts` to match the real mounting.
- Vertebrae between sensors are interpolated, not measured. **Spline** (default) runs a
  cubic Hermite curve through the four sensor orientations, so the bend changes smoothly
  along the column with no kink at a sensor. **Linear** slerps each pair of neighbouring
  sensors independently, for comparison. Both pass exactly through every sensor.
- The blue line is the sensor curve. It runs through all four pods and the matching point on
  every vertebra between them, and is rebuilt every frame, so it bends live with the sensors.
- Pod colour: green valid, amber stale (or no messages for 1 s), grey unavailable.
- **Set neutral** zeroes the current pose in the browser only. Use it when the raw sensor
  frames don't line up while standing upright. It doesn't call the backend calibration.
- **Test bend** plays a built-in loop in the browser (forward bend, extension, side bends,
  twists, then bend plus twist) in place of live data. Pods turn violet while it runs.
  Click **Stop test** to go back to the sensors. It works with no backend running.
- **Sensor axes** shows each sensor's local X/Y/Z (red/green/blue) for checking mounting.
