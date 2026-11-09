# InSite pitch deck (local web app)

The six-page ThunderClock / InSite pitch (Team, Numbers, How it works, Live sensors, Simulation training,
What comes next) as a local Vite app. The **Live sensors** page connects to the real backend telemetry
WebSocket, the same way the spine frontend (`../frontend`) does, and falls back to the simulated replay when
no backend is reachable.

Everything is served from this folder: three.js r147, fflate, the Archivo font and the data files are vendored
under `public/`, so the deck runs with no internet at the venue.

## Run it

```sh
cd pitch
npm install
npm run dev          # http://127.0.0.1:5180/
```

`npm run build` writes a static copy to `dist/`; `npm run preview` serves it at http://127.0.0.1:5181/.
Open the deck over http (dev or preview), not as a `file://` page: it loads its data with `fetch`.

### Start the backend for the live page

From `../backend`, with Python 3.11+ and its virtual environment active (see
`docs/TELEMETRY_PROTOCOL.md`, "Connect and try the demo"):

```sh
python -m pip install -r requirements-api.txt
TELEMETRY_SOURCE=simulated python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For the vest, start it with `TELEMETRY_SOURCE=hardware` and the device settings from that document.
The detection hub (`detection/hub`) serves the same WebSocket and adds `fall_state`, which the live page shows.

The deck connects to `ws://127.0.0.1:8000/ws/telemetry`. Point it somewhere else with `?ws=`:

```
http://127.0.0.1:5180/?ws=ws://192.168.1.20:8000/ws/telemetry#live
```

## The live page

- **Source badge** (top right of page 4):
  - `Live: vest`: hardware stream from the vest.
  - `Live: backend simulated`: the backend's synthetic stream (`connection.source` is `simulated`).
  - `Live: backend replay`: a recording played through the backend or the detection hub demo.
  - `Live: backend up, vest not connected`: the backend answers but no wearable data flows.
  - `Offline: simulated replay`: no backend; the deck plays the bundled ThunderSim runs.
- **Auto / Live / Replay** toggle. Auto shows the backend stream whenever it carries wearer data and falls
  back to the replay 2 s after it stops. The socket keeps retrying in the background (1 s, doubling to 5 s),
  so starting the backend mid-talk switches the page to live on its own. Live and Replay force one source.
  The `L` key cycles the three modes; `?mode=live|replay|auto` sets it from the URL. The choice is remembered
  in this browser.
- **Spine:** the four IMU quaternions (`[w, x, y, z]`, frame +X wearer-right, +Y up, +Z posterior). When the
  backend sends heading-free `segments` (gravity model), they are drawn directly, as the frontend does.
  Otherwise each raw orientation is corrected by a neutral taken from its first valid sample
  (`q * inverse(q_neutral)`); **Set upright** in the spine toolbar re-takes that neutral. Pod colours follow
  each IMU's freshness; with no message for 1 s, valid sensors turn stale.
- **Right column:** heart rate, temperature, humidity, ENS160 AQI / eCO2 / TVOC and relative noise from
  `heart_rate_bpm` and `environment`, shown only while their `freshness` group is valid. A missing value shows
  a dash and the reason (not in this stream, stale, no data). Traces show the last 45 s.
- **Status line:** `fall_state` from the detection hub when present; otherwise the backend's posture alerts
  and trunk posture zone.

## Presenting

- Fullscreen: F11 (Windows/Linux) or Ctrl+Cmd+F (macOS).
- Next / previous page: arrow keys, Page Up / Page Down, Space; Home and End jump to the first and last page.
- Direct links: `#team`, `#numbers`, `#how`, `#live`, `#simulation`, `#close`.
- Start the backend before the talk if you can; if it is not running, the live page quietly shows the replay.

## Layout

```
index.html          deck markup and styles (the published deck, as a full HTML document)
src/main.js         entry: imports the page scripts in order
src/live.js         page 4: spine view, live WebSocket mode, replay
src/sim.js          page 5: simulation grid (Anny + anatomy)
src/how.js          page 3: Anny in the vest
public/vendor/      three.js r147 (build + OrbitControls, GLTFLoader, RoundedBoxGeometry), fflate 0.8.2
public/fonts/       Archivo variable woff2 (latin, latin-ext, roman and italic) + OFL.txt
public/data/        runs, spine model, Anny body and anatomy (JSON)
```

## Credits

- Anny body model: NAVER LABS Europe, Apache-2.0. Mesh, rig and weights from MakeHuman / MPFB2, CC0.
- Vertebrae and anatomy: BodyParts3D, The Database Center for Life Science (DBCLS), CC BY 4.0.
- Archivo: The Archivo Project Authors (Omnibus-Type), SIL Open Font License 1.1 (`public/fonts/OFL.txt`).
- three.js r147: three.js authors, MIT (`public/vendor/three/LICENSE`).
- fflate 0.8.2: Arjun Barrett, MIT (`public/vendor/fflate/LICENSE`).

All runs on the Simulation and replay pages are simulated. This is a prototype, not a medical device.

## Connect the vest from the deck

The Live sensors page has a **Connect vest** button with a format menu (default **Quaternions + raw**, the
backend's `both` format; also Quaternions, or Raw fused on the laptop). It calls the backend's
`POST /device/connect?device_format=...` and `POST /device/disconnect`, the same calls the spine frontend's
Connect button makes, and shows the backend's error text if the scan fails (Bluetooth off, no permission,
vest not found).

- Start the backend in manual mode so it waits for the button:
  `cd backend && TELEMETRY_SOURCE=manual .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
- The deck reaches the backend's HTTP API through the Vite proxy at `/backend` (no CORS setup needed). Point the
  proxy elsewhere with `BACKEND_HTTP=http://host:port npm run dev`. With `?ws=...` the calls go to that host directly.
- `?format=quaternion|both|raw_fusion` preselects the format.
- The button is disabled when the backend runs a simulated or replay source.
