# SignAI — Sign Language Translator

## What This Project Is

A real-time ASL (American Sign Language) web app that lets a **deaf person** communicate with a **hearing person** over a video call. The deaf person signs into their camera; the app detects the sign and displays it as text. The hearing person can speak (speech-to-text) or press sign buttons to show the deaf person a GIF demonstrating how to do the sign.

Deployed on **Railway** via Docker. GitHub repo: `ET-32/sign-language-app`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask, Flask-SocketIO |
| ML / Vision | MediaPipe (hand landmarks), scikit-learn (SVM) |
| Frontend | Vanilla JS, HTML/CSS (single file: `assets/index.html`) |
| Real-time | Socket.IO (video relay + signaling) |
| Deployment | Docker → Railway |

---

## Project Structure

```
sign_language_app/
├── main.py                  # Entry point — calls web_app.run()
├── Dockerfile               # python:3.11-slim, installs libgl1 for MediaPipe
├── requirements.txt         # mediapipe, opencv, scikit-learn, flask, flask-socketio
├── assets/
│   ├── index.html           # Entire frontend (UI, JS, CSS — all in one file)
│   └── signs/               # ASL GIFs served by Flask
│       ├── a.gif … z.gif    # Fingerspelling letters
│       ├── good.gif         # Standalone GOOD sign (Dr. Bill Vicars / lifeprint.com)
│       └── ok.gif           # Standalone OK sign (Dr. Bill Vicars / lifeprint.com)
├── src/
│   ├── web_app.py           # Flask app + Socket.IO events
│   ├── predictor.py         # MediaPipe hand detection + SVM inference
│   ├── collect_data.py      # Data collection script (webcam → CSV)
│   ├── train_model.py       # Train SVM from CSV → models/sign_model.pkl
│   └── renormalize_csv.py   # Re-normalize existing CSV without recollecting data
├── models/
│   └── sign_model.pkl       # Trained SVM model (committed to git for Railway)
└── data/
    └── signs.csv            # Training data: 126 features + label per row
```

---

## Signs the Model Recognises (13 total)

`HELLO`, `GOOD`, `THANK YOU`, `YES`, `NO`, `OK`, `I LOVE YOU`, `CALL ME`, `PEACE`, `SORRY`, `PLEASE`, `HELP`, `STOP`

Model: SVM (rbf kernel, C=10, gamma='scale') with StandardScaler. ~98.5% test accuracy after landmark normalization.

---

## Key Architecture Decisions

### 1. Video Relay (not WebRTC)
WebRTC was removed. Video is relayed via Socket.IO:
- Client captures camera frame → draws to canvas (capped at 640px wide) → encodes as JPEG (quality 0.80) → emits `video-frame` to server → server relays as `peer-video-frame` to partner → partner displays in `<img>` tag.
- **Ack-gated send guard**: next frame only sent after server acknowledges the previous one. This naturally throttles to network speed and prevents buffer overflow (which caused a ~10-second freeze bug).
- Connection quality badge (🟢🟡🔴) based on rolling average of ack round-trip times.

### 2. Landmark Normalization
Raw MediaPipe x/y/z coordinates are camera-position and scale dependent. The model normalizes each hand by:
1. Subtracting wrist (landmark 0) — makes origin = wrist
2. Dividing by wrist-to-middle-MCP distance (landmark 9) — makes scale = palm size

This makes predictions invariant to hand position in frame, distance from camera, and hand size. Without this, the model only worked well for the person/device it was trained on.

### 3. Sign It Feature
The hearing person opens a "Sign It" panel, presses a sign button (e.g. HELLO), and:
1. A GIF demonstrating the sign plays locally so they can learn it.
2. A `sign-trigger` socket event is emitted to the server.
3. The server relays it as `peer-sign-trigger` to the deaf partner.
4. A slide-up popup appears on the deaf partner's screen showing the same GIF + sign name ("Partner is signing to you"). Auto-dismisses after 5 seconds.

GIF sources: Giphy (Sign with Robert, ASL Nook, CSDRMS) for most signs. `good.gif` and `ok.gif` are served locally from `assets/signs/` (sourced from lifeprint.com / Dr. Bill Vicars — most authoritative ASL resource).

---

## Flask Routes

| Route | Purpose |
|---|---|
| `GET /` | Serves `assets/index.html` |
| `GET /create-room` | Returns a new random 6-char room code |
| `GET /signs/<letter>` | Serves single-letter ASL GIF (a–z) from `assets/signs/` |
| `POST /predict` | Accepts base64 JPEG, returns `{label, confidence, image}` |

**Note:** `/signs/<letter>` currently only allows 1-char names. `good.gif` and `ok.gif` are in `assets/signs/` but the route still blocks them — this is a known pending fix (extend route to allow alpha-only names up to 20 chars).

## Socket.IO Events

| Event (client→server) | What it does |
|---|---|
| `join-room` | Join or create a room (max 2 people) |
| `video-frame` | Send a camera frame to relay to partner |
| `call-speech` | Send speech transcript to partner |
| `sign-update` | Send detected sign label to partner |
| `sign-trigger` | Relay a Sign It button press to partner |

| Event (server→client) | What it does |
|---|---|
| `room-joined` | Confirms room join, returns count |
| `peer-ready` | Tells first user that partner joined → start relay |
| `peer-video-frame` | Partner's relayed camera frame |
| `peer-sign` | Partner's detected sign label |
| `peer-speech` | Partner's speech text |
| `peer-sign-trigger` | Partner pressed a Sign It button |
| `peer-left` | Partner disconnected |
| `call-error` | Room not found / room full |

---

## Frontend (assets/index.html)

Everything is in one file. Key JS globals:

- `socket` — Socket.IO connection (null until call starts)
- `roomId` — current room code
- `CONF_THRESHOLD` — confidence threshold (0.65 default, adjustable via sensitivity slider)
- `transcriptWords[]` — array of detected signs (enables undo)
- `videoRelayTimer` — setInterval handle for the ack-gated relay loop
- `SIGN_GIFS` — map of sign name → Giphy GIF URL (+ local `/signs/good` and `/signs/ok`)

Key functions:
- `activate()` — starts solo mode (camera + prediction loop)
- `openCallModal()` — opens video call setup
- `createRoom()` / `joinRoom()` — room management
- `initSocket()` — creates socket and registers all event handlers
- `startVideoRelay()` — begins the ack-gated frame relay loop
- `endCall()` — stops everything, navigates back to landing page
- `goHome()` — returns to landing page from solo mode
- `showSign(sign, btn)` — shows GIF locally + emits `sign-trigger` if in call
- `showIncomingSign(sign)` — shows popup on deaf user's screen when partner signs
- `undoTranscript()` — removes last word from transcript array

UI states:
- **Landing page** (`#landing`) — shown on load
- **Solo mode** (`#app.active` with `#panels` visible) — deaf person alone
- **Call waiting** (`#call-overlay` with waiting view) — waiting for partner
- **Call active** (`#call-screen`) — both connected, video relay running

---

## How to Run Locally

```bash
pip install -r requirements.txt
python main.py
# → http://localhost:5055
```

## How to Retrain the Model

```bash
# Collect new data (webcam)
py -m src.collect_data

# Or re-normalize existing CSV without recollecting
py -m src.renormalize_csv

# Train
py -m src.train_model
```

The model must be committed to git (`models/sign_model.pkl`) so Railway can use it — Railway has no webcam to collect data.

---

## Deployment (Railway)

- Dockerfile: `python:3.11-slim` + system libs for MediaPipe/OpenCV
- CMD: `python main.py`
- Auto-deploys on push to `main` branch of `ET-32/sign-language-app`
- No environment variables needed

---

## Auth & Subscription

- Database: SQLite (`signai.db` in project root). Uses `DATABASE_URL` env var if set (e.g. PostgreSQL on Railway).
- Secret key: `SECRET_KEY` env var (falls back to a dev default — set it in production).
- Free plan: 10 video calls/month per user. Paid (Pro): unlimited.
- `is_subscribed` field on User controls this. To manually upgrade a user: set `user.is_subscribed = True` in a Flask shell.
- Call count resets automatically at the start of each new calendar month.
- **Payment integration** not yet wired — the "Upgrade" button shows a placeholder. Connect Stripe (or similar) to the `/api/upgrade` endpoint when ready.

## Signs the Model Recognises (23 total — 10 new need data collection + retraining)

**Original 13 (model trained):** `HELLO`, `GOOD`, `THANK YOU`, `YES`, `NO`, `OK`, `I LOVE YOU`, `CALL ME`, `PEACE`, `SORRY`, `PLEASE`, `HELP`, `STOP`

**New 10 (Sign It panel ready — model needs retraining):** `WAIT`, `MORE`, `FINISH`, `WHERE`, `WHAT`, `NAME`, `EAT`, `WATER`, `BATHROOM`, `UNDERSTAND`

To make the new 10 detectable: run `py -m src.collect_data`, collect each sign, then `py -m src.train_model`.

New GIF files needed in `assets/signs/` (download from lifeprint.com):
`wait.gif`, `more.gif`, `finish.gif`, `where.gif`, `what.gif`, `name.gif`, `eat.gif`, `water.gif`, `bathroom.gif`, `understand.gif`

## Known Pending Items

1. **New sign GIFs** — Download the 10 GIFs from [lifeprint.com](https://www.lifeprint.com/asl101/pages-signs/) and place them in `assets/signs/` with matching lowercase filenames (e.g. `wait.gif`). The Flask route and frontend are already wired up.

2. **New sign detection** — Collect training data for the 10 new signs with `py -m src.collect_data`, then retrain with `py -m src.train_model`. Commit the new `models/sign_model.pkl` for Railway.

3. **Upgrade payment** — Wire Stripe (or similar) to complete Pro subscriptions. The UI and `is_subscribed` field are ready.

4. **Railway SQLite persistence** — SQLite data is lost on redeploy without a volume. For production: set `DATABASE_URL` to a PostgreSQL instance and add `psycopg2-binary` to `requirements.txt`.

5. **Sign detection accuracy** — Model trained on one person. Sensitivity slider (40–90%) helps. Re-training with diverse data would help more.
