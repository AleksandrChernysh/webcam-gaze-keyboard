# Webcam Gaze Tracker and Gaze-Controlled Keyboard

A real-time eye tracker that runs on an ordinary laptop webcam, measures its own spatial accuracy and precision, and lets you type with your eyes on an on-screen keyboard.

<!-- TODO: add a short GIF of calibration and typing here, e.g. docs/demo.gif -->

## How it works

1. **Face and eye landmarks.** OpenCV reads webcam frames; MediaPipe Face Landmarker finds the face, eye and iris landmarks and the head pose.
2. **Features.** The tracker turns landmarks into eye-position, head-pose and compact eye-appearance features (a small grid sampled from each eye image).
3. **Calibration.** The user looks at 15 points (3 × 5 grid), twice. Recording pauses automatically when the face or head is not stable.
4. **Gaze model.** One multi-output Ridge regression (`StandardScaler` + `Ridge`, α = 10) maps features to screen coordinates. Round 1 is used to validate on round 2, then the model is refit on both rounds.
5. **Verification.** A separate 9-target test measures spatial accuracy and precision on unsmoothed predictions, in degrees of visual angle, using the screen geometry in `config.py`.
6. **Keyboard.** Gaze is lightly smoothed; a key is selected after a 1 s dwell. Letters are grouped so that targets stay large enough for webcam-level accuracy.

## Results

Single user (the author), MacBook webcam, three verification runs without moving the laptop, 9 targets × 2 repetitions each, about 430 mm from the screen:

| Metric | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| Mean spatial accuracy | 2.07° | 2.20° | 2.10° | **2.12°** |
| RMS sample-to-sample precision | 1.27° | 1.22° | 1.19° | **1.23°** |
| Valid predictions | 100% | 100% | 100% | 100% |
| Processing time per frame (MediaPipe + model) | 7.4 ms | 7.3 ms | 7.3 ms | **~7 ms** |

These numbers are from one person in one setup; accuracy for other users, lighting and head positions has not been tested.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# MediaPipe face model (not stored in this repository)
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python main.py
```

Before running, set your screen size, window size and viewing distance in `config.py`; the degree values depend on them.

Controls:

| Key | Action |
|---|---|
| `SPACE` | New calibration → verification → keyboard (required on first run) |
| `V` | Load saved model → verification → keyboard |
| `L` | Load saved model → keyboard |
| `D` | Show/hide camera preview on the keyboard |
| `Q` | Quit |

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Runs the application and connects all stages |
| `calibration.py` | 15-point, two-round calibration |
| `gaze_model.py` | Cleans calibration data, trains, saves and loads the Ridge model |
| `verification.py` | Independent accuracy and precision test |
| `metrics.py` | Pixel-to-degree conversion and metric aggregation |
| `keyboard.py` | Gaze-controlled keyboard |
| `tracker.py`, `backends/` | MediaPipe landmarks and eye-appearance features |
| `config.py` | Screen geometry, timing and model settings |

Recorded data, results and the per-user model are written to `data/`, `outputs/` and `models/`, which are not committed because they contain the user's face and eye features.
