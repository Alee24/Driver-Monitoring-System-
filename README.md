# Driver Monitoring System

This repository contains a complete driver‑monitoring system written in Python.  It analyses a driver’s face using a standard webcam and MediaPipe Face Mesh to estimate drowsiness, yawning, distraction and (optionally) phone usage.  Alerts are delivered via offline text‑to‑speech and an on‑screen heads‑up display (HUD).

## Features

* **Face & landmarks:** Uses [MediaPipe Face Mesh](https://developers.google.com/mediapipe/solutions/vision/face_mesh) to detect a single face and extract 3D landmarks on every frame.
* **Drowsiness detection:**
  * Computes the **Eye Aspect Ratio (EAR)**; if it drops below a configurable threshold the eye is considered closed.
  * Maintains a rolling window of eye states and calculates **PERCLOS** (the percentage of closed eyes).  If the ratio exceeds a threshold the driver is flagged as drowsy.
  * Computes the **Mouth Aspect Ratio (MAR)** and flags a yawn if it exceeds a threshold.
* **Distraction detection:**
  * Estimates head pose (yaw/pitch/roll) using `solvePnP` against a simple 3D face model.
  * Raises an alert if the driver looks away from the road for longer than a configured duration (`gaze_secs`) or looks down past a downward pitch threshold.
* **Phone detection (optional):** If enabled, loads a YOLOv8n model to detect “cell phone” objects in the lower portion of the frame.  Detection runs at a lower rate (≈2 Hz) to save CPU.  If no YOLO weights are present the feature silently disables itself.
* **Audio alerts:** Spoken alerts via [pyttsx3](https://pyttsx3.readthedocs.io/) with a configurable cooldown.  If text‑to‑speech fails the system emits a simple beep.
* **Heads‑up overlay:**
  * Displays EAR, PERCLOS, MAR, yaw and pitch in the top‑left corner.
  * Shows coloured banners (e.g. **DROWSY**, **DISTRACTED**, **PHONE**) when alerts are active.
  * Draws phone bounding boxes when detected.
* **Configuration & CLI:**
  * Default parameters live in `config/default.yaml`.
  * You may provide a `.env` file to override defaults with environment variables.
  * Command‑line flags override both YAML and environment values.
* **Cross‑platform:** Runs on Linux, macOS and Windows and only requires CPU.  Phone detection and text‑to‑speech are optional.

## Installation

1. **Clone and enter the repository**

   ```sh
   git clone <this‑repo‑url> dms_path_a
   cd dms_path_a
   ```

2. **Create and activate a virtual environment**

   ```sh
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **(Optional) Download YOLO weights**

   Phone detection uses the YOLOv8n weights file (`yolov8n.pt`).  If you wish to enable phone detection, download the weights from the [Ultralytics releases page](https://github.com/ultralytics/ultralytics/releases) and place the file in the repository root or specify its path via `--yolo-model`.  Otherwise phone detection will be disabled gracefully.

## Usage

Use the provided scripts or run the module directly.  The example below runs the system on the default camera and disables phone detection:

```sh
python -m src.dms.app --source 0 --enable-phone false
```

Command‑line flags override any values defined in `config/default.yaml` or your `.env` file.  Common flags include:

| Flag | Description | Default |
|-----|-------------|---------|
| `--source` | Camera index or video file path. | `0` |
| `--width`, `--height` | Frame resolution. | `1280`, `720` |
| `--target-fps` | Target frames per second.  Frames are dropped if processing is slower. | `30` |
| `--ear-thresh` | Threshold below which an eye is considered closed. | `0.23` |
| `--perclos-window` | Number of frames used to compute PERCLOS. | `120` |
| `--perclos-thresh` | PERCLOS threshold above which the driver is drowsy. | `0.35` |
| `--mar-yawn` | MAR threshold to detect yawns. | `0.65` |
| `--gaze-secs` | Time (in seconds) the driver may look away before being flagged. | `1.0` |
| `--yaw-abs-deg` | Absolute yaw angle (degrees) beyond which distraction is flagged. | `25` |
| `--down-pitch-deg` | Downward pitch angle (degrees) beyond which distraction is flagged. | `18` |
| `--enable-phone` | Enable phone detection (requires YOLO weights). | `false` |
| `--yolo-model` | Path to YOLOv8n weights file. | `yolov8n.pt` |
| `--speak-rate` | Text‑to‑speech words per minute. | `165` |
| `--headless` | Run without a display window. | `false` |

For a complete list of flags, run:

```sh
python -m src.dms.app --help
```

## Configuration

Parameters are loaded in the following order (later values override earlier ones):

1. **`config/default.yaml`** – Baseline configuration.
2. **Environment variables** – Use upper‑case names prefixed with `DMS_` (e.g. `DMS_EAR_THRESH=0.25`).  See `.env.example` for a template.
3. **Command‑line arguments** – Highest priority.

This hierarchy allows you to define persistent defaults in YAML, temporary overrides in the environment and one‑off changes via the CLI.

## Running on Windows

On Windows you may need to install additional packages for MediaPipe to access your camera.  If you encounter errors when opening the camera, install the pre‑compiled [MediaPipe wheel for Windows](https://github.com/google/mediapipe/blob/master/README.md).  You can then run the included PowerShell script:

```ps1
.\u200brun_local.ps1 -Source 0 -EnablePhone $false
```

## Troubleshooting

* **No face detected:** Ensure the camera is pointing at the driver’s face and that lighting is adequate.  The system only processes one face at a time.
* **Slow performance:** Try reducing the frame resolution or disabling phone detection.  YOLO is throttled to ~2 Hz to save CPU.
* **Audio not spoken:** If `pyttsx3` fails, the system falls back to a simple beep.  You can also disable audio by setting a high cooldown or muting your speakers.

## How to Extend

### Replace heuristics with machine learning

The current system uses simple geometric thresholds (EAR, PERCLOS, MAR, head pose).  To extend this approach:

* **Collect data:** Capture video frames and label drowsy vs. alert behaviour.
* **Train a classifier:** Feed facial features (e.g. facial landmark coordinates, EAR/MAR sequences) into a machine‑learning model (e.g. SVM, XGBoost).  Replace the heuristics in `video_loop.py` with predictions from your classifier.

### Use OpenVINO models

Intel’s OpenVINO toolkit provides optimised models for advanced driver‑assistance systems (ADAS), including face detection, head pose estimation and gaze estimation.  You can replace the MediaPipe and solvePnP components with OpenVINO models to improve accuracy and performance, especially on Intel hardware.

### Add IR camera support

The current implementation assumes visible light.  For night‑time driving, consider adding an infrared camera and an IR illuminator.  You may need to adjust thresholds and pre‑processing; MediaPipe generally works on IR images but performance may vary.

## License

This project is licensed under the MIT License.  See `LICENSE` for details.
