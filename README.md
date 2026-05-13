# Temperature Forecasting

> A Python script that polls live temperature readings from a Firebase Realtime Database and feeds them into a TensorFlow Lite regression model to predict humidity in real time.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [File Descriptions](#file-descriptions)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Firebase Setup](#firebase-setup)
- [Project Setup](#project-setup)
- [Configuration](#configuration)
- [Running the Script](#running-the-script)
- [Expected Output](#expected-output)
- [Version Changelog](#version-changelog)
- [Related Projects](#related-projects)
- [Security Notice](#security-notice)

---

## Overview

This project is the **machine learning inference layer** of a larger IoT monitoring system. It continuously watches a Firebase Realtime Database for new temperature sensor readings. Whenever a new (changed) temperature value is detected, it is passed as input into a pre-trained TensorFlow Lite regression model that outputs a predicted humidity value.

This script is designed to run on a host machine (laptop, desktop, or single-board computer like a Raspberry Pi) that has Python and network access to the Firebase project.

---

## How It Works

1. **Firebase Connection** — The script authenticates with Firebase using a service account credentials file and listens to the node at `Data → Temperature → value`.
2. **Change Detection** — The latest temperature entry is polled on a fixed interval. The model is only invoked when the temperature value has changed since the last check, avoiding redundant predictions.
3. **TFLite Inference** — The new temperature value is formatted as a float32 NumPy array and fed into a TensorFlow Lite interpreter loaded with a pre-trained regression model (`regModel.tflite`).
4. **Output** — The predicted humidity value is printed to the console.

---

## System Architecture

```
┌───────────────────────────────┐
│   IoT Hardware (e.g. NodeMCU) │
│   - Reads temp/humidity sensor│
│   - Pushes to Firebase        │
└──────────────┬────────────────┘
               │ HTTPS (Firebase SDK)
               ▼
┌───────────────────────────────┐
│  Firebase Realtime Database   │
│  /Data/Temperature/value/     │
└──────────────┬────────────────┘
               │ firebase_admin (Python SDK)
               ▼
┌───────────────────────────────┐
│   TempHum.py (this script)    │
│   - Polls for new readings    │
│   - Change detection filter   │
│   - TFLite model inference    │
│   - Prints predicted humidity │
└───────────────────────────────┘
```

> This script is part of the broader **TerraGuard** / **TempHum** IoT ecosystem. The hardware side (NodeMCU + Arduino that pushes sensor data to Firebase) is maintained in a separate repository.

---

## File Descriptions

| File | Description |
|---|---|
| `TempHum.v5.py` | Version 5 — polls every 25 seconds; tracks both `latest_temp` and `latest_humidity` variables; basic console output |
| `TempHum.v6.py` | Version 6 — polls every 10 seconds; formatted output with separator lines; simplified variable tracking — **recommended** |
| `firebase_credentials.json` | *(Not included — must be provided)* Firebase service account key for authentication |
| `regModel.tflite` | *(Not included — must be provided)* Pre-trained TensorFlow Lite regression model |

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Cloud Database | Firebase Realtime Database |
| Firebase SDK | `firebase-admin` |
| ML Runtime | TensorFlow Lite (`tensorflow`) |
| Numerical Processing | NumPy |

---

## Prerequisites

- Python 3.7 or later
- pip (Python package manager)
- A Firebase project with Realtime Database enabled
- A Firebase service account credentials JSON file
- A trained TFLite regression model file (`regModel.tflite`)

---

## Firebase Setup

**1. Create a Firebase project** at [https://console.firebase.google.com](https://console.firebase.google.com) if you have not already.

**2. Enable the Realtime Database** under Build → Realtime Database.

**3. Ensure your database has the following structure:**

```
/
└── Data/
    └── Temperature/
        └── value/
            └── <key>: <float temperature value>
```

**4. Generate a service account key:**

- Go to Firebase Console → Project Settings → Service Accounts
- Click **Generate new private key**
- Save the downloaded file as `firebase_credentials.json` in the same directory as the script

---

## Project Setup

**1. Clone the repository:**

```bash
git clone https://github.com/devdex-dev/Temperature-Forecasting.git
cd Temperature-Forecasting
```

**2. Install the required Python packages:**

```bash
pip install firebase-admin tensorflow numpy
```

> On a Raspberry Pi or resource-constrained device, you may use `tflite-runtime` instead of the full `tensorflow` package:
> ```bash
> pip install tflite-runtime
> ```
> Then replace `import tensorflow as tf` and `tf.lite.Interpreter(...)` with:
> ```python
> import tflite_runtime.interpreter as tflite
> interpreter = tflite.Interpreter(model_path="regModel.tflite")
> ```

**3. Place the required files** in the project root:

```
Temperature-Forecasting/
├── TempHum.v5.py
├── TempHum.v6.py
├── firebase_credentials.json   ← add this (from Firebase Console)
└── regModel.tflite             ← add this (your trained model)
```

---

## Configuration

Both scripts share the same configurable values:

| Setting | Code Location | Description |
|---|---|---|
| Credentials file | `credentials.Certificate("firebase_credentials.json")` | Path to your service account JSON |
| Database URL | `'databaseURL': 'https://...'` | Your Firebase Realtime DB URL |
| Database path | `ref.child('Data').child('Temperature').child('value')` | Path to temperature values in the DB |
| Model path | `model_path="regModel.tflite"` | Path to your `.tflite` model file |
| Poll interval | `time.sleep(10)` / `time.sleep(25)` | Seconds between Firebase checks |

---

## Running the Script

Run the recommended version (v6):

```bash
python TempHum.v6.py
```

Or run v5:

```bash
python TempHum.v5.py
```

The script runs in an infinite loop. Press `Ctrl + C` to stop.

---

## Expected Output

**v6 output (formatted):**

```
----------------------------------------

Latest temperature reading: 28.5
Predicted humidity: 74.31265

----------------------------------------
```

**v5 output (plain):**

```
Latest temperature reading: 28.5
Predicted humidity: 74.31265
```

The model is only called when the temperature value has changed since the previous poll. If the reading is unchanged, no output is produced for that cycle.

---

## Version Changelog

### v6 — `TempHum.v6.py` *(Current)*

- Reduced poll interval from 25s to **10 seconds**
- Added formatted **separator lines** around each prediction block
- Removed unused `latest_humidity` variable
- Cleaner, more readable change-detection logic

### v5 — `TempHum.v5.py`

- Initial working version
- 25-second poll interval
- Tracked both `latest_temp` and `latest_humidity` variables
- Basic unformatted console output

---

## Related Projects

This script is part of a broader IoT pipeline. See the related repository:

- **[TerraGuard-Hardware-System-Codes](https://github.com/devdex-dev/TerraGuard-Hardware-System-Codes)** — Arduino Uno + NodeMCU firmware that reads the physical sensor and pushes temperature/humidity data to the same Firebase database this script reads from.

---

## Security Notice

> ⚠️ **Do not commit `firebase_credentials.json` to version control.** This file contains a private service account key that grants admin access to your Firebase project.

Add the following to your `.gitignore`:

```gitignore
firebase_credentials.json
regModel.tflite
```

If credentials have been accidentally exposed, revoke the key immediately under Firebase Console → Project Settings → Service Accounts, then generate a new one.

---

*Temperature Forecasting — Firebase-connected TFLite Humidity Prediction*
