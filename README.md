<<<<<<< HEAD
# AI-Road-Damage-Detection
=======
# AI-Based Road Damage Detection and Geographical Mapping System

A Windows-friendly student project using Python, Ultralytics YOLO, OpenCV, SQLite, GPS/NMEA, Streamlit and Folium.

## What is included

- `app_live.py`: webcam inference, annotated video, optional serial GPS, SQLite reports and saved images.
- `dashboard.py`: report filters, category chart, GPS map, image previews and CSV export.
- `train.py`: YOLO fine-tuning entry point for a correctly prepared dataset.
- `requirements.txt`: Python dependencies.

## Important model requirement

The live application expects a **road-damage-trained YOLO model** at `models/best.pt`. A generic YOLO checkpoint such as `yolo11n.pt` detects general objects and is not guaranteed to detect potholes or road cracks. Train/fine-tune on an annotated road-damage dataset or obtain compatible weights from a trustworthy source and verify the class labels.

## Windows installation

Recommended: Python 3.11 or 3.12.

Open PowerShell in this folder:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the same PowerShell window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Run the live camera

1. Put your trained model at `models\best.pt`.
2. Connect a webcam.
3. Run:

```powershell
python app_live.py
```

Press `Q` in the video window to stop. If the webcam does not open, edit `CAMERA_INDEX` in `app_live.py` or set environment variable `CAMERA_INDEX`.

## Enable serial GPS (optional)

The code reads NMEA GGA sentences from a USB serial GPS receiver. Check Device Manager > Ports (COM & LPT) for the receiver's COM port. Set it in PowerShell before launching:

```powershell
$env:GPS_PORT = "COM3"
$env:GPS_BAUD_RATE = "9600"
python app_live.py
```

Use the actual COM port and baud rate for your receiver. A valid satellite fix is required. Without GPS, reports are still saved with empty coordinates.

A standard laptop webcam does not provide GPS. Phone GPS requires a separate integration; it is not enabled by this code.

## Open the dashboard

In another PowerShell window, activate the same virtual environment and run:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run dashboard.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Prepare data and train

`train.py` assumes a YOLO-format dataset. A dataset YAML should point to train and validation images and list the exact class IDs in order. Example:

```yaml
path: C:/Users/YourUser/Desktop/RoadDamageAI/road_dataset
train: images/train
val: images/val
names:
  0: longitudinal_crack
  1: transverse_crack
  2: alligator_crack
  3: pothole
```

This is only an example. Ensure your dataset annotations use exactly these classes and order. Some road damage datasets use different labels or formats and must be converted to YOLO format before training.

Then run:

```powershell
python train.py
```

Copy the resulting `best.pt` into `models\best.pt`. If training is slow or memory runs out, reduce `batch=8` in `train.py` to `4` or `2`. A compatible GPU is optional, not required.

## Limitations and safety

- GPS markers represent the camera's position at detection time, not the exact defect position. Exact geolocation needs calibration, camera orientation, distance estimation and time synchronization.
- Detection confidence is not damage severity.
- The per-class cooldown reduces repeated saves but does not perform robust duplicate-location matching.
- Test while stationary or use a passenger-operated camera. Do not operate a laptop while driving.
- Review dataset licensing and privacy requirements before deployment.
>>>>>>> 26ac8827 (Initial upload of AI Road Damage Detection System)
