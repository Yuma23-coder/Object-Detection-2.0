# Object Detection and Tracking

A real-time object detection and tracking system using **YOLOv8** and **DeepSORT** with live HTML dashboard reporting.

## ✨ Features

- **Real-time Object Detection**: YOLOv8 nano model for fast inference
- **Multi-Object Tracking**: DeepSORT tracker with fallback SimpleTracker
- **80 COCO Classes**: Person, vehicle, animal, and 77+ other object types
- **Live Dashboard**: HTML report with metrics, detections, and tracking data
- **Video Support**: Webcam, video files, or RTSP streams
- **Video Export**: Save annotated output videos
- **Configurable**: Adjustable confidence, IoU threshold, device selection
- **GPU Acceleration**: CUDA support for faster processing
- **Cross-Platform**: Works on Windows, macOS, and Linux

---

## 🔄 System Flowchart

```
┌─────────────────┐
│  Video Input    │
│ (Webcam/File)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  YOLOv8 Detection   │
│ (Extract boxes)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  DeepSORT Tracker   │
│ (Assign IDs)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Draw Annotations   │
│ (Boxes + IDs)       │
└────────┬────────────┘
         │
    ┌────┴────┐
    │          │
    ▼          ▼
┌────────┐ ┌──────────────┐
│Display │ │Update HTML   │
│Window  │ │Report/Stats  │
└────────┘ └──────────────┘
    │              │
    └──────┬───────┘
           │
           ▼
    ┌─────────────────┐
    │ Save Video      │
    │ (if enabled)    │
    └─────────────────┘
```

---

## 📦 Installation

### Prerequisites
- **Python 3.8+**
- **pip** (Python package manager)

### Step 1: Clone or Download
```bash
cd "Object Detection and Tracking"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python numpy ultralytics deep-sort-realtime
```

### Step 3: Download Model (if needed)
The YOLOv8 nano model will auto-download on first run. Pre-included: `yolov8n.pt`

---

## 🚀 Quick Start

### Basic Usage - Webcam Detection
```bash
python creatte.py --show
```

### Video File Processing
```bash
python creatte.py --source "path/to/video.mp4" --show
```

### Save Output Video
```bash
python creatte.py --source "path/to/video.mp4" --output "output.mp4" --show
```

### GPU Acceleration
```bash
python creatte.py --device cuda --show
```

### Adjust Confidence Threshold
```bash
python creatte.py --conf 0.5 --show
```

---

## 📋 Command-Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Webcam index, video path, or RTSP URL |
| `--model` | `yolov8n.pt` | YOLO model (yolov8n, yolov8s, yolov8m, etc.) |
| `--conf` | `0.45` | Detection confidence threshold (0-1) |
| `--iou` | `0.5` | NMS IoU threshold (0-1) |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--show` | - | Display live video window |
| `--output` | - | Save annotated video path |
| `--html-output` | `object_detection_report.html` | HTML report path |
| `--max-frames` | - | Stop after N frames |

---

## 📊 HTML Dashboard Features

The system generates a **live HTML report** with:

- ✅ **Session Overview**: Model, frames processed, detection count
- ✅ **Live Detections**: Real-time object list with confidence scores
- ✅ **Tracked Objects**: Confirmed tracks with unique IDs and positions
- ✅ **History**: Last 10 frames with detection trends
- ✅ **Configuration Panel**: Current settings and thresholds
- ✅ **Dark Theme Dashboard**: Modern, responsive UI

**View Report**: Open `object_detection_report.html` in any browser

---

## 💻 Architecture

```
creatte.py
├── Constants
│   └── COCO_CLASS_NAMES (80 classes)
├── Classes
│   └── SimpleTracker (fallback tracker)
├── Functions
│   ├── parse_args()          - CLI argument parsing
│   ├── get_class_name()      - Class ID to name conversion
│   ├── build_detections()    - Extract YOLO detections
│   ├── write_html_report()   - Generate HTML dashboard
│   └── main()                - Main processing loop
└── Models
    └── yolov8n.pt (nano model)
```

---

## 🎯 Example Usage Scenarios

### Scenario 1: Security Monitoring
```bash
python creatte.py --source "rtsp://camera_ip:554/stream" --conf 0.6 --output security_recording.mp4
```

### Scenario 2: Webcam Detection
```bash
python creatte.py --show
```

### Scenario 3: Batch Video Processing
```bash
python creatte.py --source "surveillance_video.mp4" --output "annotated.mp4" --max-frames 1000
```

### Scenario 4: High-Precision Detection
```bash
python creatte.py --device cuda --conf 0.7 --iou 0.6 --show
```

---

## 🔍 Object Detection Classes (COCO)

**People & Animals**: person, bicycle, cat, dog, horse, sheep, cow, bear, zebra, etc.

**Vehicles**: car, truck, bus, train, motorcycle, aeroplane, boat, etc.

**Household**: chair, sofa, dining table, bed, toilet, tv, laptop, keyboard, etc.

**Sports**: ball, baseball bat, tennis racket, skateboard, surfboard, etc.

**Food**: banana, apple, orange, donut, pizza, sandwich, etc.

**Other**: umbrella, handbag, tie, backpack, scissors, vase, clock, etc.

---

## 📈 Performance Tips

| Optimization | Command |
|--------------|---------|
| **Faster Processing** | `--device cuda --conf 0.5` |
| **Better Accuracy** | `--conf 0.7 --iou 0.6` |
| **Lower Latency** | `--max-frames 100` (test first) |
| **Lighter Model** | Use `yolov8n.pt` (included) |
| **Heavier Model** | Switch to `yolov8m.pt` or `yolov8l.pt` |

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **Model not downloading** | Place `yolov8n.pt` in same directory |
| **No video source** | Check camera index or file path |
| **Low FPS** | Switch to `--device cpu` if CUDA issues, or reduce resolution |
| **DeepSORT error** | Falls back to SimpleTracker automatically |
| **HTML not updating** | Check file path permissions |

---

## 📝 Output Example

```
Starting object detection and tracking...
HTML report will be written to: /path/to/object_detection_report.html
Processed 500 frames at 25.32 FPS
```

---

## 🤝 Contributing

Feel free to fork, modify, and submit improvements!

---

## 📄 License

MIT License - Free for personal and commercial use.

---

## 🎓 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DeepSORT Tracker](https://github.com/mikel-brostrom/yolo_tracking)
- [OpenCV](https://opencv.org/)

---

**Built with ❤️ for real-time object detection and tracking**
