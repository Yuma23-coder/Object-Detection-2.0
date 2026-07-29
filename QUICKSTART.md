# Quick Start Guide

## ⚡ 30-Second Setup

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Immediately
```bash
python creatte.py --show
```

---

## 🎯 Common Commands

| Use Case | Command |
|----------|---------|
| **Webcam Detection** | `python creatte.py --show` |
| **Video File** | `python creatte.py --source video.mp4 --show` |
| **Save Output** | `python creatte.py --source video.mp4 --output result.mp4` |
| **GPU Speed** | `python creatte.py --device cuda --show` |
| **High Precision** | `python creatte.py --conf 0.7 --show` |
| **RTSP Stream** | `python creatte.py --source rtsp://ip:554/stream --show` |

---

## 📊 What You Get

✅ **Live Bounding Boxes** with object IDs  
✅ **HTML Dashboard** at `object_detection_report.html`  
✅ **80 Object Types** (person, car, dog, etc.)  
✅ **Real-time Tracking** with DeepSORT  
✅ **FPS Counter** on video  
✅ **Annotated Video Output** (optional)  

---

## 🔧 All Options

```bash
python creatte.py \
  --source 0 \                    # 0 = webcam, or video.mp4
  --model yolov8n.pt \            # Model to use
  --conf 0.45 \                   # Confidence threshold
  --iou 0.5 \                     # NMS IoU threshold
  --device cpu \                  # cpu or cuda
  --show \                        # Display window
  --output output.mp4 \           # Save video
  --html-output report.html \     # HTML report path
  --max-frames 500                # Stop after N frames
```

---

## 🆘 Troubleshooting

| Problem | Fix |
|---------|-----|
| **Model errors** | Ensure `yolov8n.pt` is in same directory |
| **Camera not found** | Try `--source 1` or different index |
| **Slow processing** | Add `--device cuda` (if GPU available) |
| **DeepSORT not working** | Falls back to SimpleTracker automatically |

---

## 📈 Performance

- **YOLOv8 Nano**: ~25-30 FPS on CPU
- **With CUDA**: ~100+ FPS on modern GPU
- **Output HTML**: Updates every frame

---

## 📂 Files Generated

```
Object Detection and Tracking/
├── creatte.py                          # Main script
├── yolov8n.pt                          # Model
├── requirements.txt                    # Dependencies
├── README.md                           # Full documentation
├── QUICKSTART.md                       # This file
└── object_detection_report.html        # Live dashboard (auto-generated)
```

---

## 🚀 Next Steps

1. ✅ Install: `pip install -r requirements.txt`
2. ✅ Test: `python creatte.py --show`
3. ✅ Check: Open `object_detection_report.html` in browser
4. ✅ Customize: Use command options above

**That's it! You're tracking objects in real-time.** 🎉
