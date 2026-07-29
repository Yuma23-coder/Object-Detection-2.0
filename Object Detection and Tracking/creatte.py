import argparse
import time
from datetime import datetime
from html import escape
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        active = []
        for box, conf, cls_id in detections:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            track_id = self.next_id
            self.next_id += 1
            active.append((track_id, (x1, y1, x2, y2), conf, cls_id))
        self.tracks = {track_id: (bbox, conf, cls_id) for track_id, bbox, conf, cls_id in active}
        return active


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 + DeepSORT object tracking")
    parser.add_argument("--source", default=0, help="Video source: webcam index, video path, or RTSP URL")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu or cuda")
    parser.add_argument("--show", action="store_true", help="Display the video window")
    parser.add_argument("--output", default=None, help="Optional output video path")
    parser.add_argument("--html-output", default="object_detection_report.html", help="HTML report path to show the current detection summary")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames")
    return parser.parse_args()


def get_class_name(cls_id):
    if 0 <= cls_id < len(COCO_CLASS_NAMES):
        return COCO_CLASS_NAMES[cls_id]
    return f"class_{cls_id}"


def write_html_report(html_path, args, frame_count, detections, track_items, frame_shape, history):
    output_path = Path(html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detection_rows = []
    for idx, (box, conf, cls_id) in enumerate(detections, 1):
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        detection_rows.append(
            f"<tr><td>{idx}</td><td>{escape(get_class_name(cls_id))}</td><td>{conf:.2f}</td><td>({x1}, {y1}) to ({x2}, {y2})</td></tr>"
        )

    if not detection_rows:
        detection_rows.append('<tr><td colspan="4">No objects detected yet.</td></tr>')

    track_rows = []
    for track_id, (x1, y1, x2, y2) in track_items:
        track_rows.append(
            f"<tr><td>{track_id}</td><td>({x1}, {y1}) to ({x2}, {y2})</td></tr>"
        )

    if not track_rows:
        track_rows.append('<tr><td colspan="2">No confirmed tracks yet.</td></tr>')

    history_rows = []
    for entry in history[-10:]:
        history_rows.append(
            f"<tr><td>{entry['frame']}</td><td>{entry['detections']}</td><td>{entry['tracks']}</td><td>{entry['time']}</td></tr>"
        )

    if not history_rows:
        history_rows.append('<tr><td colspan="4">No history yet.</td></tr>')

    height, width = frame_shape[:2]
    css_content = """
:root {
    color-scheme: dark;
    color: #e2e8f0;
    background: #020617;
    font-family: 'Poppins', sans-serif;
    --bg: #03081d;
    --panel: rgba(15, 23, 42, 0.95);
    --panel-strong: rgba(15, 23, 42, 1);
    --border: rgba(148, 163, 184, 0.18);
    --accent: #38bdf8;
    --accent-2: #8b5cf6;
    --surface: rgba(15, 23, 42, 0.92);
    --text-muted: #94a3b8;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    min-height: 100vh;
    background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 30%),
                radial-gradient(circle at bottom right, rgba(139, 92, 246, 0.14), transparent 28%),
                linear-gradient(180deg, #08101f 0%, #020512 100%);
    color: #e2e8f0;
}

body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at top right, rgba(56, 189, 248, 0.12), transparent 25%),
                radial-gradient(circle at bottom left, rgba(139, 92, 246, 0.12), transparent 22%);
    pointer-events: none;
}

.page {
    width: min(1200px, calc(100% - 32px));
    margin: 0 auto;
    padding: 32px 0 56px;
}

.header {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 18px;
    align-items: center;
    margin-bottom: 32px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.brand-mark {
    width: 56px;
    height: 56px;
    border-radius: 18px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    font-weight: 700;
    letter-spacing: -0.04em;
    font-size: 1.4rem;
    color: #020617;
}

.brand-copy h1 {
    font-size: clamp(1.9rem, 2.7vw, 2.7rem);
    line-height: 1.05;
}

.brand-copy p {
    margin-top: 10px;
    color: var(--text-muted);
    max-width: 640px;
    font-size: 0.98rem;
}

.actions {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

.button {
    border: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.8);
    color: inherit;
    padding: 0.95rem 1.25rem;
    border-radius: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
}

.button:hover {
    transform: translateY(-1px);
    border-color: rgba(56, 189, 248, 0.45);
    background: rgba(15, 23, 42, 1);
}

.hero-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 22px;
    margin-bottom: 32px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 26px;
    backdrop-filter: blur(18px);
    overflow: hidden;
}

.panel h2 {
    font-size: 1.1rem;
    margin-bottom: 16px;
    color: #f8fafc;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
}

.metric {
    border-radius: 22px;
    background: rgba(15, 23, 42, 0.7);
    padding: 18px 20px;
    border: 1px solid rgba(148, 163, 184, 0.12);
}

.metric span {
    display: block;
    color: var(--text-muted);
    font-size: 0.92rem;
    margin-bottom: 8px;
}

.metric strong {
    font-size: 1.8rem;
    display: block;
    color: #edf2f7;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0.7rem 1rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 18px;
}

.grid-layout {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 24px;
}

.content-panel {
    display: grid;
    gap: 24px;
}

.content-panel .tab-switcher {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}

.tab-button {
    border-radius: 16px;
    padding: 0.95rem 1rem;
    border: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.75);
    color: inherit;
    text-align: center;
    cursor: pointer;
    transition: transform 0.18s ease, background 0.18s ease;
}

.tab-button.active,
.tab-button:hover {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(139, 92, 246, 0.24));
    border-color: rgba(56, 189, 248, 0.38);
}

.card-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 28px;
}

.card-panel h3 {
    margin-bottom: 18px;
    font-size: 1.15rem;
}

.table-wrapper {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    min-width: 640px;
}

thead tr {
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
}

th,
td {
    text-align: left;
    padding: 14px 16px;
}

th {
    color: #cbd5e1;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}

tbody tr {
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
}

tbody tr:last-child {
    border-bottom: none;
}

td {
    color: #e2e8f0;
    font-size: 0.96rem;
}

.status-pill {
    display: inline-flex;
    padding: 0.45rem 0.75rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.14);
    color: #38bdf8;
    font-size: 0.86rem;
    font-weight: 600;
}

.panel-aside {
    display: grid;
    gap: 18px;
}

.panel-aside .panel {
    padding: 24px;
}

.panel-aside h2 {
    margin-bottom: 16px;
}

.panel-aside p,
.panel-aside li {
    color: var(--text-muted);
    line-height: 1.8;
}

@media (max-width: 980px) {
    .grid-layout {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .header,
    .hero-grid,
    .topbar {
        grid-template-columns: 1fr;
    }

    .tab-button {
        font-size: 0.92rem;
    }
}
"""

    js_content = """
const panels = document.querySelectorAll('[data-panel]');
const buttons = document.querySelectorAll('[data-target]');

function switchPanel(target) {
    panels.forEach(panel => panel.hidden = panel.dataset.panel !== target);
    buttons.forEach(button => button.classList.toggle('active', button.dataset.target === target));
}

function copyReport() {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(window.location.href).then(() => alert('Report URL copied to clipboard.'));
    } else {
        alert('Copy is not supported in this browser.');
    }
}

document.addEventListener('DOMContentLoaded', () => switchPanel('overview'));
"""

    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Object Detection and Tracking Report</title>
  <style>{css_content}</style>
</head>
<body>
  <div class=\"page\">
    <div class=\"header\">
      <div class=\"brand\">
        <div class=\"brand-mark\">TB</div>
        <div class=\"brand-copy\">
          <h1>Object Detection & Tracking Dashboard</h1>
          <p>Realtime analytics from YOLOv8 with tracked objects, timestamps, and live system metrics.</p>
        </div>
      </div>
      <div class=\"actions\">
        <button class=\"button\" type=\"button\" onclick=\"switchPanel('overview')\">Overview</button>
        <button class=\"button\" type=\"button\" onclick=\"switchPanel('objects')\">Objects</button>
        <button class=\"button\" type=\"button\" onclick=\"switchPanel('tracks')\">Tracks</button>
        <button class=\"button\" type=\"button\" onclick=\"copyReport()\">Copy Link</button>
      </div>
    </div>

    <div class=\"hero-grid\">
      <section class=\"panel\">
        <div class=\"badge\">Live report</div>
        <h2>Session overview</h2>
        <div class=\"metric-grid\">
          <div class=\"metric\">
            <span>Model</span>
            <strong>{escape(args.model)}</strong>
          </div>
          <div class=\"metric\">
            <span>Active frame</span>
            <strong>{frame_count}</strong>
          </div>
          <div class=\"metric\">
            <span>Detected objects</span>
            <strong>{len(detections)}</strong>
          </div>
          <div class=\"metric\">
            <span>Confirmed tracks</span>
            <strong>{len(track_items)}</strong>
          </div>
        </div>
        <p class=\"status-pill\">Source: {escape(str(args.source))}</p>
      </section>

      <aside class=\"panel panel-aside\">
        <h2>Configuration</h2>
        <ul>
          <li>Confidence threshold: {args.conf}</li>
          <li>IoU threshold: {args.iou}</li>
          <li>Frame resolution: {width} × {height}</li>
          <li>Output report: {escape(str(args.html_output))}</li>
        </ul>
      </aside>
    </div>

    <div class=\"grid-layout\">
      <div class=\"content-panel\">
        <div class=\"tab-switcher\">
          <button class=\"tab-button active\" data-target=\"overview\" onclick=\"switchPanel('overview')\">Overview</button>
          <button class=\"tab-button\" data-target=\"objects\" onclick=\"switchPanel('objects')\">Objects</button>
          <button class=\"tab-button\" data-target=\"tracks\" onclick=\"switchPanel('tracks')\">Tracks</button>
          <button class=\"tab-button\" data-target=\"history\" onclick=\"switchPanel('history')\">History</button>
        </div>

        <section class=\"card-panel\" data-panel=\"overview\">
          <h3>Detection Summary</h3>
          <div class=\"table-wrapper\">
            <table>
              <thead>
                <tr><th>Metric</th><th>Value</th></tr>
              </thead>
              <tbody>
                <tr><td>Source</td><td>{escape(str(args.source))}</td></tr>
                <tr><td>Model</td><td>{escape(args.model)}</td></tr>
                <tr><td>Confidence threshold</td><td>{args.conf}</td></tr>
                <tr><td>IoU threshold</td><td>{args.iou}</td></tr>
                <tr><td>Frame size</td><td>{width} × {height}</td></tr>
                <tr><td>Updated</td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class=\"card-panel\" data-panel=\"objects\" hidden>
          <h3>Detected Objects</h3>
          <div class=\"table-wrapper\">
            <table>
              <thead>
                <tr><th>#</th><th>Class</th><th>Confidence</th><th>Location</th></tr>
              </thead>
              <tbody>
                {''.join(detection_rows)}
              </tbody>
            </table>
          </div>
        </section>

        <section class=\"card-panel\" data-panel=\"tracks\" hidden>
          <h3>Active Tracks</h3>
          <div class=\"table-wrapper\">
            <table>
              <thead>
                <tr><th>Track ID</th><th>Bounding Box</th></tr>
              </thead>
              <tbody>
                {''.join(track_rows)}
              </tbody>
            </table>
          </div>
        </section>

        <section class=\"card-panel\" data-panel=\"history\" hidden>
          <h3>Recent Frame History</h3>
          <div class=\"table-wrapper\">
            <table>
              <thead>
                <tr><th>Frame</th><th>Detections</th><th>Tracks</th><th>Time</th></tr>
              </thead>
              <tbody>
                {''.join(history_rows)}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <aside class=\"panel panel-aside\">
        <h2>Notes & next steps</h2>
        <p>Use the tabs to inspect detected objects, tracking assignments, and recent history. The live report is styled with a modern dashboard layout and client-side interactivity.</p>
        <ul>
          <li>Refresh the file in the browser after each update.</li>
          <li>Click <strong>Copy Link</strong> to share the current report URL.</li>
          <li>Run the script again to regenerate the report with fresh data.</li>
        </ul>
      </aside>
    </div>
  </div>
  <script>{js_content}</script>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")


def build_detections(results, frame_shape):
    h, w = frame_shape[:2]
    detections = []

    if not results or len(results) == 0:
        return detections

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        conf = float(box.conf[0].cpu())
        cls_id = int(box.cls[0].cpu())

        if conf < 0.1:
            continue

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        w_box = max(0, x2 - x1)
        h_box = max(0, y2 - y1)

        if w_box <= 0 or h_box <= 0:
            continue

        detections.append(([x1, y1, w_box, h_box], conf, cls_id))

    return detections


def main():
    args = parse_args()

    print("Starting object detection and tracking...")
    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    model = YOLO(args.model)
    try:
        tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet")
    except Exception:
        tracker = None

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, 20.0, (width, height))

    frame_count = 0
    fps_start = time.time()
    simple_tracker = SimpleTracker()
    history = []
    print(f"HTML report will be written to: {Path(args.html_output).resolve()}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if args.max_frames and frame_count > args.max_frames:
            break

        results = model(frame, stream=False, conf=args.conf, iou=args.iou, agnostic_nms=True)
        detections = build_detections(results, frame.shape)

        if tracker is not None and detections:
            tracks = tracker.update_tracks(detections, frame=frame)
            track_items = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                if track.time_since_update > 1:
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                track_items.append((track.track_id, (x1, y1, x2, y2)))
        else:
            track_items = [(item[0], item[1]) for item in simple_tracker.update(detections)]

        for track_id, (x1, y1, x2, y2) in track_items:
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Frames: {frame_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {len(track_items)}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        history.append({
            "frame": frame_count,
            "detections": len(detections),
            "tracks": len(track_items),
            "time": datetime.now().strftime("%H:%M:%S")
        })
        if len(history) > 20:
            history.pop(0)

        write_html_report(args.html_output, args, frame_count, detections, track_items, frame.shape, history)

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("Object Detection & Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - fps_start
    if elapsed > 0:
        fps = frame_count / elapsed
        print(f"Processed {frame_count} frames at {fps:.2f} FPS")


if __name__ == "__main__":
    main()