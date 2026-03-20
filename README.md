# ⚽ Football Analysis System

A computer vision pipeline that uses **YOLO object detection** and **ByteTrack** multi-object tracking to detect and annotate players, referees, and the ball in football match footage.

---

## 🎯 Features

- **Object Detection** — Detects players, referees, goalkeepers, and the ball using a custom-trained YOLO model
- **Multi-Object Tracking** — Tracks each detected object across frames with persistent IDs using ByteTrack
- **Visual Annotations** — Draws ellipses under players/referees and triangles above the ball with track ID labels
- **Stub Caching** — Saves/loads tracking results to avoid redundant inference on repeated runs
- **Batch Inference** — Processes video frames in batches for efficient GPU utilization

---

## 🗂️ Project Structure

```
football_analysis/
├── main.py                  # Entry point — runs the full pipeline
├── yolo_inference.py        # Standalone YOLO inference script for quick testing
│
├── trackers/
│   └── tracker.py           # Tracker class: detection, tracking, and annotation logic
│
├── utils/
│   ├── video_utils.py       # read_video() and save_video() helpers
│   └── bbox_utils.py        # Bounding box utility functions
│
├── models/                  # YOLO model weights (.pt files)
├── training/                # Training notebooks and dataset configs
├── input_videos/            # Source match footage
├── output_videos/           # Annotated output videos
├── stubs/                   # Cached tracking results (pickle files)
└── runs/                    # YOLO training/inference run logs
```

---

## ⚙️ How It Works

```
Input Video → Frame Extraction → YOLO Detection (batched)
           → ByteTrack Tracking → Annotation Drawing → Output Video
```

1. **Read** — Video frames are extracted using OpenCV
2. **Detect** — YOLO model runs inference in batches of 20 frames at a time (confidence threshold: 0.1)
3. **Track** — ByteTrack associates detections across frames, assigning consistent IDs
4. **Classify** — Goalkeepers are reclassified as players for unified rendering
5. **Annotate** — Each entity is drawn on-frame:
   - 🔵 **Players** → Colored ellipse at feet + track ID label
   - 🟡 **Referees** → Yellow ellipse at feet
   - 🟢 **Ball** → Green triangle indicator
6. **Save** — Annotated frames are written back to an `.avi` video at 24 FPS

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics supervision opencv-python numpy
```

### Run the Pipeline

```bash
python main.py
```

By default this reads `input_videos/08fd33_4.mp4`, loads cached tracks from `stubs/track_stubs.pkl`, and writes the annotated result to `output_videos/output_video.avi`.

### Quick YOLO Inference Test

```bash
python yolo_inference.py
```

Runs the model on the input video and prints raw detection boxes to stdout.

---

## 🏋️ Training

Custom model training is done with YOLOv5 using the notebook:

```
training/football_training_yolo_v5.ipynb
```

The dataset configuration is located in `training/football-players-detection-1/`.

---

## 📦 Available Models

| File | Description |
|------|-------------|
| `models/best.pt` | Custom-trained football detection model (used by default) |
| `base/yolov8x.pt` | YOLOv8 Extra Large pretrained weights |
| `base/yolov5xu.pt` | YOLOv5 Extra Large pretrained weights |
| `base/yolo26n.pt` | Nano variant |
| `base/yolo26x.pt` | Extra Large variant |

---

## 🛠️ Key Modules

### `Tracker` (`trackers/tracker.py`)
| Method | Description |
|--------|-------------|
| `detect_frames(frames)` | Runs batched YOLO inference |
| `get_object_tracks(frames, ...)` | Returns per-frame tracking dicts for players, referees, and ball |
| `draw_annotations(frames, tracks)` | Renders ellipses and triangles onto each frame |
| `draw_ellipse(frame, bbox, color, track_id)` | Draws player/referee marker |
| `draw_traingle(frame, bbox, color)` | Draws ball indicator |

### `utils/bbox_utils.py`
| Function | Description |
|----------|-------------|
| `get_center_of_bbox(bbox)` | Returns `(cx, cy)` of a bounding box |
| `get_bbox_width(bbox)` | Returns the width of a bounding box |
| `get_foot_position(bbox)` | Returns the bottom-center point of a bbox |
| `measure_distance(p1, p2)` | Euclidean distance between two points |
| `measure_xy_distance(p1, p2)` | Component-wise distance `(dx, dy)` |
