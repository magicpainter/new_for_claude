# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time basketball training analytics system running on NVIDIA Jetson hardware. Uses dual-camera input to analyze dribbling and shooting with TensorRT-accelerated YOLO11 detection, multi-object tracking, and AI-generated coaching feedback.

## Running the Application

**Main entry point:**
```bash
python3 offline_feature_parallel_q.py
```

**Via Docker:**
```bash
docker-compose up
```

The application requires:
- NVIDIA Jetson device with CUDA/TensorRT support
- X11 display and PulseAudio
- Pre-calibrated camera corners (`corners_left.npy`, `corners_right.npy`)
- TensorRT engine files (`.engine` models in project root)

## Testing

```bash
# Basic console test
python3 test_console.py

# BEV ONNX model test
python3 BEV/test_onnx.py

# Simulators (for debugging feature logic)
python3 shooting_simulator.py
python3 dribble_simulator.py
```

No formal test framework is configured. Pytest cache exists but no test suite.

## Architecture

### Multi-Threaded Pipeline

```
Main Thread
├── GUI Thread (PyQt5 + OpenGL rendering)
├── Inference Thread (TensorRT YOLO11 detection)
├── Feature Extraction Threads
│   ├── DribbleFeature (trajectory analysis, dribbles-per-minute)
│   ├── ShootingFeature (shot mechanics, success detection)
│   └── ByteTrack (Kalman filter + Hungarian algorithm tracking)
└── Post-Processing Threads (recording, disk management, upload)
```

### Queue-Based Communication

Threads communicate via queues defined in `offline_feature_parallel_q.py`:
- `cmd_queue`: GUI → Backend commands
- `tex_queue`: GPU texture updates
- `res_queue`: Inference results → GUI
- `inf_q`: Inference results for feature extraction
- `gui2infer_queue`: GUI settings → Inference

### GPU Acceleration

- TensorRT engines with FP16 precision (`.engine` files)
- CUDA GL interop for zero-copy GPU→GPU rendering (`Gl_Monitor.py`)
- PBO (Pixel Buffer Object) pool for efficient texture updates
- Pagelocked host memory for PCIe transfers

### Key Modules

| File | Purpose |
|------|---------|
| `offline_feature_parallel_q.py` | Main orchestrator, spawns all threads |
| `class_OD_offline_parallel_q.py` | TensorRT inference engine, frame processing |
| `GUI/GUI_q.py` | PyQt5 application, user interface |
| `GUI/Gl_Monitor.py` | OpenGL widget with CUDA interop |
| `features/bytetrack.py` | Multi-object tracking |
| `features/dribble_feature.py` | Dribble analysis and scoring |
| `features/shooting_feature.py` | Shot analysis and scoring |
| `compute_AI_advice.py` | AI coaching recommendations |

### Camera Configuration

Dual-camera setup at 1200x1920 resolution:
- Left camera: Shooting analysis (crop around left basket)
- Right camera: Shooting analysis (crop around right basket)
- Center region: Dribble analysis

Crop regions and basket positions defined in `class_OD_offline_parallel_q.py`:
```python
crop_shoot_left = [left_basekt_x-320, 0, left_basekt_x+320, 640]
crop_shoot_right = [right_basket_x-320, 0, right_basket_x+320, 640]
crop_dribble = [600, 40, 1320, 1100]
```

### BEV (Bird's-Eye-View) System

Homography-based 2D→BEV projection using calibration files:
- `H_final_left.npy`, `H_final_right.npy`: Homography matrices
- `corners_left.npy`, `corners_right.npy`: Calibration points

Manual calibration tools: `manual_bev.py`, `manual_bev_left.py`, `manual_bev_right.py`

## Configuration

**Inference config** (in `offline_feature_parallel_q.py`):
```python
cfg = {
    "engine_file_path_shooting": "last_640_yolo11n_100e_map_86p8_b4_fp16_nms.engine",
    "engine_file_path_dribble": "best_dribble_640_yolo11n_500e_map_98p3_b4_fp16_nms.engine",
    "class_names": ["ball", "bib", "player"],
    "score_threshold": 0.2,
    "batch_size": 4,
    "input_size": (640, 640),
    "framerate": 60
}
```

**Language flag** (in `calibrations.py`): `f_pinyin` toggles Chinese/English UI

**Environment** (`.env`): Display, audio, and runtime paths for Docker

## Key Dependencies

- PyQt5 (GUI framework)
- OpenCV, NumPy (image processing)
- TensorRT, PyCUDA (GPU inference)
- Jetson Utils (NVIDIA Jetson utilities)
- Pygame (audio feedback)
- SciPy (interpolation for trajectories)
