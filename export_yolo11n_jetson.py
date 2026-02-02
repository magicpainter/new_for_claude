#!/usr/bin/env python3
"""
YOLO11n TensorRT Export for Jetson Orin Nano (2-Camera Setup)
==============================================================

Optimized for:
- Model: YOLO11n (fastest)
- Cameras: 2
- Batch size: 2 (1 frame per camera)
- Target: 25-40 FPS per camera

Usage on Jetson:
    # Recommended: FP16, 640px (best accuracy/speed balance)
    python export_yolo11n_jetson.py --model best.pt --preset balanced

    # Faster: FP16, 480px
    python export_yolo11n_jetson.py --model best.pt --preset fast

    # Fastest: INT8, 480px (requires calibration data)
    python export_yolo11n_jetson.py --model best.pt --preset fastest
"""

import argparse
import os
import sys
import time


def export_yolo11n(args):
    """Export YOLO11n to TensorRT for 2-camera Jetson setup."""
    from ultralytics import YOLO

    print("=" * 60)
    print("YOLO11n EXPORT FOR JETSON ORIN NANO (2-Camera)")
    print("=" * 60)

    # Preset configurations
    presets = {
        "balanced": {
            "precision": "fp16",
            "imgsz": 640,
            "batch": 4,
            "desc": "Best accuracy/speed balance",
            "expected_latency": "35-45ms",
            "expected_fps": "22-28 FPS/camera"
        },
        "fast": {
            "precision": "fp16",
            "imgsz": 480,
            "batch": 2,
            "desc": "Faster with slightly lower accuracy",
            "expected_latency": "25-35ms",
            "expected_fps": "28-40 FPS/camera"
        },
        "fastest": {
            "precision": "int8",
            "imgsz": 480,
            "batch": 2,
            "desc": "Maximum speed (INT8 quantization)",
            "expected_latency": "18-25ms",
            "expected_fps": "40-55 FPS/camera"
        }
    }

    # Apply preset or use manual settings
    if args.preset:
        config = presets[args.preset]
        precision = config["precision"]
        imgsz = config["imgsz"]
        batch = config["batch"]
        print(f"\nPreset: {args.preset}")
        print(f"  {config['desc']}")
        print(f"  Expected batch latency: {config['expected_latency']}")
        print(f"  Expected FPS: {config['expected_fps']}")
    else:
        precision = args.precision
        imgsz = args.imgsz
        batch = args.batch

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Precision: {precision.upper()}")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Batch size: {batch}")
    print("-" * 60)

    # Load model
    model = YOLO(args.model)

    # Export settings
    half = precision.lower() == "fp16"
    int8 = precision.lower() == "int8"

    print(f"\nExporting to TensorRT...")
    start_time = time.time()

    try:
        engine_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            int8=int8,
            dynamic=False,      # Static for best performance
            simplify=True,
            workspace=4,        # 4GB workspace
            batch=batch,
            device=0,
        )

        export_time = time.time() - start_time
        print(f"\nExport completed in {export_time/60:.1f} minutes")
        print(f"Engine saved to: {engine_path}")

        # File sizes
        pt_size = os.path.getsize(args.model) / 1e6
        engine_size = os.path.getsize(engine_path) / 1e6
        print(f"\nModel sizes:")
        print(f"  PyTorch: {pt_size:.1f} MB")
        print(f"  TensorRT: {engine_size:.1f} MB")

        return engine_path

    except Exception as e:
        print(f"\nExport failed: {e}")
        print("\nIf running on desktop, export to ONNX first:")
        print(f"  python export_yolo11n_jetson.py --model {args.model} --onnx-only")
        return None


def export_onnx_only(args):
    """Export to ONNX for transfer to Jetson."""
    from ultralytics import YOLO

    print("=" * 60)
    print("YOLO11n ONNX EXPORT (for transfer to Jetson)")
    print("=" * 60)

    model = YOLO(args.model)

    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=17,
        dynamic=False,
        half=False,
    )

    print(f"\nONNX exported to: {onnx_path}")
    print(f"\nNext steps:")
    print(f"  1. Copy {onnx_path} to Jetson")
    print(f"  2. On Jetson, run:")
    print(f"     python export_yolo11n_jetson.py --model {os.path.basename(onnx_path)} --preset balanced")

    return onnx_path


def benchmark(engine_path, imgsz, batch):
    """Quick benchmark of exported model."""
    from ultralytics import YOLO
    import numpy as np
    import torch

    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)

    model = YOLO(engine_path)

    # Create batch of dummy inputs (simulating 2 cameras)
    dummy_batch = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(batch)]

    # Warmup
    print("Warming up...")
    for _ in range(20):
        model(dummy_batch, verbose=False)

    # Benchmark
    print("Benchmarking (100 iterations)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(100):
        start = time.perf_counter()
        model(dummy_batch, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    batch_latency = np.mean(times)
    per_cam_fps = 1000 / batch_latency  # Since batch=2 for 2 cams, each cam gets 1 frame per batch

    print(f"\nResults (batch={batch}, 2 cameras):")
    print(f"  Batch latency:     {batch_latency:.1f} ms (±{np.std(times):.1f})")
    print(f"  Per-camera FPS:    {per_cam_fps:.1f}")
    print(f"  Total throughput:  {(batch * 1000 / batch_latency):.1f} FPS")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11n TensorRT Export for Jetson (2-Camera Setup)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets for 2-camera setup on Jetson Orin Nano:

  balanced  FP16, 640px, batch=2  →  22-28 FPS/camera (recommended)
  fast      FP16, 480px, batch=2  →  28-40 FPS/camera
  fastest   INT8, 480px, batch=2  →  40-55 FPS/camera

Examples:
  # On Jetson - use preset
  python export_yolo11n_jetson.py --model best.pt --preset balanced

  # On desktop - export ONNX first
  python export_yolo11n_jetson.py --model best.pt --onnx-only
        """
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Path to model (.pt or .onnx)")
    parser.add_argument("--preset", type=str, choices=["balanced", "fast", "fastest"],
                        help="Optimization preset for 2-camera setup")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp32", "fp16", "int8"],
                        help="TensorRT precision (default: fp16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=2,
                        help="Batch size (default: 2 for 2-camera setup)")
    parser.add_argument("--onnx-only", action="store_true",
                        help="Export to ONNX only (for desktop)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark after export")

    args = parser.parse_args()

    if args.onnx_only:
        export_onnx_only(args)
    else:
        engine_path = export_yolo11n(args)

        if engine_path and args.benchmark:
            imgsz = args.imgsz
            batch = args.batch
            if args.preset:
                presets = {"balanced": (640, 2), "fast": (480, 2), "fastest": (480, 2)}
                imgsz, batch = presets[args.preset]
            benchmark(engine_path, imgsz, batch)


if __name__ == "__main__":
    main()
