import cv2
import time
import threading
from pathlib import Path
from datetime import datetime
from calibrations import f_pinyin

"""
Dual eCAM25 capture -> separate videos per camera

- Captures each CSI/MIPI camera via a leaky GStreamer appsink (low latency)
- Always uses the latest frame (drops if writer lags)
- Writes one video per camera

Encoders (software only)
------------------------
1) OpenCV writer:
   - CODEC="MJPG" (fast lossy JPEG, .avi)
   - CODEC="FFV1" (lossless, very large, .avi)
   - CODEC="YUY2" (raw, enormous, .avi)

2) GStreamer x264enc (CPU):
   - BACKEND="GSTREAMER_X264"
   - Uses x264enc in ultrafast, all-intra mode for speed
   - Saves as .mkv via matroskamux

Notes
-----
* Jetson Orin Nano has no hardware encoder. All encoding is CPU.
* For highest FPS, use BACKEND="GSTREAMER_X264" with ultrafast preset.
"""

# ===================== User Settings =====================
DEVICE_LEFT = "/dev/video0"
DEVICE_RIGHT = "/dev/video1"
WIDTH, HEIGHT = 1920, 1200     # per-camera capture resolution

# Output video paths (timestamped)
# out_root = Path.home() / "Downloads"
out_root = Path.home() / "Videos"
out_root.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# # Target container FPS (metadata). Writer will push frames as fast as possible.
# WRITER_FPS = 30

# Pick the writer backend: "OPENCV" | "GSTREAMER_X264"
BACKEND = "GSTREAMER_X264"

# x264 encoder settings (when BACKEND=="GSTREAMER_X264")
X264_PRESET = "ultrafast"   # ultrafast, superfast, veryfast, etc.
X264_TUNE = "zerolatency"
X264_BITRATE_KBPS = 80000   # target bitrate in kbps (e.g., 40000 = 40 Mbps)

# Show preview window (False for maximum speed)
DISPLAY = False

# =========================================================

# Shared latest frames + timestamps
class Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.t_wall = 0.0  # wall-clock seconds

latest_left = Latest()
latest_right = Latest()
latest_both = Latest()
stop_event = threading.Event()


def gst_pipeline(device: str, w: int, h: int) -> str:
    return (
        f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
        f"video/x-raw, format=UYVY, width={w}, height={h}, framerate=\\(fraction\\)0/1 ! "
        f"nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"queue max-size-buffers=1 leaky=downstream ! "
        f"appsink emit-signals=false max-buffers=1 drop=true sync=false"
    )



def capture_worker(name: str, device: str, bucket: Latest):
    cap = cv2.VideoCapture(gst_pipeline(device, WIDTH, HEIGHT), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"[{name}] ERROR: cannot open {device}")
        return

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    frames = 0
    t0 = time.time()

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.001)
            continue
        t_wall = time.time()
        with bucket.lock:
            bucket.frame = frame
            bucket.t_wall = t_wall

        frames += 1
        if frames % 200 == 0:
            now = time.time()
            fps = 200.0 / (now - t0)
            t0 = now
            print(f"[{name}] capture ~{fps:.2f} fps")

    cap.release()


def _build_gst_x264_pipeline(path: str, w: int, h: int, fps: int) -> str:
    # Build a GStreamer pipeline for x264enc (CPU)
    caps = f"video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1"
    enc = (
        f"videoconvert ! x264enc speed-preset={X264_PRESET} tune={X264_TUNE} "
        f"bitrate={X264_BITRATE_KBPS} key-int-max=1 byte-stream=false ! "
        f"h264parse ! matroskamux ! filesink location={path} sync=false"
    )
    return f"appsrc format=time is-live=true block=true caps=\"{caps}\" ! {enc}"


def _create_video_writer(path: str, w: int, h: int, fps: int):
    """Try multiple backends to create a working VideoWriter."""
    # Try 1: GStreamer with x264
    gst = _build_gst_x264_pipeline(path, w, h, fps)
    wr = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, fps, (w, h))
    if wr.isOpened():
        return wr, path

    # Try 2: FFmpeg backend with H.264 (mp4v)
    mp4_path = path.replace('.mkv', '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    wr = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
    if wr.isOpened():
        return wr, mp4_path

    # Try 3: MJPG fallback (larger files but very compatible)
    avi_path = path.replace('.mkv', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    wr = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
    if wr.isOpened():
        return wr, avi_path

    return None, path

# top-level (reuse existing Latest buckets)
def push_frame(side: str, frame_np):
    if side == 'left':
        bucket = latest_left
    elif side == 'right':
        bucket = latest_right
    else:
        bucket = latest_both

    with bucket.lock:
        bucket.frame = frame_np      # must be BGR uint8 (HxWx3)
        bucket.t_wall = time.time()


# new globals
rec_left_evt  = threading.Event()
rec_right_evt = threading.Event()
rec_both_evt = threading.Event()

# Mode prefix for video filenames (shared between threads)
_mode_prefix_lock = threading.Lock()
_mode_prefix = {"left": "", "right": "", "both": ""}

def set_mode_prefix(side: str, prefix: str):
    """Set the mode prefix for video filenames."""
    with _mode_prefix_lock:
        _mode_prefix[side] = prefix

def get_mode_prefix(side: str) -> str:
    """Get the mode prefix for video filenames."""
    with _mode_prefix_lock:
        return _mode_prefix.get(side, "")

def _translate_side(side: str) -> str:
    """Translate side name to Chinese if f_pinyin is True."""
    if f_pinyin:
        return {"left": "左", "right": "右", "both": "双"}.get(side, side)
    return side

def recording_writer(name, bucket, rec_evt, WRITER_FPS):
    # wait for first frame to get w,h
    while not stop_event.is_set():
        with bucket.lock:
            frame = bucket.frame
        if frame is not None:
            h, w = frame.shape[:2]
            break
        time.sleep(0.005)

    while not stop_event.is_set():
        # idle until recording toggled on
        if not rec_evt.is_set():
            time.sleep(0.05)
            continue

        # open a fresh file with mode prefix and translated side name
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = get_mode_prefix(name)
        side_name = _translate_side(name)
        if prefix:
            out_path = str(out_root / f"{prefix}_{side_name}_{ts}.mkv")
        else:
            out_path = str(out_root / f"{side_name}_{ts}.mkv")
        wr, actual_path = _create_video_writer(out_path, w, h, WRITER_FPS)

        if wr is None or not wr.isOpened():
            print(f"[writer-{name}] ERROR: all video backends failed")
            time.sleep(0.5)
            continue

        out_path = actual_path  # update to actual file path used

        # write until event cleared
        while rec_evt.is_set() and not stop_event.is_set():
            with bucket.lock:
                frame = bucket.frame
            if frame is not None:
                wr.write(frame)
            else:
                time.sleep(0.001)

        wr.release()

def start_recorders(gui2infer_queue):
    # writers only; they sleep until rec_*_evt is set
    tWL = threading.Thread(target=recording_writer, args=("left",  latest_left,  rec_left_evt, 30),  daemon=True)
    tWR = threading.Thread(target=recording_writer, args=("right", latest_right, rec_right_evt, 30), daemon=True)
    tWB = threading.Thread(target=recording_writer, args=("both", latest_both, rec_both_evt, 18), daemon=True)
    tWL.start(); tWR.start(); tWB.start()

    try:
        while not stop_event.is_set():
            try:
                msg = gui2infer_queue.get(timeout=0.1)
            except Exception:
                continue

            if msg.get("cmd") == "record":
                side, on = msg.get("side"), bool(msg.get("on"))
                mode = msg.get("mode", "")
                if side == 'left':
                    evt = rec_left_evt
                elif side == 'right':
                    evt = rec_right_evt
                else:
                    evt = rec_both_evt

                if on:
                    set_mode_prefix(side, mode)
                    evt.set()
                else:
                    evt.clear()

            elif msg.get("cmd") == "shutdown":
                stop_event.set()
    finally:
        tWL.join(); tWR.join(); tWB.join()


# if __name__ == "__main__":
#     start_recorders()
