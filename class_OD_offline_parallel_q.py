import threading
from pathlib import Path
from pycuda import gl as cudagl
import jetson_utils as jutils
from features.dribble_feature import DribbleFeature
import torch
import tensorrt as trt
from GUI.Gl_Monitor import MAP_LOCK
from features.shooting_feature import ShootingFeature
import os, ctypes, queue, time, traceback
import numpy as np, pycuda.driver as cuda
from PyQt5 import QtGui
from jetson_utils import cudaFont


# ============================================================================
# Inference Engine Configuration
# ============================================================================
class InferenceConfig:
    """Configuration constants for inference engine."""
    # Idle detection
    IDLE_CHECK_INTERVAL_SEC = 5.0       # How often to check for idle state

    # TensorRT defaults
    DEFAULT_DET_MAX = 300               # Default max detections
    DEFAULT_DET_DIM = 6                 # Default detection dimensions

    # Batch sizes
    SHOOTING_BATCH_SIZE = 4
    DRIBBLE_BATCH_SIZE = 2

    # Input dimensions
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640

    # Timing
    SHOOTING_DT = 0.067                 # ~15 FPS frame timing for shooting

    # Player detection
    PLAYER_CONFIDENCE_THRESHOLD = 0.3   # Confidence threshold for player counting
    BALL_CONFIDENCE_THRESHOLD = 0.3     # Confidence threshold for ball detection

    # Memory management
    CACHE_CLEANUP_INTERVAL = 500        # Batches between CUDA cache cleanup
    OD_FPS_REPORT_INTERVAL = 50         # Batches between OD FPS reports


# Load calibration files with error handling
def _load_calibration_file(filepath, description):
    """Load a numpy calibration file with error handling."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Calibration file not found: {filepath}. "
                                f"Please run camera calibration first.")
    try:
        return np.load(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to load {description}: {filepath}. Error: {e}")


corners_left = _load_calibration_file("corners_left.npy", "left camera corners")
corners_right = _load_calibration_file("corners_right.npy", "right camera corners")
left_basekt_x = int(corners_left[4][0])    # left basket X position
right_basket_x = int(corners_right[4][0])  # right basket X position
crop_shoot_left = [left_basekt_x-320, 0, left_basekt_x+320, 640]   # 640x640 crop
crop_shoot_right = [right_basket_x-320, 0, right_basket_x+320, 640]
crop_dribble = [600, 40, 1320, 1100]  # dribble analysis region

FP32, I32 = 4, 4  # bytes per float32 and int32

def _letterbox_chw(x_chw, new_hw, pad_val=114/255.0):
    # x_chw: (3,H,W) float32 in [0,1]
    import torch, torch.nn.functional as F
    _, H, W = x_chw.shape
    dh, dw = new_hw
    r = min(dh / H, dw / W)
    nh, nw = int(round(H * r)), int(round(W * r))
    pad_h, pad_w = dh - nh, dw - nw
    top, left = pad_h // 2, pad_w // 2
    x = F.interpolate(x_chw.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False).squeeze(0)
    x = F.pad(x.unsqueeze(0), (left, pad_w - left, top, pad_h - top), value=pad_val).squeeze(0)
    meta = dict(r=r, pad=(left, top), in_hw=(H, W), out_hw=(dh, dw))
    return x, meta

def _scale_back_xyxy_np(xyxy, meta):
    # xyxy: (N,4) in letterbox space (dh,dw)
    left, top = meta["pad"]
    r = meta["r"]
    H, W = meta["in_hw"]
    out = xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / r
    out[:, 0] = np.clip(out[:, 0], 0, W - 1)
    out[:, 2] = np.clip(out[:, 2], 0, W - 1)
    out[:, 1] = np.clip(out[:, 1], 0, H - 1)
    out[:, 3] = np.clip(out[:, 3], 0, H - 1)
    return out

class InferenceEngine(threading.Thread):
    """TensorRT inference engine with 3 bindings.

    Bindings:
      0: INPUT  FP32  (N,3,640,640) - RGB normalized to [0,1], CHW format
      1: OUTPUT FP32  (N,det,6)     - xyxy + score + class
      2: OUTPUT INT32 (N,det)       - class ids
    """

    def __init__(self, cfg, lr_q, oq, done_q, error_q, cmd_queue, tex_queue, run_event,res_queue, ctx_q,
                 ctx_ready, gui2infer_queue, pbo_pool, infer_frame_queue):
        super().__init__()
        self.cfg = cfg
        self.ctx_q = ctx_q
        self.ctx_ready = ctx_ready
        self.pbo_pool = pbo_pool
        self.infer_frame_queue = infer_frame_queue
        self.overlay_left = None
        self.overlay_right = None

        self.lr_q = lr_q
        self.oq = oq
        self.gui2infer_queue = gui2infer_queue
        self.view_mode = 'both'
        self.msg_feature = 'game'
        self.msg_style = 'none'


        self.done_q, self.error_q = done_q, error_q
        self.cmd_queue = cmd_queue
        self.tex_queue = tex_queue
        self.run_event = run_event
        self.winner_evt = threading.Event()
        self.res_queue = res_queue
        self.dev_id = int(cfg["gpu_id"])
        self.src_H, self.src_W = cfg["original_size"][0], cfg["original_size"][1]  # original size
        self.dst_H, self.dst_W = cfg["input_size"][0], cfg["input_size"][1]
        self.time_pre = time.time()
        self.font = cudaFont()
        self._rgba_bytes = self.src_W * self.src_H * 4

        self.feature_left = None
        self.feature_right = None

        # Idle detection for system resource management (dribble/shooting modes)
        # Note: Game mode idle detection is handled in GUI since game mode doesn't send frames to inference
        self.last_idle_check_time = time.time()
        self.idle_check_interval = InferenceConfig.IDLE_CHECK_INTERVAL_SEC
        self.idle_timeout_sent = False  # Prevent sending multiple idle messages per session


    # -------- Ping-pong buffer allocation for outputs -------- #
    @staticmethod
    def _new_pair(bytes_box, bytes_lab):
        """Allocate a host/device buffer pair for detection outputs."""
        host_box = cuda.pagelocked_empty(bytes_box // 4, dtype=np.float32)
        dev_box  = cuda.mem_alloc(host_box.nbytes)
        host_lab = cuda.pagelocked_empty(bytes_lab // 4, dtype=np.int32)
        dev_lab  = cuda.mem_alloc(host_lab.nbytes)
        return dict(
            host_box=host_box, dev_box=dev_box,
            host_lab=host_lab, dev_lab=dev_lab,
            bindings=[None, int(dev_box), int(dev_lab)],  # bindings[0] set later with input tensor
            meta=None, det_dim=None,
        )

    def create_engine(self, engine_file, BS, dst_W, dst_H):
        self.dst_W = dst_W
        self.dst_H = dst_H

        ctypes.CDLL(
            "libmmdeploy_tensorrt_ops.so",
            mode=ctypes.RTLD_GLOBAL)
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")

        with open(self.cfg[engine_file], "rb") as f:
            engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        context.set_binding_shape(0, (BS, 3, self.dst_H, self.dst_W))

        # -------- Input batch buffers (Torch Tensors) -------- #
        gpu_batches = [
            torch.empty((BS, 3, self.dst_H, self.dst_W),
                        device=f"cuda:{self.dev_id}", dtype=torch.float32),
            torch.empty((BS, 3, self.dst_H, self.dst_W),
                        device=f"cuda:{self.dev_id}", dtype=torch.float32)
        ]

        out_shape = context.get_binding_shape(1)  # e.g., (BS, 300, 6) for YOLOv11 + EfficientNMS
        if len(out_shape) >= 3 and out_shape[1] > 0 and out_shape[2] > 0:
            DET_MAX, DET_DIM = int(out_shape[1]), int(out_shape[2])
        else:
            DET_MAX = InferenceConfig.DEFAULT_DET_MAX
            DET_DIM = InferenceConfig.DEFAULT_DET_DIM
        bytes_box = BS * DET_MAX * DET_DIM * FP32
        bytes_lab = BS * DET_MAX * I32
        bufs = [self._new_pair(bytes_box, bytes_lab),
                self._new_pair(bytes_box, bytes_lab)]

        for b in bufs:
            nbind = engine.num_bindings
            b["bindings"] = [None] * nbind
            b["bindings"][1] = int(b["dev_box"])
            if nbind >= 3:
                b["bindings"][2] = int(b["dev_lab"])

        return engine, context, bufs, gpu_batches, DET_MAX

    def _check_and_handle_idle(self):
        """Periodically check if features are idle and notify GUI to return to home.
        Note: Game mode idle detection is handled in GUI since game mode doesn't send frames here.
        """
        # Don't send multiple idle messages for the same session
        if self.idle_timeout_sent:
            return

        now = time.time()
        if now - self.last_idle_check_time < self.idle_check_interval:
            return
        self.last_idle_check_time = now

        is_idle = False

        if self.msg_feature == 'dribble':
            # Check dribble feature idle (2 min no dribble)
            if self.feature_left is not None and self.feature_left.is_idle():
                is_idle = True
            elif self.feature_right is not None and self.feature_right.is_idle():
                is_idle = True

        elif self.msg_feature == 'shooting':
            # Check shooting feature idle (2 min no shooting)
            if self.feature_left is not None and self.feature_left.is_idle():
                is_idle = True
            elif self.feature_right is not None and self.feature_right.is_idle():
                is_idle = True

        # Game mode idle detection is handled in GUI (_check_idle_timeout)
        # since game mode doesn't send frames to inference

        if is_idle:
            # Send idle message to GUI - it will call _handle_end()
            try:
                self.res_queue.put_nowait({"idle_timeout": True, "feature": self.msg_feature})
                self.idle_timeout_sent = True  # Only send once per session
                print(f"[InferenceEngine] Idle timeout detected for {self.msg_feature}, notifying GUI")
            except queue.Full:
                pass  # Queue full, will retry next check

    # -------- Thread Entry Point -------- #
    def run(self):
        self.run_event.wait()
        self.ctx_ready.wait()  # Block until GUI completes ctx_ready.set()

        # Step 1: Get the shared OpenGL context
        share_ctx = QtGui.QOpenGLContext.globalShareContext()
        if share_ctx is None:
            raise RuntimeError("globalShareContext() is None - "
                               "ensure AA_ShareOpenGLContexts is called before QApplication")

        # Step 2: Create an offscreen context for this thread
        off_ctx = QtGui.QOpenGLContext()
        off_ctx.setShareContext(share_ctx)
        off_ctx.setFormat(share_ctx.format())  # Use same pixel format
        off_ctx.create()

        # Step 3: Create a 1x1 pbuffer surface (size doesn't matter for compute)
        off_surf = QtGui.QOffscreenSurface()
        off_surf.setFormat(off_ctx.format())
        off_surf.create()

        self.off_ctx = off_ctx
        self.off_surf = off_surf

        ctx = self.ctx_q.get()
        cuda.init()  # Ensure CUDA driver is initialized

        self.off_ctx.makeCurrent(self.off_surf)
        ctx.push()

        try:
            os.sched_setaffinity(0, [3])
            torch.cuda.set_device(self.dev_id)

            # Create default engine (shooting mode)
            cfg = InferenceConfig
            engine_file = "engine_file_path_shooting"
            BS = cfg.SHOOTING_BATCH_SIZE
            dst_W, dst_H = cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT
            engine, context, bufs, gpu_batches, DET_MAX = self.create_engine(engine_file, BS, dst_W, dst_H)
            cur_bs = BS

            s_prep  = cuda.Stream()
            torch_prep = torch.cuda.ExternalStream(int(s_prep.handle))
            s_exec  = cuda.Stream()
            s_exec_ptr = int(s_exec.handle)
            s_copy  = cuda.Stream()
            self.s_copy = s_copy

            evt_prep_done = cuda.Event()
            evt_exec_end  = cuda.Event()
            evt_copy_done = [cuda.Event() for _ in range(2)]

            batch_id = 0

            # OD FPS tracking
            od_fps_batch_count = 0
            od_fps_time_start = time.time()

            rgba_buf_left = jutils.cudaImage(self.src_W, self.src_H, format='rgba8')
            rgba_buf_right = jutils.cudaImage(self.src_W, self.src_H, format='rgba8')

            self.overlay_left = jutils.cudaImage(self.src_W, self.src_H, format='rgba8')
            self.overlay_right = jutils.cudaImage(self.src_W, self.src_H, format='rgba8')

            rgb_buf_left = jutils.cudaImage(self.src_W,self.src_H, format='rgb8')
            rgb_buf_right = jutils.cudaImage(self.src_W, self.src_H, format='rgb8')

            pending_frames = {'left': None, 'right': None}

            # ----------------------- Main Loop ----------------------- #

            while True:
                meta = []
                gpu_in = gpu_batches[batch_id & 1]
                buf = bufs[batch_id & 1]

                # Check the gui2infer queue to see if any feature is requested
                try:
                    msg = self.gui2infer_queue.get_nowait()
                    self.view_mode = msg['view_value']
                    self.msg_feature = msg['feature']
                    self.msg_style = msg['style']

                    # Clean up CUDA memory before switching features to prevent OOM
                    torch.cuda.empty_cache()

                    # Create engine for dribble or shooting mode
                    if self.msg_feature == 'dribble':
                        engine_file = "engine_file_path_dribble"
                        BS = cfg.DRIBBLE_BATCH_SIZE
                        dst_W, dst_H = cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT
                        engine, context, bufs, gpu_batches, DET_MAX = self.create_engine(engine_file, BS, dst_W, dst_H)
                        cur_bs = BS
                        batch_id = 0
                        gpu_in = gpu_batches[batch_id & 1]
                        buf = bufs[batch_id & 1]
                    else:
                        engine_file = "engine_file_path_shooting"
                        BS = cfg.SHOOTING_BATCH_SIZE
                        dst_W, dst_H = cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT
                        engine, context, bufs, gpu_batches, DET_MAX = self.create_engine(engine_file, BS, dst_W, dst_H)
                        cur_bs = BS
                        batch_id = 0
                        gpu_in = gpu_batches[batch_id & 1]
                        buf = bufs[batch_id & 1]

                    if self.msg_feature == 'dribble':
                        if self.view_mode == 'left':
                            self.feature_left = DribbleFeature(self.winner_evt, self.src_W,
                                                               self.src_H,msg["style"],msg["count"])
                            self.feature_right = None  # Clear to avoid stale idle check
                        elif self.view_mode == 'right':
                            self.feature_right = DribbleFeature(self.winner_evt,self.src_W,
                                                                self.src_H,msg["style"],msg["count"])
                            self.feature_left = None  # Clear to avoid stale idle check
                        elif self.view_mode == 'both':
                            self.feature_left = DribbleFeature(self.winner_evt, self.src_W,
                                                               self.src_H, msg["style"], msg["count"])
                            self.feature_right = DribbleFeature(self.winner_evt, self.src_W,
                                                                self.src_H, msg["style"], msg["count"])

                        self.winner_evt.clear()
                        # Reset idle timer for new session
                        self.last_idle_check_time = time.time()
                        self.idle_timeout_sent = False

                    elif self.msg_feature == 'shooting':
                        dt = cfg.SHOOTING_DT  # Frame timing for shooting analysis
                        if self.view_mode == 'left':
                            self.feature_left = ShootingFeature(self.src_W, self.src_H, dt, self.view_mode, msg["style"])
                            self.feature_right = None  # Clear to avoid stale idle check

                        elif self.view_mode == 'right':
                            self.feature_right = ShootingFeature(self.src_W, self.src_H, dt, self.view_mode, msg["style"])
                            self.feature_left = None  # Clear to avoid stale idle check

                        elif self.view_mode == 'both':
                            self.feature_left = ShootingFeature(self.src_W, self.src_H, dt, 'left', msg["style"])
                            self.feature_right = ShootingFeature(self.src_W, self.src_H, dt, 'right', msg["style"])

                        self.winner_evt.clear()

                        # Attach AI bridge for coaching feedback (set by GUI)
                        bridge = getattr(self, "ai_bridge", None)
                        if bridge is not None:
                            if self.feature_left is not None:
                                self.feature_left.ai_bridge = bridge
                            if self.feature_right is not None:
                                self.feature_right.ai_bridge = bridge

                        # Reset idle timer for new session
                        self.last_idle_check_time = time.time()
                        self.idle_timeout_sent = False

                    # Game mode idle detection is handled in GUI since game mode
                    # doesn't send frames to inference engine

                except queue.Empty:
                    pass  # No command pending, continue current operation

                try:
                    frame_data = self.infer_frame_queue.get(timeout=0.1)
                    side = frame_data['side']

                    # Release older pending frame (if any) for this side
                    old = pending_frames.get(side)
                    if old is not None:
                        try:
                            self.pbo_pool.release(old['pbo_id'])
                        except Exception:
                            pass

                    pending_frames[side] = frame_data
                    need_left  = (self.view_mode in ("left", "both"))
                    need_right = (self.view_mode in ("right", "both"))

                    if (need_left and pending_frames["left"] is None) or \
                       (need_right and pending_frames["right"] is None):
                        continue

                    left_frame = pending_frames["left"] if need_left else None
                    right_frame = pending_frames["right"] if need_right else None

                    left_pbo_id = left_frame["pbo_id"] if left_frame else None
                    right_pbo_id = right_frame["pbo_id"] if right_frame else None

                    self.off_ctx.makeCurrent(self.off_surf)
                    ctx.push()
                    left_released = right_released = False
                    try:
                        if left_pbo_id is not None:
                            left_res = self.pbo_pool.regbufs[left_pbo_id]
                            with MAP_LOCK:
                                m = left_res.map(s_copy)
                                left_dev_ptr, _ = m.device_ptr_and_size()
                                cuda.memcpy_dtod_async(int(rgba_buf_left.ptr), left_dev_ptr,
                                                       self.src_W * self.src_H * 4, s_copy)
                                s_copy.synchronize()
                                m.unmap(s_copy)
                            self.pbo_pool.release(left_pbo_id)
                            left_released = True

                        if right_pbo_id is not None:
                            right_res = self.pbo_pool.regbufs[right_pbo_id]
                            with MAP_LOCK:
                                m = right_res.map(s_copy)
                                right_dev_ptr, _ = m.device_ptr_and_size()
                                cuda.memcpy_dtod_async(int(rgba_buf_right.ptr), right_dev_ptr,
                                                       self.src_W * self.src_H * 4, s_copy)
                                s_copy.synchronize()
                                m.unmap(s_copy)
                            self.pbo_pool.release(right_pbo_id)
                            right_released = True

                    finally:
                        if not left_released:
                            try:
                                self.pbo_pool.release(left_pbo_id)
                            except Exception:
                                pass
                        if not right_released:
                            try:
                                self.pbo_pool.release(right_pbo_id)
                            except Exception:
                                pass

                    if need_left:
                        pending_frames["left"] = None
                    if need_right:
                        pending_frames["right"] = None

                    ctx.pop()
                    self.off_ctx.doneCurrent()

                    if need_left:
                        jutils.cudaConvertColor(rgba_buf_left, rgb_buf_left)
                    if need_right:
                        jutils.cudaConvertColor(rgba_buf_right, rgb_buf_right)

                    cuda_img0 = rgb_buf_left if need_left else None
                    cuda_img1 = rgb_buf_right if need_right else None


                except queue.Empty:
                    continue
                except Exception as e:
                    if 'frame_data' in locals() and 'pbo_id' in frame_data:
                        self.pbo_pool.release(frame_data['pbo_id'])
                    raise

                t0 = time.time()

                with torch.cuda.stream(torch_prep):
                    if self.msg_feature =='shooting':
                        if self.view_mode == 'left':  # this is to only inference the needed side.
                            batch_order = [
                                ("left", "full"),  # index 0
                                ("left", "crop")]  # index 1
                        elif self.view_mode == 'right':
                            batch_order = [
                                ("right", "full"),  # index 2
                                ("right", "crop")]
                        elif self.view_mode == 'both':
                            batch_order = [
                                ("left", "full"),  # index 0
                                ("left", "crop"),  # index 1
                                ("right", "full"),  # index 2
                                ("right", "crop")  # index 3
                            ]

                    else:  # dribble
                        if self.view_mode == "left":
                            batch_order = [("left", "full")]
                        elif self.view_mode == "right":
                            batch_order = [("right", "full")]
                        else:
                            batch_order = [("left", "full"), ("right", "full")]


                    # Map side->cuda image
                    side_to_cuda = {}
                    if need_left:  side_to_cuda["left"] = cuda_img0
                    if need_right: side_to_cuda["right"] = cuda_img1

                    # Centers

                    for bidx, (side, view) in enumerate(batch_order):
                        cuda_img = side_to_cuda[side]
                        # Get numpy view
                        np_img = jutils.cudaToNumpy(cuda_img) if isinstance(cuda_img,
                                                                            jutils.cudaImage) else cuda_img  # HxWxC uint8
                        H, W = np_img.shape[:2]

                        if view == "full":
                            # Full frame
                            src_img = np_img
                            crop_ofs = (0, 0)  # no offset
                        else:
                            # Crop around tuned center
                            if self.msg_feature =='shooting':
                                if side == 'left':
                                    x1, y1, x2, y2 = crop_shoot_left
                                else:
                                    x1, y1, x2, y2 = crop_shoot_right
                            else:  # dribble
                                x1, y1, x2, y2 = crop_dribble

                            src_img = np_img[y1:y2, x1:x2, :]
                            crop_ofs = (x1, y1)

                        # To torch CHW, [0,1]
                        t = torch.from_numpy(src_img).to(f'cuda:{self.dev_id}', dtype=torch.float32, non_blocking=True)
                        t = (t / 255.0).permute(2, 0, 1).contiguous()
                        # Letterbox to model input (dst_H,dst_W)
                        t, lbox = _letterbox_chw(t, (self.dst_H, self.dst_W), pad_val=114 / 255.0)
                        # Fill the batch slot
                        gpu_in[bidx].copy_(t, non_blocking=True)
                        # Meta
                        meta.append({
                            "shm_index": 0,
                            "original_size": (W if view == "full" else src_img.shape[1],
                                              H if view == "full" else src_img.shape[0]),
                            "letterbox": lbox,
                            "path": f"{side}_{view}",
                            "side": side,
                            "view": view,  # NEW: 'full' | 'crop'
                            "crop_ofs": crop_ofs,  # (x1,y1) in ORIGINAL full-frame coords
                            "fname": None
                        })

                evt_prep_done.record(s_prep)
                s_exec.wait_for_event(evt_prep_done)

                # ---------- TensorRT Inference ---------- #
                buf["bindings"][0] = gpu_in.data_ptr()  # Bind input tensor directly
                context.execute_async_v2(buf["bindings"], s_exec_ptr)
                evt_exec_end.record(s_exec)
                s_copy.wait_for_event(evt_exec_end)
                det_cur, dim_cur = context.get_binding_shape(1)[1:]
                det_cur = DET_MAX if det_cur < 0 else det_cur
                buf["det_dim"] = (det_cur, dim_cur)

                need_box = cur_bs * det_cur * dim_cur * FP32
                need_label = cur_bs * det_cur * I32
                cuda.memcpy_dtoh_async(
                    buf["host_box"].view(np.uint8)[:need_box],
                    buf["dev_box"], s_copy)
                if engine.num_bindings >= 3:
                    cuda.memcpy_dtoh_async(
                        buf["host_lab"].view(np.uint8)[:need_label],
                        buf["dev_lab"], s_copy)
                    bi = batch_id & 1
                    evt_copy_done[bi].record(s_copy)
                else:
                    arr = buf["host_box"][:cur_bs * det_cur * dim_cur].reshape(cur_bs, det_cur, dim_cur)
                    buf["host_lab"][:cur_bs * det_cur] = arr[:, :, 5].astype(np.int32).ravel()

                # ---------- Process Previous Batch Output ---------- #
                if batch_id >= 1:
                    prev = bufs[(batch_id - 1) & 1]
                    evt_copy_done[(batch_id - 1) & 1].synchronize()
                    n_prev, (dn, dm) = len(prev["meta"]), prev["det_dim"]
                    boxes = prev["host_box"][:n_prev * dn * dm].reshape(n_prev, dn, dm)
                    labels = prev["host_lab"][:n_prev * dn].reshape(n_prev, dn)

                    # Accumulators per side
                    acc = {
                        "left": {"boxes": [], "scores": [], "labels": []},
                        "right": {"boxes": [], "scores": [], "labels": []},
                    }

                    for itm, b, l in zip(prev["meta"], boxes, labels):
                        side_i = itm["side"]  # 'left' / 'right'
                        if self.view_mode != 'both' and side_i != self.view_mode:
                            continue

                        view_i = itm.get("view", "full")
                        lb_meta = itm["letterbox"]

                        # 1) Map from letterbox-input to the *crop* (or full) image coords:
                        xyxy_in_image = _scale_back_xyxy_np(b[:, :4], lb_meta)
                        ## Additional steps: remove bbox from full if it is in the cropped area
                        if view_i == "full" and self.msg_feature != 'dribble':
                            if side_i == 'left':
                                x1, y1, x2, y2 = crop_shoot_left
                            else:
                                x1, y1, x2, y2 = crop_shoot_right
                            for index, xyxy in enumerate(xyxy_in_image):
                                if xyxy[0]> x1 and xyxy[1] > y1 and xyxy[2] < x2 and xyxy[3] < y2:
                                    b[index, 4] = 0  # set it 0 confidence to bypass it in tracker

                        # 2) If this is a crop, shift into ORIGINAL full-frame coords:
                        if view_i == "crop":
                            ox, oy = itm["crop_ofs"]  # (x1,y1) crop's top-left in full frame
                            xyxy_in_image[:, [0, 2]] += ox
                            xyxy_in_image[:, [1, 3]] += oy

                        # 3) Filter invalid / low score if needed (optional; keep as-is if plugin already NMS+thres)
                        scores_i = b[:, 4]
                        labels_i = l.astype(np.int32)

                        # 4) Accumulate per side
                        acc[side_i]["boxes"].append(xyxy_in_image)
                        acc[side_i]["scores"].append(scores_i)
                        acc[side_i]["labels"].append(labels_i)

                    # === Now finalize per side (merge full + crop) ===
                    for side_i in ("left", "right"):
                        if self.view_mode != 'both' and side_i != self.view_mode:
                            continue
                        if not acc[side_i]["boxes"]:
                            continue
                        Bx = np.concatenate(acc[side_i]["boxes"], axis=0)
                        Sc = np.concatenate(acc[side_i]["scores"], axis=0)
                        Lb = np.concatenate(acc[side_i]["labels"], axis=0)

                        # Build the 'cur' payload ONCE per side, with merged detections
                        cur = {
                            "cfg": self.cfg,
                            "bboxes": Bx,
                            "scores": Sc,
                            "labels": Lb,
                            "original_size": (self.src_W, self.src_H),  # full frame
                            "path": f"{side_i}_merged",
                            "side": side_i,
                            "fname": None,
                            "shm_index": 0,
                            "cuda_img": self.overlay_left if side_i == "left" else self.overlay_right,
                            "display": None,
                            "category_map": {0: 'ball', 1: 'bib', 2: 'bob', 3: 'player'},
                            "cur_time": time.time() - t0,
                            "segment_time": 20,
                            "src_W": self.src_W,
                            "src_H": self.src_H,
                            "font": self.font,
                            "res_queue": self.res_queue,
                            "cuda_ctx": ctx,
                        }

                        # Clear only the overlay for the side we are about to redraw
                        with MAP_LOCK:
                            if side_i == "left":
                                cuda.memset_d8(
                                    int(self.overlay_left.ptr),
                                    0,
                                    self.overlay_left.width * self.overlay_left.height * self.overlay_left.channels,
                                )
                                overlay_img = self.overlay_left
                            else:
                                cuda.memset_d8(
                                    int(self.overlay_right.ptr),
                                    0,
                                    self.overlay_right.width * self.overlay_right.height * self.overlay_right.channels,
                                )
                                overlay_img = self.overlay_right



                            # ---- feature calls (once per side) ----
                            if self.msg_feature != 'game':
                                if side_i == 'left' and (self.view_mode in ('left', 'both')):
                                    self.feature_left.on_frame(cur)
                                    self.feature_left.check_activity()
                                if side_i == 'right' and (self.view_mode in ('right', 'both')):
                                    self.feature_right.on_frame(cur)
                                    self.feature_right.check_activity()

                                # Idle detection check (dribble/shooting only)
                                self._check_and_handle_idle()
                            else:
                                # Game mode: count players and detect ball for idle detection
                                # Process both sides in "both" mode, otherwise only the active side
                                should_process = (side_i == 'right' and self.view_mode in ('right', 'both')) or \
                                                 (side_i == 'left' and self.view_mode in ('left', 'both'))
                                if should_process:
                                    # Count players (label 2) above confidence threshold
                                    player_count = int(np.sum((Lb == 2) & (Sc > cfg.PLAYER_CONFIDENCE_THRESHOLD)))
                                    # Detect ball (label 0) above confidence threshold
                                    has_ball = bool(np.any((Lb == 0) & (Sc > cfg.BALL_CONFIDENCE_THRESHOLD)))
                                    try:
                                        self.res_queue.put_nowait({
                                            "game_player_count": player_count,
                                            "game_has_ball": has_ball,
                                            "game_side": side_i  # Include side for proper tracking in "both" mode
                                        })
                                    except queue.Full:
                                        pass

                            # cuda.Context.synchronize()  # Removed - was blocking GPU/CPU parallelism

                buf["meta"] = meta
                batch_id += 1
                od_fps_batch_count += 1

                # Calculate and send OD FPS periodically
                if od_fps_batch_count >= cfg.OD_FPS_REPORT_INTERVAL:
                    od_fps_elapsed = time.time() - od_fps_time_start
                    if od_fps_elapsed > 0:
                        od_fps = od_fps_batch_count / od_fps_elapsed
                        try:
                            self.res_queue.put_nowait({"od_fps": od_fps})
                        except queue.Full:
                            pass
                    od_fps_batch_count = 0
                    od_fps_time_start = time.time()

                # Periodic CUDA cache cleanup to prevent memory fragmentation
                if batch_id % cfg.CACHE_CLEANUP_INTERVAL == 0:
                    torch.cuda.empty_cache()


        except Exception as e:
            self.error_q.put(("InferenceEngine", traceback.format_exc()))
            traceback.print_exc()

        finally:
            try: s_exec.synchronize(); s_copy.synchronize(); s_prep.synchronize()
            except Exception: pass
            try: del context, engine
            except Exception: pass
            for b in bufs:
                for k in ("dev_box", "dev_lab"):
                    try: b[k].free()
                    except Exception: pass
            try:
                self.oq.close(); self.oq.cancel_join_thread()
            except Exception: pass

            # Release any pending frame PBOs
            for side in ('left', 'right'):
                pf = pending_frames.get(side)
                if pf:
                    try:
                        self.pbo_pool.release(pf['pbo_id'])
                    except Exception:
                        pass

            # Clean up CUDA memory to prevent OOM on next run
            try:
                torch.cuda.empty_cache()
                print("[InferenceEngine] CUDA cache cleared")
            except Exception:
                pass

            ctx.pop()
            self.off_ctx.doneCurrent()



