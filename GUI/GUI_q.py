import os
import sys
#remove Qt and let pyQT search Qt itself, to prevent conflict
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path
import random
from manual_bev import MANUAL_BEV
import cv2, time, numpy as np, queue, threading
import jetson_utils as jutils
import pycuda.driver as cuda
from GUI.Gl_Monitor import MAP_LOCK
from pycuda import gl as cudagl
from jetson_utils import saveImage, cudaToNumpy
from GUI.Gl_Monitor import GLMonitor
from OpenGL import GL
from video_recording_w_option import start_recorders, push_frame
import subprocess, pathlib
from read_from_folder import FolderFrameSource
from select_files import VideoPickerDialog, VideoPlayerDialog, WifiConnectionDialog, VolumeControlDialog
from upload_baidu import popup_queue_to_baidupcs_drop
from calibrations import f_pinyin
import calibrations  # For dynamic language switching
from ai_advice_popup import AiAdviceBridge, show_ai_advice_popup
from PyQt5.QtWidgets import QMessageBox
from disk_space_manager import (
    check_and_cleanup_disk, get_available_disk_space_gb, needs_cleanup,
    DEFAULT_MIN_FREE_SPACE_GB, DEFAULT_TARGET_FREE_SPACE_GB
)

# Constants
DRIBBLE_STYLES = ['Crossover challenge low', 'Left dribble challenge low', 'Right dribble challenge low',
                  'Crossover challenge high', 'Left dribble challenge high', 'Right dribble challenge high',
                  'Behind back', 'Cross leg', 'Left V', 'Right V']
SHOOT_MODES = ['mid-range baseline', '3-points baseline', 'mid-range star',
               '3-points star', 'close-range single-hand', 'free throw',
               '3-points challenge', 'layup']

# Shared control button settings
CONTROL_BTN_WIDTH = 300
CONTROL_BTN_HEIGHT = 80
CONTROL_BTN_FONT_SIZE = 48  # px

WINDOW_GRADIENT = '''
QMainWindow, QWidget {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #1e3c72, stop:1 #2a5298);
}
'''

BUTTON_STYLE = '''
QPushButton {
    color: white;
    font-size: 26px;
    padding: 12px 20px;
    border-radius: 8px;
    background: #ff6f00;
}
QPushButton:hover { background: #ffa040; }
QPushButton:pressed { background: #e65c00; }
'''

GROUPBOX_STYLE = '''
QGroupBox {
    color: white;
    font-size: 35px;
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 10px;
    padding: 15px;
}
'''

def cuda_to_bgr_np(cuda_img, width, height):
    # cuda_img is jetson.utils.cudaImage (RGBA, 4 channels)
    rgba = cudaToNumpy(cuda_img, width, height, 4)       # HxWx4, uint8
    # Ensure contiguous (opencv prefers C-contig)
    rgba = np.array(rgba, copy=False)                                 # usually already contiguous
    bgr  = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)                     # HxWx3
    return bgr


class CriticalDiskSpaceDialog(QtWidgets.QDialog):
    """Warning dialog for critically low disk space during active recording."""
    finished_early = QtCore.pyqtSignal()  # Emitted when user clicks "Finish Now"

    def __init__(self, available_gb: float, timeout_sec: int, parent=None):
        super().__init__(parent)
        self.timeout_sec = timeout_sec
        self.available_gb = available_gb

        self.setWindowTitle("磁盘空间不足" if calibrations.f_pinyin else "Low Disk Space Warning")
        self.setModal(False)  # Non-modal to allow continued use
        self.setMinimumSize(500, 250)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.setStyleSheet("""
            QDialog {
                background: #fff3cd;
                border: 3px solid #ffc107;
                border-radius: 10px;
            }
            QLabel { color: #856404; }
            QPushButton {
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton#finishBtn {
                background: #dc3545;
                color: white;
                border: none;
            }
            QPushButton#finishBtn:hover { background: #c82333; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Warning icon and title
        title_layout = QtWidgets.QHBoxLayout()
        warning_icon = QtWidgets.QLabel("⚠️")
        warning_icon.setStyleSheet("font-size: 48px;")
        title_layout.addWidget(warning_icon)

        title_text = "磁盘空间严重不足！" if calibrations.f_pinyin else "Critical Disk Space Warning!"
        title_label = QtWidgets.QLabel(title_text)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Info message
        if calibrations.f_pinyin:
            info_text = f"可用空间仅剩 {available_gb:.1f} GB，系统将在 {timeout_sec} 秒后\n自动返回主页进行磁盘清理优化。"
        else:
            info_text = f"Only {available_gb:.1f} GB available. System will automatically\nreturn to Home in {timeout_sec} seconds for disk cleanup."
        self.info_label = QtWidgets.QLabel(info_text)
        self.info_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.info_label)

        # Countdown label
        self.countdown_label = QtWidgets.QLabel()
        self.countdown_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #dc3545;")
        self.countdown_label.setAlignment(QtCore.Qt.AlignCenter)
        self.update_countdown(timeout_sec)
        layout.addWidget(self.countdown_label)

        # Finish Now button
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        finish_btn = QtWidgets.QPushButton("立即结束" if calibrations.f_pinyin else "Finish Now")
        finish_btn.setObjectName("finishBtn")
        finish_btn.clicked.connect(self._on_finish_now)
        btn_layout.addWidget(finish_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def update_countdown(self, remaining_sec: int):
        """Update the countdown display."""
        minutes = remaining_sec // 60
        seconds = remaining_sec % 60
        if calibrations.f_pinyin:
            self.countdown_label.setText(f"剩余时间: {minutes:02d}:{seconds:02d}")
        else:
            self.countdown_label.setText(f"Time remaining: {minutes:02d}:{seconds:02d}")

    def _on_finish_now(self):
        """User clicked Finish Now button."""
        self.finished_early.emit()
        self.close()


class TrainingAIApp(QtWidgets.QMainWindow, threading.Thread):
    def __init__(self,ctx_q, ctx_ready,cfg , cmd_queue=None, tex_queue=None, res_queue=None,
                 run_event=None,gui2infer_queue=None, pbo_pool=None, infer_frame_queue=None,
                 error_q=None, engine=None):
        super().__init__()
        self.time_pre = time.time()
        self.time_pre_check = self.time_pre
        self.cmd_queue = cmd_queue
        self.tex_queue = tex_queue
        self.res_queue = res_queue
        self.run_event = run_event
        self.gui2infer_queue = gui2infer_queue
        self.pbo_pool = pbo_pool
        self.infer_frame_queue = infer_frame_queue
        self.engine = engine
        # --- AI advice popup bridge (GUI thread) ---
        self.ai_bridge = AiAdviceBridge()

        # Ensure the slot runs on GUI thread even if emit happens from worker thread
        self.ai_bridge.show_advice.connect(self._on_ai_advice, QtCore.Qt.QueuedConnection)

        # Expose bridge to the backend/inference side (engine owns features)
        if self.engine is not None:
            self.engine.ai_bridge = self.ai_bridge


        self.pipeline_running = False
        self.cnt = 0
        self.src_W, self.src_H = cfg["original_size"][1], cfg["original_size"][0]
        self.framerate = cfg["framerate"]
        self.frame_counter = 0
        self.rgba_left = None
        self.rgba_right = None
        self._view_mode = "left"

        self.ctx_q = ctx_q
        self.ctx_ready = ctx_ready
        self.cuda_ctx = None
        self.copy_stream = None
        self.f_dribble = False
        self.f_shoot = False
        self.f_game = False
        self.f_left_cam = False
        self.f_right_cam = False
        self.f_both_cam = False
        self._started = False
        self._stoped_game= True
        self._rec_left_on = False
        self._rec_right_on = False
        self._rec_both_on = False
        # Game mode idle detection (5 minutes with no ball OR < 2 players)
        self._game_last_active_time = None  # Last time activity detected (ball + >= 2 players)
        self._game_idle_timeout_sec = 300  # 5 minutes
        # Track left and right sides separately for "both" mode
        self._game_left_player_count = 0
        self._game_left_has_ball = False
        self._game_right_player_count = 0
        self._game_right_has_ball = False
        self._game_infer_interval = 1800  # Send frame to inference every 1800 frames (~30s at 60fps)
        self._game_infer_counter = 0
        img_i = random.randint(1, 6)
        img_name = f"UI_Button/startup_{img_i}.png"
        self.bg_num = img_i
        self.start_img = cv2.imread(img_name)
        self.dribble_style = random.choice(DRIBBLE_STYLES)
        self.shoot_mode = 'free throw'  # random.choice(SHOOT_MODES)
        # Source selection
        #
        self.cap_left = jutils.videoSource(
            "/dev/video0",
            options={'width': self.src_W, 'height': self.src_H, 'framerate': self.framerate})
        self.cap_right = jutils.videoSource(
            "/dev/video1",
            options={'width': self.src_W, 'height': self.src_H, 'framerate': self.framerate})
        self.infer_interval = 4  # Send 1 frame per 4 display frames

        self.resize(2500, 1500)
        self.setMinimumSize(800, 600)
        self._apply_styles()
        self._init_ui()
        self._start_stream()
        # have a queue you already pass to recorder for commands
        self.recorder_queue = queue.Queue()
        threading.Thread(target=start_recorders, args=(self.recorder_queue,), daemon=True).start()

        self.installEventFilter(self)
        self.error_q = error_q
        self.timer.timeout.connect(self._check_errors)
        self.msg_box = None  # keep a reference
        self.has_displayed_desktop = False

        # AI advice popup references (for dual-player "both" mode)
        self._ai_popup = None
        self._ai_popup_left = None
        self._ai_popup_right = None

        # Disk space management - check every 60 seconds when idle
        self._disk_check_timer = QtCore.QTimer(self)
        self._disk_check_timer.setInterval(60000)  # 60 seconds
        self._disk_check_timer.timeout.connect(self._check_disk_space)
        self._disk_check_timer.start()
        self._disk_cleanup_in_progress = False

        # Critical disk space warning (< 10GB during active mode)
        self._critical_disk_space_gb = 10  # Trigger warning when < 10GB
        self._critical_warning_timeout_sec = 120  # 2 minutes to finish
        self._critical_warning_shown = False
        self._critical_warning_start_time = None
        self._critical_warning_dialog = None

        # FPS tracking for display
        self._display_fps = 0.0
        self._od_fps = 0.0

    def _on_ai_advice(self, result, acc, f_pinyin_flag, side='single'):
        """Handle AI advice popup with proper positioning for dual-player mode.

        Args:
            result: AI advice result dict
            acc: Accuracy percentage
            f_pinyin_flag: Use Chinese if True
            side: 'left', 'right', or 'single'. In 'both' mode, positions window accordingly.
        """
        popup = show_ai_advice_popup(self, result, acc, f_pinyin_flag, side=side)

        # Keep separate references for left/right to allow both popups simultaneously
        if side == 'left':
            self._ai_popup_left = popup
        elif side == 'right':
            self._ai_popup_right = popup
        else:
            self._ai_popup = popup

        popup.show()
        popup.raise_()
        popup.activateWindow()

    def _is_idle_at_home(self) -> bool:
        """Check if GUI is at home page and not executing any feature."""
        # Not running pipeline (dribble/shoot/game)
        if self.pipeline_running:
            return False
        # Not in started state (user hasn't pressed start)
        if self._started:
            return False
        # No feature is actively selected and running
        return True

    def _check_disk_space(self):
        """Check disk space and trigger cleanup if needed (only when idle at home)."""
        if not self._is_idle_at_home():
            return
        if self._disk_cleanup_in_progress:
            return

        video_dir = Path.home() / "Videos"

        # Check if available disk space is below threshold
        if needs_cleanup(video_dir, DEFAULT_MIN_FREE_SPACE_GB):
            available = get_available_disk_space_gb(video_dir)
            self._disk_cleanup_in_progress = True
            print(f"[DiskCheck] Available space {available:.1f} GB < {DEFAULT_MIN_FREE_SPACE_GB} GB, starting cleanup...")
            print("[DiskCheck] Prioritizing Dribble and Shooting videos for deletion")
            try:
                check_and_cleanup_disk(video_dir, DEFAULT_MIN_FREE_SPACE_GB, DEFAULT_TARGET_FREE_SPACE_GB, self)
            finally:
                self._disk_cleanup_in_progress = False

    def closeEvent(self, event):
        """Clean up camera resources before closing to prevent stale state."""
        print("[GUI] closeEvent: releasing camera resources...")

        # Stop the frame update timer
        if hasattr(self, 'timer') and self.timer is not None:
            self.timer.stop()

        # Close camera sources to release /dev/video* devices
        try:
            if hasattr(self, 'cap_left') and self.cap_left is not None:
                self.cap_left.Close()
                print("[GUI] cap_left closed")
        except Exception as e:
            print(f"[GUI] Error closing cap_left: {e}")

        try:
            if hasattr(self, 'cap_right') and self.cap_right is not None:
                self.cap_right.Close()
                print("[GUI] cap_right closed")
        except Exception as e:
            print(f"[GUI] Error closing cap_right: {e}")

        # Signal backend to stop if run_event exists
        if self.run_event is not None:
            self.run_event.clear()

        super().closeEvent(event)

    def _left_enabled(self) -> bool:
        return bool(self.f_left_cam or self.f_both_cam)

    def _right_enabled(self) -> bool:
        return bool(self.f_right_cam or self.f_both_cam)

    def _get_current_mode(self) -> str:
        """Return the current training mode for video filename prefix."""
        if self.f_dribble:
            return "运球" if calibrations.f_pinyin else "dribble"
        elif self.f_shoot:
            return "投篮" if calibrations.f_pinyin else "shooting"
        elif self.f_game:
            return "比赛" if calibrations.f_pinyin else "game"
        return ""

    def _check_errors(self):
        if not self.error_q:
            return
        while not self.error_q.empty():
            proc, msg = self.error_q.get_nowait()
            print(f"[GUI] {proc} error:\n{msg}")  # appears in your terminal too
            # Optionally also persist:
            from datetime import datetime
            with open("inference_errors.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {proc}\n{msg}\n")
            # Surface to the user:
            QtWidgets.QMessageBox.critical(self, "Inference Error", msg[:8000])

    def eventFilter(self, obj, ev):
        from PyQt5 import QtCore
        if ev.type() == QtCore.QEvent.Show and obj is self:
            QtCore.QTimer.singleShot(0, self._prime_gl_contexts)
            self.removeEventFilter(self)
        return super().eventFilter(obj, ev)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        # PyQt5: QtCore.Qt.Key_Escape
        # PyQt6: QtCore.Qt.Key.Key_Escape
        if e.key() == QtCore.Qt.Key_Escape:
            # example: exit fullscreen
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            # pass unhandled keys to base class
            super().keyPressEvent(e)

    def _prime_gl_contexts(self):
        """Ensure at least one GL context is initialized, then allocate PBOPool buffers under it."""
        # Choose monitors that should be active
        mons = []
        if getattr(self, "f_left_cam", False) or getattr(self, "f_both_cam", False):
            mons.append(self.left_monitor)
        if getattr(self, "f_right_cam", False) or getattr(self, "f_both_cam", False):
            mons.append(self.right_monitor)
        if not mons:
            mons = [self.left_monitor]  # fallback

        # Force GPU widgets to actually initialize GL
        for m in mons:
            m.setVisible(True)
            m.setCurrentIndex(1)              # GPU page
            m.gpu_view.setVisible(True)
            m.gpu_view.update()
            m.gpu_view.repaint()              # more aggressive than update()

        QtWidgets.QApplication.processEvents()

        # Allocate PBOPool buffers (must run with a current GL context)
        if self.pbo_pool is not None:
            gv = mons[0].gpu_view
            gv.makeCurrent()
            try:
                self.pbo_pool._ensure_buffers()
            finally:
                gv.doneCurrent()

        # Back to CPU page
        for m in mons:
            m.setCurrentIndex(0)


    def _apply_styles(self):
        self.setStyleSheet(WINDOW_GRADIENT + BUTTON_STYLE)

    def _init_ui(self):
        self.recording = False
        self.stack = QtWidgets.QStackedWidget(self)
        self.stack.addWidget(self._create_main_page())
        self.setCentralWidget(self.stack)

    def _start_stream(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_frames)
        self.timer.timeout.connect(self._check_commands)
        self.timer.timeout.connect(self._check_idle_timeout)
        self.timer.timeout.connect(self._check_critical_disk_space)
        #self.timer.timeout.connect(self._check_results)
        self.timer.start(16)  # time delay before next update

    def _check_idle_timeout(self):
        """Check for idle timeout messages from inference engine and game mode timeout."""
        # Check messages from inference engine
        if self.res_queue is not None:
            try:
                while True:
                    msg = self.res_queue.get_nowait()
                    if isinstance(msg, dict):
                        # Dribble/Shooting idle timeout
                        if msg.get("idle_timeout"):
                            feature = msg.get("feature", "unknown")
                            print(f"[GUI] Idle timeout detected for {feature}, returning to home")
                            if self._started:
                                self._on_start_clicked()  # This returns to home page with full UI update
                            else:
                                self._handle_end()
                            return
                        # Game mode player count and ball detection update
                        if msg.get("game_player_count") is not None:
                            player_count = msg["game_player_count"]
                            has_ball = msg.get("game_has_ball", False)
                            side = msg.get("game_side", "left")
                            # Track each side separately for "both" mode
                            if side == "left":
                                self._game_left_player_count = player_count
                                self._game_left_has_ball = has_ball
                            else:
                                self._game_right_player_count = player_count
                                self._game_right_has_ball = has_ball
                            # Activity detected if EITHER side has:
                            # - ball AND >= 2 players, OR
                            # - > 4 players (ball may be occluded by many players)
                            left_active = (self._game_left_has_ball and self._game_left_player_count >= 2) or self._game_left_player_count > 4
                            right_active = (self._game_right_has_ball and self._game_right_player_count >= 2) or self._game_right_player_count > 4
                            if left_active or right_active:
                                self._game_last_active_time = time.time()
                        # OD FPS update from inference engine
                        if msg.get("od_fps") is not None:
                            self._od_fps = msg["od_fps"]
            except:
                pass  # Queue empty or other error

        # Check game mode idle timeout (no ball OR < 2 players for 5 minutes)
        if self.f_game and self._game_last_active_time is not None:
            # Idle if NEITHER side is active (for "both" mode, either side active = not idle)
            # Active: (ball AND >= 2 players) OR (> 4 players, ball may be occluded)
            left_active = (self._game_left_has_ball and self._game_left_player_count >= 2) or self._game_left_player_count > 4
            right_active = (self._game_right_has_ball and self._game_right_player_count >= 2) or self._game_right_player_count > 4
            is_idle = not left_active and not right_active
            if is_idle:
                elapsed = time.time() - self._game_last_active_time
                if elapsed > self._game_idle_timeout_sec:
                    reason = "no activity on either side"
                    print(f"[GUI] Game mode idle: {reason} for {self._game_idle_timeout_sec}s, returning to home")
                    if self._started:
                        self._on_start_clicked()  # This returns to home page with full UI update
                    else:
                        self._handle_end()
                    return

    def _check_critical_disk_space(self):
        """Check for critically low disk space during active modes and warn user."""
        # Only check during active modes
        if not self.pipeline_running:
            # Reset warning state when not running
            if self._critical_warning_shown:
                self._critical_warning_shown = False
                self._critical_warning_start_time = None
                if self._critical_warning_dialog:
                    self._critical_warning_dialog.close()
                    self._critical_warning_dialog = None
            return

        # Check every ~5 seconds (300 frames at 60fps) to reduce overhead
        if self.frame_counter % 300 != 0:
            # But always check countdown if warning is active
            if self._critical_warning_shown and self._critical_warning_start_time:
                elapsed = time.time() - self._critical_warning_start_time
                remaining = max(0, self._critical_warning_timeout_sec - elapsed)

                # Update countdown in dialog
                if self._critical_warning_dialog and self._critical_warning_dialog.isVisible():
                    self._critical_warning_dialog.update_countdown(int(remaining))

                # Time's up - force return to home
                if elapsed >= self._critical_warning_timeout_sec:
                    print("[GUI] Critical disk space timeout - forcing return to home")
                    if self._critical_warning_dialog:
                        self._critical_warning_dialog.close()
                        self._critical_warning_dialog = None
                    self._critical_warning_shown = False
                    self._critical_warning_start_time = None
                    if self._started:
                        self._on_start_clicked()  # This returns to home page with full UI update
                    else:
                        self._handle_end()
            return

        # Check available disk space
        video_dir = Path.home() / "Videos"
        available = get_available_disk_space_gb(video_dir)

        if available < self._critical_disk_space_gb and not self._critical_warning_shown:
            # Show warning dialog
            self._critical_warning_shown = True
            self._critical_warning_start_time = time.time()
            print(f"[GUI] Critical disk space warning: {available:.1f} GB < {self._critical_disk_space_gb} GB")

            self._critical_warning_dialog = CriticalDiskSpaceDialog(
                available_gb=available,
                timeout_sec=self._critical_warning_timeout_sec,
                parent=self
            )
            self._critical_warning_dialog.finished_early.connect(self._on_critical_warning_accepted)
            self._critical_warning_dialog.show()  # Non-modal to allow continued use

    def _on_critical_warning_accepted(self):
        """User acknowledged warning and wants to finish early."""
        print("[GUI] User acknowledged critical disk space warning - returning to home")
        self._critical_warning_shown = False
        self._critical_warning_start_time = None
        self._critical_warning_dialog = None
        if self._started:
            self._on_start_clicked()  # This returns to home page with full UI update
        else:
            self._handle_end()

    def _update_frames(self):
        if self.cuda_ctx is None:
            if not self.ctx_q.empty():
                self.cuda_ctx = self.ctx_q.get()
                self.ctx_q.put(self.cuda_ctx)
            elif 'GL_CUDA_CTX' in globals():
                self.cuda_ctx = globals()['GL_CUDA_CTX']
            if self.cuda_ctx and self.copy_stream is None:
                self.cuda_ctx.push()
                try:
                    self.copy_stream = cuda.Stream()
                finally:
                    self.cuda_ctx.pop()
        
        if not self.pipeline_running:
            if not self.has_displayed_desktop:
               # even if one monitor is hidden, this is cheap and keeps state consistent
                self.left_monitor.show_cpu_frame(cv_img=self.start_img)
                self.right_monitor.show_cpu_frame(cv_img=self.start_img)
                self.has_displayed_desktop = True
            self.frame_counter += 1
            return

        # Read live from cameras
        if self.frame_counter > 0 and self.frame_counter % 200 == 0:
            time_inter = time.time() - self.time_pre
            self._display_fps = 200.0 / time_inter
            print(f"realtime display has {self._display_fps:.1f} FPS")
            self.time_pre = time.time()
            # Update FPS label when modes are active (format: display/od)
            if self.f_dribble or self.f_shoot or self.f_game:
                if self.f_game:
                    self._fps_label.setText(f"{self._display_fps:.1f}/--")
                else:
                    self._fps_label.setText(f"{self._display_fps:.1f}/{self._od_fps:.1f}")

        left_on = self._left_enabled()
        right_on = self._right_enabled()

        # --- capture only enabled sides (safe) ---
        cuda_left = None
        cuda_right = None
        if left_on:
            try:
                cuda_left = self.cap_left.Capture()
            except Exception as e:
                print(f"[cap_left] capture failed: {e}")
                cuda_left = None
        if right_on:
            try:
                cuda_right = self.cap_right.Capture()
            except Exception as e:
                print(f"[cap_right] capture failed: {e}")
                cuda_right = None

        if self.cuda_ctx:
            self.cuda_ctx.push()
        try:
            # LEFT
            if cuda_left is not None:
                if self.rgba_left is None:
                    self.rgba_left = jutils.cudaAllocMapped(
                        width=self.src_W,
                        height=self.src_H,
                        format="rgba8",
                    )
                jutils.cudaConvertColor(cuda_left, self.rgba_left)
                # Dribble/Shooting: send at normal interval
                # Game mode: send periodically for idle detection (player count)
                if not self.f_game and self.frame_counter % self.infer_interval == 0:
                    self._send_frame_to_inference(self.rgba_left, 'left')
                elif self.f_game and self._game_infer_counter % self._game_infer_interval == 0:
                    self._send_frame_to_inference(self.rgba_left, 'left')

                
            # RIGHT
            if cuda_right is not None:
                if self.rgba_right is None:
                    self.rgba_right = jutils.cudaAllocMapped(
                        width=self.src_W,
                        height=self.src_H,
                        format="rgba8",
                    )
                jutils.cudaConvertColor(cuda_right, self.rgba_right)
                # Dribble/Shooting: send at normal interval
                # Game mode: send periodically for idle detection (player count)
                if not self.f_game and self.frame_counter % self.infer_interval == 0:
                    self._send_frame_to_inference(self.rgba_right, 'right')
                elif self.f_game and self._game_infer_counter % self._game_infer_interval == 0:
                    self._send_frame_to_inference(self.rgba_right, 'right')

            # Increment game mode inference counter
            if self.f_game:
                self._game_infer_counter += 1

            # consider to record image here to get video without overlay.

            # Apply overlay for left
            if self.engine and self.engine.overlay_left is not None:
                # Optional: only if left camera is actually shown
                if left_on and not self.f_game and self.rgba_left is not None:
                    with MAP_LOCK:
                        jutils.cudaOverlay(self.engine.overlay_left,
                                           self.rgba_left, 0, 0)

            # Apply overlay for right
            if self.engine and self.engine.overlay_right is not None:
                if right_on and not self.f_game and self.rgba_right is not None:
                    with MAP_LOCK:
                        jutils.cudaOverlay(self.engine.overlay_right,
                                           self.rgba_right, 0, 0)

            # Sync CUDA to ensure overlay is fully applied before recording
            # This prevents flickering in recorded video due to async GPU operations
            cuda.Context.synchronize()

            # Display only enabled/visible sides.
            shown_any = False
            if left_on and self.rgba_left is not None and self._view_mode in ("left", "both"):
                self.left_monitor.show_cuda_frame(
                    int(self.rgba_left.ptr),
                    self.src_W,
                    self.src_H,
                    self.rgba_left.width * self.rgba_left.height * self.rgba_left.channels
                )
                shown_any = True

            if right_on and self.rgba_right is not None and self._view_mode in ("right", "both"):
                self.right_monitor.show_cuda_frame(
                    int(self.rgba_right.ptr),
                    self.src_W,
                    self.src_H,
                    self.rgba_right.width * self.rgba_right.height * self.rgba_right.channels
                )
                shown_any = True

            if shown_any:
                self.has_displayed_desktop = False

            # trigger recording (only if CUDA images are valid)
            if self.pipeline_running and self.f_left_cam and self.rgba_left is not None:
                bgr_left = cuda_to_bgr_np(self.rgba_left, self.src_W, self.src_H)
                push_frame("left", bgr_left)
                if not self._rec_left_on:
                    self.recorder_queue.put({"cmd": "record", "side": "left", "on": 1, "mode": self._get_current_mode()})
                    self._rec_left_on = True

            if self.pipeline_running and self.f_right_cam and self.rgba_right is not None:
                bgr_right = cuda_to_bgr_np(self.rgba_right, self.src_W, self.src_H)
                push_frame("right", bgr_right)
                if not self._rec_right_on:
                    self.recorder_queue.put({"cmd": "record", "side": "right", "on": 1, "mode": self._get_current_mode()})
                    self._rec_right_on = True

            if self.pipeline_running and self.f_both_cam and self.rgba_left is not None and self.rgba_right is not None:
                bgr_right = cuda_to_bgr_np(self.rgba_right, self.src_W, self.src_H)
                bgr_left = cuda_to_bgr_np(self.rgba_left, self.src_W, self.src_H)
                bgr_both = np.hstack((bgr_left, bgr_right))
                push_frame("both", bgr_both)
                if not self._rec_both_on:
                    self.recorder_queue.put({"cmd": "record", "side": "both", "on": 1, "mode": self._get_current_mode()})
                    self._rec_both_on = True

        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

        self.frame_counter += 1

    def _send_frame_to_inference(self, rgba_img, side):
        if not self.pbo_pool or not self.infer_frame_queue:
            return
        if self.copy_stream is None:
            return
        
        pbo_id = None
        enqueued = False
        try:
            # Don't block GUI thread if inference is back-pressured
            if self.infer_frame_queue.full():
                return

            # Acquire without waiting long; skip if pool is empty.
            pbo_id = self.pbo_pool.acquire(timeout=0.0)

            cuda_res = self.pbo_pool.regbufs.get(pbo_id)
            if cuda_res is None:
                cuda_res = cudagl.RegisteredBuffer(
                    int(pbo_id),
                    cudagl.graphics_map_flags.WRITE_DISCARD
                )
                self.pbo_pool.regbufs[pbo_id] = cuda_res

            with MAP_LOCK:
                mapping = cuda_res.map(self.copy_stream)
                dev_ptr, size = mapping.device_ptr_and_size()
                cuda.memcpy_dtod_async(
                    dev_ptr,
                    int(rgba_img.ptr),
                    rgba_img.width * rgba_img.height * rgba_img.channels,
                    self.copy_stream
                )
                self.copy_stream.synchronize()
                mapping.unmap(self.copy_stream)

            # Put without blocking; if full, release PBO immediately.
            self.infer_frame_queue.put_nowait({
                'pbo_id': pbo_id,
                'side': side,
                'width': rgba_img.width,
                'height': rgba_img.height,
                'timestamp': time.time()
            })
            enqueued = True

        except queue.Empty:
            print(f"[GUI] No free PBO available, skipping frame")
        except queue.Full:
             print("[GUI] Inference queue full, skipping frame")
        except Exception as e:
            print(f"[GUI] Error sending frame to inference: {e}")
        finally:
            if pbo_id is not None and not enqueued:
                # NEW: we acquired but didn't enqueue -> release
                self.pbo_pool.release(pbo_id)

    def _create_main_page(self):
        page = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        # === Monitors ===
        self.left_monitor = DualMonitor(self.ctx_q, self.ctx_ready, side="left",
                                        cmd_queue=self.cmd_queue, tex_queue=self.tex_queue,
                                        pbo_pool=self.pbo_pool,
                                        src_W=self.src_W, src_H=self.src_H)
        self.left_monitor.setAlignment(QtCore.Qt.AlignCenter)
        self.left_monitor.setScaledContents(True)
        self.left_monitor.setStyleSheet('background: rgba(0,0,0,1);')

        self.right_monitor = DualMonitor(self.ctx_q, self.ctx_ready, side="right",
                                         cmd_queue=self.cmd_queue, tex_queue=self.tex_queue,
                                         pbo_pool=self.pbo_pool,
                                         src_W=self.src_W, src_H=self.src_H)
        self.right_monitor.setAlignment(QtCore.Qt.AlignCenter)
        self.right_monitor.setScaledContents(True)
        self.right_monitor.setStyleSheet('background: rgba(0,0,0,1);')

        # === Controls overlay (2 rows) ===
        controls = QtWidgets.QWidget()
        ctl = QtWidgets.QGridLayout(controls)
        ctl.setContentsMargins(12, 12, 12, 12)
        ctl.setHorizontalSpacing(40)
        ctl.setVerticalSpacing(40)

        # -------- Row 0: Left / Right / Both (exclusive) --------
        self.left_btn = QtWidgets.QPushButton()
        self.right_btn = QtWidgets.QPushButton()
        self.both_btn = QtWidgets.QPushButton()
        for b in (self.left_btn, self.right_btn, self.both_btn):
            b.setIconSize(QtCore.QSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT))
            b.setFixedSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT)
            b.setCheckable(True)

        icon = QtGui.QIcon()
        if f_pinyin:
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/left_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/left.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        else:
            icon.addPixmap(QtGui.QPixmap("UI_Button/left_court_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("UI_Button/left_court_active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.left_btn.setIcon(icon)
        icon = QtGui.QIcon()

        if f_pinyin:
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/right_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/right.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        else:
            icon.addPixmap(QtGui.QPixmap("UI_Button/right_court_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("UI_Button/right_court_active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.right_btn.setIcon(icon)
        icon = QtGui.QIcon()

        if f_pinyin:
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/both_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/both.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        else:
            icon.addPixmap(QtGui.QPixmap("UI_Button/full_court_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("UI_Button/full_court_active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.both_btn.setIcon(icon)

        self.view_group = QtWidgets.QButtonGroup(self)
        self.view_group.setExclusive(True)
        for b in (self.left_btn, self.right_btn, self.both_btn):
            self.view_group.addButton(b)

        self.left_btn.clicked.connect(self._handle_left_cam)
        self.right_btn.clicked.connect(self._handle_right_cam)
        self.both_btn.clicked.connect(self._handle_both_cam)

        ctl.addWidget(self.left_btn, 0, 0)
        ctl.addWidget(self.right_btn, 0, 2)
        ctl.addWidget(self.both_btn, 0, 1)

        # -------- Row 1: Dribble / Shoot / Game (exclusive) --------
        self.dribble_btn = QtWidgets.QPushButton()
        self.shoot_btn = QtWidgets.QPushButton()
        self.game_btn = QtWidgets.QPushButton()
        for b in (self.dribble_btn, self.shoot_btn, self.game_btn):
            b.setIconSize(QtCore.QSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT))  # scale the image
            b.setFixedSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT)
            b.setCheckable(True)

        icon = QtGui.QIcon()
        if f_pinyin:
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/dribble_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/dribble.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        else:
            icon.addPixmap(QtGui.QPixmap("UI_Button/dribble_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("UI_Button/dribble_active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.dribble_btn.setIcon(icon)
        icon = QtGui.QIcon()
        if f_pinyin:
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/shoot_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/shoot.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        else:
            icon.addPixmap(QtGui.QPixmap("UI_Button/shoot_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("UI_Button/shoot_active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.shoot_btn.setIcon(icon)
        icon = QtGui.QIcon()
        if f_pinyin:
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/game_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("Chinese_UI/game.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        else:
            icon.addPixmap(QtGui.QPixmap("UI_Button/game_inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("UI_Button/game_active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.game_btn.setIcon(icon)

        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for b in (self.dribble_btn, self.shoot_btn, self.game_btn):
            self.mode_group.addButton(b)

        self.dribble_btn.clicked.connect(self._handle_dribble)
        self.shoot_btn.clicked.connect(self._handle_shoot)
        self.game_btn.clicked.connect(self._handle_game)

        ctl.addWidget(self.dribble_btn, 1, 0)
        ctl.addWidget(self.shoot_btn, 1, 1)
        ctl.addWidget(self.game_btn, 1, 2)

        controls.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        controls.setStyleSheet(" QWidget { background: transparent; }")

        # --- START button in bottom-right ---
        self.start_btn = QtWidgets.QPushButton()
        start_icon = QtGui.QIcon()
        if f_pinyin:
            start_icon.addPixmap(QtGui.QPixmap("Chinese_UI/start.png"), QtGui.QIcon.Normal)
        else:
            start_icon.addPixmap(QtGui.QPixmap("UI_Button/start_active.png"), QtGui.QIcon.Normal)
        self.start_btn.setIcon(start_icon)
        self.start_btn.setIconSize(QtCore.QSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT))
        self.start_btn.setFixedSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT)
        self.start_btn.setEnabled(False)
        # remove inner padding/border so icons can touch edge-to-edge
        self.start_btn.setStyleSheet("""
            QPushButton { background: transparent; border: none; padding: 0px; }
        """)
        self.start_btn.clicked.connect(self._on_start_clicked)

        # --- Next button ---
        self.next_btn = QtWidgets.QPushButton()
        next_icon = QtGui.QIcon()
        if f_pinyin:
            next_icon.addPixmap(QtGui.QPixmap("Chinese_UI/next.png"), QtGui.QIcon.Normal)
        else:
            next_icon.addPixmap(QtGui.QPixmap("UI_Button/next.png"), QtGui.QIcon.Normal)
        self.next_btn.setIcon(next_icon)
        self.next_btn.setIconSize(QtCore.QSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT))
        self.next_btn.setFixedSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT)
        self.next_btn.setVisible(False)
        self.next_btn.setStyleSheet("""
            QPushButton { background: transparent; border: none; padding: 0px; }
        """)

        self.next_btn.clicked.connect(self._handle_start)  # this is intentional.

        # --- Stop Game button in bottom-right ---
        self.stop_game_btn = QtWidgets.QPushButton()
        stop_game_icon = QtGui.QIcon()
        if f_pinyin:
            stop_game_icon.addPixmap(QtGui.QPixmap("Chinese_UI/stop_game.png"), QtGui.QIcon.Normal)
        else:
            stop_game_icon.addPixmap(QtGui.QPixmap("UI_Button/stop_game_active.png"), QtGui.QIcon.Normal)
        self.stop_game_btn.setIcon(stop_game_icon)
        self.stop_game_btn.setIconSize(QtCore.QSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT))
        self.stop_game_btn.setFixedSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT)
        self.stop_game_btn.setVisible(False)
        self.stop_game_btn.setEnabled(True)
        # remove inner padding/border so icons can touch edge-to-edge
        self.stop_game_btn.setStyleSheet("""
                            QPushButton { background: transparent; border: none; padding: 0px; }
                        """)
        self.stop_game_btn.clicked.connect(self._handle_stop_game)

        # --- bottom-right container ---
        right_corner = QtWidgets.QWidget()
        row = QtWidgets.QVBoxLayout(right_corner)  # simpler than GridLayout for a row
        row.setContentsMargins(0, 0, 0, 0)  # no outer margin
        row.setSpacing(20)  #  gap 20pix between widgets  apply to secondary interface
        right_corner.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        right_corner.setStyleSheet("background: transparent;")

        # order decides which is left/right
        row.addStretch(1)
        row.addWidget(self.stop_game_btn)
        row.addWidget(self.next_btn)
        row.addWidget(self.start_btn)


        # --- Display area container ---
        self.display = QtWidgets.QWidget()
        self.display.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.display.setStyleSheet("background: #000;")  # whatever you like

        self.dlayout = QtWidgets.QGridLayout(self.display)
        self.dlayout.setContentsMargins(0, 0, 0, 0)
        self.dlayout.setSpacing(0)

        # add both to the container (we'll lay them out by mode)
        self.left_monitor.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.right_monitor.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # initial placement (we'll call _set_view_mode next)
        self.dlayout.addWidget(self.left_monitor, 0, 0)
        self.dlayout.addWidget(self.right_monitor, 0, 1)
        self.dlayout.setColumnStretch(0, 1)
        self.dlayout.setColumnStretch(1, 1)

        # create buttons for game page

        # --- view-full button ---
        self.view_full_btn = QtWidgets.QPushButton()
        view_full_icon = QtGui.QIcon()
        if f_pinyin:
            view_full_icon.addPixmap(QtGui.QPixmap("Chinese_UI/replay_active.png"), QtGui.QIcon.Normal)
        else:
            view_full_icon.addPixmap(QtGui.QPixmap("UI_Button/replay_active.png"), QtGui.QIcon.Normal)
        self.view_full_btn.setIcon(view_full_icon)
        if f_pinyin:
            self.view_full_btn.setIconSize(QtCore.QSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT))
            self.view_full_btn.setFixedSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT)
        else:
            self.view_full_btn.setIconSize(QtCore.QSize(250, 70)) # this button is bigger than expected.
            self.view_full_btn.setFixedSize(250, 70)
        self.view_full_btn.setVisible(False)
        self.view_full_btn.setEnabled(False)
        self.view_full_btn.setStyleSheet("""
                                            QPushButton { background: transparent; border: none; padding: 0px; }
                                        """)
        self.view_full_btn.clicked.connect(self._handle_view_full)

        # --- Upload button in bottom-right ---
        self.upload_game_btn = QtWidgets.QPushButton()
        upload_game_icon = QtGui.QIcon()
        if f_pinyin:
            upload_game_icon.addPixmap(QtGui.QPixmap("Chinese_UI/upload_active.png"), QtGui.QIcon.Normal)
        else:
            upload_game_icon.addPixmap(QtGui.QPixmap("UI_Button/upload_active.png"), QtGui.QIcon.Normal)
        self.upload_game_btn.setIcon(upload_game_icon)
        self.upload_game_btn.setIconSize(QtCore.QSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT))
        self.upload_game_btn.setFixedSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT)
        self.upload_game_btn.setVisible(False)
        self.upload_game_btn.setEnabled(True)
        # remove inner padding/border so icons can touch edge-to-edge
        self.upload_game_btn.setStyleSheet("""
                                    QPushButton { background: transparent; border: none; padding: 0px; }
                                """)
        self.upload_game_btn.clicked.connect(self._handle_upload_game)

        # --- center-right container ---
        game_btns = QtWidgets.QWidget()
        col = QtWidgets.QHBoxLayout(game_btns)  # simpler than GridLayout for a row
        col.setContentsMargins(0, 0, 0, 0)  # no outer margin
        col.setSpacing(20)  # no gap between widgets
        game_btns.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        game_btns.setStyleSheet("background: transparent;")

        # order decides which is left/right
        col.addStretch(1)
        col.addWidget(self.upload_game_btn)
        col.addWidget(self.view_full_btn)

        # add MP logo
        mp_logo = QtWidgets.QPushButton()
        logo_icon = QtGui.QIcon()
        logo_icon.addPixmap(QtGui.QPixmap("UI_Button/mp_logo.jpg"), QtGui.QIcon.Normal)
        mp_logo.setIcon(logo_icon)
        mp_logo.setVisible(False)
        mp_logo.setIconSize(QtCore.QSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT))
        mp_logo.setFixedSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT)
        mp_logo.setStyleSheet("""
                            QPushButton { background: transparent; border: none; padding: 0px; }
                        """)

        # top-level layout: add display first, then overlays/buttons so they render above
        self.controls = controls
        self.right_corner = right_corner
        self.mp_logo = mp_logo
        # self.game_btns = game_btns

        # --- SETTINGS button (top-right) ---
        self.setting_btn = QtWidgets.QPushButton()
        setting_icon = QtGui.QIcon()
        if f_pinyin:
            setting_icon.addPixmap(QtGui.QPixmap("Chinese_UI/setting_active.png"), QtGui.QIcon.Normal)
        else:
            setting_icon.addPixmap(QtGui.QPixmap("UI_Button/setting_active.png"), QtGui.QIcon.Normal)
        self.setting_btn.setIcon(setting_icon)
        self.setting_btn.setIconSize(QtCore.QSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT))
        self.setting_btn.setFixedSize(CONTROL_BTN_WIDTH, CONTROL_BTN_HEIGHT)
        self.setting_btn.setVisible(True)
        self.setting_btn.setEnabled(True)
        # remove inner padding/border so icons can touch edge-to-edge
        self.setting_btn.setStyleSheet("""
                                    QPushButton { background: transparent; border: none; padding: 0px; }
                                """)
        self.setting_btn.clicked.connect(self._show_settings_menu)

        # --- FPS display labels (top-right corner) ---
        fps_container = QtWidgets.QWidget()
        fps_layout = QtWidgets.QVBoxLayout(fps_container)
        fps_layout.setContentsMargins(10, 10, 10, 10)
        fps_layout.setSpacing(5)
        fps_container.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        fps_container.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        fps_container.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 0.6);
                border-radius: 8px;
            }
        """)

        self._fps_label = QtWidgets.QLabel("--/--")
        self._fps_label.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 18px;
                font-weight: bold;
                background: transparent;
            }
        """)

        fps_layout.addWidget(self._fps_label)
        fps_container.setVisible(False)  # Hidden by default, shown when modes are active
        self._fps_container = fps_container

        grid.addWidget(self.display, 0, 0)
        grid.addWidget(self.controls, 0, 0,
                       alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter)
        grid.addWidget(self.right_corner, 0, 0,
                       alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight)
        grid.addWidget(game_btns, 0, 0,
                       alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter)
        grid.addWidget(mp_logo, 0, 0,
                       alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)
        grid.addWidget(self.setting_btn, 0, 0,
                       alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)
        grid.addWidget(fps_container, 0, 0,
                       alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)

        # choose a default view
        self._set_view_mode('left')  # or 'both' if that’s your default
        # show full screen and get rid of top menu
        self.showFullScreen()
        return page

    # state flag
    def _set_view_mode(self, mode: str):
        # remove current placements (doesn't delete widgets)
        self._view_mode = mode
        self.dlayout.removeWidget(self.left_monitor)
        self.dlayout.removeWidget(self.right_monitor)

        if mode == 'left':
            self.left_monitor.setVisible(True)
            self.right_monitor.setVisible(False)
            self.left_monitor.raise_()
            self.controls.raise_()
            self.right_corner.raise_()
            # left spans both columns (full width)
            self.dlayout.addWidget(self.left_monitor, 0, 0, 1, 2)
            self.dlayout.setColumnStretch(0, 1)
            self.dlayout.setColumnStretch(1, 1)
            self.left_monitor.gpu_view.clear_crop()
            self.right_monitor.gpu_view.clear_crop()

        elif mode == 'right':
            self.left_monitor.setVisible(False)
            self.right_monitor.setVisible(True)
            self.right_monitor.raise_()
            self.controls.raise_()
            self.right_corner.raise_()
            # right spans both columns (full width)
            self.dlayout.addWidget(self.right_monitor, 0, 0, 1, 2)
            self.dlayout.setColumnStretch(0, 1)
            self.dlayout.setColumnStretch(1, 1)
            self.left_monitor.gpu_view.clear_crop()
            self.right_monitor.gpu_view.clear_crop()

        else:  # 'both'

            self.left_monitor.setVisible(True)
            self.right_monitor.setVisible(True)
            # side-by-side
            self.dlayout.addWidget(self.left_monitor, 0, 0, 1, 1)
            self.dlayout.addWidget(self.right_monitor, 0, 1, 1, 1)
            self.dlayout.setColumnStretch(0, 1)
            self.dlayout.setColumnStretch(1, 1)
            self.left_monitor.gpu_view.set_crop_pixels(380, 1540)
            self.right_monitor.gpu_view.set_crop_pixels(380, 1540)
            self.right_corner.raise_()
            # self.left_monitor.setCurrentIndex(1)
            # self.right_monitor.setCurrentIndex(1)

        # optional: force an immediate relayout/repaint
        self.display.update()

    def _on_start_clicked(self):
        if not self._started:
            # START -> hide the 6 buttons and flip label to END
            self._started = True
            icon = QtGui.QIcon()
            if calibrations.f_pinyin:
                icon.addPixmap(QtGui.QPixmap("Chinese_UI/home.png"), QtGui.QIcon.Normal)
            else:
                icon.addPixmap(QtGui.QPixmap("UI_Button/home_active.png"), QtGui.QIcon.Normal)
            self.start_btn.setIcon(icon)
            for b in (self.left_btn, self.right_btn, self.both_btn,
                      self.dribble_btn, self.shoot_btn, self.game_btn):
                b.setVisible(False)
            # call your callback exactly once on start
            self.next_btn.setVisible(True)
            self._handle_start()
            self.mp_logo.setVisible(True)
            icon = QtGui.QIcon()
            if calibrations.f_pinyin:
                icon.addPixmap(QtGui.QPixmap("Chinese_UI/stop_game.png"), QtGui.QIcon.Normal)
            else:
                icon.addPixmap(QtGui.QPixmap("UI_Button/stop_game_active.png"), QtGui.QIcon.Normal)
            self.stop_game_btn.setIcon(icon)
            self._stoped_game = True
        else:
            # END -> show buttons back and flip label to START
            self._started = False
            for b in (self.left_btn, self.right_btn, self.both_btn,
                      self.dribble_btn, self.shoot_btn, self.game_btn):
                b.setVisible(True)
            # If you also want a stop callback, implement self._handle_end()
            self._handle_end()
            self.next_btn.setVisible(False)
            self.stop_game_btn.setVisible(False)
            self.view_full_btn.setVisible(False)
            self.upload_game_btn.setVisible(False)
            self.mp_logo.setVisible(False)
            icon = QtGui.QIcon()
            if calibrations.f_pinyin:
                icon.addPixmap(QtGui.QPixmap("Chinese_UI/start.png"), QtGui.QIcon.Normal)
            else:
                icon.addPixmap(QtGui.QPixmap("UI_Button/start_active.png"), QtGui.QIcon.Normal)
            self.start_btn.setIcon(icon)


    def _handle_dribble(self):
        self.f_dribble = True
        self.f_shoot = False
        self.f_game = False
        if self.f_left_cam or self.f_both_cam or self.f_right_cam:
            self.start_btn.setEnabled(True)

    def _handle_shoot(self):
        self.f_shoot = True
        self.f_dribble = False
        self.f_game = False
        if self.f_left_cam or self.f_both_cam or self.f_right_cam:
            self.start_btn.setEnabled(True)

    def _handle_game(self):
        self.f_game = True
        self.f_dribble = False
        self.f_shoot = False
        if self.f_left_cam or self.f_both_cam or self.f_right_cam:
            self.start_btn.setEnabled(True)

    def _handle_left_cam(self):
        self.f_left_cam = True
        self.f_both_cam = False
        self.f_right_cam = False
        if self.f_dribble or self.f_shoot or self.f_game:
            self.start_btn.setEnabled(True)

    def _handle_right_cam(self):
        self.f_left_cam = False
        self.f_both_cam = False
        self.f_right_cam = True
        if self.f_dribble or self.f_shoot or self.f_game:
            self.start_btn.setEnabled(True)

    def _handle_both_cam(self):
        self.f_left_cam = False
        self.f_both_cam = True
        self.f_right_cam = False
        if self.f_dribble or self.f_shoot or self.f_game:
            self.start_btn.setEnabled(True)

    def _handle_start(self):
        self.pipeline_running = False
        self.setting_btn.setVisible(False)

        # 1) Decide view mode and install monitors into layout first
        if self.f_left_cam:
            view_value = 'left'
            self._set_view_mode('left')
        elif self.f_right_cam:
            view_value = 'right'
            self._set_view_mode('right')
        else:
            view_value = 'both'
            self._set_view_mode('both')

        # 2) Prime GL AFTER the monitor(s) are visible/in-layout
        self._prime_gl_contexts()

        
        if self.f_dribble:
            self.stop_game_btn.setVisible(False)
            self.view_full_btn.setVisible(False)
            forbidden = self.dribble_style  # example
            choices = [x for x in DRIBBLE_STYLES if x != forbidden]
            style = random.choice(choices)
            print(f'Dribble style selected: {style}')
            self.cmd_queue.put({"cmd": "set_style", "style": style})
            self.run_event.set()
            self.gui2infer_queue.put({"feature": "dribble", "style": style, "count": 20, 'view_value': view_value})
            self.dribble_style = style
            self.next_btn.setVisible(True)
            # Use faster interval for both-side mode
            self.infer_interval = 3 if view_value == 'both' else 4
        elif self.f_shoot:
            self.stop_game_btn.setVisible(False)
            self.view_full_btn.setVisible(False)
            forbidden = self.shoot_mode  # example
            choices = [x for x in SHOOT_MODES if x != forbidden]
            mode = random.choice(choices)
            # mode = "free throw"

            print(f'Shoot mode selected: {mode}')
            self.run_event.set()
            self.gui2infer_queue.put({"feature": "shooting", "style": mode, "count": 0, 'view_value': view_value})
            self.shoot_mode = mode
            self.next_btn.setVisible(True)
            # Use faster interval for both-side mode
            self.infer_interval = 3 if view_value == 'both' else 4
        elif self.f_game:
            self.next_btn.setVisible(False)
            self.stop_game_btn.setVisible(True)
            self.stop_game_btn.setEnabled(True)
            mode = "random"
            print(f'Shoot mode selected: {mode}')
            self.run_event.set()
            self.gui2infer_queue.put({"feature": "game", "style": mode, "count": 0, 'view_value': view_value})
            # Reset game mode idle detection (both sides assumed active at start)
            self._game_last_active_time = time.time()
            self._game_left_player_count = 2  # Assume active at start
            self._game_left_has_ball = True   # Assume ball present at start
            self._game_right_player_count = 2
            self._game_right_has_ball = True
            self._game_infer_counter = 0

        self.pipeline_running = True
        # Show FPS display when modes are active
        self._fps_container.setVisible(True)
        self._fps_container.raise_()

    def _handle_end(self):  # this is home button actually
        self.pipeline_running = False
        self.view_full_btn.setEnabled(False)
        self.stop_game_btn.setVisible(False)
        self.setting_btn.setVisible(True)
        self.view_full_btn.setVisible(False)
        self._set_view_mode('left')
        # Reset game mode idle detection
        self._game_last_active_time = None
        self._game_left_player_count = 0
        self._game_left_has_ball = False
        self._game_right_player_count = 0
        self._game_right_has_ball = False
        self._game_infer_counter = 0
        # Hide FPS display and reset values
        self._fps_container.setVisible(False)
        self._display_fps = 0.0
        self._od_fps = 0.0
        self._fps_label.setText("--/--")

        # Reset critical disk space warning
        self._critical_warning_shown = False
        self._critical_warning_start_time = None
        if self._critical_warning_dialog:
            self._critical_warning_dialog.close()
            self._critical_warning_dialog = None

        if self._rec_left_on:
            self.recorder_queue.put({"cmd": "record", "side": "left", "on": 0})
            self._rec_left_on = False

        if self._rec_right_on:
            self.recorder_queue.put({"cmd": "record", "side": "right", "on": 0})
            self._rec_right_on = False

        if self._rec_both_on:
            self.recorder_queue.put({"cmd": "record", "side": "both", "on": 0})
            self._rec_both_on = False

    def _handle_stop_game(self):
        if self._stoped_game:
            self._stoped_game = False
            self.view_full_btn.setVisible(True)
            self.upload_game_btn.setVisible(True)
            self.mp_logo.setVisible(False)
            # need to process the video before the button is enabled!
            self.view_full_btn.setEnabled(True)
            self.pipeline_running = False
            icon = QtGui.QIcon()
            if calibrations.f_pinyin:
                icon.addPixmap(QtGui.QPixmap("Chinese_UI/start_game.png"), QtGui.QIcon.Normal)
            else:
                icon.addPixmap(QtGui.QPixmap("UI_Button/start_game.png"), QtGui.QIcon.Normal)
            self.stop_game_btn.setIcon(icon)

            if self._rec_left_on:
                self.recorder_queue.put({"cmd": "record", "side": "left", "on": 0})
                self._rec_left_on = False

            if self._rec_right_on:
                self.recorder_queue.put({"cmd": "record", "side": "right", "on": 0})
                self._rec_right_on = False

            if self._rec_both_on:
                self.recorder_queue.put({"cmd": "record", "side": "both", "on": 0})
                self._rec_both_on = False

        else:
            self._stoped_game = True
            self.view_full_btn.setVisible(False)
            self.upload_game_btn.setVisible(False)
            icon = QtGui.QIcon()
            if calibrations.f_pinyin:
                icon.addPixmap(QtGui.QPixmap("Chinese_UI/stop_game.png"), QtGui.QIcon.Normal)
            else:
                icon.addPixmap(QtGui.QPixmap("UI_Button/stop_game_active.png"), QtGui.QIcon.Normal)
            self.stop_game_btn.setIcon(icon)
            self._handle_start()
            self.frame_counter = 0

    def _handle_upload_game(self):
        video_dir = Path.home() / "Videos"

        parent = QtWidgets.QApplication.activeWindow()
        f_upload = True
        dlg = VideoPickerDialog(f_upload, video_dir, parent=parent)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        chosen = dlg.selected_paths() # this enable multiple paths
        if not chosen:
            QtWidgets.QMessageBox.warning(parent, "Not found", "Selected file no longer exists.")
            return
        
        print(chosen)
        popup_queue_to_baidupcs_drop(
            self,
            gate_conf_path="upload_gate.conf",
            files_to_upload=chosen,
            move=False,   # False = copy instead of move
        )


    @staticmethod
    def _handle_view_full():
        video_dir = Path.home() / "Videos"
        parent = QtWidgets.QApplication.activeWindow()
        f_upload = False
        dlg = VideoPickerDialog(f_upload, video_dir, parent=parent)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        chosen = dlg.selected_path()
        if not chosen or not chosen.exists():
            QtWidgets.QMessageBox.warning(parent, "Not found", "Selected file no longer exists.")
            return

        # Launch video player dialog with controls
        player = VideoPlayerDialog(chosen, parent=parent)
        player.exec_()

    def _is_valid_frame(self, frame_np, min_std=10.0):
        """Check if frame is valid (not corrupted/blank)."""
        if frame_np is None or frame_np.size == 0:
            return False
        # Check if image has reasonable variance (not all black/white/uniform)
        if frame_np.std() < min_std:
            return False
        # Check if image is not all zeros
        if np.all(frame_np == 0):
            return False
        return True

    def _quick_capture_frame(self, side):
        """Quick frame capture for refresh - camera is already warm."""
        cap = self.cap_left if side == 'left' else self.cap_right
        path = f"/tmp/_cal_frame_{side}.png"

        # Delete old file first to avoid caching issues
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

        # Just flush a few frames (camera is already streaming)
        for _ in range(3):
            try:
                cap.Capture()
            except Exception:
                pass

        # Try to capture a valid frame with fewer retries
        for attempt in range(5):
            try:
                cuda_frame = cap.Capture()
                if cuda_frame is None:
                    continue
                frame_np = cudaToNumpy(cuda_frame).astype(np.uint8)
                frame_np = frame_np[..., ::-1]  # RGBA/RGB -> BGR for cv2

                if self._is_valid_frame(frame_np):
                    # Write to temp file first, then rename (atomic)
                    temp_path = path + ".tmp"
                    cv2.imwrite(temp_path, frame_np)
                    os.sync()
                    os.rename(temp_path, path)
                    os.sync()
                    return path
            except Exception:
                pass

        return None

    def _save_calibration_frame(self, side):
        """Capture current frame from already-open camera and save to a temp PNG.

        Flushes the camera buffer and validates the frame to avoid black/corrupted images.
        """
        cap = self.cap_left if side == 'left' else self.cap_right
        path = f"/tmp/_cal_frame_{side}.png"

        # Flush camera buffer by discarding several frames
        warmup_frames = 15
        for _ in range(warmup_frames):
            try:
                cap.Capture()
            except Exception:
                pass

        # Try to capture a valid frame with retries
        max_retries = 10
        frame_np = None
        for attempt in range(max_retries):
            try:
                cuda_frame = cap.Capture()
                if cuda_frame is None:
                    continue
                frame_np = cudaToNumpy(cuda_frame).astype(np.uint8)
                frame_np = frame_np[..., ::-1]  # RGBA/RGB -> BGR for cv2

                if self._is_valid_frame(frame_np):
                    cv2.imwrite(path, frame_np)
                    return path
            except Exception:
                pass

        # Last resort: save whatever we have
        if frame_np is not None:
            cv2.imwrite(path, frame_np)
            return path

        return None

    def start_calibration(self):
        # Build list of (script, side) pairs
        if self.f_left_cam and not self.f_both_cam:
            entries = [("manual_bev_left.py", "left")]
        elif self.f_right_cam and not self.f_both_cam:
            entries = [("manual_bev_right.py", "right")]
        else:  # either f_both_cam was set or by default it runs cals for both cams
            entries = [("manual_bev_left.py", "left"), ("manual_bev_right.py", "right")]

        self._cal_entries = list(entries)
        self._run_next_calibration_script()

    def _run_next_calibration_script(self):
        """Launch the next calibration script as a non-blocking subprocess."""
        if not self._cal_entries:
            self._hide_loading_overlay()
            self._cal_current_side = None
            return
        script, side = self._cal_entries.pop(0)
        self._cal_current_side = side  # Track which side is being calibrated
        # Save a frame from the already-open camera so the subprocess
        # doesn't need to re-open the device (which would fail).
        img_path = self._save_calibration_frame(side)
        cmd = [sys.executable, script]
        if img_path:
            cmd += ["--image", img_path]
        self._cal_proc = subprocess.Popen(cmd)
        # Poll for completion and refresh requests without blocking the Qt event loop
        self._cal_timer = QtCore.QTimer(self)
        self._cal_timer.timeout.connect(self._poll_calibration)
        self._cal_timer.start(100)  # Poll more frequently to handle refresh requests

    def _poll_calibration(self):
        """Check if the current calibration subprocess has finished or needs a refresh."""
        # Check for refresh request from calibration window
        if hasattr(self, '_cal_current_side') and self._cal_current_side:
            refresh_signal = f"/tmp/_cal_frame_{self._cal_current_side}.png.refresh"
            done_signal = f"/tmp/_cal_frame_{self._cal_current_side}.png.done"
            if os.path.exists(refresh_signal):
                try:
                    os.remove(refresh_signal)
                    # Use quick capture for refresh (camera is already warm)
                    path = self._quick_capture_frame(self._cal_current_side)
                    if not path:
                        path = self._save_calibration_frame(self._cal_current_side)
                    # Create done signal to notify calibration script
                    if path:
                        with open(done_signal, 'w') as f:
                            f.write("done")
                except Exception:
                    pass

        if self._cal_proc.poll() is not None:
            self._cal_timer.stop()
            self._cal_current_side = None
            self._run_next_calibration_script()

    def _check_commands(self):
        while not self.cmd_queue.empty():
            msg = self.cmd_queue.get_nowait()

            if msg.get("cmd") == "shutdown":
                print("[GUI] backend finished, closing window")
                QtWidgets.QApplication.quit()  # ´¥·¢ÍË³ö
                return

    def _show_loading_overlay(self, message_cn: str, message_en: str):
        """
        Show a loading overlay label on the main window.
        Works for both mouse and touch screen users.

        Args:
            message_cn: Chinese text to display
            message_en: English text to display
        """
        # Also set busy cursor for mouse users
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # Create overlay label
        text = message_cn if calibrations.f_pinyin else message_en
        self._loading_label = QtWidgets.QLabel(text, self)
        self._loading_label.setAlignment(QtCore.Qt.AlignCenter)
        self._loading_label.setStyleSheet("""
            QLabel {
                background-color: rgba(30, 144, 255, 220);
                color: white;
                font-size: 28px;
                font-weight: bold;
                padding: 30px 50px;
                border-radius: 20px;
            }
        """)
        self._loading_label.adjustSize()

        # Center on screen
        screen_geo = self.geometry()
        x = (screen_geo.width() - self._loading_label.width()) // 2
        y = (screen_geo.height() - self._loading_label.height()) // 2
        self._loading_label.move(x, y)
        self._loading_label.raise_()
        self._loading_label.show()
        QtWidgets.QApplication.processEvents()

    def _hide_loading_overlay(self):
        """Hide the loading overlay and restore cursor."""
        QtWidgets.QApplication.restoreOverrideCursor()
        if hasattr(self, '_loading_label') and self._loading_label:
            self._loading_label.hide()
            self._loading_label.deleteLater()
            self._loading_label = None

    def _open_wifi_dialog(self):
        """Open WiFi dialog with loading indicator."""
        self._show_loading_overlay("正在扫描WiFi...", "Scanning WiFi...")
        try:
            dlg = WifiConnectionDialog(self)
            self._hide_loading_overlay()
            dlg.exec_()
        except Exception as e:
            self._hide_loading_overlay()
            print(f"[WiFi] Error: {e}")

    def _open_calibration(self):
        """Start camera calibration with loading indicator."""
        self._show_loading_overlay("正在启动标定...", "Starting calibration...")
        self.start_calibration()

    def _show_settings_menu(self):
        """Popup menu with Volume control, WIFI connection, and Language toggle."""
        menu = QtWidgets.QMenu(self)

        # Set font (family, size, bold, etc.)
        font = QtGui.QFont("Arial", 18)
        font.setBold(True)
        menu.setFont(font)

        # Style via stylesheet
        menu.setStyleSheet("""
               QMenu {
                   background-color: #232323;      /* menu background */
                   color: #FFFFFF;                 /* text color */
                   border: 1px solid #444444;
               }
               QMenu::item {
                   padding: 6px 24px;              /* top/bottom, left/right */
                   background-color: transparent;
               }
               QMenu::item:selected {
                   background-color: #ffaa00;      /* hover/selected background */
                   color: #000000;                 /* hover/selected text color */
               }
               QMenu::separator {
                   height: 4px;
                   background: #555555;
                   margin: 4px 0px 4px 0px;
               }
           """)

        # Use current language setting from calibrations module
        is_chinese = calibrations.f_pinyin

        # OD overlay toggle - show action (what clicking will do)
        od_overlay_on = calibrations.f_od_overlay

        if is_chinese:
            act_replay = menu.addAction("回放录像")
            act_volume = menu.addAction("音量调节")
            act_wifi = menu.addAction("WI-FI设置")
            act_calibration = menu.addAction("相机标定")
            act_desktop = menu.addAction("换壁纸")
            act_od_overlay = menu.addAction("关闭检测叠加" if od_overlay_on else "开启检测叠加")
            act_language = menu.addAction("语言: 中文 → English")
            act_help = menu.addAction("帮助")
        else:
            act_replay = menu.addAction("Replay video")
            act_volume = menu.addAction("Volume control")
            act_wifi = menu.addAction("WIFI connection")
            act_calibration = menu.addAction("Camera calibration")
            act_desktop = menu.addAction("Change background")
            act_od_overlay = menu.addAction("Disable OD Overlay" if od_overlay_on else "Enable OD Overlay")
            act_language = menu.addAction("Language: English → 中文")
            act_help = menu.addAction("Help")

        # Position menu under the settings button
        pos = self.setting_btn.mapToGlobal(self.setting_btn.rect().topRight())
        pos += QtCore.QPoint(-100, -320)  # Adjusted for additional menu items
        chosen = menu.exec_(pos)

        if chosen is act_volume:
            dlg = VolumeControlDialog(self)
            dlg.exec_()
        elif chosen is act_wifi:
            self._open_wifi_dialog()
        elif chosen is act_calibration:
            self._open_calibration()
        elif chosen is act_desktop:
            self.change_desktop()
            self.left_monitor.show_cpu_frame(cv_img=self.start_img)
        elif chosen is act_language:
            self._toggle_language()
        elif chosen is act_od_overlay:
            calibrations.toggle_od_overlay()
        elif chosen is act_help:
            if calibrations.f_pinyin:
                title = "帮助"
                text = "版本: 1.0.0, 01/30/2026.\nmagic painter, Inc.\n电话: 181-3778-1612\nliang.ma@magicpainter.net"
            else:
                title = "Help"
                text = "Version: 1.0.0, 01/30/2026.\nProduct of magic painter, Inc.\nTel: 248-8948640\nliang.ma@magicpainter.net"
            self.show_info_popup(title, text, calibrations.f_pinyin)
        elif chosen is act_replay:
            self._handle_view_full()

    def _toggle_language(self):
        """Toggle between English and Chinese language."""
        calibrations.toggle_language()

        # Update all button icons immediately
        self._update_button_icons()

    def _update_button_icons(self):
        """Update all button icons based on current language setting."""
        use_chinese = calibrations.f_pinyin

        # Helper to create icon with active/inactive states
        def make_icon(cn_inactive, cn_active, en_inactive, en_active):
            icon = QtGui.QIcon()
            if use_chinese:
                icon.addPixmap(QtGui.QPixmap(cn_inactive), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                icon.addPixmap(QtGui.QPixmap(cn_active), QtGui.QIcon.Normal, QtGui.QIcon.On)
            else:
                icon.addPixmap(QtGui.QPixmap(en_inactive), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                icon.addPixmap(QtGui.QPixmap(en_active), QtGui.QIcon.Normal, QtGui.QIcon.On)
            return icon

        # Helper for single-state icons
        def make_single_icon(cn_path, en_path):
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(cn_path if use_chinese else en_path), QtGui.QIcon.Normal)
            return icon

        # View buttons (left/right/both court)
        self.left_btn.setIcon(make_icon(
            "Chinese_UI/left_inactive.png", "Chinese_UI/left.png",
            "UI_Button/left_court_inactive.png", "UI_Button/left_court_active.png"))
        self.right_btn.setIcon(make_icon(
            "Chinese_UI/right_inactive.png", "Chinese_UI/right.png",
            "UI_Button/right_court_inactive.png", "UI_Button/right_court_active.png"))
        self.both_btn.setIcon(make_icon(
            "Chinese_UI/both_inactive.png", "Chinese_UI/both.png",
            "UI_Button/full_court_inactive.png", "UI_Button/full_court_active.png"))

        # Mode buttons (dribble/shoot/game)
        self.dribble_btn.setIcon(make_icon(
            "Chinese_UI/dribble_inactive.png", "Chinese_UI/dribble.png",
            "UI_Button/dribble_inactive.png", "UI_Button/dribble_active.png"))
        self.shoot_btn.setIcon(make_icon(
            "Chinese_UI/shoot_inactive.png", "Chinese_UI/shoot.png",
            "UI_Button/shoot_inactive.png", "UI_Button/shoot_active.png"))
        self.game_btn.setIcon(make_icon(
            "Chinese_UI/game_inactive.png", "Chinese_UI/game.png",
            "UI_Button/game_inactive.png", "UI_Button/game_active.png"))

        # Action buttons (single state)
        self.start_btn.setIcon(make_single_icon("Chinese_UI/start.png", "UI_Button/start_active.png"))
        self.next_btn.setIcon(make_single_icon("Chinese_UI/next.png", "UI_Button/next.png"))
        self.stop_game_btn.setIcon(make_single_icon("Chinese_UI/stop_game.png", "UI_Button/stop_game_active.png"))
        self.view_full_btn.setIcon(make_single_icon("Chinese_UI/replay_active.png", "UI_Button/replay_active.png"))
        self.upload_game_btn.setIcon(make_single_icon("Chinese_UI/upload_active.png", "UI_Button/upload_active.png"))
        self.setting_btn.setIcon(make_single_icon("Chinese_UI/setting_active.png", "UI_Button/setting_active.png"))

    def show_info_popup(self, title: str, text: str, f_pinyin: bool = False):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setWindowTitle(title)
        box.setText(text)
        box.setMinimumSize(600, 400)  # 2x bigger window

        box.setStyleSheet("""
            QMessageBox { background-color: #f3f5f7; min-width: 600px; min-height: 400px; }
            QLabel { color: #111; font-size: 32px; background: #dbeafe; padding: 20px; }
            QPushButton {
                background: #dbeafe;
                color: #111;
                border: 1px solid #cfd6df;
                border-radius: 16px;
                padding: 16px 24px;
                font-size: 28px;
                min-width: 120px;
            }
            QPushButton:hover { background: #eef2f7; }
        """)
        box.exec_()

    def change_desktop(self):
        curr = self.bg_num  # example
        value = (curr + 1)
        if value == 7:
            value = 1
        img_name = f"UI_Button/startup_{value}.png"
        self.start_img = cv2.imread(img_name)
        self.bg_num = value


class DualMonitor(QtWidgets.QStackedWidget):
    """
    index 0 ¡ú QLabel   £¨CPU Â·¾¶£©
    index 1 ¡ú GLMonitor£¨GPU Â·¾¶£¬¼ûÉÏ´Î¸øÄãµÄÊµÏÖ£©
    """
    gpu_update = QtCore.pyqtSignal(object, int, int, int)  # ptr, w, h, pit

    def __init__(self,ctx_q, ctx_ready, *, side, cmd_queue,tex_queue, pbo_pool, parent=None, src_W, src_H):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.cpu_label = QtWidgets.QLabel()
        self.cpu_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cpu_label.setScaledContents(True)

        self.ctx_q = ctx_q
        self.ctx_ready = ctx_ready
        self.side = side  # ¼ÇÂ¼×óÓÒ
        self.cmd_queue = cmd_queue
        self.tex_queue = tex_queue

        self.pbo_pool = pbo_pool
        self.gpu_view = GLMonitor(self.ctx_q, self.ctx_ready, side=self.side,
                                  tex_queue=self.tex_queue, src_W=src_W, src_H=src_H)

        self.addWidget(self.cpu_label)  # 0
        self.addWidget(self.gpu_view)   # 1
        self.setCurrentIndex(0)         # ÏÈÓÃ CPU ÊÓÍ¼

        self.gpu_update.connect(
            self.gpu_view.update_from_cuda,
            QtCore.Qt.QueuedConnection  # ¹Ø¼ü£ºÅÅ¶Óµ½ GUI Ïß³Ì
        )

    def setAlignment(self, alignment):
        """Pass©\through so old code that expected a QLabel still works."""
        self.cpu_label.setAlignment(alignment)

    def setScaledContents(self, flag: bool):
        self.cpu_label.setScaledContents(flag)

    def show_cpu_frame(self, cv_img):
        """cv_img: BGR ndarray on host"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(cv_img.data, w, h, bytes_per_line,
                            QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.cpu_label.setPixmap(pix.scaled(self.size(),
                                            QtCore.Qt.KeepAspectRatio,
                                            QtCore.Qt.SmoothTransformation))
        self.setCurrentIndex(0)

    def show_cuda_frame(self, dev_ptr, w, h, pitch):
        self.gpu_update.emit(dev_ptr, w, h, pitch)  # ·¢ËÍÒì²½ÐÅºÅ
        self.setCurrentIndex(1)

    def notify_frame_ready(self):
        self.setCurrentIndex(1)
        self.gpu_view.update()

class PBOPool:
    def __init__(self, count, width, height):
        self.free_queue = queue.Queue()
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.regbufs = {}
        self.pbo_ids = []
        self._to_generate = count
        self._in_use = set()

    def _ensure_buffers(self):
        if self._to_generate == 0:
            return
        ids = GL.glGenBuffers(self._to_generate)
        if isinstance(ids, int):
            ids = [ids]
        for pbo_id in ids:
            if pbo_id == 0:
                continue
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo_id)
            GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER,
                            self.width * self.height * 4,
                            None,
                            GL.GL_DYNAMIC_DRAW)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            self.free_queue.put(pbo_id)
            self.pbo_ids.append(pbo_id)
        self._to_generate = 0


    def acquire(self, timeout=None):
        self._ensure_buffers()
        # Avoid handing out the same PBO twice if someone double-released.
        # Loop a few times to skip duplicates if they exist.
        tries = 0
        while True:
            pbo_id = self.free_queue.get(timeout=timeout)
            if pbo_id == 0:
                raise RuntimeError("PBOPool handed out invalid PBO id 0 – GL context missing during allocation")
            with self.lock:
                if pbo_id not in self._in_use:
                    self._in_use.add(pbo_id)
                    break
            tries += 1
            if tries >= 4:
                # Something is badly wrong; don't spin forever.
                raise RuntimeError(f"PBOPool corruption: PBO {pbo_id} appears already in use repeatedly")


        print(f"[PBOPool] Acquired PBO {pbo_id}, remaining: {self.free_queue.qsize()}")

        return pbo_id

    def release(self, pbo_id):
        if not pbo_id or pbo_id == 0:
            return
        with self.lock:
            if pbo_id not in self._in_use:
                # Ignore double-release; it will otherwise corrupt the pool.
                # print(f"[PBOPool] WARNING: double-release or unknown PBO {pbo_id}, ignoring")
                return
            self._in_use.remove(pbo_id)
        self.free_queue.put(pbo_id)
        # print(f"[PBOPool] Released PBO {pbo_id}, available: {self.free_queue.qsize()}")



def run_gui(cmd_queue, tex_queue, res_queue, run_event,ctx_q, ctx_ready,cfg ,gui2infer_queue, pbo_pool,
            infer_frame_queue, error_q, engine):
    #os.sched_setaffinity(0, [0])
    #os.nice(1)
    import sys
    from PyQt5 import QtCore
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)
    win = TrainingAIApp(ctx_q, ctx_ready,cfg,
                         cmd_queue=cmd_queue,
                        tex_queue = tex_queue,
                         res_queue=res_queue,
                         run_event=run_event,
                        gui2infer_queue=gui2infer_queue,
                        pbo_pool=pbo_pool,
                        infer_frame_queue=infer_frame_queue,
                        error_q=error_q,
                        engine=engine
                        )
    win.show()
    sys.exit(app.exec_())




if __name__ == '__main__':
    run_gui()
