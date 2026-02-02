from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path
import os
import re
import subprocess
import hashlib
import json
import socket
import tempfile
import calibrations

# Keep backward compatibility
f_pinyin = calibrations.f_pinyin

def tr(zh: str, en: str) -> str:
    """Tiny i18n helper controlled by global f_pinyin (reads dynamically)."""
    return zh if calibrations.f_pinyin else en


class VideoPlayerDialog(QtWidgets.QDialog):
    """Video player with mpv and control buttons at bottom of screen."""

    CONTROL_HEIGHT = 90

    def __init__(self, video_path: Path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.mpv_process = None
        self.ipc_socket_path = None
        self._paused = False
        self._check_timer = None
        self._hidden_parent = None

        # Hide parent window so mpv is visible
        if parent is not None:
            self._hidden_parent = parent.window()
            self._hidden_parent.hide()

        # Get screen geometry
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()

        # Control panel at bottom of screen
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        # Position at right edge, arranged vertically
        btn_panel_width = 220
        btn_panel_height = 400  # Height for 4 stacked buttons
        bottom_margin = 80  # Space for mpv OSC progress bar
        self.setGeometry(self.screen_width - btn_panel_width,
                         self.screen_height - btn_panel_height - bottom_margin,
                         btn_panel_width, btn_panel_height)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QDialog {
                background: transparent;
            }
            QPushButton {
                background-color: rgba(30, 144, 255, 180);
                color: white;
                border: none;
                border-radius: 12px;
                font-family: "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Microsoft YaHei", sans-serif;
                font-size: 36px;
                font-weight: bold;
                padding: 15px 33px;
                min-width: 165px;
            }
            QPushButton:hover {
                background-color: rgba(58, 160, 255, 200);
            }
            QPushButton:pressed {
                background-color: rgba(0, 102, 204, 200);
            }
        """)

        control_layout = QtWidgets.QVBoxLayout(self)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(10)

        # Backward 10s button
        self.btn_backward = QtWidgets.QPushButton("◀◀ 10")
        self.btn_backward.clicked.connect(self._seek_backward)
        control_layout.addWidget(self.btn_backward)

        # Play/Pause button
        self.btn_playpause = QtWidgets.QPushButton(tr("暂停", "Pause"))
        self.btn_playpause.clicked.connect(self._toggle_playpause)
        control_layout.addWidget(self.btn_playpause)

        # Forward 10s button
        self.btn_forward = QtWidgets.QPushButton("10 ▶▶")
        self.btn_forward.clicked.connect(self._seek_forward)
        control_layout.addWidget(self.btn_forward)

        # Close button
        self.btn_close = QtWidgets.QPushButton(tr("关闭", "Close"))
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: rgba(220, 53, 69, 180);
                color: white;
                border: none;
                border-radius: 12px;
                font-family: "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Microsoft YaHei", sans-serif;
                font-size: 36px;
                font-weight: bold;
                padding: 15px 33px;
                min-width: 165px;
            }
            QPushButton:hover {
                background-color: rgba(224, 69, 85, 200);
            }
            QPushButton:pressed {
                background-color: rgba(176, 42, 55, 200);
            }
        """)
        self.btn_close.clicked.connect(self._close_player)
        control_layout.addWidget(self.btn_close)

        # Start mpv in fullscreen
        self._start_mpv()

    def _start_mpv(self):
        """Start mpv in fullscreen mode."""
        self.ipc_socket_path = tempfile.mktemp(prefix="mpv_ipc_", suffix=".sock")

        cmd = [
            "mpv",
            "--fullscreen",
            "--no-border",
            "--ontop=no",  # Don't let mpv stay on top
            f"--input-ipc-server={self.ipc_socket_path}",
            "--keep-open=yes",
            "--osc=yes",
            str(self.video_path)
        ]

        try:
            self.mpv_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Timer to check if mpv is still running
            self._check_timer = QtCore.QTimer(self)
            self._check_timer.timeout.connect(self._check_mpv_status)
            self._check_timer.start(500)

            # Raise control panel once after mpv starts
            QtCore.QTimer.singleShot(500, self._raise_controls)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                tr("错误", "Error"),
                tr(f"无法启动视频播放器: {e}", f"Failed to start video player: {e}")
            )
            self.reject()

    def _raise_controls(self):
        """Raise control panel to top."""
        self.show()
        self.raise_()
        self.activateWindow()

    def _check_mpv_status(self):
        """Check if mpv is still running, close dialog if not."""
        if self.mpv_process is None:
            self._close_player()
            return

        ret = self.mpv_process.poll()
        if ret is not None:
            self.mpv_process = None
            self._close_player()
        else:
            # Keep control panel above mpv
            self._raise_controls()

    def _send_mpv_command(self, command: list):
        """Send a command to mpv via IPC socket."""
        if not self.ipc_socket_path:
            return None

        # Wait a bit for socket to be ready on first command
        for _ in range(10):
            if os.path.exists(self.ipc_socket_path):
                break
            import time
            time.sleep(0.1)

        if not os.path.exists(self.ipc_socket_path):
            return None

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(self.ipc_socket_path)

            msg = json.dumps({"command": command}) + "\n"
            sock.sendall(msg.encode('utf-8'))

            # Read response
            response = sock.recv(4096).decode('utf-8')
            sock.close()

            return json.loads(response) if response else None
        except Exception:
            return None

    def _seek_backward(self):
        """Seek backward 10 seconds."""
        self._send_mpv_command(["seek", -10, "relative"])

    def _seek_forward(self):
        """Seek forward 10 seconds."""
        self._send_mpv_command(["seek", 10, "relative"])

    def _toggle_playpause(self):
        """Toggle play/pause state."""
        self._send_mpv_command(["cycle", "pause"])
        self._paused = not self._paused
        if self._paused:
            self.btn_playpause.setText(tr("播放", "Play"))
        else:
            self.btn_playpause.setText(tr("暂停", "Pause"))

    def _close_player(self):
        """Close the video player."""
        # Stop the check timer
        if self._check_timer:
            self._check_timer.stop()
            self._check_timer = None
        self._stop_mpv()
        # Restore parent window
        if self._hidden_parent is not None:
            self._hidden_parent.showFullScreen()
            self._hidden_parent = None
        self.accept()

    def _stop_mpv(self):
        """Stop mpv process and cleanup."""
        if self.mpv_process:
            try:
                # Try graceful quit first
                self._send_mpv_command(["quit"])
                self.mpv_process.wait(timeout=2)
            except:
                # Force kill if needed
                try:
                    self.mpv_process.terminate()
                    self.mpv_process.wait(timeout=1)
                except:
                    try:
                        self.mpv_process.kill()
                    except:
                        pass

            self.mpv_process = None

        # Cleanup IPC socket
        if self.ipc_socket_path and os.path.exists(self.ipc_socket_path):
            try:
                os.remove(self.ipc_socket_path)
            except:
                pass
            self.ipc_socket_path = None

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self._close_player()
        elif key == QtCore.Qt.Key_Space:
            self._toggle_playpause()
        elif key == QtCore.Qt.Key_Left:
            self._seek_backward()
        elif key == QtCore.Qt.Key_Right:
            self._seek_forward()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Ensure mpv is stopped when dialog is closed."""
        if self._check_timer:
            self._check_timer.stop()
            self._check_timer = None
        self._stop_mpv()
        # Restore parent window
        if self._hidden_parent is not None:
            self._hidden_parent.showFullScreen()
            self._hidden_parent = None
        super().closeEvent(event)


class _ThumbSignals(QtCore.QObject):
    done = QtCore.pyqtSignal(str, str)   # (video_path, png_path)
    fail = QtCore.pyqtSignal(str, str)   # (video_path, reason)


class _ThumbMultiSignals(QtCore.QObject):
    done = QtCore.pyqtSignal(str, object)   # (video_path, list of png_paths)
    fail = QtCore.pyqtSignal(str, str)      # (video_path, reason)


class _ThumbJob(QtCore.QRunnable):
    def __init__(self, video_path: str, cache_png: str, icon_w: int, icon_h: int):
        super().__init__()
        self.video_path = video_path
        self.cache_png = cache_png
        self.icon_w = icon_w
        self.icon_h = icon_h
        self.signals = _ThumbSignals()

    def run(self):
        try:
            _ensure_thumbnail(self.video_path, self.cache_png, self.icon_w, self.icon_h)
            # DO NOT create QPixmap here (worker thread)
            self.signals.done.emit(self.video_path, self.cache_png)
        except Exception as e:
            self.signals.fail.emit(self.video_path, str(e))


class _ThumbMultiJob(QtCore.QRunnable):
    """Generate multiple thumbnails at different timestamps for animated preview."""
    def __init__(self, video_path: str, cache_dir: str, icon_w: int, icon_h: int,
                 num_frames: int = 6, signals: _ThumbMultiSignals = None,
                 cancel_check: callable = None):
        super().__init__()
        self.setAutoDelete(False)  # prevent premature deletion before signal emission
        self.video_path = video_path
        self.cache_dir = cache_dir
        self.icon_w = icon_w
        self.icon_h = icon_h
        self.num_frames = num_frames
        # Use provided signals or create new one (signals should be created on main thread)
        self.signals = signals if signals else _ThumbMultiSignals()
        self._cancel_check = cancel_check  # callable returning True if job should abort

    def run(self):
        try:
            # Check cancellation before starting
            if self._cancel_check and self._cancel_check():
                return
            png_paths = _ensure_multi_thumbnails(
                self.video_path, self.cache_dir, self.icon_w, self.icon_h, self.num_frames,
                cancel_check=self._cancel_check
            )
            # Check cancellation before emitting (dialog may have closed)
            if self._cancel_check and self._cancel_check():
                return
            if png_paths:  # Only emit if we got results
                self.signals.done.emit(self.video_path, png_paths)
        except Exception as e:
            if self._cancel_check and self._cancel_check():
                return  # Don't emit errors for cancelled jobs
            self.signals.fail.emit(self.video_path, str(e))



def _which(cmd: str) -> bool:
    try:
        subprocess.check_output(["bash", "-lc", f"command -v {cmd}"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _safe_name_key(video_path: str, mtime_ns: int) -> str:
    # Stable + invalidates when file changes
    h = hashlib.sha1(f"{video_path}|{mtime_ns}".encode("utf-8")).hexdigest()
    return h


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(out)
    except Exception:
        return 30.0  # fallback to 30 seconds


def _ensure_thumbnail(video_path: str, out_png: str, icon_w: int, icon_h: int):
    vp = Path(video_path)
    if not vp.exists():
        raise FileNotFoundError(video_path)

    os.makedirs(str(Path(out_png).parent), exist_ok=True)

    # If thumbnail exists, it's already keyed by mtime_ns, so it's valid.
    if Path(out_png).exists():
        return

    # Prefer ffmpegthumbnailer (usually faster)
    if _which("ffmpegthumbnailer"):
        # -t 10% grabs a frame not at the very start (often black)
        # -s 0 keeps aspect ratio, chooses reasonable size; we still scale in list view.
        cmd = [
            "ffmpegthumbnailer",
            "-i", video_path,
            "-o", out_png,
            "-t", "10%",      # position
            "-s", "0",        # auto size
            "-q", "8"         # quality (png ignores, but fine)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    # Fallback to ffmpeg
    if _which("ffmpeg"):
        # Pick 5 seconds in; if video shorter, ffmpeg will still usually produce something.
        # Scale to something a bit larger than icon size for sharper downscale.
        scale_w = max(icon_w * 2, 256)
        cmd = [
            "ffmpeg", "-y",
            "-ss", "00:00:05",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale={scale_w}:-1",
            out_png
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    raise RuntimeError(tr("未找到 ffmpegthumbnailer 或 ffmpeg，请安装其中之一。", "Neither ffmpegthumbnailer nor ffmpeg found. Install one of them."))


def _ensure_multi_thumbnails(video_path: str, cache_dir: str, icon_w: int, icon_h: int,
                             num_frames: int = 6, cancel_check: callable = None) -> list:
    """Generate multiple thumbnails at different timestamps for animated preview.

    Args:
        cancel_check: Optional callable that returns True if generation should abort.
    """
    vp = Path(video_path)
    if not vp.exists():
        raise FileNotFoundError(video_path)

    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique key for this video
    mtime_ns = vp.stat().st_mtime_ns
    base_key = _safe_name_key(str(vp), mtime_ns)

    png_paths = []
    # Generate frames at 10%, 25%, 40%, 55%, 70%, 85% of duration
    percentages = [10, 25, 40, 55, 70, 85][:num_frames]

    # Check if all frames already exist
    all_exist = True
    for i, pct in enumerate(percentages):
        out_png = os.path.join(cache_dir, f"{base_key}_f{i}.png")
        png_paths.append(out_png)
        if not Path(out_png).exists():
            all_exist = False

    if all_exist:
        return png_paths

    # Check cancellation before expensive ffprobe call
    if cancel_check and cancel_check():
        return []

    # Get video duration for timestamp calculation
    duration = _get_video_duration(video_path)

    # Generate missing frames
    scale_w = max(icon_w * 2, 256)
    for i, pct in enumerate(percentages):
        # Check cancellation before each ffmpeg call
        if cancel_check and cancel_check():
            return []

        out_png = png_paths[i]
        if Path(out_png).exists():
            continue

        timestamp = duration * pct / 100.0
        # Format as HH:MM:SS.mmm
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = timestamp % 60
        ts_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        if _which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y",
                "-ss", ts_str,
                "-i", video_path,
                "-frames:v", "1",
                "-vf", f"scale={scale_w}:-1",
                out_png
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                # If specific timestamp fails, try using the first frame thumbnail
                if i == 0:
                    raise
                # Copy first frame as fallback
                if png_paths[0] and Path(png_paths[0]).exists():
                    import shutil
                    shutil.copy(png_paths[0], out_png)

    return png_paths


class DotCheckDelegate(QtWidgets.QStyledItemDelegate):
    """Draw a white checkbox with a dot when checked (touch-friendly, theme-proof)."""

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex):
        opt = QtWidgets.QStyleOptionViewItem(option)

        # Remove default checkbox drawing so we can render our own
        has_check = bool(opt.features & QtWidgets.QStyleOptionViewItem.HasCheckIndicator)
        check_state = index.data(QtCore.Qt.CheckStateRole)
        if has_check:
            opt.features &= ~QtWidgets.QStyleOptionViewItem.HasCheckIndicator

        # Draw icon/text normally
        super().paint(painter, opt, index)

        if not has_check:
            return

        # Custom checkbox rect (fixed size) in the item area (top-left)
        m = 4
        size = 18
        r = QtCore.QRect(opt.rect.left() + m, opt.rect.top() + m, size, size)

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # White background + gray border
        painter.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        painter.drawRoundedRect(r, 6, 6)

        # Dot when checked
        if check_state == QtCore.Qt.Checked:
            dot = QtCore.QRectF(r.center().x() - 5, r.center().y() - 5, 10, 10)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 200)))
            painter.drawEllipse(dot)

        painter.restore()

class VideoPickerDialog(QtWidgets.QDialog):
    VIDEO_EXTS = (".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v")

    ROLE_BASE_PIX = QtCore.Qt.UserRole + 1   # QPixmap base (no overlay) - first frame
    ROLE_FRAME_PIXMAPS = QtCore.Qt.UserRole + 2  # list of QPixmaps for animation
    ROLE_FRAME_INDEX = QtCore.Qt.UserRole + 3    # current frame index for animation

    def __init__(self, f_upload, video_dir: Path, parent=None):
        super().__init__(parent)

        self.f_upload = bool(f_upload)
        self._hover_item = None  # currently hovered item for animation
        self._anim_timer = None  # QTimer for frame cycling

        self.setWindowTitle(tr("选择比赛录像", "Select a video to operate"))
        self.resize(900, 550)
        self.setStyleSheet("""
                            QDialog {
                                background: #f3f5f7;
                            }
                            QLabel {
                                color: #ffffff;
                                font-size: 16px;
                                background: #1E90FF;
                            }
                            QListWidget {
                                background: #ffffff;
                                color: #111;
                                border: 1px solid #cfd6df;
                                border-radius: 10px;
                            }
                            QListWidget::item {
                                padding: 5px;
                                height: 100px;
                            }
                            QListWidget::item:selected {
                                background: #dbeafe;
                                color: #ffffff;
                            }
                            QLineEdit {
                                background: #1E90FF;
                                color: #111;
                                border: 1px solid #cfd6df;
                                border-radius: 10px;
                                padding: 10px;
                                font-size: 16px;
                            }
                            QPushButton {
                                font-size: 16px;
                                padding: 10px 14px;
                                background: #1E90FF;
                            }
                            QListWidget QScrollBar:vertical {
                            background: #dbeafe;
                            }
                            QListWidget QScrollBar::handle:vertical {
                                background: #60A5FA;
                            }
                        """)

        self.video_dir = Path(video_dir)
        self._selected_paths = []  # list[Path]

        # thumbnail cache
        self.cache_dir = Path.home() / ".cache" / "mp_video_thumbs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # thread pool for thumbnail generation
        self._jobs = {}
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self._path_to_item = {}   # video_path(str) -> QListWidgetItem

        # Shared signals object for multi-thumbnail jobs (prevents GC issues)
        self._multi_thumb_signals = _ThumbMultiSignals(self)
        self._multi_thumb_signals.done.connect(self._on_multi_thumb_done)
        self._multi_thumb_signals.fail.connect(self._on_thumb_fail)


        self._closed = False
        self._filter_prefix = ""  # Current filter prefix for video names

        main = QtWidgets.QVBoxLayout(self)

        # Filter buttons row
        filter_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton(tr("全部", "All"))
        self.btn_dribble = QtWidgets.QPushButton(tr("运球", "Dribble"))
        self.btn_shooting = QtWidgets.QPushButton(tr("投篮", "Shooting"))
        self.btn_game = QtWidgets.QPushButton(tr("比赛", "Game"))
        self.btn_cloud = QtWidgets.QPushButton(tr("云端", "Cloud"))

        for btn in (self.btn_all, self.btn_dribble, self.btn_shooting, self.btn_game, self.btn_cloud):
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton { background: #3B82F6; color: white; border-radius: 5px; padding: 8px 16px; }
                QPushButton:checked { background: #1D4ED8; border: 2px solid #FCD34D; }
                QPushButton:hover { background: #2563EB; }
            """)
            filter_row.addWidget(btn)

        self.btn_all.setChecked(True)
        self.btn_all.clicked.connect(lambda checked: self._set_filter(""))
        self.btn_dribble.clicked.connect(lambda checked: self._set_filter(tr("运球", "dribble")))
        self.btn_shooting.clicked.connect(lambda checked: self._set_filter(tr("投篮", "shooting")))
        self.btn_game.clicked.connect(lambda checked: self._set_filter(tr("比赛", "game")))
        self.btn_cloud.clicked.connect(lambda checked: self._set_filter("cloud"))

        filter_row.addStretch(1)
        main.addLayout(filter_row)

        self.listw = QtWidgets.QListWidget()
        # Touch-friendly list
        self.listw.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)  # selection not needed anymore
        self.listw.setUniformItemSizes(True)

        # Hide the built-in checkbox indicator (it can render as a black square on Jetson themes)
        # We'll draw our own white box + dot on top of the thumbnail icon instead.
        self.listw.setStyleSheet("QListView::indicator { width: 0px; height: 0px; }")

        # Custom checkbox rendering (white box + dot) for dark themes
        # self.listw.setItemDelegate(DotCheckDelegate(self.listw))  # not reliable in IconMode
        # Make checkbox easier to tap
        # Selection highlight is disabled (touch uses checkbox overlay)
        if self.f_upload:
            # Touch: tap anywhere on the item to toggle the checkbox
            self.listw.itemClicked.connect(self._toggle_item_check)
        else:
            self.listw.itemClicked.connect(self._set_single_checked)   # <-- add this
            self.listw.itemDoubleClicked.connect(self._accept_current)

        # icon grid view
        self.listw.setViewMode(QtWidgets.QListView.IconMode)
        self.listw.setResizeMode(QtWidgets.QListView.Adjust)
        self.listw.setMovement(QtWidgets.QListView.Static)
        self.listw.setSpacing(12)
        self.listw.setWrapping(True)
        self.listw.setWordWrap(True)

        self.icon_w = 160
        self.icon_h = 90
        self.listw.setIconSize(QtCore.QSize(self.icon_w, self.icon_h))
        self.listw.setGridSize(QtCore.QSize(self.icon_w + 90, self.icon_h + 85))

        # Enable mouse tracking for hover events
        self.listw.setMouseTracking(True)
        self.listw.viewport().setMouseTracking(True)
        self.listw.viewport().installEventFilter(self)

        # Animation timer for cycling frames on hover
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setInterval(400)  # 400ms per frame = ~2.5 FPS preview
        self._anim_timer.timeout.connect(self._on_anim_tick)

        main.addWidget(self.listw, 1)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn_play = QtWidgets.QPushButton(tr("确定", "Confirm"))
        self.btn_cancel = QtWidgets.QPushButton(tr("取消", "Cancel"))
        self.btn_play.clicked.connect(self._accept_current)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_play)
        btns.addWidget(self.btn_cancel)
        main.addLayout(btns)

        # placeholder icon while thumbnails generate
        self._placeholder_icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)

        QtCore.QTimer.singleShot(0, self._populate)


    def closeEvent(self, e):
        self._closed = True
        if self._anim_timer:
            self._anim_timer.stop()
        # Clear pending jobs from the pool (jobs not yet started)
        self.pool.clear()
        # Clear job references so running jobs can detect cancellation via _is_cancelled()
        self._jobs.clear()
        super().closeEvent(e)

    def _is_cancelled(self) -> bool:
        """Check if dialog has been closed (used by thumbnail jobs to abort early)."""
        return self._closed

    def eventFilter(self, obj, event):
        """Handle hover events on list items for animated preview."""
        if obj is self.listw.viewport():
            if event.type() == QtCore.QEvent.MouseMove:
                pos = event.pos()
                item = self.listw.itemAt(pos)
                if item != self._hover_item:
                    self._stop_animation()
                    self._hover_item = item
                    if item is not None:
                        self._start_animation(item)
            elif event.type() == QtCore.QEvent.Leave:
                self._stop_animation()
                self._hover_item = None
        return super().eventFilter(obj, event)

    def _start_animation(self, item: QtWidgets.QListWidgetItem):
        """Start cycling through frames for the hovered item."""
        pixmaps = item.data(self.ROLE_FRAME_PIXMAPS)
        if not pixmaps or len(pixmaps) < 2:
            return  # No animation if only one frame
        item.setData(self.ROLE_FRAME_INDEX, 0)
        self._anim_timer.start()

    def _stop_animation(self):
        """Stop animation and reset to first frame."""
        self._anim_timer.stop()
        if self._hover_item is not None:
            # Reset to first frame
            pixmaps = self._hover_item.data(self.ROLE_FRAME_PIXMAPS)
            if pixmaps:
                self._hover_item.setData(self.ROLE_FRAME_INDEX, 0)
                self._set_item_base_pixmap(self._hover_item, pixmaps[0])

    def _on_anim_tick(self):
        """Advance to next frame in the animation."""
        if self._hover_item is None:
            self._anim_timer.stop()
            return

        pixmaps = self._hover_item.data(self.ROLE_FRAME_PIXMAPS)
        if not pixmaps:
            self._anim_timer.stop()
            return

        # Get current index and advance
        idx = self._hover_item.data(self.ROLE_FRAME_INDEX) or 0
        idx = (idx + 1) % len(pixmaps)
        self._hover_item.setData(self.ROLE_FRAME_INDEX, idx)

        # Update the icon with the new frame
        self._set_item_base_pixmap(self._hover_item, pixmaps[idx])

    def _set_single_checked(self, item: QtWidgets.QListWidgetItem):
        # Replay mode: exactly one "selected" item (shows dot)
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            new_state = QtCore.Qt.Checked if it is item else QtCore.Qt.Unchecked
            if it.checkState() != new_state:
                it.setCheckState(new_state)
                self._refresh_item_icon(it)


    def selected_paths(self) -> list[Path]:
        """For upload mode: return all selected paths (may be empty)."""
        return list(self._selected_paths)

    def selected_path(self) -> Path | None:
        """Backward-compatible: return the first selected path (or None)."""
        return self._selected_paths[0] if self._selected_paths else None

    def _set_filter(self, prefix: str):
        """Set the filter prefix and refresh the video list."""
        self._filter_prefix = prefix
        # Update button states - use blockSignals to prevent recursive calls
        for btn in (self.btn_all, self.btn_dribble, self.btn_shooting, self.btn_game, self.btn_cloud):
            btn.blockSignals(True)
        self.btn_all.setChecked(prefix == "")
        self.btn_dribble.setChecked(prefix in ("运球", "dribble"))
        self.btn_shooting.setChecked(prefix in ("投篮", "shooting"))
        self.btn_game.setChecked(prefix in ("比赛", "game"))
        self.btn_cloud.setChecked(prefix == "cloud")
        for btn in (self.btn_all, self.btn_dribble, self.btn_shooting, self.btn_game, self.btn_cloud):
            btn.blockSignals(False)
        # Stop animation and clear hover state before repopulating
        self._stop_animation()
        self._hover_item = None
        # Refresh the list
        self._populate()

    def _videos(self):
        if not self.video_dir.exists():
            return []

        # Minimum file size for replay mode (100 MB) - filter out corrupted/failed recordings
        MIN_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

        files = []
        for p in self.video_dir.iterdir():
            if p.is_file() and p.suffix.lower() in self.VIDEO_EXTS:
                # Apply name filter if set
                if self._filter_prefix:
                    if not p.name.lower().startswith(self._filter_prefix.lower()):
                        continue
                # Apply size filter for replay mode (skip small/corrupted files)
                if not self.f_upload:
                    try:
                        if p.stat().st_size < MIN_SIZE_BYTES:
                            continue
                    except OSError:
                        continue  # Skip files we can't stat
                files.append(p)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files

    def _thumb_cache_path(self, p: Path) -> Path:
        # cache key includes mtime_ns so it auto-invalidates when file changes
        mtime_ns = p.stat().st_mtime_ns
        key = _safe_name_key(str(p), mtime_ns)
        return self.cache_dir / f"{key}.png"

    def _make_icon_pixmap_with_overlay(self, base_pix: QtGui.QPixmap, checked: bool) -> QtGui.QPixmap:
        """Return a copy of base_pix with a white checkbox + optional dot drawn on top-left."""
        # Ensure correct size for icon
        pix = base_pix
        if pix.width() != self.icon_w or pix.height() != self.icon_h:
            pix = pix.scaled(self.icon_w, self.icon_h, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
            # Crop to exact icon size
            x = max(0, (pix.width() - self.icon_w) // 2)
            y = max(0, (pix.height() - self.icon_h) // 2)
            pix = pix.copy(x, y, self.icon_w, self.icon_h)

        out = pix.copy()
        painter = QtGui.QPainter(out)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Checkbox rect
        m = 4
        size = 18
        r = QtCore.QRect(m, m, size, size)

        painter.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 240)))
        painter.drawRoundedRect(r, 6, 6)

        if checked:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 200)))
            cx = r.center().x()
            cy = r.center().y()
            painter.drawEllipse(QtCore.QPointF(cx, cy), 4.0, 4.0)

        painter.end()
        return out

    def _set_item_base_pixmap(self, item: QtWidgets.QListWidgetItem, base_pix: QtGui.QPixmap):
        """Store base pixmap (without overlay) in item data and refresh icon."""
        item.setData(self.ROLE_BASE_PIX, base_pix)
        self._refresh_item_icon(item)

    def _refresh_item_icon(self, item: QtWidgets.QListWidgetItem):
        base = item.data(self.ROLE_BASE_PIX)
        if isinstance(base, QtGui.QPixmap) and not base.isNull():
            checked = (item.checkState() == QtCore.Qt.Checked)
            out = self._make_icon_pixmap_with_overlay(base, checked)
            item.setIcon(QtGui.QIcon(out))

    

    def _toggle_item_check(self, item: QtWidgets.QListWidgetItem):
        # Toggle check state on tap (better for touchscreen)
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)
        # Refresh the icon overlay so the dot/box changes immediately
        self._refresh_item_icon(item)

    def checked_paths(self) -> list[Path]:
        out = []
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                out.append(Path(it.data(QtCore.Qt.UserRole)))
        return out

    def _populate(self):
        # Stop any running animation before clearing
        self._stop_animation()
        self._hover_item = None

        # Cancel pending thumbnail jobs before clearing the list
        self.pool.clear()
        self._jobs.clear()
        self._path_to_item.clear()
        self.listw.clear()

        vids = self._videos()

        for p in vids:
            ts = p.stat().st_mtime
            caption = f"{p.stem}\n{self._fmt_mtime(ts)}"

            item = QtWidgets.QListWidgetItem(self._placeholder_icon, caption)
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            item.setToolTip(str(p))
            item.setData(QtCore.Qt.UserRole, str(p))

            # enable checkbox
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.listw.addItem(item)

            # Store a base pixmap (thumbnail without overlay) and paint our overlay checkbox
            base_pix = self._placeholder_icon.pixmap(self.icon_w, self.icon_h)
            self._set_item_base_pixmap(item, base_pix)

            video_path = str(p)
            self._path_to_item[video_path] = item

            # Schedule multi-thumbnail job for animated preview (use shared signals)
            # Pass _is_cancelled so jobs can abort early when dialog closes
            job = _ThumbMultiJob(video_path, str(self.cache_dir), self.icon_w, self.icon_h,
                                 num_frames=6, signals=self._multi_thumb_signals,
                                 cancel_check=self._is_cancelled)

            # IMPORTANT: keep job alive until done/fail
            self._jobs[video_path] = job
            self.pool.start(job)

        if self.listw.count() > 0:
            self.listw.setCurrentRow(0)

        # Process events to keep UI responsive
        QtWidgets.QApplication.processEvents()


  
    @QtCore.pyqtSlot(str, str)
    def _on_thumb_done(self, video_path: str, png_path: str):
        if self._closed:
            return

        item = self._path_to_item.get(video_path)
        if item is None:
            return

        pix = QtGui.QPixmap(png_path)
        if pix.isNull():
            print(f"[Thumb] QPixmap failed to load: {png_path}")
            return

                # Store thumbnail as base pixmap and repaint overlay checkbox
        self._set_item_base_pixmap(item, pix)
        self._jobs.pop(video_path, None)

    @QtCore.pyqtSlot(str, object)
    def _on_multi_thumb_done(self, video_path: str, png_paths):
        """Handle completion of multi-frame thumbnail generation."""
        if self._closed:
            return

        item = self._path_to_item.get(video_path)
        if item is None:
            return

        # Load all pixmaps
        pixmaps = []
        for png_path in png_paths:
            pix = QtGui.QPixmap(png_path)
            if not pix.isNull():
                pixmaps.append(pix)

        if not pixmaps:
            print(f"[Thumb] No valid pixmaps loaded for: {video_path}")
            self._jobs.pop(video_path, None)
            return

        # Store frame pixmaps for animation
        item.setData(self.ROLE_FRAME_PIXMAPS, pixmaps)
        item.setData(self.ROLE_FRAME_INDEX, 0)

        # Use first frame as base pixmap
        self._set_item_base_pixmap(item, pixmaps[0])
        self._jobs.pop(video_path, None)



    @QtCore.pyqtSlot(str, str)
    def _on_thumb_fail(self, video_path: str, reason: str):
        print(f"[Thumb FAIL] {video_path}\n  reason: {reason}")
        self._jobs.pop(video_path, None)


    def _apply_filter(self, text: str):
        text = (text or "").strip().lower()
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            path_str = (it.data(QtCore.Qt.UserRole) or "")
            name = Path(path_str).name.lower()
            it.setHidden(bool(text) and (text not in name))

    def _accept_current(self):
        if self.f_upload:
            paths = self.checked_paths()
            if not paths:
                QtWidgets.QMessageBox.information(
                    self,
                    tr("未选择", "No selection"),
                    tr("请勾选一个或多个视频。", "Please check one or more videos.")
                )
                return
            self._selected_paths = paths
            self.accept()
            return
        else:
            paths = self.checked_paths()
            if paths:
                self._selected_paths = [paths[0]]
                self.accept()
                return

        it = self.listw.currentItem()
        if not it:
            QtWidgets.QMessageBox.information(
                self,
                tr("未选择", "No selection"),
                tr("请选择一个视频。", "Please select a video.")
            )
            return
        self._selected_paths = [Path(it.data(QtCore.Qt.UserRole))]
        self.accept()



    @staticmethod
    def _fmt_mtime(ts: float) -> str:
        dt = QtCore.QDateTime.fromSecsSinceEpoch(int(ts))
        return dt.toString("yyyy-MM-dd HH:mm:ss")
    
class VolumeControlDialog(QtWidgets.QDialog): 
    """ Simple system volume control dialog using pactl. On Linux, this requires PulseAudio / PipeWire with pactl available. """ 
    def __init__(self, parent=None): 
        super().__init__(parent) 
        self.setWindowTitle(tr("音量控制", "Volume control")) 
        self.setModal(True)
        self.setMinimumSize(900, 100)
        self.setStyleSheet("""
                    QDialog {
                        background: #f3f5f7;
                    }
                    QLabel {
                        color: #111;
                        font-size: 28px;
                        font-weight: 600;
                        background: #d9dde3;
                    }
                    QSlider::groove:horizontal {
                        height: 18px;
                        background: #d9dde3;
                        border-radius: 9px;
                    }
                    QSlider::sub-page:horizontal {
                        background: #6aa9ff;
                        border-radius: 9px;
                    }
                    QSlider::add-page:horizontal {
                        background: #d9dde3;
                        border-radius: 9px;
                    }
                    QSlider::handle:horizontal {
                        width: 34px;
                        height: 34px;
                        margin: -10px 0;   /* centers handle on groove */
                        border-radius: 17px;
                        background: #ffffff;
                        border: 2px solid #9aa4b2;
                    }
                """)

        layout = QtWidgets.QVBoxLayout(self) 
        self.label = QtWidgets.QLabel(tr("音量大小： -- %", "Volume: -- %")) 
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal) 
        self.slider.setRange(0, 100) 
        layout.addWidget(self.label) 
        layout.addWidget(self.slider) 
        # Try to read current volume, fall back to 50% 
        self.slider.setValue(self._get_system_volume()) 
        self._update_label(self.slider.value()) 
        self.slider.valueChanged.connect(self._on_value_changed) 
    
    def _update_label(self, value: int): 
        self.label.setText(f"{tr('音量', 'Volume')}： {value} %") 
    
    def _on_value_changed(self, value: int): 
        self._update_label(value) 
        self._set_system_volume(value) 
    
    def _get_system_volume(self) -> int: 
        """Read volume from pactl, return 0–100, default 50 if fails.""" 
        import re, subprocess 
        try: 
            out = subprocess.check_output( "pactl get-sink-volume @DEFAULT_SINK@", shell=True, stderr=subprocess.DEVNULL, ) 
            m = re.search(rb"(\d+)%", out) 
            if m: 
                return int(m.group(1)) 
        except Exception as e: 
                print(f"[VolumeControl] get volume failed: {e}") 
        return 50 
    
    def _set_system_volume(self, percent: int):
        """Set system volume via pactl. Requires pactl in container/host.""" 
        import subprocess 
        try: 
            subprocess.run( f"pactl set-sink-volume @DEFAULT_SINK@ {percent}%", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, ) 
        except Exception as e:
            print(f"[VolumeControl] set volume failed: {e}")



class _WifiConnectThread(QtCore.QThread):
    """Background thread to run nmcli wifi connect without blocking UI."""
    finished_signal = QtCore.pyqtSignal(bool, str, str)  # (success, ssid, detail)

    def __init__(self, cmd: list, ssid: str, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.ssid = ssid

    def run(self):
        import subprocess
        try:
            result = subprocess.run(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                self.finished_signal.emit(True, self.ssid, "")
            else:
                detail = (result.stderr or result.stdout or "").strip()
                self.finished_signal.emit(False, self.ssid, detail)
        except Exception as e:
            self.finished_signal.emit(False, self.ssid, str(e))


class WifiConnectionDialog(QtWidgets.QDialog):
    """
    WiFi connection dialog using nmcli.
    Needs NetworkManager + nmcli inside the environment that can actually control WiFi.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("WIFI连接", "WIFI connection"))
        self.setModal(True)
        self.setMinimumSize(400, 600)
        self.setStyleSheet("""
                    QDialog {
                        background: #f3f5f7;
                    }
                    QLabel {
                        color: #ffffff;
                        font-size: 16px;
                        background: #1E90FF;
                    }
                    QListWidget {
                        background: #ffffff;
                        color: #111;
                        border: 1px solid #cfd6df;
                        border-radius: 10px;
                    }
                    QListWidget::item {
                        padding: 10px;
                        height: 10px;
                    }
                    QListWidget::item:selected {
                        background: #1E90FF;
                        color: #ffffff;
                        font-weight: bold;
                    }
                    QLineEdit {
                        background: #1E90FF;
                        color: #111;
                        border: 1px solid #cfd6df;
                        border-radius: 10px;
                        padding: 10px;
                        font-size: 16px;
                    }
                    QPushButton {
                        font-size: 16px;
                        padding: 10px 14px;
                        background: #1E90FF;
                    }
                """)

        main_layout = QtWidgets.QVBoxLayout(self)

        # Available networks
        main_layout.addWidget(QtWidgets.QLabel(tr("WiFi列表:", "Available WiFi networks:")))
        self.network_list = QtWidgets.QListWidget()
        self.network_list.setStyleSheet("background-color: #ffffff;")
        f = self.network_list.font()
        f.setPointSize(20)  # bump up as needed (e.g., 20/22)
        self.network_list.setFont(f)
        main_layout.addWidget(self.network_list)

        # Selected SSID + password
        form = QtWidgets.QFormLayout()
        self.ssid_label = QtWidgets.QLabel(tr("<无>", "<none>"))
        txt = tr("选定：", "Selected:")
        form.addRow(txt, self.ssid_label)

        self.password_edit = QtWidgets.QLineEdit()
        self.password_edit.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.password_edit.setAttribute(QtCore.Qt.WA_InputMethodEnabled, True)
        self.password_edit.setStyleSheet("background-color: #ffffff;")
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self._osk_proc = None
        self.password_edit.installEventFilter(self)
        txt = tr("密码：", "Password:")
        form.addRow(txt, self.password_edit)

        main_layout.addLayout(form)

        # Progress bar for connection (hidden by default)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)

        self.progress_label = QtWidgets.QLabel(tr("正在连接...", "Connecting..."))
        self.progress_label.setStyleSheet("background: transparent; color: #1E90FF; font-weight: bold;")
        self.progress_label.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_label.hide()
        main_layout.addWidget(self.progress_label)

        # Status + buttons
        bottom_layout = QtWidgets.QHBoxLayout()
        status_text = QtWidgets.QLabel(tr("状态:", "Status:"))
        status_text.setStyleSheet("background-color: #1E90FF")

        bottom_layout.addWidget(status_text)

        self.status_dot = QtWidgets.QLabel()
        self.status_dot.setFixedSize(16, 16)
        self._set_status_color("gray")  # initial: not connected
        bottom_layout.addWidget(self.status_dot)

        bottom_layout.addStretch(1)

        self.refresh_btn = QtWidgets.QPushButton(tr("刷新", "Refresh"))
        self.connect_btn = QtWidgets.QPushButton(tr("连接", "Connect"))
        self.close_btn = QtWidgets.QPushButton(tr("关闭", "Close"))
        bottom_layout.addWidget(self.refresh_btn)
        bottom_layout.addWidget(self.connect_btn)
        bottom_layout.addWidget(self.close_btn)

        main_layout.addLayout(bottom_layout)

        # Signals
        self.network_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.refresh_btn.clicked.connect(self.scan_networks)
        self.connect_btn.clicked.connect(self.connect_selected)
        self.close_btn.clicked.connect(self.close)

        # Initial scan
        self.scan_networks()

        # -------------------- On-screen keyboard helpers --------------------

    def closeEvent(self, e):
        self._hide_osk()
        super().closeEvent(e)

    def eventFilter(self, obj, event):
        if obj is self.password_edit:
            if event.type() in (
                    QtCore.QEvent.MouseButtonPress,
                    QtCore.QEvent.FocusIn,
                    QtCore.QEvent.TouchBegin,
            ):
                QtCore.QTimer.singleShot(0, self._show_osk)
        return super().eventFilter(obj, event)

    def _show_osk(self):
        """Best-effort: start an on-screen keyboard if one is installed."""
        import shutil
        print("[OSK] show requested")
        for exe in ("onboard", "florence", "matchbox-keyboard"):
            print("[OSK] which", exe, "=", shutil.which(exe))
        # already running?
        if self._osk_proc is not None and self._osk_proc.state() != QtCore.QProcess.NotRunning:
            return

        candidates = [
            ("onboard", ["onboard"]),  # GNOME onboard
            ("florence", ["florence"]),  # florence
            ("matchbox-keyboard", ["matchbox-keyboard"]),
        ]

        for exe, cmd in candidates:
            if shutil.which(exe):
                p = QtCore.QProcess(self)
                # start detached works too, but QProcess lets us track state
                p.start(cmd[0], cmd[1:])
                self._osk_proc = p
                return

        # If none found, do nothing (no keyboard installed)

    def _hide_osk(self):
        """Try to hide/close the on-screen keyboard (best-effort)."""
        import subprocess, shutil

        # If you prefer NOT hiding it, just `return` here.
        # return

        # onboard supports --hide on many setups
        if shutil.which("onboard"):
            subprocess.Popen(["onboard", "--hide"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        # Otherwise, if we launched something with QProcess, terminate it
        if self._osk_proc is not None and self._osk_proc.state() != QtCore.QProcess.NotRunning:
            self._osk_proc.terminate()
            self._osk_proc = None

    def _set_status_color(self, color: str):
        # green dot on success, red on failure, gray idle
        self.status_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 8px;"
        )

    def scan_networks(self):
        """Use nmcli to scan WiFi networks."""
        import subprocess

        self.network_list.clear()
        self._set_status_color("gray")

        try:
            # -t: terse, -f SSID,SECURITY: get SSID and security type
            out = subprocess.check_output(
                ["nmcli", "-t", "-f", "SSID,SECURITY", "dev", "wifi"],
                stderr=subprocess.STDOUT,
            ).decode("utf-8", "ignore")
            seen_ssids = set()
            networks = []  # list of (ssid, is_open)
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Format: SSID:SECURITY (e.g., "MyNetwork:WPA2" or "OpenNet:")
                parts = line.split(":", 1)
                ssid = parts[0].strip()
                security = parts[1].strip() if len(parts) > 1 else ""
                if ssid and ssid not in seen_ssids:
                    seen_ssids.add(ssid)
                    # Network is open if security field is empty or "--"
                    is_open = (security == "" or security == "--")
                    networks.append((ssid, is_open))

            if not networks:
                self.network_list.addItem(tr("(未发现WiFi网络)", "(no networks found)"))
                self.network_list.setEnabled(False)
            else:
                self.network_list.setEnabled(True)
                for ssid, is_open in networks:
                    if is_open:
                        display_text = f"{ssid}  [{tr('开放', 'Open')}]"
                    else:
                        display_text = ssid
                    item = QtWidgets.QListWidgetItem(display_text)
                    item.setData(QtCore.Qt.UserRole, ssid)  # Store actual SSID
                    item.setData(QtCore.Qt.UserRole + 1, is_open)  # Store if open
                    self.network_list.addItem(item)

        except Exception as e:
            self.network_list.addItem(tr(f"错误: {e}", f"Error: {e}"))
            self.network_list.setEnabled(False)

    def _on_selection_changed(self):
        items = self.network_list.selectedItems()
        if items:
            item = items[0]
            ssid = item.data(QtCore.Qt.UserRole)
            is_open = item.data(QtCore.Qt.UserRole + 1)
            # If data not set (old-style item), use text
            if ssid is None:
                ssid = item.text()
                is_open = False
            self.ssid_label.setText(ssid)
            # Update password field based on whether network is open
            if is_open:
                self.password_edit.setPlaceholderText(tr("无需密码", "No password required"))
                self.password_edit.setEnabled(False)
                self.password_edit.clear()
            else:
                self.password_edit.setPlaceholderText("")
                self.password_edit.setEnabled(True)
        else:
            self.ssid_label.setText(tr("<无>", "<none>"))
            self.password_edit.setPlaceholderText("")
            self.password_edit.setEnabled(True)

    def _set_connecting_ui(self, connecting: bool):
        """Show/hide progress bar and enable/disable buttons during connection."""
        self.progress_bar.setVisible(connecting)
        self.progress_label.setVisible(connecting)
        self.connect_btn.setEnabled(not connecting)
        self.refresh_btn.setEnabled(not connecting)
        self.network_list.setEnabled(not connecting)
        self.password_edit.setEnabled(not connecting)

    def connect_selected(self):
        items = self.network_list.selectedItems()
        if not items:
            msg = tr("没有选择网络，请选择一个WiFi网络。", "No network selected. Please select a WiFi network.")
            QtWidgets.QMessageBox.warning(self, tr("提示", "Warning"), msg)
            return

        item = items[0]
        ssid = item.data(QtCore.Qt.UserRole)
        is_open = item.data(QtCore.Qt.UserRole + 1)

        # If data not set (old-style item), use text
        if ssid is None:
            ssid = item.text()
            is_open = False

        if ssid.startswith("(") or ssid.startswith("Error"):
            return  # not a valid SSID

        pwd = self.password_edit.text().strip()

        # Only require password for secured networks
        if not is_open and not pwd:
            msg = tr("请输入密码。", "Password required. Please enter WiFi password.")
            QtWidgets.QMessageBox.warning(self, tr("提示", "Warning"), msg)
            return

        self._set_status_color("gray")
        self._set_connecting_ui(True)

        # Build command based on whether password is needed
        if is_open or not pwd:
            cmd = ["nmcli", "dev", "wifi", "connect", ssid]
        else:
            cmd = ["nmcli", "dev", "wifi", "connect", ssid, "password", pwd]

        # Run connection in a thread to not block UI
        self._connect_thread = _WifiConnectThread(cmd, ssid)
        self._connect_thread.finished_signal.connect(self._on_connect_finished)
        self._connect_thread.start()

    def _on_connect_finished(self, success: bool, ssid: str, detail: str):
        """Handle connection result from background thread."""
        self._set_connecting_ui(False)

        if success:
            self._set_status_color("green")
            msg = tr(f"连接成功，已连接到：{ssid}", f"Connected to {ssid} successfully.")
            QtWidgets.QMessageBox.information(self, tr("连接成功", "Connected"), msg)
        else:
            self._set_status_color("red")
            if not detail:
                detail = tr("未知错误", "Unknown error")
            msg = tr(f"连接失败：{detail}", f"Connection failed: {detail}")
            QtWidgets.QMessageBox.critical(self, tr("连接失败", "Connection failed"), msg)
