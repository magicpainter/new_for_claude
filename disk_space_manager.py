# disk_space_manager.py
# Manages disk space for video recordings by deleting oldest videos when space is low

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from PyQt5 import QtCore, QtWidgets, QtGui
from calibrations import f_pinyin

# Video extensions to consider for cleanup
VIDEO_EXTS = (".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v")

# Default settings
DEFAULT_VIDEO_DIR = Path.home() / "Videos"
DEFAULT_MIN_FREE_SPACE_GB = 20  # Trigger cleanup when available space < 20GB
DEFAULT_TARGET_FREE_SPACE_GB = 30  # Target to free up to this much space


def get_available_disk_space_gb(path: Path) -> float:
    """Get available disk space in GB for the partition containing path."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024 ** 3)
    except OSError:
        return float('inf')  # Return large value if can't determine


def get_folder_size_bytes(folder: Path) -> int:
    """Get total size of all files in folder (non-recursive for Videos)."""
    total = 0
    if not folder.exists():
        return 0
    for f in folder.iterdir():
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def get_folder_size_gb(folder: Path) -> float:
    """Get folder size in GB."""
    return get_folder_size_bytes(folder) / (1024 ** 3)


def _get_video_priority(filename: str) -> int:
    """
    Get deletion priority for a video file.
    Lower number = delete first.

    Priority:
        0 - Dribble videos (delete first)
        1 - Shooting videos
        2 - Game videos (delete last)
        3 - Unknown/other
    """
    name_lower = filename.lower()

    if 'dribble' in name_lower:
        return 0
    elif 'shoot' in name_lower or 'shooting' in name_lower:
        return 1
    elif 'game' in name_lower:
        return 2
    else:
        return 3  # Unknown files have lowest priority (kept longer)


def get_videos_sorted_for_deletion(folder: Path) -> List[Tuple[Path, float, int, int]]:
    """
    Get list of video files sorted by deletion priority.

    Sort order:
    1. First by priority (Dribble > Shooting > Game > Other)
    2. Then by modification time (oldest first within each priority)

    Returns: List of (path, mtime, size_bytes, priority)
    """
    videos = []
    if not folder.exists():
        return videos

    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            try:
                stat = f.stat()
                priority = _get_video_priority(f.name)
                videos.append((f, stat.st_mtime, stat.st_size, priority))
            except OSError:
                pass

    # Sort by priority first, then by mtime (oldest first)
    videos.sort(key=lambda x: (x[3], x[1]))
    return videos


def calculate_deletion_plan(
    folder: Path,
    min_free_space_gb: float = DEFAULT_MIN_FREE_SPACE_GB,
    target_free_space_gb: float = DEFAULT_TARGET_FREE_SPACE_GB
) -> Tuple[List[Tuple[Path, int]], float]:
    """
    Calculate which files to delete to get available space above target.
    Prioritizes Dribble and Shooting videos before Game videos.

    Args:
        folder: Video folder path
        min_free_space_gb: Threshold to trigger cleanup
        target_free_space_gb: Target free space after cleanup

    Returns:
        Tuple of (List of (path, size_bytes) to delete, current_free_space_gb)
    """
    current_free = get_available_disk_space_gb(folder)

    # No cleanup needed if we have enough free space
    if current_free >= min_free_space_gb:
        return [], current_free

    # Calculate how much we need to free
    need_to_free_bytes = int((target_free_space_gb - current_free) * (1024 ** 3))

    if need_to_free_bytes <= 0:
        return [], current_free

    videos = get_videos_sorted_for_deletion(folder)
    to_delete = []
    freed_so_far = 0

    for path, mtime, size, priority in videos:
        if freed_so_far >= need_to_free_bytes:
            break
        to_delete.append((path, size))
        freed_so_far += size

    return to_delete, current_free


class DiskCleanupWorker(QtCore.QObject):
    """Worker to delete files in background thread."""
    progress = QtCore.pyqtSignal(int, int, str)  # (current, total, filename)
    finished = QtCore.pyqtSignal(int, int)  # (deleted_count, freed_bytes)
    error = QtCore.pyqtSignal(str)

    def __init__(self, files_to_delete: List[Tuple[Path, int]]):
        super().__init__()
        self.files_to_delete = files_to_delete
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        deleted_count = 0
        freed_bytes = 0
        total = len(self.files_to_delete)

        for i, (path, size) in enumerate(self.files_to_delete):
            if self._cancelled:
                break

            self.progress.emit(i + 1, total, path.name)

            try:
                path.unlink()
                deleted_count += 1
                freed_bytes += size
            except Exception as e:
                self.error.emit(f"Failed to delete {path.name}: {e}")

        self.finished.emit(deleted_count, freed_bytes)


class DiskCleanupDialog(QtWidgets.QDialog):
    """Progress dialog for disk cleanup."""

    def __init__(
        self,
        folder: Path,
        min_free_space_gb: float = DEFAULT_MIN_FREE_SPACE_GB,
        target_free_space_gb: float = DEFAULT_TARGET_FREE_SPACE_GB,
        parent=None
    ):
        super().__init__(parent)
        self.folder = folder
        self.min_free_space_gb = min_free_space_gb
        self.target_free_space_gb = target_free_space_gb
        self.worker = None
        self.thread = None

        self._setup_ui()
        self._calculate_and_start()

    def _setup_ui(self):
        self.setWindowTitle("磁盘清理" if f_pinyin else "Disk Cleanup")
        self.setModal(True)
        self.setMinimumSize(500, 200)
        self.setStyleSheet("""
            QDialog { background: #f3f5f7; }
            QLabel { color: #111; font-size: 14px; }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background: #4CAF50;
                border-radius: 4px;
            }
            QPushButton {
                font-size: 14px;
                padding: 8px 16px;
                background: #1E90FF;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover { background: #1976D2; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)

        # Status label
        self.status_label = QtWidgets.QLabel(
            "正在计算需要删除的文件..." if f_pinyin else "Calculating files to delete..."
        )
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.status_label)

        # Info label
        self.info_label = QtWidgets.QLabel("")
        layout.addWidget(self.info_label)

        # Current file label
        self.file_label = QtWidgets.QLabel("")
        self.file_label.setStyleSheet("color: #666;")
        layout.addWidget(self.file_label)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Button row
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QtWidgets.QPushButton("取消" if f_pinyin else "Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)

        self.close_btn = QtWidgets.QPushButton("关闭" if f_pinyin else "Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setVisible(False)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    def _calculate_and_start(self):
        files_to_delete, current_free = calculate_deletion_plan(
            self.folder, self.min_free_space_gb, self.target_free_space_gb
        )

        if not files_to_delete:
            self.status_label.setText(
                "磁盘空间充足，无需清理" if f_pinyin else "Disk space is OK, no cleanup needed"
            )
            self.info_label.setText(
                f"可用空间: {current_free:.1f} GB (阈值: {self.min_free_space_gb:.0f} GB)" if f_pinyin else
                f"Available: {current_free:.1f} GB (threshold: {self.min_free_space_gb:.0f} GB)"
            )
            self.progress_bar.setValue(100)
            self.cancel_btn.setVisible(False)
            self.close_btn.setVisible(True)
            return

        total_to_free = sum(size for _, size in files_to_delete)
        total_to_free_gb = total_to_free / (1024 ** 3)

        # Count by type for info display
        dribble_count = sum(1 for p, _ in files_to_delete if 'dribble' in p.name.lower())
        shooting_count = sum(1 for p, _ in files_to_delete if 'shoot' in p.name.lower())
        game_count = sum(1 for p, _ in files_to_delete if 'game' in p.name.lower())

        self.status_label.setText(
            f"正在删除 {len(files_to_delete)} 个旧视频..." if f_pinyin else
            f"Deleting {len(files_to_delete)} old videos..."
        )

        type_info = f"(运球:{dribble_count} 投篮:{shooting_count} 比赛:{game_count})" if f_pinyin else \
                    f"(Dribble:{dribble_count} Shooting:{shooting_count} Game:{game_count})"
        self.info_label.setText(
            f"可用: {current_free:.1f} GB | 将释放: {total_to_free_gb:.1f} GB {type_info}" if f_pinyin else
            f"Available: {current_free:.1f} GB | Will free: {total_to_free_gb:.1f} GB {type_info}"
        )

        self.progress_bar.setMaximum(len(files_to_delete))
        self.progress_bar.setValue(0)

        # Start worker thread
        self.thread = QtCore.QThread()
        self.worker = DiskCleanupWorker(files_to_delete)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self.thread.quit)

        self.thread.start()

    def _on_progress(self, current: int, total: int, filename: str):
        self.progress_bar.setValue(current)
        self.file_label.setText(
            f"正在删除: {filename}" if f_pinyin else f"Deleting: {filename}"
        )

    def _on_finished(self, deleted_count: int, freed_bytes: int):
        freed_gb = freed_bytes / (1024 ** 3)
        new_free = get_available_disk_space_gb(self.folder)

        self.status_label.setText(
            "清理完成!" if f_pinyin else "Cleanup complete!"
        )
        self.info_label.setText(
            f"已删除 {deleted_count} 个文件，释放 {freed_gb:.1f} GB | 可用空间: {new_free:.1f} GB" if f_pinyin else
            f"Deleted {deleted_count} files, freed {freed_gb:.1f} GB | Available: {new_free:.1f} GB"
        )
        self.file_label.setText("")
        self.progress_bar.setValue(self.progress_bar.maximum())

        self.cancel_btn.setVisible(False)
        self.close_btn.setVisible(True)

    def _on_error(self, msg: str):
        print(f"[DiskCleanup] Error: {msg}")

    def _on_cancel(self):
        if self.worker:
            self.worker.cancel()
        self.reject()

    def closeEvent(self, event):
        if self.worker:
            self.worker.cancel()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)
        super().closeEvent(event)


def check_and_cleanup_disk(
    folder: Path = DEFAULT_VIDEO_DIR,
    min_free_space_gb: float = DEFAULT_MIN_FREE_SPACE_GB,
    target_free_space_gb: float = DEFAULT_TARGET_FREE_SPACE_GB,
    parent: QtWidgets.QWidget = None
) -> bool:
    """
    Check disk usage and show cleanup dialog if needed.
    Triggers when available disk space falls below min_free_space_gb.
    Prioritizes deleting Dribble and Shooting videos before Game videos.

    Args:
        folder: Video folder to clean
        min_free_space_gb: Trigger cleanup when available space < this (default: 20GB)
        target_free_space_gb: Target free space after cleanup (default: 30GB)
        parent: Parent widget for dialog

    Returns True if cleanup was needed and performed.
    """
    current_free = get_available_disk_space_gb(folder)

    if current_free >= min_free_space_gb:
        return False

    # Show cleanup dialog
    dlg = DiskCleanupDialog(folder, min_free_space_gb, target_free_space_gb, parent)
    dlg.exec_()
    return True


def needs_cleanup(
    folder: Path = DEFAULT_VIDEO_DIR,
    min_free_space_gb: float = DEFAULT_MIN_FREE_SPACE_GB
) -> bool:
    """
    Quick check if disk cleanup is needed (for periodic background checks).
    Returns True if available space < min_free_space_gb.
    """
    return get_available_disk_space_gb(folder) < min_free_space_gb
