import hashlib
import hmac
import shutil
from pathlib import Path
from PyQt5 import QtCore, QtWidgets
from calibrations import f_pinyin 

def _parse_kv_config(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def verify_gate_password(conf_path: str, entered_password: str) -> tuple[bool, str]:
    """
    Returns (ok, drop_dir_or_error).
    Config keys:
      gate_salt=HEX
      gate_sha256=HEX(sha256(salt + password))
      drop_dir=/home/mp/NetdiskAutoUpload   (optional, default below)
    """
    cfg = _parse_kv_config(conf_path)
    gate_salt = cfg.get("gate_salt", "")
    gate_sha256 = cfg.get("gate_sha256", "")
    drop_dir = cfg.get("drop_dir", "").strip() or "/home/mp/NetdiskAutoUpload"

    if not gate_salt or not gate_sha256:
        return False, "Config missing gate_salt / gate_sha256"

    try:
        salt_bytes = bytes.fromhex(gate_salt)
    except Exception:
        return False, "gate_salt is not valid hex"

    computed = hashlib.sha256(salt_bytes + entered_password.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(computed, gate_sha256):
        return False, "Password incorrect"

    return True, str(Path(drop_dir).expanduser())


def queue_upload(local_file: str, drop_dir: Path, move: bool = True) -> str:
    src = Path(local_file)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    drop_dir.mkdir(parents=True, exist_ok=True)
    dst = drop_dir / src.name

    # Avoid overwrite by auto-suffix
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            cand = drop_dir / f"{stem}__{i}{suf}"
            if not cand.exists():
                dst = cand
                break
            i += 1

    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

    return str(dst)


class _StageWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int)      # done, total
    finished = QtCore.pyqtSignal(bool, str)     # ok, message

    def __init__(self, files: list[str], drop_dir: str, move: bool = True):
        super().__init__()
        self.files = [str(Path(f)) for f in files]
        self.drop_dir = Path(drop_dir)
        self.move = move
        self._cancel = False

    @QtCore.pyqtSlot()
    def run(self):
        try:
            total = len(self.files)
            done = 0
            self.progress.emit(done, total)

            for f in self.files:
                if self._cancel:
                    self.finished.emit(False, "Canceled.")
                    return
                queue_upload(f, self.drop_dir, move=self.move)
                done += 1
                self.progress.emit(done, total)

            txt = (
                "操作成功，另一个程序在上传比赛录像"
                if f_pinyin
                else f"Queued {total} file(s), cloud upload is in progress with another APP"
            )

            self.finished.emit(
                True,
                txt
            )
        except Exception as e:
            self.finished.emit(False, str(e))

    def cancel(self):
        self._cancel = True


def popup_queue_to_baidupcs_drop(
    parent,
    gate_conf_path: str,
    files_to_upload,
    move: bool = True,
):
    """
    Minimal UI:
      1) password prompt
      2) progress dialog while moving/copying into drop folder
    """
    # normalize input
    if isinstance(files_to_upload, (str, Path)):
        files = [str(files_to_upload)]
    else:
        files = [str(p) for p in files_to_upload]

    if not files:
        title = "上传" if f_pinyin else "Upload"
        msg   = "未选择任何文件。" if f_pinyin else "No files selected."
        QtWidgets.QMessageBox.warning(parent, title, msg)
        return False


    pwd, ok = QtWidgets.QInputDialog.getText(
        parent,
        "上传验证" if f_pinyin else "Upload Gate",
        "请输入上传密码：" if f_pinyin else "Enter upload password:",
        QtWidgets.QLineEdit.Password,
    )

    if not ok:
        return False

    ok_pwd, drop_or_err = verify_gate_password(gate_conf_path, pwd)
    if not ok_pwd:
        QtWidgets.QMessageBox.warning(
            parent,
            "上传验证" if f_pinyin else "Upload Gate",
            drop_or_err
        )
        return False

    drop_dir = drop_or_err

    prog = QtWidgets.QProgressDialog(
        "正在加入队列..." if f_pinyin else "Queueing files...",
        "取消" if f_pinyin else "Cancel",
        0,
        len(files),
        parent
    )
    prog.setWindowTitle("队列上传" if f_pinyin else "Queue Upload")
    prog.setWindowModality(QtCore.Qt.ApplicationModal)
    prog.setMinimumDuration(0)
    prog.setAutoClose(False)
    prog.setAutoReset(False)

    thread = QtCore.QThread(parent)
    worker = _StageWorker(files, drop_dir=drop_dir, move=move)
    worker.moveToThread(thread)

    def on_progress(done, total):
        prog.setMaximum(max(total, 1))
        prog.setValue(done)
        if f_pinyin:
            prog.setLabelText(f"已加入队列 {done}/{total}\n{drop_dir}")
        else:
            prog.setLabelText(f"Queued {done}/{total}\n{drop_dir}")

    def on_cancel():
        worker.cancel()

    def on_finished(ok2, msg):
        prog.setValue(prog.maximum())
        thread.quit()
        thread.wait(2000)
        prog.close()
        if ok2:
            QtWidgets.QMessageBox.information(
                parent,
                "已加入队列" if f_pinyin else "Queued",
                msg
            )
        else:
            QtWidgets.QMessageBox.critical(
                parent,
                "失败" if f_pinyin else "Failed",
                msg
            )

    prog.canceled.connect(on_cancel)
    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    thread.started.connect(worker.run)
    thread.start()

    return True

