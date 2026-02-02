import os, glob, time
import cv2
import jetson_utils as jutils

class FolderFrameSource:
    def __init__(self, folder, width, height, fps=30, loop=True, sort_numerically=True):
        self.folder = folder
        exts = ('*.jpg','*.jpeg','*.png','*.bmp')
        files = []
        for e in exts:
            files += glob.glob(os.path.join(folder, e))
        if not files:
            raise FileNotFoundError(f"No images found in {folder}")

        if sort_numerically:
            # natural sort like img_1, img_2, ..., img_10
            import re
            def keyfn(s):
                return [int(t) if t.isdigit() else t.lower()
                        for t in re.split(r'(\d+)', os.path.basename(s))]
            files.sort(key=keyfn)
        else:
            files.sort()

        self.files = files
        self.w, self.h = int(width), int(height)
        self.loop = bool(loop)
        self.fps = float(fps) if fps and fps > 0 else None
        self._idx = 0
        self._streaming = True
        self._next_t = time.perf_counter()

    def IsStreaming(self):
        return self._streaming

    def Close(self):
        self._streaming = False

    def _throttle(self):
        if self.fps is None:
            return
        now = time.perf_counter()
        if now < self._next_t:
            time.sleep(self._next_t - now)
        self._next_t = max(now, self._next_t) + 1.0 / self.fps

    def Capture(self):
        if not self._streaming:
            return None

        if self._idx >= len(self.files):
            if self.loop:
                self._idx = 0
            else:
                self._streaming = False
                return None

        path = self.files[self._idx]
        self._idx += 1

        # Load BGR -> resize -> RGBA uint8
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            # skip unreadable file
            return self.Capture()

        if (bgr.shape[1], bgr.shape[0]) != (self.w, self.h):
            bgr = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # HxWx3 uint8
        # Upload to CUDA (uchar4). Returns jetson.utils.cudaImage (RGBA)
        cuda_img = jutils.cudaFromNumpy(rgb)
        self._throttle()
        return cuda_img
