# Gl_Monitor.py
from PyQt5 import QtGui, QtWidgets, QtCore
from OpenGL import GL
from collections import deque

import pycuda.driver as cuda
from pycuda import gl as cudagl
import threading, queue

MAP_LOCK = threading.Lock()

class GLMonitor(QtWidgets.QOpenGLWidget):
    """ÏÔÊ¾ CUDA Device ÄÚ´æÖÐµÄ RGBA8 Ö¡£¨GPU¡úGPU 0-copy£©"""

    def __init__(self,ctx_q, ctx_ready, *, side, tex_queue, parent=None, src_W, src_H):
        super().__init__(parent)

        self._tex_ids = []  # [GLuint, GLuint]
        self._cuda_res = []  # [RegisteredImage, RegisteredImage]
        self._active_idx = 0

        self._img_w = src_W
        self._img_h = src_H
        self._ready = False
        self.setMinimumSize(400, 300)
        self.ctx_q = ctx_q
        self.ctx_ready = ctx_ready
        self.side = side
        self.tex_queue = tex_queue
        self._crop_px = None

    # -------------------------------------------------------------------------
    # OpenGL life-cycle
    # -------------------------------------------------------------------------
    def initializeGL(self):
        """´´½¨Õ¼Î»ÎÆÀí£¨1¡Á1£©£¬ÏÈ **²»×¢²á** ¸ø CUDA¡£"""
        super().initializeGL()
        print(f"[GLMonitor] {self.side} initializeGL")
        #print(f"[GLMonitor] {self.side} initializeGL  -> tex = {self._tex_id}")

        cuda.init()
        global GL_CUDA_CTX
        if 'GL_CUDA_CTX' not in globals():
            GL_CUDA_CTX = cudagl.make_context(cuda.Device(0))

        self._cuda_ctx = GL_CUDA_CTX
        self._cuda_ctx.push()  # µ±Ç°Ïß³Ì¼¤»î

        self._stream = cuda.Stream()  # Stream ±ØÐëÔÚ push Ö®ºó´´½¨
        self._prev_copy_evt = None  # ÓÃÓÚ¿çÖ¡Í¬²½ CUDA ¿½±´

        if self.ctx_q.empty():
            self.ctx_q.put(GL_CUDA_CTX)

        self._tex_ids = GL.glGenTextures(2)  # Éú³É 2 ¸ö tex_id

        if isinstance(self._tex_ids, int):  # PyOpenGL ¿ÉÄÜ·µ»Ø int
            self._tex_ids = [self._tex_ids]

        for tex in self._tex_ids:
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0,
                GL.GL_RGBA8,  # internal format
                self._img_w, self._img_h,
                0,
                GL.GL_RGBA,  # format
                GL.GL_UNSIGNED_BYTE,
                None
            )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self._pbo_ids = GL.glGenBuffers(2)
        self._cuda_res = []  # RegisteredBuffer ÁÐ±í

        for pbo in self._pbo_ids:
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
            GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER,
                            self._img_w * self._img_h * 4,
                            None,
                            GL.GL_DYNAMIC_DRAW)
            buf = cudagl.RegisteredBuffer(
                int(pbo),
                cudagl.graphics_map_flags.WRITE_DISCARD)
            self._cuda_res.append(buf)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        GL.glFinish()  # µÈÈ«²¿´´½¨Íê

        # °ÑÁ½·Ý×ÊÔ´·Ö±ð·¢¸øÍÆÀíÏß³Ì£¬´ø idx ±êºÅ
        for idx, res in enumerate(self._cuda_res):
            self.tex_queue.put({
                "cmd": "registered_res",
                "side": self.side,  # 'left' / 'right'
                "idx": idx,  # 0 / 1
                "res": res,
                "pbo_id": int(self._pbo_ids[idx])
            })

        self._active_idx = 0  # µ±Ç°ÕýÔÚÏÔÊ¾ÄÄÕÅ£¨paintGL ÓÃ£©

        if not self.ctx_ready.is_set():  # Ö» set Ò»´Î
           self.ctx_ready.set()

        self._ready = True
        self._cuda_ctx.pop()

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def resizeGL(self, w, h):
        print(f"[GLMonitor-{self.side}] resizeGL({w}, {h})")
        GL.glViewport(0, 0, w, h)

    def paintGL(self):

        if self._crop_px is not None and self._img_w:
            x0, x1 = self._crop_px
            # clamp and normalize to [0,1]
            x0 = max(0, min(self._img_w, x0))
            x1 = max(0, min(self._img_w, x1))
            if x1 <= x0:
                u0, u1 = 0.0, 1.0
            else:
                u0 = x0 / float(self._img_w)
                u1 = x1 / float(self._img_w)
        else:
            u0, u1 = 0.0, 1.0

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glEnable(GL.GL_TEXTURE_2D)

        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER,
                        self._pbo_ids[self._active_idx])
        GL.glBindTexture(GL.GL_TEXTURE_2D,
                         self._tex_ids[self._active_idx])
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D,
                           0, 0, 0,
                           self._img_w, self._img_h,
                           GL.GL_RGBA,
                           GL.GL_UNSIGNED_BYTE,
                           None)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_ids[self._active_idx])
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(u0, 1.0)
        GL.glVertex2f(-1, -1)
        GL.glTexCoord2f(u1, 1.0)
        GL.glVertex2f(1, -1)
        GL.glTexCoord2f(u1, 0.0)
        GL.glVertex2f(1, 1)
        GL.glTexCoord2f(u0, 0.0)
        GL.glVertex2f(-1, 1)
        GL.glEnd()


    @QtCore.pyqtSlot(int)
    def set_active_buffer(self, idx: int):
        self._active_idx = idx & 1  # ±£Ö¤ 0/1


    def _recreate_texture(self, w, h):
        """×¢Ïú¾ÉÎÆÀí ¡ú É¾³ý ¡ú ´´½¨ÐÂÎÆÀí ¡ú ×¢²á CUDA"""
        # 1. ÈôÒÑ×¢²á£¬ÏÈ×¢Ïú
        if self._cuda_res is not None:
            self._cuda_res.unregister()
            self._cuda_res = None
            GL.glDeleteTextures([self._tex_id])

        # 2. ÐÂ½¨ + ·ÖÅä´æ´¢
        self._tex_id = GL.glGenTextures(1)
        if isinstance(self._tex_id, (list, tuple)):
            self._tex_id = self._tex_id[0]

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        #GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,   # ES2 ÓÃ GL_RGB
                        w, h, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        # 3. ×¢²á¸ø CUDA
        self._cuda_res = cudagl.RegisteredImage(
            int(self._tex_id), GL.GL_TEXTURE_2D,
            cudagl.graphics_map_flags.WRITE_DISCARD
        )

        self._img_w, self._img_h = w, h

    # -------------------------------------------------------------------------
    # Public API: GPU¡úGPU ¸´ÖÆÖ¡
    # -------------------------------------------------------------------------


    def update_from_cuda(self, dev_ptr: int, width, height, bytes_to_copy):
        if not self._ready:
            return

        if self._prev_copy_evt is not None:
            self._stream.wait_for_event(self._prev_copy_evt)
            self._prev_copy_evt = None

        self._cuda_ctx.push()
        try:
            # Ê×Ö¡»ò·Ö±æÂÊ±ä»¯Ê±¶¯Ì¬ÖØ½¨ÎÆÀí/PBO
            if (width, height) != (self._img_w, self._img_h):
                self._recreate_texture(width, height)

            # ---- GPU ¡ú GPU ¿½±´£º°Ñ frame Ð´½øµ±Ç° PBO ----------
            write_idx = self._active_idx ^ 1
            buf = self._cuda_res[write_idx]

            with MAP_LOCK:
                mapping = buf.map(self._stream)
                try:
                    pbo_dev_ptr, _ = mapping.device_ptr_and_size()
                    cuda.memcpy_dtod_async(pbo_dev_ptr, dev_ptr,
                                           bytes_to_copy,self._stream)
                    self._prev_copy_evt = cuda.Event()
                    self._prev_copy_evt.record(self._stream)
                finally:
                    mapping.unmap(self._stream)

            self._active_idx = write_idx

            self._img_w, self._img_h = width, height
            self.update()  # ´¥·¢ paintGL()

        finally:
            self._cuda_ctx.pop()


    def set_crop_pixels(self, x0: int, x1: int):
        """Crop horizontally to [x0:x1] in source pixels; height unchanged."""
        self._crop_px = (int(x0), int(x1))
        self.update()

    def clear_crop(self):
        self._crop_px = None
        self.update()
