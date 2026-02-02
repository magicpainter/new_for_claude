import numpy as np
import pandas as pd
from enum import Enum
import json
import cv2
from features.bytetrack import Tracker
from jetson_utils import cudaDrawRect, cudaFont, cudaDrawLine
import pygame


class Event(Enum):
    NO_SHOOTING = 0
    SHOOTING = 1
    IN_BASKET = 2
    FAILED = 3


names = {'ball': 0, 'bib': 1, 'bob': 2, 'player': 3}
reverse_names = {v: k for k, v in names.items()}


def build_dets(ctx, detections, thr, ow, oh, f_shift=False):
    for bb, sc, lb in zip(ctx["bboxes"], ctx["scores"], ctx["labels"]):
        lb = int(lb)
        if lb < 0 or sc < thr:  # ¡û ¹Ø¼ü¹ýÂË£¡
            continue

        # convert TRT xyxy to xywh
        x1, y1, x2, y2 = bb

        x1 = max(0, min(int(x1), ow - 1))
        y1 = max(0, min(int(y1), oh - 1))
        x2 = max(0, min(int(x2), ow - 1))
        y2 = max(0, min(int(y2), oh - 1))

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        if f_shift:  # for two camera, the object in the right cam shifts w/ image width
            x1 += ow
            x2 += ow
        xyxy = [x1, y1, x2, y2]
        # 提取置信度和类别
        conf = float(sc)
        obj_class = lb
        detection = [*xyxy, conf, obj_class]
        detections.append(detection)

    return detections


class GameFeature:
    def __init__(self, w, h, dt):
        self.cuda_img_left = None
        self.cuda_img_right = None
        self.tracker = Tracker(q=1000.0, r=0.01)
        # Initialize the tracker with Constant Velocity (CV) model
        self.shooting_locs = []
        self.shooting_result = []
        self.latest_shooting_loc = None
        self.total_shooting = 0
        self.bingo_2pt_num = 0
        self.bingo_3pt_num = 0
        self.status = Event.NO_SHOOTING
        self.three_pointer = False
        self.frame_cnt = 0
        self.panning_center = [w / 2, h / 2]  # it is [x,y] format, need to convert to [row, col] during cropping
        self.shooting_start = 0
        self.shooting_start_array = []
        self.shooting_end_array = []
        self.fail_cnt = 0
        self.bib_cnt = 0
        self.ball_id = -1
        self.img_width = w
        self.img_height = h
        self.ring_loc_y = 200
        self.dt = dt
        self.pan_x0 = 0
        self.pan_x1 = 0
        self.pan_y0 = 0
        self.pan_y1 = 0
        self._fontS = cudaFont(size=32)
        self.hot_left = False
        self.f_both_activated = False
        self.shoot_du = 0

    def on_start(self, engine_ctx: dict):

        self.__init__(self.img_width, self.img_height, self.dt)

        pass

    def on_frame(self, ctx_left: dict, ctx_right: dict, side):
        if ctx_left and side == 'left':
            ctx = ctx_left
            self.cuda_img_left = ctx_left["cuda_img"]
        if ctx_right and side == 'right':
            ctx = ctx_right
            self.cuda_img_right = ctx_right["cuda_img"]
        if not ctx_left and not ctx_right:
            return

        cfg = ctx["cfg"]
        ow, oh = ctx["src_W"], ctx["src_H"]
        thr = cfg["score_threshold"]
        detections = []

        if ctx_left:  # left ctx is not empty
            detections = build_dets(ctx_left, detections, thr, ow, oh, f_shift=False)

        if ctx_right:
            if ctx_left:
                detections = build_dets(ctx_right, detections, thr, ow, oh, f_shift=True)
            else:
                detections = build_dets(ctx_right, detections, thr, ow, oh, f_shift=False)

        if ctx_left and ctx_right:  # both cameras are activated
            self.f_both_activated = True
            if side == 'right':
                self.calc_event(detections)
                self.calc_panning_xy()
        elif ctx_left or ctx_right:
            self.f_both_activated = False
            self.calc_event(detections)
            self.calc_panning_xy()

        self.cuda_draw_name_id(side)

    # calculate events based on tracks and 3pts map and basket location
    def calc_event(self, detections):
        self.tracker.update(detections, self.dt)
        tracks = self.tracker.get_tracks()
        self.frame_cnt += 1
        if self.frame_cnt == 251:
            print(1)
        # the new logic shall start from event, then go thru all the tracks
        if self.status == Event.NO_SHOOTING:
            f_shooting = self.search_shooting(tracks)
            if f_shooting:
                self.status = Event.SHOOTING
                self.shoot_du = 1

        elif self.status == Event.SHOOTING:
            tid = self.ball_id
            all_tids = tracks.keys()
            if tid not in all_tids:  # the shooting ball is lost
                self.status = Event.NO_SHOOTING
                self.save_results()
            else:
                ball_track = tracks[tid]
                if ball_track['object_class'] == names['bib']:
                    # distance = np.linalg.norm(tracks[tid]['pos'] - self.basket_xy)
                    if self.bib_cnt == 0:
                        self.bib_cnt = 2  # wait for two more frames to confirm it is a make
                    elif self.bib_cnt == 1:
                        self.status = Event.IN_BASKET
                        self.save_results()
                        self.bingo_2pt_num += 1
                        self.three_pointer = False
                        self.bib_cnt = 0
                    else:
                        self.bib_cnt -= 1

                elif ball_track['object_class'] == names['bob'] and self.shoot_du > 10:
                    # distance = np.linalg.norm(tracks[tid]['pos'] - self.basket_xy)
                    # if distance < 100:
                    self.status = Event.FAILED
                    self.save_results()

                elif ball_track['object_class'] == names['ball']:
                    if self.bib_cnt > 0:  # indicate it was confirmed "bib", now it left the basket
                        self.status = Event.IN_BASKET
                        self.save_results()
                        self.bingo_2pt_num += 1
                        self.three_pointer = False
                        self.bib_cnt = 0
                    else:
                        ball_pos = ball_track['pos']
                        if ball_pos[1] > self.ring_loc_y + 80:  # indicate the ball has fallen off
                            self.status = Event.FAILED
                            self.save_results()

        elif self.status == Event.FAILED:
            if self.fail_cnt == 0:
                self.fail_cnt = 3
            elif self.fail_cnt == 1:
                self.status = Event.NO_SHOOTING
                self.fail_cnt = 0
            else:
                self.fail_cnt -= 1

        elif self.status == Event.IN_BASKET:
            if self.bib_cnt == 0:
                self.bib_cnt = 5
            elif self.bib_cnt == 1:
                self.status = Event.NO_SHOOTING
                self.bib_cnt = 0
            else:
                self.bib_cnt -= 1

    def save_results(self):
        self.shooting_result.append(self.status)
        self.total_shooting += 1
        self.shooting_locs.append(self.latest_shooting_loc)
        self.shooting_start_array.append(self.shooting_start)
        self.shooting_end_array.append(self.frame_cnt)

    def search_shooting(self, tracks):
        f_shooting = False
        for tid, tr in tracks.items():
            if tr['object_class'] == names['ball'] and tr['coasted'] is False and tr['static_age'] < 3:   # don't allow coasted ball to trigger shooting
                ball_pos = tr['pos']
                if ball_pos[1] < self.ring_loc_y:  # it is higher than the ring
                    f_shooting = True
                    self.ball_id = tid
                    self.shooting_start = self.frame_cnt
                    self.find_player(tracks, ball_pos)
                    break

        return f_shooting

    def find_player(self, tracks, ball_pos):  # when shooting is detected, we need to find the shooter
        smallest_dist = 1e9
        for tid, tr in tracks.items():
            if tr['object_class'] == names['player']:
                player_pos = tr['pos']
                w, h = tr['wh']
                foot_loc = [player_pos[0]+w/4, player_pos[1]+h/2] # make the foot location biased toward camera
                dist = np.linalg.norm(ball_pos - player_pos)
                if dist < smallest_dist and h / w > 2:  # only, locating the shooter
                    smallest_dist = dist
                    self.latest_shooting_loc = foot_loc

    def calc_panning(self, image):
        # img = image.copy()
        self.calc_panning_xy()
        panned_image = image[self.pan_y0:self.pan_y1, self.pan_x0:self.pan_x1]
        return panned_image

    def calc_panning_xy(self):
        tracks = self.tracker.get_tracks()
        all_tids = tracks.keys()
        if self.status != Event.NO_SHOOTING and self.ball_id in all_tids:  # only track ball
            ball_track = tracks[self.ball_id]
            ball_pos = ball_track['pos']  # [x, y]
            self.panning_center[0] = self.panning_center[0] * 0.8 + ball_pos[0] * 0.2
            self.panning_center[1] = self.panning_center[1] * 0.8 + ball_pos[1] * 0.2
        else:  # track players
            pos_all = []
            for tid, trk in tracks.items():
                if trk['object_class'] == names['player'] and trk['conf'] > 0.2 and trk['age'] > 20:
                    w, h = trk['wh']
                    if w * h > 6000:  # this is to make sure this player is in home court
                        pos_all.append(trk['pos'])

            if len(pos_all) > 0:
                pos_median = np.median(pos_all, axis=0)
                self.panning_center[0] = self.panning_center[0] * 0.92 + pos_median[0] * 0.08
                self.panning_center[1] = self.panning_center[1] * 0.92 + pos_median[1] * 0.08

        pan_center_x = int(self.panning_center[0])
        pan_center_y = int(self.panning_center[1])
        # determine the hot area side.
        switch_thresh_ratio = 0.1

        if self.hot_left:
            width_bound_left = 0
            width_bound_right = self.img_width
            if pan_center_x > self.img_width * (1.0 - switch_thresh_ratio):
                self.hot_left = False
                self.panning_center[0] = self.img_width * (1.0 + switch_thresh_ratio)

        else:  # right side
            width_bound_left = self.img_width
            width_bound_right = 2 * self.img_width
            if pan_center_x < self.img_width * (1.0 + switch_thresh_ratio) - 100:  # 20 is a buffer
                self.hot_left = True
                self.panning_center[0] = self.img_width * (1.0 - switch_thresh_ratio) - 100

        x0 = pan_center_x - 640
        x1 = pan_center_x + 640
        if x1 > width_bound_right:
            x1 = width_bound_right
            x0 = x1 - 1280
        elif x0 < width_bound_left:
            x0 = width_bound_left
            x1 = x0 + 1280

        y0 = pan_center_y - 360
        y1 = pan_center_y + 360  # panned image size 1280 x 720
        if y1 > self.img_height:
            y1 = self.img_height
            y0 = y1 - 720
        elif y0 < 0:
            y0 = 0
            y1 = y0 + 720

        self.pan_x0 = x0
        self.pan_x1 = x1
        self.pan_y0 = y0
        self.pan_y1 = y1

    def cuda_draw_name_id(self, side):
        """Draw tracker bounding boxes and label each with class name & ID."""
        tracks = self.tracker.get_tracks()

        for tid, tr in tracks.items():
            cx, cy = tr['pos']
            w_box, h_box = tr['wh']
            x1 = int(cx - w_box / 2)
            y1 = int(cy - h_box / 2)
            x2 = x1 + int(w_box)
            y2 = y1 + int(h_box)
            if side == 'left':
                if x1 >= self.img_width:
                    continue
            elif self.f_both_activated and side == 'right':
                if x1 < self.img_width:
                    continue

            # 4?line hollow rectangle (green, semi?transparent)
            self._cuda_draw_hollow_rect(side,(x1, y1, x2, y2),
                                        (0, 255, 0, 200),
                                        line_width=2
                                        )

            # Compose label text
            cls_name = reverse_names.get(tr['object_class'], str(tr['object_class']))
            label = f"{cls_name} #{tid}"

            # Rough text size estimate for background box
            text_w = len(label) * 18  # 18px per char @ size?32 font (empirical)
            text_h = 32
            pad = 4
            bx1 = x1
            by1 = max(0, y1 - text_h - 2 * pad)
            bx2 = bx1 + text_w + 2 * pad
            by2 = by1 + text_h + 2 * pad

            # Black translucent background
            self._cuda_draw_hollow_rect(side, (bx1, by1, bx2, by2),
                                        (0, 0, 0, 160), label=label
                                        )
            #

    def _cuda_draw_hollow_rect(self, side, rect, color, line_width=2, label=None):
        """Draw rectangle outline on a cudaImage using 4 lines."""
        x1, y1, x2, y2 = rect
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        W, H = self.img_width, self.img_height
        if side == 'left':
            cuda_img = self.cuda_img_left
        elif side == 'right':
            cuda_img = self.cuda_img_right
            if self.f_both_activated:
                x1 -= W
                x2 -= W

        # 2) clamp to image bounds

        x1 = int(max(0, min(x1, W - 1)))
        x2 = int(max(0, min(x2, W - 1)))
        y1 = int(max(0, min(y1, H - 1)))
        y2 = int(max(0, min(y2, H - 1)))

        # 3) skip degenerate/empty boxes or invalid widths
        if x2 <= x1 or y2 <= y1 or line_width <= 0 or cuda_img is None:
            return
        if label is None:
            # ÉÏ±ß
            cudaDrawLine(cuda_img, (x1, y1), (x2, y1),
                         color, line_width=line_width)
            # ÓÒ±ß
            cudaDrawLine(cuda_img, (x2, y1), (x2, y2),
                         color, line_width=line_width)
            # ÏÂ±ß
            cudaDrawLine(cuda_img, (x2, y2), (x1, y2),
                         color, line_width=line_width)
            # ×ó±ß
            cudaDrawLine(cuda_img, (x1, y2), (x1, y1),
                         color, line_width=line_width)
        else:
            cudaDrawRect(cuda_img, (x1, y1, x2, y2), (0, 0, 0, 160))

        if label is not None:
            # Yellow text overlay
            pad = 4  # padding around text
            self._fontS.OverlayText(
                cuda_img,
                W,
                H,
                label,
                x1 + pad,
                y1 + pad,
                (255, 255, 0, 255),
            )
