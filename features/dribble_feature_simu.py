import cv2
import os
from typing import List, Tuple
import time
from collections import deque
import pygame
from features.bytetrack import Tracker
from collections import defaultdict
import random
import math
import numpy as np

names = {'ball': 0, 'player': 3}


class DribbleFeature:
    def __init__(self, winner_evt, ow, oh, style, count, crop_area, colour=(0, 255, 255)):
        self.colour = colour  # line colour (BGR)
        self.dribble_count: int = 0
        self._last_dribble_times: deque[float] = deque(maxlen=5)
        self.dpm: float = 0.0  # dribbles per minute (rolling window)
        self.winner_evt = winner_evt
        self.ow = ow
        self.oh = oh
        self.score: int = 0  # 目标训练得分

        self.traj_map = []
        self.bottom_x = None
        self.bottom_y = None
        self.up_map = float('inf')
        self.left_map = float('inf')
        self.right_map = float('-inf')
        self.bottom_map = float('inf')
        self.prev_dy_map = None

        self.miss_cnt = 0
        # OpenCV will handle font rendering instead of cudaFont
        self.font_scale_big = 1.67
        self.font_thickness_big = 3
        self.font_scale_small = 1.0
        self.font_thickness_small = 2

        self.job_done = False
        self.is_winner = False

        # 初始化pygame音频
        pygame.mixer.init()

        # 加载音效文件
        self.match_sound = pygame.mixer.Sound('match.wav')     # 匹配时的音效
        self.mismatch_sound = pygame.mixer.Sound('mismatch.wav')  # 不匹配时的音效
        self.mismatch_cnt = 0

        self.tracker = Tracker(q=1000.0, r=0.01, base_dist_thresh=500)

        # 目标区域相关属性（基于帧号）
        self.target_circle = None  # 格式: (center_x, center_y, radius, spawn_frame)
        self.target_radius = 25
        self.next_target_frame = 0  # 下一个目标生成帧号
        self.frame_count = 0  # 当前帧计数器
        self.time_pre = time.time()
        self.fps = 15
        self.target_duration_frames = 5 * self.fps
        self.target_spawn_delay_frames = round(0.1 * self.fps)

        # player info before detect
        self.player_xmin = 0
        self.player_ymin = 0
        self.player_w = 0
        self.player_h = 0
        self.target_center_x = 0
        self.target_center_y = 0
        self.crop_area = crop_area
        self.blue_cnt = 0
        self.player_id = -1
        self.ball_id = -1

        # ---- Ground hit target image using OpenCV instead of cuda ----
        ground_hitting_img = cv2.imread('UI_Button/ground-hitting.png', cv2.IMREAD_UNCHANGED)
        if ground_hitting_img is None:
            raise FileNotFoundError("UI_Button/ground-hitting.png not found")

        # Ensure it is 4-channel BGRA for alpha blend
        if ground_hitting_img.shape[2] == 3:
            h, w = ground_hitting_img.shape[:2]
            alpha = np.full((h, w, 1), 200, dtype=np.uint8)
            ground_hitting_img = np.concatenate([ground_hitting_img, alpha], axis=2)

        self.ground_tgt = cv2.resize(ground_hitting_img, (160, 120), interpolation=cv2.INTER_AREA)
        self.ground_tgt_x, self.ground_tgt_y = 0, 0
        self.ball_locs = []

        self.score_sound = pygame.mixer.Sound('match.wav')  # 得分音效
        self.target_type = 'full v'
        self.dribble_target(style)
        self.target_count = count
        self.cudaDrawGroundHit()  # get the locations for ground hit tgt

    def on_start(self, style, count):
        # WARNING: your original signature here dropped crop_area; keep that in mind.
        self.__init__(self.winner_evt, self.ow, self.oh, style, count, self.crop_area, self.colour)

    # ------------- Core per-frame logic (unchanged, but now on OpenCV frame) -------------
    def on_frame(self, ctx):
        t0 = time.time()
        if self.frame_count > 0 and self.frame_count % 200 == 0:
            time_inter = time.time() - self.time_pre
            print(f"Dribble OD Inference {200 / time_inter:.1f} FPS")
            self.time_pre = time.time()

        self.frame_count += 1

        ow, oh = self.ow, self.oh
        thr = 0.2

        # NOW: cuda_img is a normal OpenCV BGR frame (np.ndarray)
        frame = ctx["cuda_img"]
        img_name = ctx["img_name"]

        print(f"Extract inputs from dictionary takes {time.time() - t0} sec")
        detections = []
        for bb, sc, lb in zip(ctx["bboxes"], ctx["scores"], ctx["labels"]):
            lb = int(lb)
            if lb < 0 or sc < thr:
                continue
            elif lb == 1:
                lb = int(3)  # convert it back to 3 so that it can share the same tracker as Shooting

            x1, y1, x2, y2 = bb
            x1 = max(0, min(int(x1), ow - 1))
            y1 = max(0, min(int(y1), oh - 1))
            x2 = max(0, min(int(x2), ow - 1))
            y2 = max(0, min(int(y2), oh - 1))
            w = x2 - x1
            h = y2 - y1

            if (lb == 3 and w * h < 100000) or (lb == 0 and w * h > 100000):
                continue
            detections.append([x1, y1, x2, y2, float(sc), lb])

        self.cuda_draw_name_id(detections, frame)

        # Draw ground hit target with alpha blending
        self.overlay_image_alpha(frame, self.ground_tgt, self.ground_tgt_x, self.ground_tgt_y)

        foot_top = None
        foot_width = None

        self.tracker.update(detections, dt=0.1)
        tracks = self.tracker.get_tracks()

        player_area_max = 100000
        f_ball_in_hands = False

        if self.player_id != -1 and self.player_id in tracks and tracks[self.player_id]['object_class'] == names['player']:
            tr = tracks[self.player_id]
            w_box, h_box = tr['wh']
            foot_top = (int(tr['pos'][0]), int(tr['pos'][1] + tr['wh'][1] / 2 * 0.85))
            foot_width = tr['wh'][0]
            self.player_xmin = tr['pos'][0] - w_box / 2
            self.player_ymin = tr['pos'][1] - h_box / 2
            self.player_w = w_box
            self.player_h = h_box
        else:
            self.player_id = -1
            for tid, tr in tracks.items():
                if tr['object_class'] == names['player']:
                    w_box, h_box = tr['wh']
                    if w_box * h_box > player_area_max:
                        foot_top_candidate = (int(tr['pos'][0]), int(tr['pos'][1] + tr['wh'][1] / 2 * 0.85))
                        foot_width_candidate = tr['wh'][0]
                        foot_x_in_region = abs(foot_top_candidate[0] - 960) < 100
                        foot_y_in_region = abs(foot_top_candidate[1] - 1000) < 150
                        if foot_x_in_region and foot_y_in_region:
                            player_area_max = w_box * h_box
                            self.player_id = tid
                            self.player_xmin = tr['pos'][0] - w_box / 2
                            self.player_ymin = tr['pos'][1] - h_box / 2
                            self.player_w = w_box
                            self.player_h = h_box
                            foot_top = foot_top_candidate
                            foot_width = foot_width_candidate

        if (
            self.player_id > 0
            and self.ball_id > 0
            and self.ball_id in tracks
            and tracks[self.ball_id]['object_class'] == names['ball']
        ):
            f_ball_in_hands = True
        else:
            ball_id = -1
            if self.player_id >= 0 and self.player_id in tracks:
                tr = tracks[self.player_id]
                w_box, h_box = tr['wh']
                x_bound_top_left = tr['pos'][0] - w_box / 2
                y_bound_top_left = tr['pos'][1] - h_box / 2
                x_bound_bottom_right = tr['pos'][0] + w_box / 2
                y_bound_bottom_right = tr['pos'][1] + h_box / 2
                x_left_bound = max(0, x_bound_top_left - 100)
                x_right_bound = min(x_bound_bottom_right + 100, 1920)
                y_top_bound = max(0, y_bound_top_left - 100)
                y_bottom_bound = min(y_bound_bottom_right + 100, 1200)

                ball_area_max = -float('inf')
                for bid, trk in tracks.items():
                    if trk['object_class'] == names['ball'] and trk['static_age'] <= 5:
                        w, h = trk['wh']
                        if w * h > 200000 or w * h < 5000:
                            continue
                        x = trk['pos'][0]
                        y = trk['pos'][1]
                        if (
                            x_left_bound < x < x_right_bound
                            and y_top_bound < y < y_bottom_bound
                            and w * h > ball_area_max
                        ):
                            f_ball_in_hands = True
                            ball_id = bid
                            ball_area_max = w * h

                if self.ball_id != ball_id:
                    self.traj_map = []
                    self.left_map = float('inf')
                    self.right_map = float('-inf')
                    self.bottom_map = float('inf')
                    self.prev_dy_map = None
                    self.up_map = float('inf')
                    self.ball_id = ball_id

        # draw working area rect; color depends on state
        foot_x_in_region, foot_y_in_region = False, False
        if foot_top is not None and f_ball_in_hands:
            if self.blue_cnt > 0:
                foot_x_in_region = abs(foot_top[0] - 960) < 300
                foot_y_in_region = abs(foot_top[1] - 1000) < 200
            else:
                foot_x_in_region = abs(foot_top[0] - 960) < 100
                foot_y_in_region = abs(foot_top[1] - 1000) < 100

        if f_ball_in_hands and foot_x_in_region and foot_y_in_region:
            color = (0, 255, 0, 200)  # green with alpha
            self.blue_cnt = min(20, self.blue_cnt + 1)
        else:
            color = (255, 255, 255, 200)  # white
            self.blue_cnt = max(0, self.blue_cnt - 1)
            self.player_id = -1
            self.ball_id = -1
            f_ball_in_hands = False

        if self.blue_cnt < 20:
            x1, y1, x2, y2 = self.crop_area
            self.cudaDrawHollowRect(frame, (x1, y1, x2, y2), color, line_width=2)

        if self.blue_cnt > 0 and f_ball_in_hands:
            tr_ball = tracks[self.ball_id]
            ball_center = [int(tr_ball['pos'][0]), int(tr_ball['pos'][1])]
            tr_player = tracks[self.player_id]
            player_center = [int(tr_player['pos'][0]), int(tr_player['pos'][1])]

            kind = self.history_update(ball_center, player_center, foot_top, foot_width, self.target_type)
            self.draw_trajectory(frame)

        self.draw_warning(frame, ow, oh)
        ## save images with results in the image for debugging purpose
        filename = os.path.basename(img_name)  # "frame_000123.png"
        name_only = os.path.splitext(filename)[0]  # "frame_000123"
        processed_img = '/home/mp/Downloads/dribble_processed/' + name_only + '.png'
        os.makedirs('/home/mp/Downloads/dribble_processed', exist_ok=True)
        cv2.imwrite(processed_img, frame)

    # -------------------- History / classification logic (unchanged) --------------------
    def history_update(self, ball_centre, player_center, foot_top, foot_width, target_type):
        dribble_type = None
        now = time.time()
        self.traj_map.append([ball_centre, now])
        cutoff = now - 1.0

        while len(self.traj_map) > 0 and self.traj_map[0][1] < cutoff:
            self.traj_map.pop(0)

        if ball_centre[0] < self.left_map:
            self.left_map = ball_centre[0]
        if ball_centre[0] > self.right_map:
            self.right_map = ball_centre[0]

        if len(self.traj_map) >= 2:
            dy = ball_centre[1] - self.traj_map[-2][0][1]  # +ve = moving down
            if self.prev_dy_map is not None:
                if self.prev_dy_map < 0 <= dy:
                    if self.bottom_map - self.up_map > 200:
                        dribble_type = self._classify_and_count(player_center, foot_top, foot_width, target_type)
                    else:
                        dribble_type = 'unknown'
                    self.reset_trajectory()
                elif self.prev_dy_map > 0 >= dy:
                    self.bottom_map = self.traj_map[-2][0][1]
                    self.bottom_x = (self.traj_map[-2][0][0] + self.traj_map[-2][0][0]) / 2
                    self.bottom_y = (self.traj_map[-2][0][1] + self.traj_map[-2][0][1]) / 2

            self.prev_dy_map = dy
        return dribble_type

    def cuda_draw_name_id(self, dets, frame):
        """Draw bounding boxes for ball/player with OpenCV."""
        for det in dets:
            x1, y1, x2, y2, conf, det_class = det
            if det_class == names['ball']:
                color = (255, 0, 255, 200)  # purple-ish with alpha
            else:
                color = (0, 0, 255, 200)    # red with alpha
            self.cudaDrawHollowRect(frame, (x1, y1, x2, y2), color, line_width=2)

    def reset_trajectory(self):
        self.traj_map = self.traj_map[-2:]
        self.left_map = float('inf')
        self.right_map = float('-inf')
        self.bottom_map = float('inf')
        self.prev_dy_map = None
        self.up_map = self.traj_map[0][0][1]

    def _classify_and_count(self, player_center, foot_top, foot_width, target_type):
        calc_type = "unknown"

        tgt_x, tgt_y = self.ground_tgt_x, self.ground_tgt_y
        tgt_x += 80
        tgt_y += 60

        if (
            self.bottom_x is None
            or abs(self.bottom_x - tgt_x) > 100
            or abs(self.bottom_y - tgt_y) > 200
        ):
            return calc_type
        elif self.right_map < foot_top[0]:
            if (player_center[1] - self.up_map) / abs(player_center[1] - foot_top[1]) > -0.1:
                calc_type = "left high"
            else:
                calc_type = "left low"

        elif self.left_map > foot_top[0]:
            if (player_center[1] - self.up_map) / abs(player_center[1] - foot_top[1]) > -0.1:
                calc_type = "right high"
            else:
                calc_type = "right low"

        elif self.right_map > (foot_top[0] + foot_width / 8) and self.left_map < (foot_top[0] - foot_width / 8):
            if (player_center[1] - self.up_map) / abs(player_center[1] - foot_top[1]) > -0.1:
                calc_type = "crossover high"
            else:
                calc_type = "crossover low"

            if target_type in ("left in-and-out", "right in-and-out", "behind-the-back", "cross-the-legs"):
                calc_type = target_type

        if target_type == calc_type:
            self.dribble_count += 1

        if calc_type == self.target_type:
            self.target_count -= 1
            self.match_sound.play()
            self.mismatch_cnt = 0
        else:
            self.mismatch_cnt += 1
            if self.mismatch_cnt % 3 == 0:
                self.mismatch_sound.play()
        if self.target_count <= 0:
            self.job_done = True

        return calc_type

    # --------------------- OpenCV versions of former cuda* drawing ---------------------
    def draw_trajectory(self, frame) -> None:
        """Overlay trajectory on the frame using cv2.line."""
        pts = [pt for pt, _ in self.traj_map]
        if len(pts) >= 2:
            for pt1, pt2 in zip(pts, pts[1:]):
                p1 = (int(pt1[0]), int(pt1[1]))
                p2 = (int(pt2[0]), int(pt2[1]))
                # Yellow in BGR
                cv2.line(frame, p1, p2, (0, 255, 255), 2)

    def draw_warning(
        self,
        frame,
        ow,
        oh,
        colour: tuple = (0, 255, 255),
        font_scale: float = 1.67,
        thickness: int = 3,
        y_margin: int = 50,
        pad: int = 6,
        line_spacing: int = 10
    ) -> None:

        if not self.job_done:
            lines = [f"{self.target_count} {self.target_type} to go!"]
        elif (not self.winner_evt.is_set()) or self.is_winner:
            lines = ["Winner!"]
            self.is_winner = True
            self.winner_evt.set()
        else:
            lines = ["Finished!"]

        h, w = oh, ow
        y = y_margin

        for text in lines:
            (tw, th), h_max = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            baseline_y = y + h_max
            x = (w - tw) // 2

            tl = (x - pad, baseline_y - th - pad)
            br = (x + tw + pad, baseline_y + pad)

            x1, y1 = tl
            x2, y2 = br
            rect_coords = (x1 - 5, y1 - 5, x2, y2)

            # semi-transparent white background
            bg_rgba = (255, 255, 255, 120)
            self._draw_rect_with_alpha(frame, rect_coords, bg_rgba)

            # Draw text in BGR
            cv2.putText(
                frame,
                text,
                (x, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                colour,
                thickness,
                lineType=cv2.LINE_AA,
            )

            y = y2 + line_spacing

    def calculate_dpm(self):
        now = time.time()
        self._last_dribble_times.append(now)
        if len(self._last_dribble_times) >= 2:
            span_sec = self._last_dribble_times[-1] - self._last_dribble_times[0]
            if span_sec > 0:
                self.dpm = (len(self._last_dribble_times) - 1) / (span_sec / 60.0)

    def cudaDrawHollowRect(self, frame, rect, color, line_width=1):
        """
        OpenCV version of hollow rectangle.
        rect: (x1, y1, x2, y2)
        color: (r,g,b,a) or (b,g,r)
        """
        x1, y1, x2, y2 = rect
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        W, H = self.ow, self.oh
        x1 = int(max(0, min(x1, W - 1)))
        x2 = int(max(0, min(x2, W - 1)))
        y1 = int(max(0, min(y1, H - 1)))
        y2 = int(max(0, min(y2, H - 1)))

        if x2 <= x1 or y2 <= y1 or line_width <= 0 or frame is None:
            return

        if len(color) == 4:
            r, g, b, a = color
            bgr = (b, g, r)
        else:
            # assume BGR
            bgr = color

        cv2.line(frame, (x1, y1), (x2, y1), bgr, line_width)
        cv2.line(frame, (x2, y1), (x2, y2), bgr, line_width)
        cv2.line(frame, (x2, y2), (x1, y2), bgr, line_width)
        cv2.line(frame, (x1, y2), (x1, y1), bgr, line_width)

    def _draw_rect_with_alpha(self, frame, rect, color_rgba):
        """
        Filled rectangle with alpha blending using OpenCV.
        rect: (x1, y1, x2, y2)
        color_rgba: (r, g, b, a)
        """
        x1, y1, x2, y2 = rect
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        H, W = frame.shape[:2]
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H))

        if x2 <= x1 or y2 <= y1:
            return

        r, g, b, a = color_rgba
        alpha = a / 255.0
        bgr = (b, g, r)

        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), bgr, thickness=-1)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    # ---------------- Ground hit placement (logic kept) ----------------
    def cudaDrawGroundHit(self):
        if self.target_type in ('crossover high', 'crossover low'):
            loc_x, loc_y = 960, 920
        elif self.target_type in ('left high', 'left low'):
            loc_x, loc_y = 760, 920
        elif self.target_type in ('right high', 'right low'):
            loc_x, loc_y = 1160, 920
        elif self.target_type == 'behind-the-back':
            loc_x, loc_y = 960, 820
        elif self.target_type == 'cross-the-legs':
            loc_x, loc_y = 960, 920
        elif self.target_type == 'left in-and-out':
            loc_x, loc_y = 810, 920
        elif self.target_type == 'right in-and-out':
            loc_x, loc_y = 1110, 920
        else:
            loc_x, loc_y = 960, 920

        loc_x, loc_y = loc_x - 80, loc_y - 60
        loc_y += 120
        self.ground_tgt_x, self.ground_tgt_y = int(loc_x), int(loc_y)

    def dribble_target(self, style):
        if style == 'Crossover challenge high':
            self.target_type = 'crossover high'
        elif style == 'Crossover challenge low':
            self.target_type = 'crossover low'
        elif style == 'Left dribble challenge high':
            self.target_type = 'left high'
        elif style == 'Left dribble challenge low':
            self.target_type = 'left low'
        elif style == 'Right dribble challenge high':
            self.target_type = 'right high'
        elif style == 'Right dribble challenge low':
            self.target_type = 'right low'
        elif style == 'Behind back':
            self.target_type = 'behind-the-back'
        elif style == 'Cross leg':
            self.target_type = 'cross-the-legs'
        elif style == 'Left V':
            self.target_type = 'left in-and-out'
        elif style == 'Right V':
            self.target_type = 'right in-and-out'
        else:
            self.target_type = 'v'

    # ---------------- Generic overlay helper for ground_tgt ----------------
    def overlay_image_alpha(self, frame, overlay_img, x, y):
        """
        Overlay overlay_img (BGRA or BGR) onto frame at (x,y) with per-pixel or uniform alpha.
        """
        h, w = overlay_img.shape[:2]
        H, W = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)

        if x1 >= x2 or y1 >= y2:
            return

        overlay_roi = overlay_img[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]
        frame_roi = frame[y1:y2, x1:x2]

        if overlay_roi.shape[2] == 4:
            # BGRA
            overlay_bgr = overlay_roi[:, :, :3].astype(np.float32)
            alpha = overlay_roi[:, :, 3:4].astype(np.float32) / 255.0
            frame_roi_f = frame_roi.astype(np.float32)
            blended = alpha * overlay_bgr + (1.0 - alpha) * frame_roi_f
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            alpha = 0.8
            cv2.addWeighted(overlay_roi, alpha, frame_roi, 1 - alpha, 0, frame_roi)
            frame[y1:y2, x1:x2] = frame_roi
