import numpy as np
from enum import Enum
import cv2
from features.bytetrack import Tracker
from manual_bev import MANUAL_BEV
import time
from itertools import pairwise  # Python 3.10+. If on 3.8/3.9, see fallback below.
import pygame
import os
from features.shooting_feature import map_style_to_key_pts_index, chaikin
from compute_AI_advice import compute_shooting_ai_advice
from calc_shooting_metrics import analyze_sideview_shot
from calc_player import analyze_player_features
from collections import deque
import json

class Event(Enum):
    NO_SHOOTING = 0
    SHOOTING = 1
    IN_BASKET = 2
    FAILED = 3


names = {'ball': 0, 'bib': 1, 'player': 2}
reverse_names = {v: k for k, v in names.items()}
key_pts = [[468, 181], [928, 181], [468, 290], [928, 290], [468, 397], [928, 397], [468, 488], [928, 488],
         [701, 581], [701, 402], [701, 840], [327, 710], [1069, 710], [124, 443], [1273, 443], [81, 181],
         [1317, 181], [1123, 181], [275, 181], [300, 443], [1100, 443]]

corners_left = np.load("corners_left.npy")
corners_right = np.load("corners_right.npy")
left_rim_x, left_rim_y = int(corners_left[4][0]), int(corners_left[4][1])    # left side shifts to the right
right_rim_x, right_rim_y = int(corners_right[4][0]), int(corners_right[4][1])

def _as_xy(p):
    # ensure (int x, int y)
    x, y = p[:2]
    return int(x), int(y)


def rgba_to_bgr(color):
    """
    Convert an RGBA or RGB tuple (R,G,B,(A)) to BGR tuple for cv2.
    Alpha is ignored.
    """
    if color is None:
        return (0, 255, 0)
    if len(color) == 4:
        r, g, b, _ = color
    elif len(color) == 3:
        r, g, b = color
    else:
        return (0, 255, 0)
    return (b, g, r)


def draw_polyline(img, traj, color, width=2, show_endpoints=False):
    """Draws a connected polyline for a trajectory on a cv2/numpy image."""
    pts = [_as_xy(p) for p in traj if p is not None]
    if not pts:
        return
    bgr = rgba_to_bgr(color)
    if len(pts) == 1:
        # degenerate: draw a dot
        cv2.circle(img, pts[0], radius=max(1, width + 1), color=bgr, thickness=-1)
        return
    for p0, p1 in pairwise(pts):
        cv2.line(img, p0, p1, bgr, thickness=width, lineType=cv2.LINE_AA)
    if show_endpoints:
        cv2.circle(img, pts[0], radius=width + 2, color=bgr, thickness=-1)  # start
        cv2.circle(img, pts[-1], radius=width + 2, color=bgr, thickness=-1)  # end


def draw_cross_cv(dst, x: int, y: int,
                  size: int = 12, thickness: int = 2,
                  color=(255, 0, 0)):
    """Draw a cross centered at (x,y) on a cv2 image."""
    h, w = dst.shape[:2]
    x0 = max(0, x - size)
    x1 = min(w - 1, x + size)
    y0 = max(0, y - size)
    y1 = min(h - 1, y + size)

    cv2.line(dst, (x0, y), (x1, y), color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.line(dst, (x, y0), (x, y1), color, thickness=thickness, lineType=cv2.LINE_AA)


class ShootingFeature:
    def __init__(self, w, h, dt, side, style):
        self.side = side
        if side == 'left':
            self.basket_x = left_rim_x
            self.ring_loc_y = left_rim_y
        else:
            self.basket_x = right_rim_x
            self.ring_loc_y = right_rim_y
        self.style = style
        self.key_pts_index, self.shoot_times, self.shoot_dist_thresh = map_style_to_key_pts_index(style)

        # This will be a numpy BGR image now
        self.cuda_img = None

        # Load BEV image as BGR
        img_bgr = cv2.imread('BEV/RBEV_1400.png')
        if img_bgr is None:
            raise FileNotFoundError("BEV/RBEV_1400.png not found")
        self.bev_img = img_bgr.copy()           # working BEV image (BGR)
        self.original_bev_img = img_bgr.copy()  # pristine copy to reset

        # Tracker
        self.tracker = Tracker(q=1000.0, r=0.01)

        # Shooting statistics
        self.shooting_locs = []
        self.shooting_result = []
        self.latest_shooting_loc = None
        self.total_shooting = 0
        self.bingo_2pt_num = 0
        self.bingo_3pt_num = 0
        self.status = Event.NO_SHOOTING
        self.prev_status = Event.NO_SHOOTING
        self.frame_cnt = 0
        self.shooting_start = 0
        self.shooting_start_array = []
        self.shooting_end_array = []
        self.fail_cnt = 0
        self.bib_cnt = 0
        self.bib_confirmed = False
        self.bib_spatial_miss_cnt = 0
        self.ball_id = -1
        self.img_width = w
        self.img_height = h
        self.dt = dt
        self.time_pre = time.time()
        self.traj = []
        self.traj_all_make = []
        self.traj_all_fail = []

        pygame.mixer.init()
        self.match_sound = pygame.mixer.Sound('match.wav')      # success sound
        self.mismatch_sound = pygame.mixer.Sound('mismatch.wav')  # fail sound

        # BEV display target size
        self.target_w, self.target_h = 500, 500

        self.total_attempts_21 = [0] * 21
        self.total_makes_21 = [0] * 21
        self.accuracy = 0
        self.make_pos = []
        self.fail_pos = []
        self.shoot_du = 0
        self.q_ball = deque(maxlen=60)
        self.q_player = deque(maxlen=60)

        # BEV class
        H, W = 1200, 1920
        img = np.zeros((H, W, 3), dtype=np.uint8)  # dummy
        self.bev_class = MANUAL_BEV(img, None, False, side)

        # draw key pts in the BEV
        for key_pt_ind in self.key_pts_index:
            key_pt = key_pts[key_pt_ind]
            cv2.circle(self.bev_img, _as_xy(key_pt), radius=15,
                       color=(0, 255, 0), thickness=-1)

        self.drift_dir_lr_arr = []
        self.drift_dir_ud_arr = []
        self.release_height_ratio_arr = []
        self.release_lateral_ratio_arr = []
        self.bias_short_long_arr = []
        self.bias_left_right_arr = []
        self.flat_shot_arr = []

    def search_key_pt(self, x_bev, y_bev):
        min_dist = self.shoot_dist_thresh
        ind_found = -1
        for key_pt_ind in self.key_pts_index:
            key_pt = key_pts[key_pt_ind]
            diff = [key_pt[0] - x_bev, key_pt[1] - y_bev]
            dist = np.linalg.norm(diff)
            if key_pt_ind == 8:  # free throw
                dx = max(0, abs(key_pt[0] - x_bev) - 100)
                dy = abs(key_pt[1] - y_bev)
                dist = np.linalg.norm([dx, dy])

            if dist < min_dist:
                min_dist = dist
                ind_found = key_pt_ind

        return ind_found

    def check_finish(self):
        index = self.key_pts_index
        for ind in index:
            att = self.total_attempts_21[ind]
            if att < self.shoot_times:
                return False
        return True

    def calc_accuracy(self):
        sum_make = 0
        sum_att = 0
        for ind in self.key_pts_index:
            sum_make += self.total_makes_21[ind]
            sum_att += self.total_attempts_21[ind]
        if sum_att > 0:
            acc = sum_make / sum_att
        else:
            acc = 0
        return acc

    def update_basket(self, ball_track):
        # use bib track's location to update self.ring_loc_y
        w, h = ball_track['wh']
        ring_loc_y_bib = ball_track['pos'][1] - h / 2  # centroid - h/2
        if abs(ring_loc_y_bib - self.ring_loc_y) < 150:  # avoid wild points
            self.ring_loc_y = self.ring_loc_y * 0.6 + ring_loc_y_bib * 0.4
            self.basket_x = self.basket_x * 0.6 + ball_track['pos'][0] * 0.4

    def on_frame(self, ctx: dict):
        """
        ctx must provide:
          - 'side': 'left' or 'right'
          - 'bboxes', 'scores', 'labels' (from OD)
          - OPTIONAL: 'cuda_img' as a BGR numpy image for drawing
        """
        ow, oh = 1920, 1200
        thr = 0.2

        self.side = ctx["side"]
        img_name = ctx["img_name"]
        # expect a BGR image here if you want overlays
        if "cuda_img" in ctx:
            self.cuda_img = ctx["cuda_img"]

        detections = []
        for bb, sc, lb in zip(ctx["bboxes"], ctx["scores"], ctx["labels"]):
            lb = int(lb)
            if lb < 0 or sc < thr:
                continue
            x1, y1, x2, y2 = bb
            x1 = max(0, min(int(x1), ow - 1))
            y1 = max(0, min(int(y1), oh - 1))
            x2 = max(0, min(int(x2), ow - 1))
            y2 = max(0, min(int(y2), oh - 1))

            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            xyxy = [x1, y1, x2, y2]
            conf = float(sc)
            obj_class = lb
            detection = [*xyxy, conf, obj_class]
            detections.append(detection)

        self.calc_event(detections)

        if self.cuda_img is not None:
            # draw the shooting loc in the BEV if a shot just finished
            if ((self.status == Event.IN_BASKET or self.status == Event.FAILED)
                    and self.status != self.prev_status
                    and self.latest_shooting_loc is not None):
                x, y = self.latest_shooting_loc
                x_bev, y_bev = self.bev_class.calc_bev_pts(x, y)
                x_bev = 1400 - x_bev  # flip
                print(f"Bev_loc found : {(x_bev, y_bev)}, last shooting loc: {(x, y)}")
                key_pt_ind_found = self.search_key_pt(x_bev, y_bev)

                if key_pt_ind_found != -1:
                    self.total_attempts_21[key_pt_ind_found] += 1
                    if self.status == Event.IN_BASKET:
                        self.total_makes_21[key_pt_ind_found] += 1
                        self.traj_all_make.append(self.traj)
                    else:
                        self.traj_all_fail.append(self.traj)

                if self.status == Event.IN_BASKET:
                    self.make_pos.append([x_bev, y_bev])
                elif self.status == Event.FAILED:
                    self.fail_pos.append([x_bev, y_bev])

                self.traj = []
                self.cuda_draw_each_spot_result(self.status)

            # Debug state box
            bg_color = (120, 120, 120)
            x1, y1, x2, y2 = 900, 20, 1050, 50
            cv2.rectangle(self.cuda_img, (x1, y1), (x2, y2),
                          color=bg_color, thickness=-1)

            if self.status == Event.IN_BASKET:
                notes = "MAKE"
            elif self.status == Event.FAILED:
                notes = "FAILED"
            elif self.status == Event.NO_SHOOTING:
                notes = "NO_SHOOT"
            elif self.status == Event.SHOOTING:
                notes = "SHOOTING"
            else:
                notes = "HAHA"

            cv2.putText(
                self.cuda_img,
                notes,
                (905, 45),  # roughly centered vertically in [20,50]
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            # Warn user to calibrate camera if too many bib detections fail spatial check
            if self.bib_spatial_miss_cnt >= 3:
                warn_text = "Please calibrate camera from settings"
                warn_color = (0, 80, 255)  # orange in BGR
                text_size, _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                tw, th = text_size
                cv2.rectangle(self.cuda_img, (748, 52), (752 + tw, 68 + th), (0, 0, 0), -1)
                cv2.putText(self.cuda_img, warn_text, (750, 66 + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, warn_color, 2, cv2.LINE_AA)

            # Display BEV: resize & overlay
            bev_resized = cv2.resize(self.bev_img, (self.target_w, self.target_h),
                                     interpolation=cv2.INTER_LINEAR)
            if self.side == 'left':
                ox, oy = 1000, 800
            else:
                ox, oy = 420, 800
            self._overlay_image(self.cuda_img, bev_resized, ox, oy, alpha=0.8)

            # Shooting text + stats
            self.cuda_draw_shoot()

            # Draw detections on frame
            self.cuda_draw_name_id(detections)

            # Finished condition: draw summary + trajectories
            if self.check_finish():
                result = compute_shooting_ai_advice(
                    self.release_height_ratio_arr,
                    self.release_lateral_ratio_arr,
                    self.flat_shot_arr,
                    self.bias_short_long_arr,
                    self.bias_left_right_arr,
                    f_pinyin=False
                )
                accuracy = self.calc_accuracy() * 100.0

                line = f"FINISHED,Accuracy:{accuracy:.1f}%"
                cv2.putText(
                    self.cuda_img,
                    line,
                    (5, 545),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                y1 = 545
                for a in result["advice"]:
                    y1 += 80
                    metrics_s = json.dumps(a.get("metrics", {}), ensure_ascii=False)
                    # line = (
                    #     f'{a.get("title","")}-->{a.get("why","")}\n'
                    #     f'-->{metrics_s}-->{a.get("cue","")}'
                    #     )
                    line = (
                        f'{a.get("title", "")}->{a.get("cue", "")}'
                    )
                    cv2.putText(
                        self.cuda_img,  #
                        line,  # ÎÄ±¾
                        (5, y1 + 5),  # ×óÉÏ½Ç×ø±ê
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),  # ´¿»ÆÉ«×ÖÌå  Blue
                        2,
                        cv2.LINE_AA
                        )

                for traj in self.traj_all_make:
                    draw_polyline(self.cuda_img, traj,
                                  color=(173, 216, 230, 60),  # light blue
                                  width=3, show_endpoints=False)

                for traj in self.traj_all_fail:
                    draw_polyline(self.cuda_img, traj,
                                  color=(255, 182, 193, 60),  # light red
                                  width=3, show_endpoints=False)

        filename = os.path.basename(img_name)  # "frame_000123.png"
        name_only = os.path.splitext(filename)[0]  # "frame_000123"
        processed_img = '/container_folder/shoot_processed/' + name_only + '.png'
        os.makedirs('/container_folder/shoot_processed', exist_ok=True)
        cv2.imwrite(processed_img, self.cuda_img)

    def _overlay_image(self, base, overlay, x, y, alpha=0.8):
        """Alpha blend 'overlay' onto 'base' at top-left (x,y)."""
        if base is None:
            return
        h, w = overlay.shape[:2]
        H, W = base.shape[:2]
        if x >= W or y >= H:
            return
        x_end = min(W, x + w)
        y_end = min(H, y + h)
        ow = x_end - x
        oh = y_end - y
        if ow <= 0 or oh <= 0:
            return

        roi = base[y:y_end, x:x_end]
        ov = overlay[0:oh, 0:ow]

        blended = cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0)
        base[y:y_end, x:x_end] = blended

    def queue_tgt(self, tracks):

        ball_pos = []
        for _, tr in tracks.items():
            tr_vel = np.linalg.norm(tr['vel'])
            if tr['object_class'] == names['ball'] and not tr['coasted'] and tr['age'] > 5 and tr_vel > 10:
                ball_pos = tr['pos']
                x, y = ball_pos
                self.q_ball.append([x, y])
                break # assume there is only one ball for each half court
        else:
            self.q_ball.append([])  # by this way, we can make sure the ball and player are sync'ed

        for _, tr in tracks.items():
            if tr['object_class'] == names['player'] and not tr['coasted'] and tr['age'] > 10:
                player_pos = tr['pos']
                x, y = player_pos
                w, h = tr['wh']
                y0 = y + h/2  # this is foot location
                x_bev, y_bev = self.bev_class.calc_bev_pts(x, y0)
                x_bev = 1400 - x_bev # it needs to flip
                key_pt_ind_found = self.search_key_pt(x_bev, y_bev)
                if key_pt_ind_found != -1:
                    new_val = [x, y, w, h]
                    self.q_player.append(new_val)
                    break   # assume there is only one player
        else:
            if len(ball_pos) > 0:
                player_id = self.find_closest_player(tracks, ball_pos)
                if player_id >= 0:
                    tr = tracks[player_id]
                    player_pos = tr['pos']
                    x, y = player_pos
                    w, h = tr['wh']
                    new_val = [x, y, w, h]
                    self.q_player.append(new_val)
                else:
                    self.q_player.append([])
            else:
                self.q_player.append([])

    def calc_event(self, detections):
        self.frame_cnt += 1
        if self.frame_cnt == 1327:
            print(1)

        self.tracker.update(detections, self.dt)
        tracks = self.tracker.get_tracks()
        self.queue_tgt(tracks)
        if self.frame_cnt > 0 and self.frame_cnt % 200 == 0:
            time_inter = time.time() - self.time_pre
            print(f"Shooting OD Inference {200 / time_inter:.1f} FPS")
            self.time_pre = time.time()

        self.prev_status = self.status   # detect rising edge

        if self.status == Event.NO_SHOOTING:
            f_shooting = self.search_shooting(tracks)
            if f_shooting:
                self.status = Event.SHOOTING
                self.shoot_du = 1

        elif self.status == Event.SHOOTING:
            tid = self.ball_id
            self.shoot_du += 1
            if tid not in tracks:  # lost the shooting ball
                self.status = Event.NO_SHOOTING
                self.traj = []
                self.bib_confirmed = False
            else:
                ball_track = tracks[tid]
                if ball_track['object_class'] == names['bib']:
                    self.bib_confirmed = True
                    self.update_basket(ball_track)

                elif ball_track['object_class'] == names['ball']:
                    if ball_track['pos'][1] < self.ring_loc_y:
                        self.traj.append(ball_track['pos'])

                    ball_pos = ball_track['pos']
                    if self.bib_confirmed:
                        if ball_pos[1] < self.ring_loc_y:
                            # Ball bounced back above the ring — not a make
                            self.bib_confirmed = False
                        elif abs(ball_pos[0] - self.basket_x) < 80:
                            # Ball fell through below the ring near basket_x — MAKE
                            self.status = Event.IN_BASKET
                            self.save_results()
                            self.bingo_2pt_num += 1
                            self.bib_confirmed = False
                        elif ball_pos[1] > self.ring_loc_y + 80 or self.shoot_du > 60:
                            # Ball fell off but not near basket — FAIL
                            self.bib_spatial_miss_cnt += 1
                            if len(self.traj) <= 5:
                                self.status = Event.NO_SHOOTING
                                self.traj = []
                            else:
                                self.status = Event.FAILED
                                self.save_results()
                            self.bib_confirmed = False
                    else:
                        if (ball_pos[1] > self.ring_loc_y + 80) or (self.shoot_du > 60):
                            if len(self.traj) <= 5:  # this might be passing or false shooting detection
                                self.status = Event.NO_SHOOTING
                                self.traj = []
                            else:
                                self.status = Event.FAILED
                                self.save_results()

        elif self.status == Event.FAILED:
            self.mismatch_sound.play()
            if self.fail_cnt == 0:
                self.fail_cnt = 5
            elif self.fail_cnt == 1:
                self.status = Event.NO_SHOOTING
                self.fail_cnt = 0
            else:
                self.fail_cnt -= 1

        elif self.status == Event.IN_BASKET:
            self.match_sound.play()
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
        self.traj = chaikin(self.traj, iterations=1)  # this is to remove tail and smooth the traj
        metrics = analyze_sideview_shot(self.traj, self.basket_x, self.ring_loc_y)
        if metrics.ok:
            # 1) Flat shot flag (single shot)
            self.flat_shot_arr.append(metrics.flat_shot)
            # 2) For bias patterns across shots
            self.bias_short_long_arr.append(metrics.short_long)
            self.bias_left_right_arr.append(metrics.left_right)

    def search_shooting(self, tracks):
        f_shooting = False

        for tid, tr in tracks.items():
            tr_vel = np.linalg.norm(tr['vel'])
            if tr['object_class'] == names['ball'] and not tr['coasted'] and tr_vel > 50:
                ball_pos = tr['pos']
                if ball_pos[1] < self.ring_loc_y + 20:
                    f_shooting = True
                    self.ball_id = tid
                    self.shooting_start = self.frame_cnt
                    self.calc_player_derived()  # calculate player metrics while shooting is confirmed
                    self.traj.append(ball_pos)
                    break

        return f_shooting

    def find_closest_player(self, tracks, ball_pos):  # when shooting is detected, we need to find the shooter
        smallest_dist = 1e9
        player_id = -1
        for tid, tr in tracks.items():
            if tr['object_class'] == names['player']:
                player_pos = tr['pos']
                # print(f"basket x is {self.basket_x}")
                # this condition to make sure ball posn is between player and basket
                dist = np.linalg.norm(ball_pos - player_pos)
                if dist < smallest_dist:  # and h / w > 2:  # only, locating the shooter
                    smallest_dist = dist
                    player_id = tid

        return player_id

    def calc_player_derived(self):
        player_q = self.q_player
        ball_q = self.q_ball
        # need to calculate following properties
        # shooting loc, drift, release height ratio, release lateral ratio
        # need to calculate shooting loc with queue info
        feat = analyze_player_features(player_q, ball_q)
        if feat is not None:
            x, y, w, h = player_q[feat["release_idx"]]

            foot_x = x
            foot_y = y + 0.5 * h

            self.latest_shooting_loc = [foot_x, foot_y]
            self.release_height_ratio_arr.append(feat["release_height_ratio"])
            self.release_lateral_ratio_arr.append(feat["release_lateral_ratio"])

    def cuda_draw_each_spot_result(self, status):
        # reset BEV image
        self.bev_img = self.original_bev_img.copy()

        # redraw keypoints
        for key_pt_ind in self.key_pts_index:
            key_pt = key_pts[key_pt_ind]
            cv2.circle(self.bev_img, _as_xy(key_pt), radius=15,
                       color=(0, 255, 0), thickness=-1)

        # redraw shoot marks
        for pos in self.make_pos:
            x, y = int(pos[0]), int(pos[1])
            cv2.circle(self.bev_img, (x, y), radius=12,
                       color=(0, 0, 255), thickness=-1)
        for pos in self.fail_pos:
            x, y = int(pos[0]), int(pos[1])
            draw_cross_cv(self.bev_img, x, y)

        # draw text stats at each key pt
        for ind in self.key_pts_index:
            key = key_pts[ind]
            x0, y0 = key
            line = f"{self.total_makes_21[ind]}/{self.total_attempts_21[ind]}"
            cv2.putText(
                self.bev_img,
                line,
                (x0 - 30, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
                cv2.LINE_AA
            )

    def cuda_draw_shoot(self):
        if self.cuda_img is None:
            return

        lines = f"Each spot x {self.shoot_times}"
        if self.side == 'right':
            x0, y0 = 550, 1130
        else:
            x0, y0 = 1130, 1130

        cv2.putText(
            self.cuda_img,
            lines,
            (x0 + 2, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

        lines = f"Makes:{self.bingo_2pt_num + self.bingo_3pt_num} / Total:{self.total_shooting}: {self.frame_cnt}"
        if self.side == 'right':
            x0, y0 = 500, 1160
        else:
            x0, y0 = 1080, 1160

        cv2.putText(
            self.cuda_img,
            lines,
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

    def _cv_draw_hollow_rect(self, img, rect, color, line_width=2):
        """Draw rectangle outline on a cv2 image."""
        if img is None:
            return

        x1, y1, x2, y2 = rect
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        W, H = self.img_width, self.img_height
        x1 = int(max(0, min(x1, W - 1)))
        x2 = int(max(0, min(x2, W - 1)))
        y1 = int(max(0, min(y1, H - 1)))
        y2 = int(max(0, min(y2, H - 1)))

        if x2 <= x1 or y2 <= y1 or line_width <= 0:
            return

        cv2.rectangle(img, (x1, y1), (x2, y2),
                      color=color, thickness=line_width)

    def cuda_draw_name_id(self, dets):
        """Draw OD bounding boxes on self.cuda_img."""
        if self.cuda_img is None:
            return

        for det in dets:
            x1, y1, x2, y2, conf, det_class = det
            if conf > 0.2:
                if det_class == names['ball']:
                    color = (255, 0, 255)    # magenta
                elif det_class == names['player']:
                    color = (0, 255, 0)      # green
                elif det_class == names['bib']:
                    color = (0, 255, 255)    # yellow-ish
                else:
                    color = (0, 0, 255)      # red

                self._cv_draw_hollow_rect(
                    self.cuda_img,
                    (x1, y1, x2, y2),
                    color,
                    line_width=2,
                )
