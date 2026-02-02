import cv2
import os
from typing import List, Tuple
import time
from collections import deque
from pathlib import Path
from jetson_utils import cudaDrawLine, cudaDrawRect, cudaFont, cudaAllocMapped, cudaFromNumpy, cudaResize, cudaOverlay
import pygame
from features.bytetrack import Tracker
import numpy as np
import calibrations  # For dynamic language switching
from features.cuda_text_utils import overlay_text, overlay_text_centered

# Detection class names mapping
names = {'ball': 0, 'player': 2}


# ============================================================================
# Tunable Constants - adjust these for different camera setups or requirements
# ============================================================================
class DribbleConfig:
    """Configuration constants for dribble detection and tracking."""
    # Tracker parameters
    TRACKER_Q = 200.0                   # Kalman filter process noise
    TRACKER_R = 0.01                    # Kalman filter measurement noise
    TRACKER_BASE_DIST = 350             # Base distance threshold for tracking

    # Player detection
    PLAYER_AREA_MIN = 100000            # Min area for valid player detection
    PLAYER_POSITION_X_TIGHT = 200       # Tight X tolerance for initial detection
    PLAYER_POSITION_Y_TIGHT = 100       # Tight Y tolerance for initial detection
    PLAYER_POSITION_X_LOOSE = 400       # Loose X tolerance after confirmed
    PLAYER_POSITION_Y_LOOSE = 200       # Loose Y tolerance after confirmed
    PLAYER_CENTER_X = 960               # Expected player X position
    PLAYER_CENTER_Y = 1000              # Expected player Y position

    # Ball detection
    BALL_AREA_MAX = 200000              # Max area for valid ball
    BALL_AREA_MIN = 5000                # Min area for valid ball
    BALL_SEARCH_MARGIN = 200            # Margin around player to search for ball

    # Dribble detection
    MIN_DRIBBLE_HEIGHT = 100            # Min vertical travel for valid dribble
    HIT_TARGET_X_THRESHOLD = 120        # X distance threshold for hitting target
    HIT_TARGET_Y_THRESHOLD = 150        # Y distance threshold for hitting target
    SPECIAL_DRIBBLE_WIDTH = 250         # Min width for special dribbles
    TRAJECTORY_GAP_THRESHOLD = 160      # Gap threshold for behind-back detection

    # Size filtering (to distinguish ball from player misdetections)
    PLAYER_MISDETECT_AREA = 80000       # Area below which player detection is likely ball
    BALL_MISDETECT_AREA = 100000        # Area above which ball detection is likely player

    # Timing
    TRAJECTORY_WINDOW_SEC = 1.0         # Time window to keep trajectory points
    READY_GO_DURATION = 3.0             # Duration to show "Ready, Go!"
    HINT_DURATION = 2.0                 # Duration to show hints

    # Blue count (ready state) thresholds
    BLUE_CNT_MAX = 20                   # Max blue count value
    BLUE_CNT_READY = 20                 # Blue count needed to be "ready"

    # Idle detection
    IDLE_TIMEOUT_SEC = 180              # 3 minutes idle timeout
def _truncate_by_chars(text: str, max_chars: int) -> str:
    """Truncate text to max_chars with ellipsis if needed."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return text[:max_chars - 3] + "..."


def _load_image_as_cuda_rgba(filepath, alpha=200):
    """Load an image file and convert to CUDA RGBA format.

    Args:
        filepath: Path to image file
        alpha: Alpha channel value (0-255)

    Returns:
        cudaImage or None if file not found
    """
    if not Path(filepath).exists():
        print(f"[DribbleFeature] Warning: Image not found: {filepath}")
        return None
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        print(f"[DribbleFeature] Warning: Failed to read image: {filepath}")
        return None
    numpy_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = numpy_img.shape[:2]
    A = np.full((H, W), alpha, dtype=np.uint8)
    rgba_np = np.ascontiguousarray(np.dstack((numpy_img, A)).astype(np.uint8))
    return cudaFromNumpy(rgba_np)


class DribbleFeature:
    def __init__(self, winner_evt, ow, oh, style, count, colour=(0, 255, 255)):
        self.cfg = DribbleConfig()  # Store config reference
        self.colour = colour  # Line colour (BGR)
        self.dribble_count: int = 0
        self._last_dribble_times: deque[float] = deque(maxlen=5)
        self.dpm: float = 0.0  # Dribbles per minute (rolling window)
        self.winner_evt = winner_evt
        self.ow = ow
        self.oh = oh
        self.score: int = 0

        # Trajectory tracking - use deque for O(1) popleft
        self.traj_map: deque = deque()
        self.bottom_x = None
        self.bottom_y = None
        self.up_map = float('inf')
        self.left_map = float('inf')
        self.right_map = float('-inf')
        self.bottom_map = float('inf')
        self.prev_dy_map = None

        self.miss_cnt = 0
        self.fontM = cudaFont(size=48)
        self._fontS = cudaFont(size=32)
        self._fontL = cudaFont(size=72)
        self.job_done = False
        self.is_winner = False

        # Audio initialization with error handling
        self.audio_enabled = False
        self.match_sound = None
        self.mismatch_sound = None
        try:
            pygame.mixer.init()
            if Path('match.wav').exists():
                self.match_sound = pygame.mixer.Sound('match.wav')
            if Path('mismatch.wav').exists():
                self.mismatch_sound = pygame.mixer.Sound('mismatch.wav')
            self.audio_enabled = self.match_sound is not None or self.mismatch_sound is not None
        except Exception as e:
            print(f"[DribbleFeature] Audio disabled: {e}")

        self.mismatch_cnt = 0

        # Initialize tracker with config parameters
        self.tracker = Tracker(
            q=self.cfg.TRACKER_Q,
            r=self.cfg.TRACKER_R,
            base_dist_thresh=self.cfg.TRACKER_BASE_DIST
        )

        # Frame counting
        self.frame_count = 0
        self.time_pre = time.time()

        # Player tracking state
        self.player_xmin = 0
        self.player_ymin = 0
        self.player_w = 0
        self.player_h = 0
        self.target_center_x = 0
        self.target_center_y = 0
        self.crop_area = [720, 80, 1200, 1140]
        self.blue_cnt = 0
        self.player_id = -1
        self.ball_id = -1

        # Load ground hitting target image with error handling
        ground_hitting_cuda = _load_image_as_cuda_rgba('UI_Button/ground-hitting.png', alpha=200)
        if ground_hitting_cuda is not None:
            self.ground_tgt = cudaAllocMapped(width=160, height=120, format=ground_hitting_cuda.format)
            cudaResize(ground_hitting_cuda, self.ground_tgt)
        else:
            self.ground_tgt = None
        self.ground_tgt_x, self.ground_tgt_y = 0, 0

        self.target_type = 'full v'
        self.target_type_zh = '交叉运球-高手'
        self.dribble_target(style)

        # "Ready, Go!" display tracking
        self.ready_go_start_time = None
        self.ready_go_duration = self.cfg.READY_GO_DURATION
        self.was_ready = False  # Track previous ready state to detect transition

        self.target_count = count
        self.cudaDrawGroundHit()  # Get the locations for ground hit target
        self.height_hint = None
        self.height_hint_time = None
        self.height_hint_duration = self.cfg.HINT_DURATION
        self.f_no_hitting_tgt = False

        # Validation checks for behind-the-back and cross-the-legs dribbles
        self.validation_hint = None
        self.validation_hint_time = None
        self.validation_warning = None
        self.validation_warning_time = None

        # Idle detection
        self.last_activity_time = time.time()
        self.idle_timeout_sec = self.cfg.IDLE_TIMEOUT_SEC
        self._prev_dribble_count = 0

        # Track CUDA resources for cleanup
        self._cuda_resources = [self.ground_tgt]

    def __del__(self):
        """Clean up resources on destruction."""
        try:
            self._cuda_resources = None
            self.tracker = None
        except Exception:
            pass

    def on_frame(self, ctx):
        """Process a frame for dribble detection."""
        # Periodic FPS logging
        if self.frame_count > 0 and self.frame_count % 200 == 0:
            time_inter = time.time() - self.time_pre
            print(f"Dribble OD Inference {200 / time_inter:.1f} FPS")
            self.time_pre = time.time()

        self.frame_count += 1
        cfg = self.cfg
        det_cfg = ctx["cfg"]
        ow, oh = ctx["src_W"], ctx["src_H"]
        thr = det_cfg["score_threshold"]
        cuda_img = ctx["cuda_img"]

        # Build detections list
        detections = []
        for bb, sc, lb in zip(ctx["bboxes"], ctx["scores"], ctx["labels"]):
            lb = int(lb)
            if lb < 0 or sc < thr:  # Filter by score threshold
                continue
            elif lb == 1:
                lb = 2  # Convert to player class for shared tracker

            x1, y1, x2, y2 = bb
            x1 = max(0, min(int(x1), ow - 1))
            y1 = max(0, min(int(y1), oh - 1))
            x2 = max(0, min(int(x2), ow - 1))
            y2 = max(0, min(int(y2), oh - 1))
            w = x2 - x1
            h = y2 - y1

            # Filter misdetections based on area
            area = w * h
            if (lb == 2 and area < cfg.PLAYER_MISDETECT_AREA) or \
               (lb == 0 and area > cfg.BALL_MISDETECT_AREA):
                continue
            detections.append([x1, y1, x2, y2, float(sc), lb])

        self.cuda_draw_name_id(detections, cuda_img)

        # Draw the ground hit target
        if self.ground_tgt is not None:
            cudaOverlay(self.ground_tgt, cuda_img, self.ground_tgt_x, self.ground_tgt_y)

        foot_top = None
        foot_width = None

        self.tracker.update(detections, dt=0.1)
        tracks = self.tracker.get_tracks()
        self.cuda_draw_tracks(tracks, cuda_img)

        player_area_max = cfg.PLAYER_AREA_MIN
        f_ball_in_hands = False

        # Check if existing player ID is still valid
        if (self.player_id != -1 and self.player_id in tracks
                and tracks[self.player_id]['object_class'] == names['player']):
            tr = tracks[self.player_id]
            w_box, h_box = tr['wh']
            foot_top = (int(tr['pos'][0]), int(tr['pos'][1] + tr['wh'][1] / 2 * 0.85))
            foot_width = tr['wh'][0]
            self.player_xmin = tr['pos'][0] - w_box / 2
            self.player_ymin = tr['pos'][1] - h_box / 2
            self.player_w = w_box
            self.player_h = h_box
        else:
            # Search for new player
            self.player_id = -1
            for tid, tr in tracks.items():
                if tr['object_class'] == names['player']:
                    w_box, h_box = tr['wh']
                    if w_box * h_box > player_area_max:
                        foot_top = (int(tr['pos'][0]), int(tr['pos'][1] + tr['wh'][1] / 2 * 0.85))
                        foot_width = tr['wh'][0]
                        foot_x_in_region = abs(foot_top[0] - cfg.PLAYER_CENTER_X) < cfg.PLAYER_POSITION_X_TIGHT
                        foot_y_in_region = abs(foot_top[1] - cfg.PLAYER_CENTER_Y) < cfg.PLAYER_POSITION_Y_TIGHT
                        if foot_x_in_region and foot_y_in_region:
                            player_area_max = w_box * h_box
                            self.player_id = tid
                            self.player_xmin = tr['pos'][0] - w_box / 2
                            self.player_ymin = tr['pos'][1] - h_box / 2
                            self.player_w = w_box
                            self.player_h = h_box

        # Check if existing ball is still valid
        if (self.player_id > 0 and self.ball_id > 0 and self.ball_id in tracks
                and tracks[self.ball_id]['object_class'] == names['ball']):
            f_ball_in_hands = True
        else:
            ball_id = -1
            if self.player_id >= 0:
                tr = tracks[self.player_id]
                w_box, h_box = tr['wh']
                margin = cfg.BALL_SEARCH_MARGIN

                x_left_bound = max(0, tr['pos'][0] - w_box / 2 - margin)
                x_right_bound = min(tr['pos'][0] + w_box / 2 + margin, ow)
                y_top_bound = max(0, tr['pos'][1] - h_box / 2 - margin)
                y_bottom_bound = min(tr['pos'][1] + h_box / 2 + margin, oh)

                ball_area_max = -float('inf')
                for bid, trk in tracks.items():
                    if trk['object_class'] == names['ball'] and trk['static_age'] <= 5:
                        w, h = trk['wh']
                        area = w * h
                        if area > cfg.BALL_AREA_MAX or area < cfg.BALL_AREA_MIN:
                            continue
                        x, y = trk['pos'][0], trk['pos'][1]
                        if (x_left_bound < x < x_right_bound
                                and y_top_bound < y < y_bottom_bound
                                and area > ball_area_max):
                            f_ball_in_hands = True
                            ball_id = bid
                            ball_area_max = area

                if self.ball_id != ball_id:
                    # Ball changed - reset trajectory
                    self.traj_map.clear()
                    self.left_map = float('inf')
                    self.right_map = float('-inf')
                    self.bottom_map = float('inf')
                    self.prev_dy_map = None
                    self.up_map = float('inf')
                    self.ball_id = ball_id

        # Draw rect to indicate working area
        foot_x_in_region, foot_y_in_region = False, False
        if foot_top is not None and f_ball_in_hands:
            if self.blue_cnt > 0:
                # Loose tolerance after confirmed
                foot_x_in_region = abs(foot_top[0] - cfg.PLAYER_CENTER_X) < cfg.PLAYER_POSITION_X_LOOSE
                foot_y_in_region = abs(foot_top[1] - cfg.PLAYER_CENTER_Y) < cfg.PLAYER_POSITION_Y_LOOSE
            else:
                # Tight tolerance for initial detection
                foot_x_in_region = abs(foot_top[0] - cfg.PLAYER_CENTER_X) < cfg.PLAYER_POSITION_X_TIGHT
                foot_y_in_region = abs(foot_top[1] - cfg.PLAYER_CENTER_Y) < cfg.PLAYER_POSITION_Y_TIGHT

        if f_ball_in_hands and foot_x_in_region and foot_y_in_region:
            color = (0, 255, 0, 200)  # Green - ready
            self.blue_cnt += 1
            self.blue_cnt = min(cfg.BLUE_CNT_MAX, self.blue_cnt)
        else:
            color = (255, 255, 255, 200)  # White - not ready
            self.blue_cnt -= 1
            self.blue_cnt = max(0, self.blue_cnt)
            self.player_id = -1
            self.ball_id = -1
            f_ball_in_hands = False
            if self.blue_cnt == 0:
                self.was_ready = False

        if self.blue_cnt < cfg.BLUE_CNT_READY:
            x1, y1, x2, y2 = self.crop_area
            self.cudaDrawHollowRect(
                cuda_img,
                (x1, y1, x2, y2),
                color,
                line_width=2,
            )

        if self.blue_cnt > 0 and f_ball_in_hands:
            # Player is ready in position
            if not self.was_ready:
                self.ready_go_start_time = time.time()
            self.was_ready = True

            tr = tracks[self.ball_id]
            ball_center = [int(tr['pos'][0]), int(tr['pos'][1])]
            tr = tracks[self.player_id]
            player_center = [int(tr['pos'][0]), int(tr['pos'][1])]

            # Update trajectory and count
            self.history_update(ball_center, player_center, foot_top, foot_width, self.target_type)
            # Draw trajectory
            self.draw_trajectory(cuda_img)

        self.draw_warning(cuda_img, ow, oh)

    def history_update(self, ball_centre, player_center, foot_top, foot_width, target_type):
        """Add current ball centre and update trend / count."""
        cfg = self.cfg
        dribble_type = None
        now = time.time()
        self.traj_map.append([ball_centre, now])
        cutoff = now - cfg.TRAJECTORY_WINDOW_SEC

        # Use popleft for O(1) removal from deque
        while len(self.traj_map) > 0 and self.traj_map[0][1] < cutoff:
            self.traj_map.popleft()

        # also track the left most and right most
        # Track left/right extremes
        if ball_centre[0] < self.left_map:
            self.left_map = ball_centre[0]
        if ball_centre[0] > self.right_map:
            self.right_map = ball_centre[0]

        if len(self.traj_map) >= 2:
            dy = ball_centre[1] - self.traj_map[-2][0][1]  # +ve = moving down
            if self.prev_dy_map is not None:
                # Upward to Downward transition - count dribble
                if self.prev_dy_map < 0 <= dy:
                    if self.bottom_map - self.up_map > cfg.MIN_DRIBBLE_HEIGHT:
                        dribble_type = self._classify_and_count(player_center, foot_top, foot_width, target_type)
                    else:
                        dribble_type = 'unknown'
                    self.reset_trajectory()

                # Downward to Upward transition - ball hit ground
                elif self.prev_dy_map > 0 >= dy:
                    self.bottom_map = self.traj_map[-2][0][1]
                    self.bottom_x = self.traj_map[-2][0][0]
                    self.bottom_y = self.traj_map[-2][0][1]

            self.prev_dy_map = dy
        return dribble_type

    def cuda_draw_name_id(self, dets, cuda_img):
        """Draw detection bounding boxes with class-based colors."""
        if not calibrations.f_od_overlay:
            return
        for det in dets:
            x1, y1, x2, y2, conf, det_class = det
            if det_class == names['ball']:
                color = (255, 0, 255, 200)  # Magenta
            else:
                color = (0, 0, 255, 200)    # Red for player
            self.cudaDrawHollowRect(cuda_img, (x1, y1, x2, y2), color, line_width=2)

    def cuda_draw_tracks(self, tracks, cuda_img):
        """Draw track bounding boxes with class labels and dotted lines to matched detections.

        Args:
            tracks: Dict of tracks from tracker.get_tracks()
            cuda_img: CUDA image to draw on
        """
        if not calibrations.f_od_overlay:
            return

        reverse_names = {v: k for k, v in names.items()}

        # Consistent color for all tracks
        track_color = (0, 255, 255, 255)  # Cyan
        coasted_color = (0, 128, 128, 150)  # Dim cyan

        for tid, tr in tracks.items():
            pos = tr['pos']
            w, h = tr['wh']
            obj_class = tr['object_class']
            coasted = tr['coasted']
            matched_det = tr.get('matched_det')

            # Track bounding box from Kalman state
            tx1 = int(pos[0] - w / 2)
            ty1 = int(pos[1] - h / 2)
            tx2 = int(pos[0] + w / 2)
            ty2 = int(pos[1] + h / 2)

            color = coasted_color if coasted else track_color

            # Draw track bounding box
            self.cudaDrawHollowRect(cuda_img, (tx1, ty1, tx2, ty2), color, line_width=3)

            # Draw class label and track ID on top of bounding box (with background)
            class_name = reverse_names.get(obj_class, '?')
            label_text = f"{class_name}#{tid}"
            label_x = max(0, tx1)
            label_y = max(0, ty1 - 35)
            # Only draw if within image bounds
            if label_x < self.ow and label_y < self.oh:
                bg_w = len(label_text) * 22 + 45
                bg_h = 40
                cudaDrawRect(cuda_img, (label_x, label_y, label_x + bg_w, label_y + bg_h), (0, 0, 0, 160))
                self.fontM.OverlayText(cuda_img, self.ow, self.oh, label_text, label_x + 6, label_y + 4, color)

            # Draw dotted line from track center to matched detection center (ball only, skip players)
            if matched_det is not None and obj_class != names['player']:
                dx1, dy1, dx2, dy2, dconf, dclass = matched_det
                det_cx = int((dx1 + dx2) / 2)
                det_cy = int((dy1 + dy2) / 2)
                track_cx = int(pos[0])
                track_cy = int(pos[1])
                self._draw_dotted_line(cuda_img, track_cx, track_cy, det_cx, det_cy, color=(255, 255, 0, 200))

    def _draw_dotted_line(self, cuda_img, x1, y1, x2, y2, color=(255, 255, 0, 200), dash_len=8, gap_len=6):
        """Draw a dotted line between two points.

        Args:
            cuda_img: CUDA image to draw on
            x1, y1: Start point
            x2, y2: End point
            color: Line color (RGBA)
            dash_len: Length of each dash in pixels
            gap_len: Length of gap between dashes
        """
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 1:
            return

        # Normalize direction
        ux = dx / dist
        uy = dy / dist

        segment_len = dash_len + gap_len
        num_segments = int(dist / segment_len)

        for i in range(num_segments + 1):
            start_d = i * segment_len
            end_d = min(start_d + dash_len, dist)

            if start_d >= dist:
                break

            sx = int(x1 + ux * start_d)
            sy = int(y1 + uy * start_d)
            ex = int(x1 + ux * end_d)
            ey = int(y1 + uy * end_d)

            cudaDrawLine(cuda_img, (sx, sy), (ex, ey), color, line_width=2)

    def reset_trajectory(self):
        """Reset trajectory keeping only last two points."""
        # Convert to list, slice, convert back to deque
        last_two = list(self.traj_map)[-2:] if len(self.traj_map) >= 2 else list(self.traj_map)
        self.traj_map = deque(last_two)
        self.left_map = float('inf')
        self.right_map = float('-inf')
        self.bottom_map = float('inf')
        self.prev_dy_map = None
        self.up_map = self.traj_map[0][0][1] #if last_two else float('inf')

    def _count_trajectory_gaps(self, gap_threshold_px=120) -> int:
        """Count gaps in the ball trajectory.

        A gap is when the distance between consecutive points exceeds gap_threshold_px.
        For behind-the-back and cross-the-legs dribbles, ball occlusion by the body
        should create 1-2 gaps in the trajectory.

        Returns the number of gaps detected.
        """
        if len(self.traj_map) < 2:
            return 0

        gap_count = 0

        for i in range(1, len(self.traj_map)):
            prev_pos = self.traj_map[i - 1][0]  # [x, y]
            curr_pos = self.traj_map[i][0]
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance > gap_threshold_px:
                gap_count += 1

        return gap_count

    def _validate_special_dribble(self, target_type) -> tuple:
        """Validate behind-the-back and cross-the-legs dribbles.

        Performs three checks:
        1. Player height/width ratio should be close to 1.0 (threshold 1.4)
        2. Ball bottom position should be close to ground target (within 100 pixels)
        3. Trajectory should have gaps (ball occlusion by body)

        Returns:
            (failed_count, hints_list): Number of failed checks and list of hint messages
        """
        failed_checks = []
        tgt_y = self.ground_tgt_y + 60  # center of ground target

        # Check 1: Player height/width ratio (should be close to 1.0 for low stance)
        if self.player_w > 0:
            hw_ratio = self.player_h / self.player_w
            if hw_ratio > 1.8:
                if calibrations.f_pinyin:
                    failed_checks.append("压低重心")
                else:
                    failed_checks.append("Stay low")

        # Check 2: Ball bottom Y position relative to ground target
        if self.bottom_y is not None:
            y_diff = self.bottom_y - tgt_y
            if y_diff > 100:  # ball is hitting too far below target
                if calibrations.f_pinyin:
                    failed_checks.append("运球位置太低")
                else:
                    failed_checks.append("Dribble position too low")
            elif y_diff < -100:  # ball is hitting too far above target
                if calibrations.f_pinyin:
                    failed_checks.append("运球位置太高")
                else:
                    failed_checks.append("Dribble position too high")

        # Check 3: Trajectory should have gaps (ball occlusion)
        gap_count = self._count_trajectory_gaps(gap_threshold_px=self.cfg.TRAJECTORY_GAP_THRESHOLD)
        if gap_count == 0:
            if calibrations.f_pinyin:
                failed_checks.append("请让球从身后/胯下穿过")
            else:
                failed_checks.append("Ball should pass behind/under body")

        return len(failed_checks), failed_checks

    def _classify_and_count(self, player_center, foot_top, foot_width, target_type):
        # if we have valid dribble, calculate the dpm
        # self.calculate_dpm()
        # classify the dribble type
        calc_type = "unknown"
        # next need to check if the ground hit condition is met
        tgt_x, tgt_y = self.ground_tgt_x, self.ground_tgt_y
        tgt_x += 80
        tgt_y += 60  # now convert it to the center of tgt
        cfg = self.cfg
        # Check if ball hit near the ground target
        if self.bottom_x is None or abs(self.bottom_x - tgt_x) > cfg.HIT_TARGET_X_THRESHOLD or abs(self.bottom_y - tgt_y) > cfg.HIT_TARGET_Y_THRESHOLD:
            self.f_no_hitting_tgt = True
            return calc_type

        calc_types = []
        height_ratio = (player_center[1] - self.up_map) / abs(player_center[1] - foot_top[1])

        if self.right_map < foot_top[0]:
            calc_types.append("left high" if height_ratio > -0.1 else "left low")

        if self.left_map > foot_top[0]:
            calc_types.append("right high" if height_ratio > -0.1 else "right low")

        if self.right_map > (foot_top[0] + foot_width/4) and self.left_map < (foot_top[0] - foot_width/4):
            calc_types.append("crossover high" if height_ratio > -0.1 else "crossover low")

        if self.right_map - self.left_map > cfg.SPECIAL_DRIBBLE_WIDTH:
            # this place might plug in a NN to tell back or legs
            calc_types.extend(["left in-and-out", "right in-and-out",
                               "behind-the-back", "cross-the-legs"])

        if not calc_types:
            calc_types.append("unknown")

        calc_type = calc_types[0]  # primary classification for return value

        if target_type in calc_types:
            # Special validation for behind-the-back and cross-the-legs
            if target_type in ("behind-the-back", "cross-the-legs"):
                failed_count, hints = self._validate_special_dribble(target_type)

                if failed_count >= 2:
                    # Too many failures - don't count and show warning
                    self.validation_warning = hints[0] if hints else None
                    self.validation_warning_time = time.time()
                    self.validation_hint = None  # Clear single hint
                    self.mismatch_cnt += 1
                    if self.mismatch_cnt % 3 == 0 and self.audio_enabled:
                        self.mismatch_sound.play()
                elif failed_count == 1:
                    # Single failure - count but show hint
                    self.dribble_count += 1
                    self.f_no_hitting_tgt = False
                    if self.audio_enabled:
                        self.match_sound.play()
                    self.mismatch_cnt = 0
                    self.validation_hint = hints[0]
                    self.validation_hint_time = time.time()
                    self.validation_warning = None
                    self.height_hint = None
                else:
                    # All checks passed - count normally
                    self.dribble_count += 1
                    self.f_no_hitting_tgt = False
                    if self.audio_enabled:
                        self.match_sound.play()
                    self.mismatch_cnt = 0
                    self.validation_hint = None
                    self.validation_warning = None
                    self.height_hint = None
            else:
                # Regular dribble types - count normally
                self.dribble_count += 1
                self.f_no_hitting_tgt = False
                if self.audio_enabled:
                    self.match_sound.play()
                self.mismatch_cnt = 0
                self.height_hint = None
                self.validation_hint = None
                self.validation_warning = None
        else:
            self.mismatch_cnt += 1
            if self.mismatch_cnt % 3 == 0:
                if self.audio_enabled:
                    self.mismatch_sound.play()
            # Check for height mismatch: right side but wrong height
            target_base = target_type.rsplit(" ", 1)[0] if target_type.endswith((" high", " low")) else None
            if target_base is not None:
                for ct in calc_types:
                    ct_base = ct.rsplit(" ", 1)[0] if ct.endswith((" high", " low")) else None
                    if ct_base == target_base and ct != target_type:
                        # Same side, wrong height
                        self.height_hint = "Dribble higher!" if target_type.endswith("high") else "Dribble lower!"
                        self.height_hint_time = time.time()
                        break

        if self.target_count <= self.dribble_count:
            self.job_done = True

        return calc_type

    def draw_trajectory(self, cuda_img) -> None:
        """Overlay trajectory + counter on the frame."""
        # draw trajectory lines
        pts = [pt for pt, _ in self.traj_map]
        if len(pts) >=2:
            for pt1, pt2 in zip(pts, pts[1:]):
                #cv2.line(frame, p1, p2, self.colour, 2)
                cudaDrawLine(cuda_img, pt1, pt2, (255, 255, 0, 255), 2)

    def draw_warning(
        self,
        cuda_img,
        ow, oh,
        y_margin=10,
        box_h=65,
        pad=8,
        shift_right=0
    ) -> None:
        w, h = ow, oh
        font = self._fontL  # Use larger font (size 72)
        font_small = self.fontM  # For bottom instructions

        # Bright yellow color for better visibility
        yellow = (255, 255, 0, 255)
        # Semi-transparent dark background for text readability
        bg_rgba = (0, 0, 0, 160)

        f_long_ins = False
        f_ground_hit_ins = False
        f_score_status = False
        f_finished_status = False

        # Calculate remaining count
        remaining = max(0, self.target_count - self.dribble_count)

        # --- build display text ---
        if not self.job_done:
            if self.blue_cnt > 0:
                if self.f_no_hitting_tgt:
                    # Top: dribble type + remaining count
                    top_line = f"{self.target_type}  [{remaining}]"
                    bottom_line = "Score 1 point by hitting the ground target"
                    f_ground_hit_ins = True
                else:
                    top_line = f"{self.target_type}  [{remaining}]"
                    bottom_line = f"Achieved: {self.dribble_count}   Goal: {self.target_count}"
                    f_score_status = True
            else:
                top_line = f"{self.target_type}  [{remaining}]"
                bottom_line = "Stand in the white box holding the ball"
                f_long_ins = True

        elif (not self.winner_evt.is_set()) or self.is_winner:
            self.is_winner = True
            self.winner_evt.set()
            top_line = "Winner!"
            bottom_line = "Nice job!"
            f_finished_status = True
        else:
            top_line = "Finished!"
            bottom_line = "Good work!"
            f_finished_status = True

        # --- Top center: dribble type + remaining count ---
        # Calculate box width based on text length (26 pixels per char for 48pt font)
        top_box_w = len(top_line) * 26 + 40
        top_text_x = (w - top_box_w) // 2
        top_text_y = y_margin

        if not calibrations.f_pinyin:
            # Draw dark background box for text readability
            cudaDrawRect(cuda_img, (top_text_x, top_text_y, top_text_x + top_box_w, top_text_y + box_h), bg_rgba)
            # Use 48pt font (cudaFont doesn't render 72pt reliably)
            font_small.OverlayText(cuda_img, ow, oh, top_line, top_text_x + pad, top_text_y + pad+10, yellow)
        else:
            if f_finished_status:
                overlay_text_centered(cuda_img, '完成训练！', top_text_y, w,
                                      font_size=40, text_color=yellow, bg_color=bg_rgba)
            else:
                zh_top = f"{self.target_type_zh}  [{remaining}]"
                overlay_text_centered(cuda_img, zh_top, top_text_y, w,
                                      font_size=40, text_color=yellow, bg_color=bg_rgba)

        # --- "Ready, Go!" or height hint display below the dribble type ---
        hint_text_y = y_margin + box_h + 10  # shared position below dribble type
        hint_box_h = 55

        # Check for validation warning (red - dribble didn't count, highest priority)
        f_show_validation_warning = (self.validation_warning is not None
                                     and self.validation_warning_time is not None
                                     and time.time() - self.validation_warning_time < self.height_hint_duration)

        # Check for validation hint (orange - single check failed, counted but show hint)
        f_show_validation_hint = (self.validation_hint is not None
                                  and self.validation_hint_time is not None
                                  and time.time() - self.validation_hint_time < self.height_hint_duration)

        # Height hint takes priority over "Ready, Go!" since it's actionable feedback
        f_show_hint = (self.height_hint is not None and self.height_hint_time is not None
                       and time.time() - self.height_hint_time < self.height_hint_duration)

        if f_show_validation_warning:
            # Red background for validation warning (dribble didn't count)
            if calibrations.f_pinyin:
                overlay_text_centered(cuda_img, self.validation_warning, hint_text_y, w,
                                      font_size=32, text_color=yellow, bg_color=(180, 0, 0, 180))
            else:
                hint_text = self.validation_warning
                hint_box_w = len(hint_text) * 26 + 40
                hint_text_x = (w - hint_box_w) // 2
                red_bg = (180, 0, 0, 180)
                cudaDrawRect(cuda_img, (hint_text_x, hint_text_y, hint_text_x + hint_box_w, hint_text_y + hint_box_h), red_bg)
                font_small.OverlayText(cuda_img, ow, oh, hint_text, hint_text_x + pad, hint_text_y + pad, yellow)
        elif f_show_validation_hint:
            # Orange background for validation hint (counted but needs improvement)
            if calibrations.f_pinyin:
                overlay_text_centered(cuda_img, self.validation_hint, hint_text_y, w,
                                      font_size=32, text_color=yellow, bg_color=(180, 80, 0, 180))
            else:
                hint_text = self.validation_hint
                hint_box_w = len(hint_text) * 26 + 40
                hint_text_x = (w - hint_box_w) // 2
                orange_bg = (180, 80, 0, 180)
                cudaDrawRect(cuda_img, (hint_text_x, hint_text_y, hint_text_x + hint_box_w, hint_text_y + hint_box_h), orange_bg)
                font_small.OverlayText(cuda_img, ow, oh, hint_text, hint_text_x + pad, hint_text_y + pad, yellow)
        elif f_show_hint:
            hint_text = self.height_hint
            hint_box_w = len(hint_text) * 26 + 40
            hint_text_x = (w - hint_box_w) // 2
            orange_bg = (180, 80, 0, 180)
            cudaDrawRect(cuda_img, (hint_text_x, hint_text_y, hint_text_x + hint_box_w, hint_text_y + hint_box_h), orange_bg)
            font_small.OverlayText(cuda_img, ow, oh, hint_text, hint_text_x + pad, hint_text_y + pad, yellow)
        elif self.ready_go_start_time is not None:
            elapsed = time.time() - self.ready_go_start_time
            if elapsed < self.ready_go_duration:
                ready_text = "Ready, Go!"
                ready_box_w = len(ready_text) * 26 + 40  # 26 pixels per char for 48pt font
                ready_text_x = (w - ready_box_w) // 2
                # Green background for positive feedback
                green_bg = (0, 100, 0, 180)
                cudaDrawRect(cuda_img, (ready_text_x, hint_text_y, ready_text_x + ready_box_w, hint_text_y + hint_box_h), green_bg)
                font_small.OverlayText(cuda_img, ow, oh, ready_text, ready_text_x + pad, hint_text_y + pad, yellow)

        # --- Bottom center: instructions ---
        bottom_y = 1120
        bottom_box_h = 55

        if calibrations.f_pinyin and f_long_ins:
            overlay_text_centered(cuda_img, '请持球站立在白框内', bottom_y, w,
                                  font_size=36, text_color=yellow, bg_color=bg_rgba)
        elif calibrations.f_pinyin and f_ground_hit_ins:
            overlay_text_centered(cuda_img, '击中地面目标得1分', bottom_y, w,
                                  font_size=36, text_color=yellow, bg_color=bg_rgba)
        elif calibrations.f_pinyin and f_score_status:
            score_zh = f"得分：{self.dribble_count}  目标：{self.target_count}"
            overlay_text_centered(cuda_img, score_zh, bottom_y, w,
                                  font_size=36, text_color=yellow, bg_color=bg_rgba)
        elif calibrations.f_pinyin and f_finished_status:
            # Chinese: show completion message and replay hint
            overlay_text_centered(cuda_img, bottom_line if bottom_line != "Nice job!" else "做得好！", bottom_y, w,
                                  font_size=36, text_color=yellow, bg_color=bg_rgba)
        else:
            # English: draw with dark background, centered, bright yellow
            # 26 pixels per char for 48pt font
            bottom_box_w = len(bottom_line) * 26 + 40
            bottom_text_x = (w - bottom_box_w) // 2
            cudaDrawRect(cuda_img, (bottom_text_x, bottom_y, bottom_text_x + bottom_box_w, bottom_y + bottom_box_h), bg_rgba)
            font_small.OverlayText(cuda_img, ow, oh, bottom_line, bottom_text_x + pad, bottom_y + pad, yellow)

        # --- Replay hint when task is finished ---
        if f_finished_status:
            replay_y = bottom_y - 70  # Position above the bottom message
            replay_box_h = 50
            # Blue background for replay hint
            blue_bg = (30, 100, 180, 200)

            if calibrations.f_pinyin:
                replay_hint = "提示：主页 → 设置 → 回放 → 运球 查看训练录像"
                overlay_text_centered(cuda_img, replay_hint, replay_y, w,
                                      font_size=28, text_color=yellow, bg_color=blue_bg)
            else:
                replay_hint = "Tip: Home -> Setting -> Replay -> Dribble to review video"
                replay_box_w = len(replay_hint) * 20 + 40
                replay_text_x = (w - replay_box_w) // 2
                cudaDrawRect(cuda_img, (replay_text_x, replay_y, replay_text_x + replay_box_w, replay_y + replay_box_h), blue_bg)
                self._fontS.OverlayText(cuda_img, ow, oh, replay_hint, replay_text_x + pad, replay_y + pad, yellow)



    def calculate_dpm(self):
        # record dribble global time
        now = time.time()
        self._last_dribble_times.append(now)
        # calculate dpm
        if len(self._last_dribble_times) >= 2:
            span_sec = self._last_dribble_times[-1] - self._last_dribble_times[0]
            if span_sec > 0:
                # rate based on #dribbles in the deque over that span
                self.dpm = (len(self._last_dribble_times) - 1) / (span_sec / 60.0)


    def cudaDrawHollowRect(self, cuda_img,
                           rect,  # (x1, y1, x2, y2)
                           color,  # (r, g, b, a) 0-255
                           line_width=1):  # line width in pixels

        x1, y1, x2, y2 = rect
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # 2) clamp to image bounds
        W, H = self.ow, self.oh
        x1 = int(max(0, min(x1, W - 1)))
        x2 = int(max(0, min(x2, W - 1)))
        y1 = int(max(0, min(y1, H - 1)))
        y2 = int(max(0, min(y2, H - 1)))

        # 3) skip degenerate/empty boxes or invalid widths
        if x2 <= x1 or y2 <= y1 or line_width <= 0 or cuda_img is None:
            return

        # Top edge
        cudaDrawLine(cuda_img, (x1, y1), (x2, y1),
                     color, line_width=line_width)
        # Right edge
        cudaDrawLine(cuda_img, (x2, y1), (x2, y2),
                     color, line_width=line_width)
        # Bottom edge
        cudaDrawLine(cuda_img, (x2, y2), (x1, y2),
                     color, line_width=line_width)
        # Left edge
        cudaDrawLine(cuda_img, (x1, y2), (x1, y1),
                     color, line_width=line_width)

    def cudaDrawGroundHit(self):

        if self.target_type == 'crossover high' or self.target_type == 'crossover low':  # front full V
            loc_x, loc_y = 960, 920
        elif self.target_type == 'left high' or self.target_type == 'left low':
            loc_x, loc_y = 760, 920
        elif self.target_type == 'right high' or self.target_type == 'right low':
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

        loc_x, loc_y = loc_x - 80, loc_y - 60  # get its top left corner's loc
        loc_y += 160 # it seems too high

        self.ground_tgt_x, self.ground_tgt_y = loc_x, loc_y

    def dribble_target(self, style):
        """Set dribble target type and Chinese display name."""
        style_map = {
            'Crossover challenge high': ('crossover high', '交叉运球-高手'),
            'Crossover challenge low': ('crossover low', '交叉运球-低手'),
            'Left dribble challenge high': ('left high', '左手单手运球-高手'),
            'Left dribble challenge low': ('left low', '左手单手运球-低手'),
            'Right dribble challenge high': ('right high', '右手单手运球-高手'),
            'Right dribble challenge low': ('right low', '右手单手运球-低手'),
            'Behind back': ('behind-the-back', '背后运球训练'),
            'Cross leg': ('cross-the-legs', '胯下运球训练'),
            'Left V': ('left in-and-out', '左手内外运球'),
            'Right V': ('right in-and-out', '右手内外运球'),
        }
        self.target_type, self.target_type_zh = style_map.get(
            style, ('v', '交叉运球-高手')
        )

    def update_activity(self):
        """Call this to update the last activity timestamp when dribble occurs."""
        self.last_activity_time = time.time()

    def check_activity(self):
        """Check if dribble_count changed and update activity time.
        Call this periodically (e.g., each frame) to track activity."""
        if self.dribble_count != self._prev_dribble_count:
            self._prev_dribble_count = self.dribble_count
            self.update_activity()

    def is_idle(self) -> bool:
        """Returns True if no dribble activity for idle_timeout_sec."""
        return (time.time() - self.last_activity_time) > self.idle_timeout_sec

