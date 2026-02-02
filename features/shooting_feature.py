import numpy as np
from enum import Enum
import cv2
from features.bytetrack import Tracker
from jetson_utils import (cudaDrawRect, cudaFont, cudaAllocMapped, cudaResize, cudaDrawLine,
                          cudaFromNumpy, cudaDrawCircle, cudaOverlay, cudaDeviceSynchronize, cudaToNumpy)
import pygame
from pathlib import Path
from manual_bev import MANUAL_BEV
import time
from itertools import pairwise  # Python 3.10+. If on 3.8/3.9, see fallback below.
import calibrations  # For dynamic language switching
from features.cuda_text_utils import overlay_text
from collections import deque
from calc_player import analyze_player_features
from calc_shooting_metrics import analyze_sideview_shot
from compute_AI_advice import compute_shooting_ai_advice, compute_layup_ai_advice

class Event(Enum):
    NO_SHOOTING = 0
    SHOOTING = 1
    IN_BASKET = 2
    FAILED = 3

# Detection class names mapping
names = {'ball': 0, 'bib': 1, 'player': 2}
reverse_names = {v: k for k, v in names.items()}

# Key points for shooting positions on BEV court
key_pts = np.array([[468, 181], [928, 181], [468, 290], [928, 290], [468, 397], [928, 397], [468, 488], [928, 488],
                    [701, 581], [701, 402], [701, 840], [327, 710], [1069, 710], [124, 443], [1273, 443], [81, 181],
                    [1317, 181], [1123, 181], [275, 181], [300, 443], [1100, 443]], dtype=np.float32)

# Load calibration files with error handling
def _load_corners(filepath, default_value):
    """Load corner calibration file with fallback to default."""
    try:
        return np.load(filepath)
    except FileNotFoundError:
        print(f"[ShootingFeature] Warning: {filepath} not found, using default")
        return default_value

_default_corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480], [320, 100]])
corners_left = _load_corners("corners_left.npy", _default_corners)
corners_right = _load_corners("corners_right.npy", _default_corners)
left_rim_x, left_rim_y = int(corners_left[4][0]), int(corners_left[4][1])
right_rim_x, right_rim_y = int(corners_right[4][0]), int(corners_right[4][1])

# ============================================================================
# Tunable Constants - adjust these for different camera setups or requirements
# ============================================================================
class ShootingConfig:
    """Configuration constants for shooting detection and tracking."""
    # Shooting detection thresholds
    BALL_VELOCITY_THRESHOLD = 50        # Min velocity to trigger shooting detection
    BALL_VELOCITY_MIN_TRACK = 10        # Min velocity for ball tracking in queue
    RING_PROXIMITY_OFFSET = 20          # Offset above ring to detect shooting

    # Basket/make detection
    BASKET_PROXIMITY_X = 200            # X distance from basket to count as make
    BALL_FALLOFF_Y_OFFSET = 80          # Y offset below ring for ball fall-off
    SHOOT_DURATION_TIMEOUT = 100         # Max frames for a single shot attempt
    MIN_TRAJECTORY_LENGTH = 5           # Min trajectory points for valid shot

    # Tracker parameters
    TRACKER_Q = 100.0                   # Kalman filter process noise
    TRACKER_R = 0.01                     # Kalman filter measurement noise
    TRACKER_BASE_DIST = 300             # Base distance threshold for tracking

    # Player detection
    PLAYER_MIN_AGE = 10                 # Min track age for player consideration
    BALL_MIN_AGE = 5                    # Min track age for ball consideration

    # BEV dimensions
    BEV_WIDTH = 1400                    # BEV image width for coordinate flip

    # Audio/visual feedback delay (frames)
    RESULT_DISPLAY_FRAMES = 5           # Frames to display make/fail result

    # Memory management
    MAX_TRAJECTORY_HISTORY = 100        # Max stored trajectories per type
    MAX_POSITION_HISTORY = 200          # Max stored make/fail positions

    # Layup speed filtering
    MAX_LAYUP_SPEED_MS = 12             # Max realistic layup speed (m/s)
    MIN_LAYUP_SPEED_MS = 0.1            # Min realistic layup speed (m/s)


def _as_xy(p):
    """Ensure point is (int x, int y) tuple."""
    x, y = p[:2]
    return int(x), int(y)

def _load_image_as_cuda_rgba(filepath, alpha=200):
    """Load an image file and convert to CUDA RGBA format.

    Args:
        filepath: Path to image file
        alpha: Alpha channel value (0-255)

    Returns:
        cudaImage or None if file not found
    """
    if not Path(filepath).exists():
        print(f"[ShootingFeature] Warning: Image not found: {filepath}")
        return None
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        print(f"[ShootingFeature] Warning: Failed to read image: {filepath}")
        return None
    numpy_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = numpy_img.shape[:2]
    A = np.full((H, W), alpha, dtype=np.uint8)
    rgba_np = np.ascontiguousarray(np.dstack((numpy_img, A)).astype(np.uint8))
    return cudaFromNumpy(rgba_np)

def _flip_bev_x(x_bev):
    """Flip BEV x-coordinate (image is mirrored)."""
    return ShootingConfig.BEV_WIDTH - x_bev

def draw_polyline(img, traj, color, width=2, show_endpoints=False):
    """Draws a connected polyline for a trajectory."""
    pts = [_as_xy(p) for p in traj if p is not None]
    if not pts:
        return
    if len(pts) == 1:
        # degenerate: draw a dot
        cudaDrawCircle(img, pts[0], radius=max(1, width+1), color=color)
        return
    for p0, p1 in pairwise(pts):
        cudaDrawLine(img, p0, p1, color, line_width=width)
    if show_endpoints:
        cudaDrawCircle(img, pts[0],  radius=width+2, color=color)  # start
        cudaDrawCircle(img, pts[-1], radius=width+2, color=color)  # end


def map_style_to_key_pts_index(style):
    index = []
    times = 0
    dist_thresh = -1
    if style == 'mid-range baseline':
        index =  [17, 18]
        times = 10
        dist_thresh = 120
    elif style == '3-points baseline':
        index = [15, 16]
        times = 10
        dist_thresh = 150
    elif style == 'mid-range star':
        index = [17, 6, 7, 18, 8]
        times = 5
        dist_thresh = 120
    elif style == '3-points star':
        index = [16, 11, 12, 15, 10]
        times = 5
        dist_thresh = 150
    elif style == 'close-range single-hand':
        index = [0, 1, 9]
        times = 10
        dist_thresh = 120
    elif style == 'free throw':
        index = [8]
        times = 20
        dist_thresh = 120
    elif style == '3-points challenge':
        index = [10, 13, 14]
        times = 5
        dist_thresh = 150
    elif style == 'layup':
        index = []  # No specific key points for layup - uses basket proximity
        times = 20
        dist_thresh = 250  # Close to basket for layup

    return index, times, dist_thresh

def chaikin(points, iterations=1):
    pts = [tuple(map(float, p[:2])) for p in points]
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            Q = (0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1])
            R = (0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1])
            new.extend([Q, R])
        new.append(pts[-1])
        pts = new
    # Next cut the tails of hte traj
    f_peak_found = False
    ind = len(pts)
    for i in range(1, len(pts)-1):
        if pts[i+1][1] > pts[i][1] and pts[i-1][1] > pts[i][1] : # this is peak:
            f_peak_found = True
        if f_peak_found and ((pts[i+1][1] < pts[i][1] and pts[i-1][1] < pts[i][1]) or (pts[i+1][1] - pts[i][1] < 20
                and pts[i][1] > 180)): # this is the max y loc, the second cond is to remove ball's lateral movement
            ind = i+1
            break

    return [[int(x), int(y)] for x, y in pts[:ind]]

# Example:
# pts = chaikin(traj, iterations=2); draw_polyline(self.cuda_img, pts, color, width=3)


def draw_cross_cuda(dst_cuda, x: int, y: int,
                    size: int = 12, thickness: int = 2,
                    color=(255, 0, 0, 128)):
    # clamp inside image
    x0 = max(0, x - size); x1 = min(dst_cuda.width  - 1, x + size)
    y0 = max(0, y - size); y1 = min(dst_cuda.height - 1, y + size)

    # two perpendicular lines
    cudaDrawLine(dst_cuda, (x0, y), (x1, y), color, thickness)
    cudaDrawLine(dst_cuda, (x, y0), (x, y1), color, thickness)

class ShootingFeature:
    def __init__(self, w, h, dt, side, style):
        self.cfg = ShootingConfig()  # Store config reference
        self.side = side
        if side == 'left':
            self.basket_x = left_rim_x
            self.ring_loc_y = left_rim_y
        else:
            self.basket_x = right_rim_x
            self.ring_loc_y = right_rim_y

        self.style = style
        self.key_pts_index, self.shoot_times, self.shoot_dist_thresh = map_style_to_key_pts_index(style)
        self.cuda_img = None

        # Load BEV image with error handling
        self.base_bev_cuda = _load_image_as_cuda_rgba('BEV/RBEV_1400.png', alpha=200)
        if self.base_bev_cuda is None:
            raise FileNotFoundError("BEV image 'BEV/RBEV_1400.png' is required")

        self.bev_rgba_cuda = cudaAllocMapped(
            width=self.base_bev_cuda.width,
            height=self.base_bev_cuda.height,
            format=self.base_bev_cuda.format
        )
        cudaOverlay(self.base_bev_cuda, self.bev_rgba_cuda, 0, 0)

        # Initialize tracker with config parameters
        self.tracker = Tracker(
            q=self.cfg.TRACKER_Q,
            r=self.cfg.TRACKER_R,
            base_dist_thresh=self.cfg.TRACKER_BASE_DIST
        )
        # Set ring zone for Re-ID buffer (ball recovery near basket)
        self.tracker.set_ring_zone(self.basket_x, self.ring_loc_y)

        # Shooting state
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
        self._fontM = cudaFont(size=48)
        self._fontS = cudaFont(size=32)

        self.time_pre = time.time()
        self.traj = []
        self.traj_all_make = []
        self.traj_all_fail = []

        # Audio initialization with error handling
        self.audio_enabled = False
        self.match_sound = None
        self.mismatch_sound = None
        self._audio_played_this_event = False  # Prevent repeated audio playback
        try:
            pygame.mixer.init()
            if Path('match.wav').exists():
                self.match_sound = pygame.mixer.Sound('match.wav')
            if Path('mismatch.wav').exists():
                self.mismatch_sound = pygame.mixer.Sound('mismatch.wav')
            self.audio_enabled = self.match_sound is not None or self.mismatch_sound is not None
        except Exception as e:
            print(f"[ShootingFeature] Audio disabled: {e}")

        # BEV display parameters - allocate once and reuse
        self.target_w, self.target_h = 600, 600
        self.dst = cudaAllocMapped(width=self.target_w, height=self.target_h,
                                   format=self.bev_rgba_cuda.format)
        self.total_attempts_21 = [0] * 21
        self.total_makes_21 = [0] * 21
        self.accuracy = 0
        self.make_pos = []
        self.fail_pos = []
        self.shoot_du = 0
        self.q_ball = deque(maxlen=60)
        self.q_player = deque(maxlen=60)

        # Initialize BEV class
        H, W = 1200, 1920
        img = np.zeros((H, W, 3), dtype=np.uint8)  # Dummy image for initialization
        self.bev_class = MANUAL_BEV(img, None, False, side)

        # Draw key points in the BEV
        for key_pt_ind in self.key_pts_index:
            key_pt = key_pts[key_pt_ind]
            cudaDrawCircle(self.bev_rgba_cuda, (int(key_pt[0]), int(key_pt[1])),
                           radius=15, color=(0, 255, 0, 255))

        # AI feedback arrays
        self.release_height_ratio_arr = []
        self.release_lateral_ratio_arr = []
        self.bias_short_long_arr = []
        self.bias_left_right_arr = []
        self.flat_shot_arr = []

        # Load result indicator icons with error handling
        bib_cuda = _load_image_as_cuda_rgba('UI_Button/bib.png', alpha=200)
        if bib_cuda is not None:
            self.bib_icon = cudaAllocMapped(width=120, height=120, format=bib_cuda.format)
            cudaResize(bib_cuda, self.bib_icon)
        else:
            self.bib_icon = None

        bob_cuda = _load_image_as_cuda_rgba('UI_Button/bob.png', alpha=200)
        if bob_cuda is not None:
            self.bob_icon = cudaAllocMapped(width=120, height=150, format=bob_cuda.format)
            cudaResize(bob_cuda, self.bob_icon)
        else:
            self.bob_icon = None

        # AI popup bridge
        self.ai_bridge = None
        self._advice_popup_shown = False

        # Layup tracking (left-hand / right-hand based on player position relative to basket)
        self.layup_left_attempts = 0
        self.layup_left_makes = 0
        self.layup_right_attempts = 0
        self.layup_right_makes = 0

        # Layup advanced metrics tracking
        self.layup_speed_history = []   # (peak_speed_m_per_s, attempt_number, is_make, hand)
        self.layup_zone_history = []    # (zone_name, is_make, hand)
        self.layup_player_bev_positions = []  # [(x_bev, y_bev, timestamp), ...]

        # Idle detection
        self.last_activity_time = time.time()
        self.idle_timeout_sec = 180  # 3 minutes
        self._prev_total_shooting = 0

        # Current tracks for overlay drawing
        self.current_tracks = {}

        # Track allocated CUDA resources for cleanup
        self._cuda_resources = [
            self.base_bev_cuda, self.bev_rgba_cuda, self.dst,
            self.bib_icon, self.bob_icon
        ]

    def __del__(self):
        """Clean up CUDA resources on destruction."""
        # Note: cudaAllocMapped resources are typically auto-freed,
        # but explicit cleanup is good practice
        try:
            self._cuda_resources = None
            self.tracker = None
        except Exception:
            pass


    def search_key_pt(self, x_bev, y_bev):
        """Find the closest key point to the given BEV coordinates.

        Uses vectorized numpy operations for efficiency.

        Args:
            x_bev: X coordinate in BEV space
            y_bev: Y coordinate in BEV space

        Returns:
            Index of closest key point, or -1 if none within threshold
        """
        if not self.key_pts_index:
            return -1

        # Get subset of key points we care about
        indices = np.array(self.key_pts_index)
        pts = key_pts[indices]  # Shape: (N, 2)

        # Vectorized distance calculation
        pos = np.array([x_bev, y_bev], dtype=np.float32)
        diffs = pts - pos  # Shape: (N, 2)
        dists = np.linalg.norm(diffs, axis=1)  # Shape: (N,)

        # Special handling for free throw (index 8) - use rectangular zone
        if 8 in self.key_pts_index:
            ft_local_idx = self.key_pts_index.index(8)
            ft_pt = pts[ft_local_idx]
            dx = max(0, abs(ft_pt[0] - x_bev) - 100)
            dy = abs(ft_pt[1] - y_bev)
            dists[ft_local_idx] = np.sqrt(dx * dx + dy * dy)

        # Find minimum distance
        min_idx = np.argmin(dists)
        if dists[min_idx] < self.shoot_dist_thresh:
            return indices[min_idx]
        return -1

    def check_finish(self):
        # For layup mode, check total layup attempts (left + right)
        if self.style == 'layup':
            total_layup_attempts = self.layup_left_attempts + self.layup_right_attempts
            return total_layup_attempts >= self.shoot_times

        index = self.key_pts_index
        # print(self.key_pts_index)
        for ind in index:
            att = self.total_attempts_21[ind]
            # print(f"total att: {att}")
            # print(f"shooting times : {self.shoot_times}")
            if att < self.shoot_times:
                return False

        return True

    def calc_accuracy(self):
        # For layup mode, calculate from layup-specific counters
        if self.style == 'layup':
            sum_make = self.layup_left_makes + self.layup_right_makes
            sum_att = self.layup_left_attempts + self.layup_right_attempts
            return sum_make / sum_att if sum_att > 0 else 0

        sum_make = 0
        sum_att = 0
        acc = 0
        for ind in self.key_pts_index:
            sum_make += self.total_makes_21[ind]
            sum_att += self.total_attempts_21[ind]
        if sum_att > 0:
            acc = sum_make / sum_att

        return acc

    def update_basket(self, ball_track):
        # next, use bib track's location to update self.ring_loc_y
        w, h = ball_track['wh']
        ring_loc_y_bib = ball_track['pos'][1] - h / 2  # this pos is centroid, so it needs to shift up by h/2
        if abs(ring_loc_y_bib - self.ring_loc_y) < 150:  # make sure this is not a wild pt.
            self.ring_loc_y = self.ring_loc_y * 0.6 + ring_loc_y_bib * 0.4
            self.basket_x = self.basket_x * 0.6 + ball_track['pos'][0] * 0.4
            # Update tracker ring zone for Re-ID buffer
            self.tracker.set_ring_zone(self.basket_x, self.ring_loc_y)

    def on_frame(self, ctx: dict):
        cfg = ctx["cfg"]
        ow, oh = ctx["src_W"], ctx["src_H"]
        thr = cfg["score_threshold"]
        self.cuda_img = ctx["cuda_img"]
        self.side = ctx["side"]

        detections = []
        for bb, sc, lb in zip(ctx["bboxes"], ctx["scores"], ctx["labels"]):
            lb = int(lb)
            if lb < 0 or sc < thr:  # Filter by score threshold
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
            xyxy = [x1, y1, x2, y2]
            conf = float(sc)
            obj_class = lb
            detection = [*xyxy, conf, obj_class]
            detections.append(detection)

        self.calc_event(detections)

        # Draw shooting location in BEV on result state transition
        if ((self.status == Event.IN_BASKET or self.status == Event.FAILED)
                and self.status != self.prev_status
                and self.latest_shooting_loc is not None):
            x, y = self.latest_shooting_loc
            x_bev, y_bev = self.bev_class.calc_bev_pts(x, y)
            x_bev = _flip_bev_x(x_bev)

            # Handle layup mode separately
            if self.style == 'layup':
                # Determine left/right hand based on player position relative to basket
                # NOTE: Image is left/right flipped, so x > basket_x means left-hand layup
                is_left_hand = (x > self.basket_x)
                # Check if player is close enough to basket for layup
                dist_to_basket = abs(x - self.basket_x)
                if dist_to_basket < self.shoot_dist_thresh:
                    # Calculate peak speed from tracked BEV positions
                    peak_speed = self._calc_layup_peak_speed()
                    # Determine approach zone based on BEV position
                    zone = self._determine_layup_zone(x_bev, y_bev)
                    is_make = (self.status == Event.IN_BASKET)
                    hand = 'left' if is_left_hand else 'right'

                    # Store speed and zone history
                    attempt_num = self.layup_left_attempts + self.layup_right_attempts + 1
                    if peak_speed > 0:
                        self.layup_speed_history.append((peak_speed, attempt_num, is_make, hand))
                    self.layup_zone_history.append((zone, is_make, hand))

                    if is_left_hand:
                        self.layup_left_attempts += 1
                        if is_make:
                            self.layup_left_makes += 1
                            self.traj_all_make.append(self.traj)
                        else:
                            self.traj_all_fail.append(self.traj)
                    else:
                        self.layup_right_attempts += 1
                        if is_make:
                            self.layup_right_makes += 1
                            self.traj_all_make.append(self.traj)
                        else:
                            self.traj_all_fail.append(self.traj)

                # Clear BEV positions for next layup
                self.layup_player_bev_positions = []
            else:
                # next determine if the shooting location is at desired key pts
                key_pt_ind_found = self.search_key_pt(x_bev, y_bev)
                if key_pt_ind_found != -1:
                    self.total_attempts_21[key_pt_ind_found] += 1
                    if self.status == Event.IN_BASKET:
                        self.total_makes_21[key_pt_ind_found] += 1
                        self.traj_all_make.append(self.traj)   # only record the traj at designated shooting place
                    else:
                        self.traj_all_fail.append(self.traj)

            if self.status == Event.IN_BASKET:
                self.make_pos.append([x_bev, y_bev])
            elif self.status == Event.FAILED:
                self.fail_pos.append([x_bev, y_bev])

            self.traj = []
            # draw statistics for each spot:
            self.cuda_draw_each_spot_result()

        # Display current shooting state indicator
        bg_rgba = (120, 120, 120, 160)
        rect_coords = (900, 20, 1050, 50)
        cudaDrawRect(self.cuda_img, rect_coords, bg_rgba)

        if self.status == Event.IN_BASKET:
            if self.bib_icon is not None:
                cudaOverlay(self.bib_icon, self.cuda_img, 900, 5)
            notes = "MAKE"
        elif self.status == Event.FAILED:
            if self.bob_icon is not None:
                cudaOverlay(self.bob_icon, self.cuda_img, 900, 5)
            notes = "FAILED"
        elif self.status == Event.NO_SHOOTING:
            notes = "NO_SHOOT"
        elif self.status == Event.SHOOTING:
            notes = "SHOOTING"
        else:
            notes = ""

        self._fontS.OverlayText(
            self.cuda_img, self.img_width, self.img_height,
            notes, 905, 25, (0, 0, 255, 255)
        )

        # Warn user to calibrate camera if too many bib detections fail spatial check
        if self.bib_spatial_miss_cnt >= 3:
            warn_color = (255, 80, 0, 255)  # orange
            if calibrations.f_pinyin:
                warn_text = "请在设置中重新校准相机"
                overlay_text(self.cuda_img, warn_text, 750, 60, font_size=28,
                             text_color=warn_color, bg_color=(0, 0, 0, 160))
            else:
                warn_text = "Please calibrate camera from settings"
                cudaDrawRect(self.cuda_img, (748, 58, 748 + len(warn_text) * 18, 58 + 36), (0, 0, 0, 160))
                self._fontS.OverlayText(
                    self.cuda_img, self.img_width, self.img_height,
                    warn_text, 750, 60, warn_color)

        # Display BEV
        cudaResize(self.bev_rgba_cuda, self.dst)
        # overlay RBEV in the cuda img (600x600 BEV)
        if self.side == 'left':
            cudaOverlay(self.dst, self.cuda_img, 900, 700)
        else:
            cudaOverlay(self.dst, self.cuda_img, 420, 700)

        self.cuda_draw_shoot()
        self.cuda_draw_name_id(detections)
        self.cuda_draw_tracks(self.current_tracks)
        f_finished = self.check_finish()
        if f_finished:
            accuracy = self.calc_accuracy() * 100  # %
            if self.style == 'layup':
                # Compute layup-specific AI advice with left/right hand FG% and advanced metrics
                result = compute_layup_ai_advice(
                    self.layup_left_attempts,
                    self.layup_left_makes,
                    self.layup_right_attempts,
                    self.layup_right_makes,
                    speed_history=self.layup_speed_history,
                    zone_history=self.layup_zone_history,
                    f_pinyin=calibrations.f_pinyin
                )
            else:
                result = compute_shooting_ai_advice(
                        self.release_height_ratio_arr,
                        self.release_lateral_ratio_arr,
                        self.flat_shot_arr,
                        self.bias_short_long_arr,
                        self.bias_left_right_arr,
                        f_pinyin=calibrations.f_pinyin
                    )
            # print(f"result is {result}")
            # print(f"accuracy is {accuracy}")
            ## Plan to use popup window to display both English and Chinese. need to check its results.
            if not self._advice_popup_shown and self.ai_bridge is not None:
                self._advice_popup_shown = True
                self.ai_bridge.show_advice.emit(result, accuracy, calibrations.f_pinyin, self.side)

            # pts = chaikin(traj, iterations=2); draw_polyline(self.cuda_img, pts, color, width=3)
            # draw trajectory for each shooting
            for traj in self.traj_all_make:
                draw_polyline(self.cuda_img, traj, color=(255, 255, 0, 255), width=3, show_endpoints=False) #bright yellow

            for traj in self.traj_all_fail:
                draw_polyline(self.cuda_img, traj, color=(255, 255, 255, 255), width=3, show_endpoints=False) #pure white

    def calc_event(self, detections):
        """Process detections and update shooting state machine."""
        self.tracker.update(detections, self.dt)
        tracks = self.tracker.get_tracks()
        self.current_tracks = tracks  # Store for overlay drawing
        self.queue_tgt(tracks)
        self.frame_cnt += 1

        # Periodic FPS logging (debug)
        if self.frame_cnt > 0 and self.frame_cnt % 200 == 0:
            time_inter = time.time() - self.time_pre
            print(f"Shooting OD Inference {200 / time_inter:.1f} FPS")
            self.time_pre = time.time()

        cfg = self.cfg
        self.prev_status = self.status  # Detect state transitions

        if self.status == Event.NO_SHOOTING:
            f_shooting = self.search_shooting(tracks)
            if f_shooting:
                self.status = Event.SHOOTING
                self.shoot_du = 1
                self._audio_played_this_event = False  # Reset audio flag

        elif self.status == Event.SHOOTING:
            tid = self.ball_id
            self.shoot_du += 1

            # Track player BEV position for layup speed calculation
            if self.style == 'layup':
                self._track_layup_player_position(tracks)

            if tid not in tracks:  # Shooting ball lost
                self.status = Event.NO_SHOOTING
                self.traj = []
                self.bib_confirmed = False
                if self.style == 'layup':
                    self.layup_player_bev_positions = []
            else:
                ball_track = tracks[tid]
                if ball_track['object_class'] == names['bib']:
                    self.bib_confirmed = True
                    self.update_basket(ball_track)

                elif ball_track['object_class'] == names['ball']:
                    if ball_track['pos'][1] < self.ring_loc_y:
                        self.traj.append(ball_track['pos'])

                    ball_pos = ball_track['pos']
                    timeout = self.shoot_du > cfg.SHOOT_DURATION_TIMEOUT
                    ball_fell_off = ball_pos[1] > self.ring_loc_y + cfg.BALL_FALLOFF_Y_OFFSET

                    if self.bib_confirmed:
                        if ball_pos[1] < self.ring_loc_y:
                            # Ball bounced back above the ring - not a make
                            self.bib_confirmed = False
                        elif abs(ball_pos[0] - self.basket_x) < cfg.BASKET_PROXIMITY_X and ball_pos[1] > self.ring_loc_y + 20:
                            # Ball fell through near basket_x - MAKE
                            self.status = Event.IN_BASKET
                            self.save_results()
                            self.bingo_2pt_num += 1
                            self.bib_confirmed = False
                            self._audio_played_this_event = False
                        elif ball_fell_off or timeout:
                            # Ball fell off but not near basket - FAIL
                            self.bib_spatial_miss_cnt += 1
                            if len(self.traj) <= cfg.MIN_TRAJECTORY_LENGTH:
                                self.status = Event.NO_SHOOTING
                                self.traj = []
                            else:
                                self.status = Event.FAILED
                                self.save_results()
                                self._audio_played_this_event = False
                            self.bib_confirmed = False
                    else:
                        if ball_fell_off or timeout:
                            if len(self.traj) <= cfg.MIN_TRAJECTORY_LENGTH:
                                self.status = Event.NO_SHOOTING
                                self.traj = []
                            else:
                                self.status = Event.FAILED
                                self.save_results()
                                self._audio_played_this_event = False

        elif self.status == Event.FAILED:
            # Play audio only once on state entry
            if not self._audio_played_this_event and self.audio_enabled and self.mismatch_sound:
                self.mismatch_sound.play()
                self._audio_played_this_event = True

            if self.fail_cnt == 0:
                self.fail_cnt = cfg.RESULT_DISPLAY_FRAMES
            elif self.fail_cnt == 1:
                self.status = Event.NO_SHOOTING
                self.fail_cnt = 0
            else:
                self.fail_cnt -= 1

        elif self.status == Event.IN_BASKET:
            # Play audio only once on state entry
            if not self._audio_played_this_event and self.audio_enabled and self.match_sound:
                self.match_sound.play()
                self._audio_played_this_event = True

            if self.bib_cnt == 0:
                self.bib_cnt = cfg.RESULT_DISPLAY_FRAMES
            elif self.bib_cnt == 1:
                self.status = Event.NO_SHOOTING
                self.bib_cnt = 0
            else:
                self.bib_cnt -= 1

    def save_results(self):
        """Save shooting results and calculate shot metrics."""
        self.shooting_result.append(self.status)
        self.total_shooting += 1
        self.shooting_locs.append(self.latest_shooting_loc)
        self.shooting_start_array.append(self.shooting_start)
        self.shooting_end_array.append(self.frame_cnt)

        # Smooth trajectory and calculate metrics
        self.traj = chaikin(self.traj, iterations=1)
        metrics = analyze_sideview_shot(self.traj, self.basket_x, self.ring_loc_y)
        if metrics.ok:
            self.flat_shot_arr.append(metrics.flat_shot)
            self.bias_short_long_arr.append(metrics.short_long)
            self.bias_left_right_arr.append(metrics.left_right)

        # Memory management: limit stored trajectory history
        cfg = self.cfg
        if len(self.traj_all_make) > cfg.MAX_TRAJECTORY_HISTORY:
            self.traj_all_make = self.traj_all_make[-cfg.MAX_TRAJECTORY_HISTORY:]
        if len(self.traj_all_fail) > cfg.MAX_TRAJECTORY_HISTORY:
            self.traj_all_fail = self.traj_all_fail[-cfg.MAX_TRAJECTORY_HISTORY:]
        if len(self.make_pos) > cfg.MAX_POSITION_HISTORY:
            self.make_pos = self.make_pos[-cfg.MAX_POSITION_HISTORY:]
        if len(self.fail_pos) > cfg.MAX_POSITION_HISTORY:
            self.fail_pos = self.fail_pos[-cfg.MAX_POSITION_HISTORY:]

    def queue_tgt(self, tracks):
        cfg = self.cfg
        ball_pos = None

        for _, tr in tracks.items():
            if tr.get('object_class') != names['ball'] or tr.get('coasted', False):
                continue
            if tr.get('age', 0) <= cfg.BALL_MIN_AGE:
                continue

            vel = tr.get('vel', None)
            tr_vel = float(np.linalg.norm(vel)) if vel is not None else 0.0
            if tr_vel <= cfg.BALL_VELOCITY_MIN_TRACK:
                continue

            ball_pos = np.asarray(tr['pos'], dtype=float)
            x, y = ball_pos
            self.q_ball.append([float(x), float(y)])
            break
        else:
            self.q_ball.append([])

        if ball_pos is None:
            self.q_player.append([])
            return

        smallest_dist = float('inf')
        tmp_append = []

        for _, tr in tracks.items():
            if tr.get('object_class') != names['player'] or tr.get('coasted', False):
                continue
            if tr.get('age', 0) <= cfg.PLAYER_MIN_AGE:
                continue

            x, y = tr['pos']
            w, h = tr['wh']
            foot_x, foot_y = tr['foot_loc']

            x_bev, y_bev = self.bev_class.calc_bev_pts(foot_x, foot_y)
            x_bev = _flip_bev_x(x_bev)

            if self.search_key_pt(x_bev, y_bev) != -1:
                pos = np.asarray(tr['pos'], dtype=float)
                dist = float(np.linalg.norm(ball_pos - pos))
                if dist < smallest_dist:
                    smallest_dist = dist
                    tmp_append = [x, y, w, h, foot_x, foot_y]

        if tmp_append:
            self.q_player.append(tmp_append)
        else:
            player_id = self.find_closest_player(tracks, ball_pos)
            if player_id >= 0:
                tr = tracks[player_id]
                x, y = tr['pos']
                w, h = tr['wh']
                foot_x, foot_y = tr['foot_loc']
                self.q_player.append([x, y, w, h, foot_x, foot_y])
            else:
                self.q_player.append([])

    def search_shooting(self, tracks):
        """Detect if a shooting event has started.

        Returns:
            True if shooting detected, False otherwise
        """
        cfg = self.cfg

        for tid, tr in tracks.items():
            tr_vel = np.linalg.norm(tr['vel'])
            # Don't allow coasted ball to trigger shooting
            if (tr['object_class'] == names['ball'] and not tr['coasted']
                    and tr_vel > cfg.BALL_VELOCITY_THRESHOLD):
                ball_pos = tr['pos']
                # Ball is above the ring level
                if ball_pos[1] < self.ring_loc_y + cfg.RING_PROXIMITY_OFFSET:
                    self.ball_id = tid
                    self.shooting_start = self.frame_cnt
                    self.traj.append(ball_pos)
                    self.calc_player_derived()
                    return True

        return False

    def find_closest_player(self, tracks, ball_pos):
        """Find the closest player to the ball position.

        Args:
            tracks: Dictionary of tracked objects
            ball_pos: Ball position (x, y)

        Returns:
            Track ID of closest player, or -1 if none found
        """
        smallest_dist = 1e9
        player_id = -1

        for tid, tr in tracks.items():
            if tr['object_class'] == names['player']:
                player_pos = tr['pos']
                # this condition to make sure ball posn is between player and basket
                dist = np.linalg.norm(ball_pos - player_pos)
                if dist < smallest_dist:  # and h / w > 2:  # only, locating the shooter
                    smallest_dist = dist
                    player_id = tid

        return player_id
    
    def _calc_layup_peak_speed(self) -> float:
        """Calculate peak speed from tracked BEV positions during layup approach.

        Uses the BEV calibration to calculate real-world distance.

        Returns:
            Peak speed in meters per second, or 0.0 if insufficient data
        """
        cfg = self.cfg
        positions = self.layup_player_bev_positions
        if len(positions) < 2:
            return 0.0

        # Calculate instantaneous speeds between consecutive frames
        speeds = []
        for i in range(1, len(positions)):
            x1, y1, t1 = positions[i - 1]
            x2, y2, t2 = positions[i]
            dt = t2 - t1
            if dt <= 0:
                continue

            dist_m = self.bev_class.calc_distance_meters((x1, y1), (x2, y2))
            speed = dist_m / dt

            # Filter unrealistic speeds
            if cfg.MIN_LAYUP_SPEED_MS < speed < cfg.MAX_LAYUP_SPEED_MS:
                speeds.append(speed)

        if not speeds:
            return 0.0

        # Return peak speed (95th percentile to filter outliers)
        return float(np.percentile(speeds, 95)) if len(speeds) >= 3 else max(speeds)

    def _track_layup_player_position(self, tracks):
        """Track player's BEV position during layup approach for speed calculation."""
        ball_pos = None
        for _, tr in tracks.items():
            if tr['object_class'] == names['ball'] and not tr['coasted']:
                ball_pos = tr['pos']
                break

        if ball_pos is None:
            return

        player_id = self.find_closest_player(tracks, ball_pos)
        if player_id < 0:
            return

        tr = tracks[player_id]
        foot_x, foot_y = tr['foot_loc']

        # Convert to BEV coordinates
        x_bev, y_bev = self.bev_class.calc_bev_pts(foot_x, foot_y)
        x_bev = _flip_bev_x(x_bev)

        self.layup_player_bev_positions.append((x_bev, y_bev, time.time()))

    def _determine_layup_zone(self, x_bev: float, y_bev: float) -> str:
        """Determine the approach zone for layup based on BEV coordinates.
        Zones: 'center', 'left_wing', 'right_wing', 'left_corner', 'right_corner'"""
        # BEV image is 1400x1400, basket at center-top
        # x_bev is already flipped (1400 - x_bev was done earlier)
        center_x = 700

        # Define zones based on x position
        if x_bev < 300:
            return 'right_corner'  # After flip, low x_bev = right side
        elif x_bev < 550:
            return 'right_wing'
        elif x_bev < 850:
            return 'center'
        elif x_bev < 1100:
            return 'left_wing'
        else:
            return 'left_corner'

    def calc_player_derived(self):
        player_q = self.q_player
        ball_q = self.q_ball
        # need to calculate following properties
        # shooting loc, drift, release height ratio, release lateral ratio
        # need to calculate shooting loc with queue info
        feat = analyze_player_features(player_q, ball_q)
        if feat is not None:
            ri = feat["release_idx"]
            P = list(player_q)
            B = list(ball_q)

            # Smooth foot position over a window of ±3 frames around release
            half_win = 3
            lo = max(0, ri - half_win)
            hi = min(len(P), ri + half_win + 1)

            foot_xs = []
            foot_ys = []
            for i in range(lo, hi):
                p, b = P[i], B[i]
                # Queue now stores [x, y, w, h, foot_x, foot_y]
                if not (isinstance(p, (list, tuple)) and len(p) >= 4):
                    continue

                # Use foot_loc from tracker if available, otherwise fall back to centroid
                if len(p) >= 6:
                    px, py, pw, ph, p_foot_x, p_foot_y = p
                else:
                    px, py, pw, ph = p
                    p_foot_x = px
                    p_foot_y = py + 0.47 * ph

                # Apply small compensation for front toe location based on facing direction
                # Player faces basket, so front toe is slightly toward basket
                if px > self.basket_x + 200:
                    # Player is to the right of basket, front toe shifts left (toward basket)
                    toe_offset = pw * 0.08
                elif px < self.basket_x - 200:
                    # Player is to the left of basket, front toe shifts right (toward basket)
                    toe_offset = -pw * 0.08
                else:
                    # Player is near basket center line, minimal offset
                    toe_offset = 0

                foot_xs.append(p_foot_x + toe_offset)
                foot_ys.append(p_foot_y)

            if not foot_xs:
                # Fallback if no valid frames in window
                p = P[ri]
                if len(p) >= 6:
                    _, _, _, _, foot_x, foot_y = p
                else:
                    x, y, w, h = p[:4]
                    foot_x = x
                    foot_y = y + 0.5 * h
            else:
                foot_x = float(np.mean(foot_xs))
                foot_y = float(np.mean(foot_ys))

            self.latest_shooting_loc = [foot_x, foot_y]
            self.release_height_ratio_arr.append(feat["release_height_ratio"])
            self.release_lateral_ratio_arr.append(feat["release_lateral_ratio"])

    def cuda_draw_each_spot_result(self):
        """Draw shooting results on BEV overlay."""
        cudaOverlay(self.base_bev_cuda, self.bev_rgba_cuda, 0, 0)

        # Draw key points
        for key_pt_ind in self.key_pts_index:
            key_pt = key_pts[key_pt_ind]
            cudaDrawCircle(self.bev_rgba_cuda, (int(key_pt[0]), int(key_pt[1])),
                           radius=15, color=(0, 255, 0, 255))

        # Draw make/fail markers
        for pos in self.make_pos:
            cudaDrawCircle(self.bev_rgba_cuda, (int(pos[0]), int(pos[1])),
                           radius=12, color=(0, 0, 255, 128))
        for pos in self.fail_pos:
            draw_cross_cuda(self.bev_rgba_cuda, int(pos[0]), int(pos[1]))

        # Draw statistics text for each spot
        for ind in self.key_pts_index:
            key = key_pts[ind]
            x0, y0 = int(key[0]), int(key[1])

            if self.total_attempts_21[ind] >= self.shoot_times:
                color = (0, 0, 255, 255)
                line  = f"Made:{self.total_makes_21[ind]}"
            else:
                color = (255, 0, 255, 255)
                line = f"Att.:{self.total_attempts_21[ind]}"

            W, H = self.bev_rgba_cuda.width, self.bev_rgba_cuda.height
            self._fontM.OverlayText(self.bev_rgba_cuda, W, H, line, x0 - 100, y0 + 20, color)


    def cuda_draw_shoot(self):
        magenta = (255, 0, 255, 255)

        # Text positions adjusted for 600x600 BEV at (950,700) for left, (470,700) for right
        x0 = 550 if self.side == 'right' else 1030

        if self.style == 'layup':
            total_att = self.layup_left_attempts + self.layup_right_attempts
            total_made = self.layup_left_makes + self.layup_right_makes
            if calibrations.f_pinyin:
                line1 = f"上篮训练 目标{self.shoot_times}次"
                line2 = f"左手:{self.layup_left_makes}/{self.layup_left_attempts} 右手:{self.layup_right_makes}/{self.layup_right_attempts}"
            else:
                line1 = f"Layup training x {self.shoot_times} times"
                line2 = f"L:{self.layup_left_makes}/{self.layup_left_attempts} R:{self.layup_right_makes}/{self.layup_right_attempts}"
        else:
            if calibrations.f_pinyin:
                line1 = f"每个位置投球 {self.shoot_times} 次"
                line2 = f"命中：{self.bingo_2pt_num}  共投：{self.total_shooting}"
            else:
                line1 = f"Each spot x {self.shoot_times} times"
                line2 = f"Made:{self.bingo_2pt_num}  Shots:{self.total_shooting}"

        if calibrations.f_pinyin:
            overlay_text(self.cuda_img, line1, x0, 1110, font_size=28,
                         text_color=magenta, bg_color=None)
            overlay_text(self.cuda_img, line2, x0, 1150, font_size=28,
                         text_color=magenta, bg_color=None)
        else:
            self._fontS.OverlayText(
                self.cuda_img, self.img_width, self.img_height,
                line1, x0 + 2, 1110, magenta)
            self._fontS.OverlayText(
                self.cuda_img, self.img_width, self.img_height,
                line2, x0, 1150, magenta)

    def _cuda_draw_hollow_rect(self, cuda_img, rect, color, line_width=2):
        """Draw rectangle outline on a cudaImage using 4 lines."""
        x1, y1, x2, y2 = map(int, rect)

        # Clamp to image bounds
        W, H = self.img_width, self.img_height
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1 or line_width <= 0 or cuda_img is None:
            return

        # Draw four edges: top, right, bottom, left
        cudaDrawLine(cuda_img, (x1, y1), (x2, y1), color, line_width=line_width)
        cudaDrawLine(cuda_img, (x2, y1), (x2, y2), color, line_width=line_width)
        cudaDrawLine(cuda_img, (x2, y2), (x1, y2), color, line_width=line_width)
        cudaDrawLine(cuda_img, (x1, y2), (x1, y1), color, line_width=line_width)

    def cuda_draw_name_id(self, dets):
        """Draw detection bounding boxes with class-based colors."""
        if not calibrations.f_od_overlay:
            return
        for det in dets:
            x1, y1, x2, y2, conf, det_class = det
            if conf > 0.2:
                if det_class == names['ball']:
                    color = (255, 0, 255, 200)  # Magenta
                elif det_class == names['player']:
                    color = (0, 255, 0, 200)    # Green
                elif det_class == names['bib']:
                    color = (0, 255, 255, 200)  # Cyan
                else:
                    color = (0, 0, 255, 200)    # Red
                self._cuda_draw_hollow_rect(self.cuda_img, (x1, y1, x2, y2), color, line_width=2)

    def cuda_draw_tracks(self, tracks):
        """Draw track bounding boxes with class labels and dotted lines to matched detections.

        Args:
            tracks: Dict of tracks from tracker.get_tracks()
        """
        if not calibrations.f_od_overlay:
            return

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
            self._cuda_draw_hollow_rect(self.cuda_img, (tx1, ty1, tx2, ty2), color, line_width=3)

            # Draw class label and track ID on top of bounding box (with background)
            class_name = reverse_names.get(obj_class, '?')
            label_text = f"{class_name}#{tid}"
            label_x = max(0, tx1)
            label_y = max(0, ty1 - 35)
            # Only draw if within image bounds
            if label_x < self.img_width and label_y < self.img_height:
                bg_w = len(label_text) * 22 + 45
                bg_h = 40
                cudaDrawRect(self.cuda_img, (label_x, label_y, label_x + bg_w, label_y + bg_h), (0, 0, 0, 160))
                self._fontM.OverlayText(self.cuda_img, self.img_width, self.img_height, label_text, label_x + 6, label_y + 4, color)

            # Draw dotted line from track center to matched detection center (ball/bib only, skip players)
            if matched_det is not None and obj_class != names['player']:
                dx1, dy1, dx2, dy2, dconf, dclass = matched_det
                det_cx = int((dx1 + dx2) / 2)
                det_cy = int((dy1 + dy2) / 2)
                track_cx = int(pos[0])
                track_cy = int(pos[1])
                self._draw_dotted_line(track_cx, track_cy, det_cx, det_cy, color=(255, 255, 0, 200))

    def _draw_dotted_line(self, x1, y1, x2, y2, color=(255, 255, 0, 200), dash_len=8, gap_len=6):
        """Draw a dotted line between two points.

        Args:
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

            cudaDrawLine(self.cuda_img, (sx, sy), (ex, ey), color, line_width=2)

    def update_activity(self):
        """Call this to update the last activity timestamp when shooting occurs."""
        self.last_activity_time = time.time()

    def check_activity(self):
        """Check if total_shooting changed and update activity time.
        Call this periodically (e.g., each frame) to track activity."""
        if self.total_shooting != self._prev_total_shooting:
            self._prev_total_shooting = self.total_shooting
            self.update_activity()

    def is_idle(self) -> bool:
        """Returns True if no shooting activity for idle_timeout_sec."""
        return (time.time() - self.last_activity_time) > self.idle_timeout_sec
