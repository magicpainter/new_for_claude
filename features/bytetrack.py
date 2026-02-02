import math
import numpy as np
from scipy.optimize import linear_sum_assignment

# Detection class names mapping
# Note: Dribble feature uses only ball(0) and player(2)
#       Shooting feature uses ball(0), bib(1), and player(2)
names = {'ball': 0, 'bib': 1, 'player': 2}


# ============================================================================
# Tracker Configuration - adjust these for different tracking requirements
# ============================================================================
class TrackerConfig:
    """Configuration constants for ByteTrack multi-object tracker."""

    # Confidence thresholds for two-stage matching
    CONF_HIGH = 0.6                 # High confidence threshold for primary matching
    CONF_LOW = 0.2                  # Low confidence threshold for secondary matching
    CONF_BIB_SPECIAL = 0.2          # Special threshold for bib detection

    # Distance and matching thresholds
    EXTRA_DIST_MARGIN = 150         # Added to base_dist for max matching distance
    MIN_IOU = 0.1                   # Minimum IoU for player matching

    # Ball tracking adjustments
    BALL_Q_MULTIPLIER = 50          # Process noise multiplier for ball (handles bouncing)
    BALL_ACCEL_DAMPING = 0.5        # Damping factor for ball acceleration

    # Bib association
    BIB_PRIORITY_BONUS = 60         # Cost reduction for bib-to-track association
    BIB_VERTICAL_REJECT = 200       # Reject bib if track is this much below detection

    # Sky-shooting detection bounds (ball going up high)
    SKY_SHOT_Y_MIN = -200           # Min Y for sky-shooting detection
    SKY_SHOT_Y_MAX = 100            # Max Y for sky-shooting detection

    # Track lifecycle
    STATIC_VELOCITY_THRESHOLD = 10  # Velocity below this is considered static
    NEW_TRACK_MAX_COAST = 2         # Max coasted frames for new tracks (age < 5)
    MATURE_TRACK_MAX_COAST = 5      # Max coasted frames for mature tracks (age >= 5)
    MATURE_TRACK_MIN_AGE = 5        # Age threshold to be considered mature

    # Confidence smoothing (exponential moving average)
    CONF_EMA_OLD = 0.8              # Weight for existing confidence
    CONF_EMA_NEW = 0.2              # Weight for new detection confidence

    # Track ID management
    MAX_TRACK_ID = 1000             # Wrap ID after this (increased from 300 for safety)
    MIN_TRACK_ID = 1                # Starting ID after wrap

    # Ring zone for relaxed ball association (handles bouncing near basket)
    RING_ZONE_MARGIN_X = 100        # Horizontal margin around basket (pixels)
    RING_ZONE_MARGIN_Y = 100        # Vertical margin around ring (pixels)
    RING_ZONE_DIST_MULTIPLIER = 2.5 # Multiply max_dist by this in ring zone

    # Re-ID buffer for recovering killed tracks
    REID_BUFFER_MAX_AGE = 30        # Max frames to keep killed track in buffer
    REID_MAX_DISTANCE = 200         # Max distance for Re-ID matching
    REID_SIZE_TOLERANCE = 0.5       # Max relative size difference (50%)

    # Class confusion rejection (bandaid for OD misclassification: ball<->player)
    # When creating a new track, check if a nearby track of opposite class exists
    # and if the detection's size/aspect ratio matches that class instead
    CLASS_CONFUSION_DIST = 150          # Max distance to check for nearby opposite-class tracks
    CLASS_CONFUSION_SIZE_RATIO = 0.3    # Area ratio threshold - if det_area/track_area > this, sizes are "similar"
    CLASS_CONFUSION_ASPECT_PLAYER_MIN = 1.3   # Player aspect ratio (h/w) typically > this
    CLASS_CONFUSION_ASPECT_BALL_MAX = 1.8     # Ball aspect ratio (h/w) typically < this


class KalmanFilter:
    """6-state Kalman filter for position tracking with velocity and acceleration.

    State vector: [x, y, vx, vy, ax, ay]
    Observation: [x, y] (position only)
    """

    def __init__(self, q=1.0, r=1.0):
        """Initialize Kalman filter.

        Args:
            q: Process noise scaling factor
            r: Measurement noise scaling factor
        """
        self.q = q  # Process noise parameter
        self.r = r  # Measurement noise parameter

        # State vector: [x, y, vx, vy, ax, ay]
        self.x = np.zeros((6, 1))

        # Initial covariance - higher uncertainty for velocity and acceleration
        self.P = np.diag([1.0, 1.0, 100.0, 100.0, 10.0, 10.0])

        # Observation matrix - only observe position
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])

        # Measurement noise covariance
        self.R = r * np.eye(2)

    def predict(self, dt, q):
        """Predict next state using motion model.

        Args:
            dt: Time step in seconds
            q: Process noise scaling for this prediction
        """
        # State transition matrix (constant acceleration model)
        F = np.array([
            [1, 0, dt, 0, 0.5 * dt ** 2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt ** 2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Process noise covariance (discrete white noise acceleration model)
        Q = np.array([
            [dt ** 5 / 20, 0, dt ** 4 / 8, 0, dt ** 3 / 6, 0],
            [0, dt ** 5 / 20, 0, dt ** 4 / 8, 0, dt ** 3 / 6],
            [dt ** 4 / 8, 0, dt ** 3 / 3, 0, dt ** 2 / 2, 0],
            [0, dt ** 4 / 8, 0, dt ** 3 / 3, 0, dt ** 2 / 2],
            [dt ** 3 / 6, 0, dt ** 2 / 2, 0, dt, 0],
            [0, dt ** 3 / 6, 0, dt ** 2 / 2, 0, dt]
        ]) * q

        # State prediction
        self.x = F @ self.x
        # Covariance prediction
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        """Update state with new measurement.

        Args:
            z: Measurement vector [x, y]
        """
        # Ensure observation is column vector
        z = np.asarray(z).reshape(-1, 1)

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain (using pseudo-inverse for numerical stability)
        K = self.P @ self.H.T @ np.linalg.pinv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T


def _bbox_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes [x1, y1, x2, y2]
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-9)


class Tracker:
    """ByteTrack-style multi-object tracker with two-stage association.

    Uses high-confidence detections for primary matching, then low-confidence
    detections for secondary matching of remaining tracks. Supports different
    matching strategies for players (IoU-based) vs ball/bib (distance-based).
    """

    def __init__(self, q=1.0, r=1.0, base_dist_thresh=150):
        """Initialize tracker.

        Args:
            q: Kalman filter process noise
            r: Kalman filter measurement noise
            base_dist_thresh: Base distance threshold for matching
        """
        self.cfg = TrackerConfig()  # Configuration constants
        self.tracks = {}
        self.next_id = 0
        self.q = q
        self.r = r
        self.min_iou = self.cfg.MIN_IOU
        self.max_dist = base_dist_thresh + self.cfg.EXTRA_DIST_MARGIN
        self.base_dist = base_dist_thresh
        self.S_high = self.cfg.CONF_HIGH
        self.S_low = self.cfg.CONF_LOW
        self.unmatched_track_indices = set()
        self.ring_zone = None  # (x, y) center of ring zone, set by shooting_feature
        self.reid_buffer = {}  # Buffer for killed tracks: {track_id: {last_pos, wh, class, age_in_buffer}}

    def set_ring_zone(self, x, y):
        """Set ring zone center for relaxed ball association near basket.

        Args:
            x: Horizontal position of basket (basket_x)
            y: Vertical position of ring (ring_loc_y)
        """
        self.ring_zone = (x, y)

    def _is_in_ring_zone(self, pos):
        """Check if a position is within the ring zone.

        Args:
            pos: Position array [x, y]

        Returns:
            True if position is within ring zone margins
        """
        if self.ring_zone is None:
            return False
        rx, ry = self.ring_zone
        margin_x = self.cfg.RING_ZONE_MARGIN_X
        margin_y = self.cfg.RING_ZONE_MARGIN_Y
        return abs(pos[0] - rx) < margin_x and abs(pos[1] - ry) < margin_y

    def _is_class_confusion(self, det_cx, det_cy, det_class, det_w, det_h):
        """Check if detection is likely OD misclassification based on nearby tracks.

        Detects cases where:
        - A "ball" detection is actually a player (large, tall aspect ratio, near player track)
        - A "player" detection is actually a ball (small, square aspect ratio, near ball track)

        Args:
            det_cx, det_cy: Detection center coordinates
            det_class: Detection class ID
            det_w: Detection width
            det_h: Detection height

        Returns:
            True if detection is likely a misclassification and should be rejected
        """
        cfg = self.cfg
        det_aspect = det_h / (det_w + 1e-6)
        det_area = det_w * det_h
        dist_thresh_sq = cfg.CLASS_CONFUSION_DIST * cfg.CLASS_CONFUSION_DIST

        for tid, tr in self.tracks.items():
            track_class = tr['object_class']

            # Early exit: only check ball<->player confusion
            if not ((det_class == names['ball'] and track_class == names['player']) or
                    (det_class == names['player'] and track_class == names['ball'])):
                continue

            # Fast squared distance check (avoid sqrt)
            track_x = tr['kf'].x[0, 0]
            track_y = tr['kf'].x[1, 0]
            dx = det_cx - track_x
            dy = det_cy - track_y
            dist_sq = dx * dx + dy * dy
            if dist_sq > dist_thresh_sq:
                continue

            track_w, track_h = tr['wh']
            track_area = track_w * track_h
            size_ratio = det_area / (track_area + 1e-6)

            # "Ball" detection near player track - check if it looks like a player
            if det_class == names['ball']:
                if 0.8 < size_ratio < 1.2 and det_aspect > cfg.CLASS_CONFUSION_ASPECT_PLAYER_MIN:
                    return True  # Likely a misclassified player

            # "Player" detection near ball track - check if it looks like a ball
            else:
                if 0.8 < size_ratio < 1.2 and det_aspect < cfg.CLASS_CONFUSION_ASPECT_BALL_MAX:
                    return True  # Likely a misclassified ball

        return False

    def update(self, detections, dt):
        """Update tracker with new detections.

        Args:
            detections: List of [x1, y1, x2, y2, conf, class_id]
            dt: Time step since last update in seconds
        """
        cfg = self.cfg

        # Update existing tracks: predict and mark as coasted
        for _, track in self.tracks.items():
            track['age'] += 1
            track['coasted_age'] += 1
            track['coasted'] = True
            track['matched_det'] = None  # Clear matched detection until re-matched

            if track['object_class'] == names['player']:
                track['kf'].predict(dt, self.q)
            else:
                # Ball/bib: higher process noise to handle bouncing/fast motion
                track['kf'].predict(dt, self.q * cfg.BALL_Q_MULTIPLIER)
                # Damp acceleration to prevent runaway predictions
                track['kf'].x[4:] *= cfg.BALL_ACCEL_DAMPING

        # Convert detections to list if needed
        if not isinstance(detections, list):
            detections = list(detections) if hasattr(detections, '__iter__') else []

        # Handle case with no tracks - create new ones from high-conf detections
        if not self.tracks:
            for detection in detections:
                if len(detection) < 6:
                    continue  # Invalid detection format
                *xyxy, conf, det_class = detection
                if conf >= self.S_high:
                    z = np.array([[(xyxy[0] + xyxy[2]) * 0.5], [(xyxy[1] + xyxy[3]) * 0.5]])
                    self._create_new_track(xyxy, conf, det_class, z)
            return

        # Handle case with no detections
        if not detections:
            return

        # Split detections into high and low confidence sets
        dets_high = []
        dets_low = []

        for det in detections:
            if len(det) < 6:
                continue  # Invalid detection format
            x1, y1, x2, y2, conf, cls_id = det
            # Bib gets special treatment with lower threshold
            if conf > self.S_high or (cls_id == names['bib'] and conf > cfg.CONF_BIB_SPECIAL):
                dets_high.append(det)
            elif conf > self.S_low:
                dets_low.append(det)
            # else: ignore very low confidence detections

        # Initialize unmatched track indices
        track_ids = list(self.tracks.keys())
        self.unmatched_track_indices = set(range(len(track_ids)))
        # Two-stage matching: high-confidence first, then low-confidence
        self.primary_match(dets_high, dt)
        self.secondary_match(dets_low, dt)

        # Update static_age based on velocity magnitude
        static_thresh_sq = cfg.STATIC_VELOCITY_THRESHOLD * cfg.STATIC_VELOCITY_THRESHOLD
        for tid, tr in self.tracks.items():
            vx = tr['kf'].x[2, 0]
            vy = tr['kf'].x[3, 0]
            vel_sq = vx * vx + vy * vy
            if vel_sq < static_thresh_sq:
                self.tracks[tid]['static_age'] += 1
            else:
                self.tracks[tid]['static_age'] = 0

        # Age Re-ID buffer entries and remove expired ones
        expired_reid = []
        for rid, rinfo in self.reid_buffer.items():
            rinfo['age_in_buffer'] += 1
            if rinfo['age_in_buffer'] > cfg.REID_BUFFER_MAX_AGE:
                expired_reid.append(rid)
        for rid in expired_reid:
            del self.reid_buffer[rid]

        # Filter tracks based on age and coasting
        # - New tracks (age < MATURE_TRACK_MIN_AGE): allow only NEW_TRACK_MAX_COAST frames coasting
        # - Mature tracks (age >= MATURE_TRACK_MIN_AGE): allow up to MATURE_TRACK_MAX_COAST frames coasting
        # Killed tracks (exceeded coast limit) are added to Re-ID buffer for potential recovery
        filtered_tracks = {}
        for tid, tr in self.tracks.items():
            is_new_track = tr['age'] < cfg.MATURE_TRACK_MIN_AGE
            keep_track = False
            if is_new_track:
                keep_track = tr['coasted_age'] < cfg.NEW_TRACK_MAX_COAST
            else:
                keep_track = tr['coasted_age'] <= cfg.MATURE_TRACK_MAX_COAST

            if keep_track:
                filtered_tracks[tid] = tr
            else:
                # Track is being killed - add to Re-ID buffer for potential recovery
                # Only add mature tracks in ring zone (ball disappearing through net)
                # Use last_matched_pos (not drifted position) for ring zone Re-ID
                # This avoids drift from Kalman prediction during coasted frames
                last_matched_pos = tr.get('last_matched_pos', tr['kf'].x[:2].flatten())
                last_matched_vel = tr.get('last_matched_vel', tr['kf'].x[2:4].flatten())
                if not is_new_track and self._is_in_ring_zone(last_matched_pos):
                    self.reid_buffer[tid] = {
                        'last_pos': last_matched_pos.copy(),
                        'last_vel': last_matched_vel.copy(),
                        'wh': tr['wh'].copy() if isinstance(tr['wh'], list) else list(tr['wh']),
                        'object_class': tr['object_class'],
                        'age_in_buffer': 0,
                        'original_age': tr['age'],
                    }

        self.tracks = filtered_tracks

    def primary_match(self, detections, dt):
        """Match high-confidence detections to tracks and create new tracks."""
        self.unmatched_track_indices, unmatched_det_indices = self.do_matches(detections, dt)

        # Create new tracks for unmatched high-confidence detections
        for j in unmatched_det_indices:
            *xyxy, conf, det_class = detections[j]
            z = np.array([[(xyxy[0] + xyxy[2]) * 0.5], [(xyxy[1] + xyxy[3]) * 0.5]])
            self._create_new_track(xyxy, conf, det_class, z)

    def secondary_match(self, detections, dt):
        """Match remaining tracks to low-confidence detections.

        Note: Does not create new tracks for low-confidence detections.
        """
        _, _ = self.do_matches(detections, dt)

    def do_matches(self, detections, dt):
        """Perform Hungarian matching between tracks and detections.

        Uses IoU-based matching for players and distance-based matching for ball/bib.

        Args:
            detections: List of [x1, y1, x2, y2, conf, class_id]
            dt: Time step in seconds

        Returns:
            (unmatched_track_indices, unmatched_det_indices): Sets of unmatched indices
        """
        cfg = self.cfg
        track_ids = list(self.tracks.keys())
        cost_matrix = np.full((len(track_ids), len(detections)), 1e9)

        # Calculate cost matrix entries
        for i, track_id in enumerate(track_ids):
            # Only process unmatched tracks (for secondary matching)
            if i not in self.unmatched_track_indices:
                continue

            track = self.tracks[track_id]
            # Direct array access instead of flatten() to avoid array allocation
            pred_x = track['kf'].x[0, 0]
            pred_y = track['kf'].x[1, 0]
            vel_x = track['kf'].x[2, 0]
            vel_y = track['kf'].x[3, 0]

            # Special handling for ball during sky-shooting (ball going high)
            # Keep the track alive during the arc by resetting coasted_age
            if (track['object_class'] == names['ball'] and
                cfg.SKY_SHOT_Y_MIN < pred_y < cfg.SKY_SHOT_Y_MAX and
                track['coasted_age'] > 1):
                track['coasted_age'] = 1

            # Dynamic max distance based on velocity (use simple math instead of np.linalg.norm)
            vel_mag = math.sqrt(vel_x * vel_x + vel_y * vel_y)
            dynamic_max_dist = min(self.max_dist, self.base_dist + vel_mag * dt)

            # Get predicted track bounding box
            w, h = track['wh']
            track_bbox = [
                pred_x - w / 2,
                pred_y - h / 2,
                pred_x + w / 2,
                pred_y + h / 2
            ]

            for j, detection in enumerate(detections):
                *xyxy, conf, det_class = detection
                det_cx = (xyxy[0] + xyxy[2]) * 0.5
                det_cy = (xyxy[1] + xyxy[3]) * 0.5
                det_bbox = xyxy

                # Apply class constraints:
                # - Players can only match with players
                # - Ball/bib can match with each other (for shooting feature)
                # - Reject bib-to-ball match if bib is way above the ball track
                track_class = track['object_class']
                if (det_class == names['player'] and track_class != names['player']) or \
                        (track_class == names['player'] and det_class != names['player']) or \
                        (track_class == names['ball'] and det_class == names['bib']
                         and pred_y - det_cy > cfg.BIB_VERTICAL_REJECT):
                    continue  # Skip incompatible classes

                # Calculate Euclidean distance (simple math instead of numpy)
                dx = pred_x - det_cx
                dy = pred_y - det_cy
                dist = math.sqrt(dx * dx + dy * dy)

                if track_class != names['player']:
                    # Ball/bib: use distance-based cost
                    # Check if in ring zone for relaxed matching (handles bouncing)
                    in_ring_zone = (track_class == names['ball'] and
                                    (self._is_in_ring_zone((pred_x, pred_y)) or
                                     self._is_in_ring_zone((det_cx, det_cy))))
                    effective_max_dist = (dynamic_max_dist * cfg.RING_ZONE_DIST_MULTIPLIER
                                          if in_ring_zone else dynamic_max_dist)

                    if dist <= effective_max_dist:
                        cost_matrix[i, j] = dist
                        # Prioritize bib association (helps track ball through net)
                        if det_class == names['bib']:
                            cost_matrix[i, j] -= cfg.BIB_PRIORITY_BONUS
                else:
                    # Player: use IoU-based cost
                    iou = _bbox_iou(track_bbox, det_bbox)
                    if iou >= self.min_iou:
                        cost_matrix[i, j] = 1 - iou  # Lower cost for higher IoU

        # Apply Hungarian algorithm for optimal matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_track_indices = set(range(len(track_ids)))
        unmatched_det_indices = set(range(len(detections)))

        # Process valid matches
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1e9:  # Valid match
                matches.append((i, j))
                unmatched_track_indices.discard(i)
                unmatched_det_indices.discard(j)

        # Update matched tracks
        for i, j in matches:
            track_id = track_ids[i]
            *xyxy, conf, det_class = detections[j]
            z = np.array([[(xyxy[0] + xyxy[2]) * 0.5], [(xyxy[1] + xyxy[3]) * 0.5]])

            track = self.tracks[track_id]
            track['kf'].update(z)
            track['coasted_age'] = 0
            # Exponential moving average for confidence smoothing
            track['confidence'] = track['confidence'] * cfg.CONF_EMA_OLD + conf * cfg.CONF_EMA_NEW
            # Allow class mutation: detection class may change (e.g., ball<->bib)
            # This is intentional for shooting feature where bib helps track ball through net
            track['object_class'] = det_class
            track['coasted'] = False
            # Store matched detection for visualization
            track['matched_det'] = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, det_class]

            new_w = (xyxy[2] - xyxy[0])
            new_h = (xyxy[3] - xyxy[1])
            new_left = xyxy[0]
            new_right = xyxy[2]
            new_bottom = xyxy[3]

            # Update foot_loc for player objects
            # Anchor foot to the edge that moved less (handles arm raising)
            if det_class == names['player']:
                old_left = track['prev_left_edge']
                old_right = track['prev_right_edge']
                old_foot_x = track['foot_loc'][0]

                left_moved = abs(new_left - old_left)
                right_moved = abs(new_right - old_right)

                if left_moved < right_moved:
                    # Left edge is more stable, anchor foot relative to left edge
                    foot_offset_from_left = old_foot_x - old_left
                    new_foot_x = new_left + foot_offset_from_left
                else:
                    # Right edge is more stable, anchor foot relative to right edge
                    foot_offset_from_right = old_right - old_foot_x
                    new_foot_x = new_right - foot_offset_from_right

                # Clamp foot_x within the bounding box
                new_foot_x = max(new_left, min(new_right, new_foot_x))

                track['foot_loc'] = [new_foot_x, new_bottom]
                track['prev_left_edge'] = new_left
                track['prev_right_edge'] = new_right
            else:
                # For non-player objects, foot_loc is just center of bottom
                track['foot_loc'] = [(new_left + new_right) * 0.5, new_bottom]
                track['prev_left_edge'] = new_left
                track['prev_right_edge'] = new_right

            track['wh'] = [new_w, new_h]

            # Update last matched position/velocity (for Re-ID in ring zone - avoids drift)
            track['last_matched_pos'] = track['kf'].x[:2].flatten().copy()
            track['last_matched_vel'] = track['kf'].x[2:4].flatten().copy()

        return unmatched_track_indices, unmatched_det_indices

    def _try_reid_match(self, det_center, det_class, wh):
        """Try to match a detection against the Re-ID buffer.

        Args:
            det_center: Detection center [x, y]
            det_class: Detection class ID
            wh: Detection size [width, height]

        Returns:
            Matched track ID from buffer, or None if no match
        """
        cfg = self.cfg
        best_match_id = None
        best_dist = cfg.REID_MAX_DISTANCE

        for rid, rinfo in self.reid_buffer.items():
            # Must be same class
            if rinfo['object_class'] != det_class:
                continue

            # Check size similarity
            old_w, old_h = rinfo['wh']
            new_w, new_h = wh
            w_ratio = max(old_w, new_w) / (min(old_w, new_w) + 1e-6)
            h_ratio = max(old_h, new_h) / (min(old_h, new_h) + 1e-6)
            if w_ratio > (1 + cfg.REID_SIZE_TOLERANCE) or h_ratio > (1 + cfg.REID_SIZE_TOLERANCE):
                continue

            # Predict where the track would be now based on last velocity
            # Account for frames spent in buffer
            frames_in_buffer = rinfo['age_in_buffer']
            predicted_pos = rinfo['last_pos'] + rinfo['last_vel'] * frames_in_buffer * (1/60.0)

            # Calculate distance to predicted position
            dist = np.linalg.norm(predicted_pos - det_center)
            if dist < best_dist:
                best_dist = dist
                best_match_id = rid

        return best_match_id

    def _create_new_track(self, xyxy, conf, det_class, z):
        """Create a new track from a detection, or recover from Re-ID buffer.

        Args:
            xyxy: Bounding box [x1, y1, x2, y2]
            conf: Detection confidence
            det_class: Object class ID
            z: Initial position measurement [[x], [y]]
        """
        cfg = self.cfg
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        centroid_x = (xyxy[0] + xyxy[2]) * 0.5
        centroid_y = (xyxy[1] + xyxy[3]) * 0.5
        bottom_y = xyxy[3]  # bottom of bbox

        # Store edge positions to detect asymmetric width changes (for player foot tracking)
        left_edge = xyxy[0]
        right_edge = xyxy[2]

        # Check for class confusion (OD misclassification bandaid)
        # Reject if detection looks like it belongs to a nearby track of different class
        if self._is_class_confusion(centroid_x, centroid_y, det_class, w, h):
            return  # Don't create track - likely OD misclassification

        det_center = np.array([centroid_x, centroid_y])

        # Try to match against Re-ID buffer first (only in ring zone)
        reid_match = None
        if self._is_in_ring_zone(det_center):
            reid_match = self._try_reid_match(det_center, det_class, [w, h])
        if reid_match is not None:
            # Recover the old track with its original ID
            rinfo = self.reid_buffer.pop(reid_match)
            track_id = reid_match

            self.tracks[track_id] = {
                'kf': KalmanFilter(q=self.q, r=self.r),
                'wh': [w, h],
                'confidence': conf,
                'age': rinfo['original_age'] + rinfo['age_in_buffer'],  # Continue age
                'coasted': False,
                'coasted_age': 0,
                'static_age': 0,
                'object_class': det_class,
                'id': track_id,
                'foot_loc': [centroid_x, bottom_y],
                'prev_left_edge': left_edge,
                'prev_right_edge': right_edge,
                'matched_det': [xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, det_class],
                # Last matched position/velocity - used for Re-ID in ring zone
                'last_matched_pos': det_center.copy(),
                'last_matched_vel': rinfo['last_vel'].copy(),
            }
            self.tracks[track_id]['kf'].x[:2] = z
            # Initialize velocity from buffer (helps with smooth recovery)
            self.tracks[track_id]['kf'].x[2:4] = rinfo['last_vel'].reshape(2, 1)
            return

        # No Re-ID match - create brand new track
        # Find next available ID to avoid collision with existing tracks
        # This handles the case where old tracks may still exist after ID wrap
        original_next_id = self.next_id
        while self.next_id in self.tracks or self.next_id in self.reid_buffer:
            self.next_id += 1
            if self.next_id >= cfg.MAX_TRACK_ID:
                self.next_id = cfg.MIN_TRACK_ID
            if self.next_id == original_next_id:
                # All IDs are taken (extremely unlikely) - force cleanup
                print("[Tracker] Warning: All track IDs in use, forcing oldest track removal")
                oldest_tid = min(self.tracks.keys(), key=lambda t: self.tracks[t]['age'])
                del self.tracks[oldest_tid]
                break

        self.tracks[self.next_id] = {
            'kf': KalmanFilter(q=self.q, r=self.r),
            'wh': [w, h],
            'confidence': conf,
            'age': 1,
            'coasted': False,
            'coasted_age': 0,
            'static_age': 0,
            'object_class': det_class,
            'id': self.next_id,
            # Foot location tracking (for players)
            'foot_loc': [centroid_x, bottom_y],  # initially center of bottom line
            'prev_left_edge': left_edge,
            'prev_right_edge': right_edge,
            'matched_det': [xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, det_class],  # Associated detection
            # Last matched position/velocity - used for Re-ID in ring zone (avoids drift from coasting)
            'last_matched_pos': det_center.copy(),
            'last_matched_vel': np.zeros(2),
        }
        self.tracks[self.next_id]['kf'].x[:2] = z

        # Increment ID with wrap-around
        self.next_id += 1
        if self.next_id >= cfg.MAX_TRACK_ID:
            self.next_id = cfg.MIN_TRACK_ID

    def get_tracks(self):
        """Get current track states for external use.

        Returns:
            Dict mapping track_id to track state dict containing:
            - id, pos, wh, vel, acc, conf, age, coasted, coasted_age,
              object_class, static_age, foot_loc, matched_det
        """
        return {
            tid: {
                'id': tr['id'],
                'pos': tr['kf'].x[:2].flatten(),
                'wh': tr['wh'],
                'vel': tr['kf'].x[2:4].flatten(),
                'acc': tr['kf'].x[4:].flatten(),
                'conf': tr['confidence'],
                'age': tr['age'],
                'coasted': tr['coasted'],
                'coasted_age': tr['coasted_age'],
                'object_class': tr['object_class'],
                'static_age': tr['static_age'],
                'foot_loc': tr['foot_loc'],
                'matched_det': tr.get('matched_det')  # Associated detection [x1,y1,x2,y2,conf,class] or None
            }
            for tid, tr in self.tracks.items()
        }
