from manual_pts_sel import select_critical_zone_corners
import cv2
import numpy as np
from scipy.interpolate import interp1d
from jetson_utils import cudaToNumpy
import jetson_utils as jutils
import os

def _clip_xy(x, y, w, h):
    return int(max(0, min(w - 1, round(x)))), int(max(0, min(h - 1, round(y))))

class MANUAL_BEV:
    # Court dimensions for BEV scale calculation
    # Standard basketball court baseline width: 50 feet = 15.24 meters
    COURT_WIDTH_METERS = 15.24
    # RBEV template baseline width in pixels (from RBEV_TEMPLATE_PTS: 1320 - 80)
    RBEV_BASELINE_WIDTH_PX = 1240

    def __init__(self, img, DEFAULT_CORNERS, f_cals, side, image_path=None):
        self.img = img
        self.image_path = image_path  # Path to image file for refresh capability
        self.view_bgr = img.copy()
        #dst_points=((500,700),(100,100),(100,700),(500,100))  # (x_max→, x_min→, y_max→, y_min→) 现在可当"世界坐标"
        #dst_points = ((500, 700), (-300, 100), (100, 700), (900, 100)) #make 2nd and 4th pts on 3ptr curve and boundary intersection
        # dst_points = ((-300, 100), (900, 100), (500, 700), (100, 700))  # 3 pts uses
        if side == 'left':    # this is to overcome the distortion effect
            # before, it biased to left, so adjust pt1's x to left
            dst_points = ((-450, 100), (800, 100), (500, 700), (100, 700))  # 3 pts uses
        else:
            # before, it biased to right, so adjust pt1's and pt2's x to right
            dst_points = ((-200, 100), (1200, 100), (500, 700), (100, 700))

        self.world_bbox=(-2000, 2000, 0, 2000)               # <- 新：例如 (-2000, 2000, 0, 2000)，不传则走旧逻辑
        self.dst_world = np.array(tuple((float(x), float(y)) for (x, y) in dst_points), dtype=np.float32)
        self.px_per_unit = 1.0      # <- 新：1 世界单位=1像素
        # —— 2b-2) 世界→像素 仿射，把 world_bbox 装进画布
        x_min, x_max, y_min, y_max = self.world_bbox
        # 注意：若你要“端点也占一列/行像素”，可加 +1
        self.bev_w = int(round((x_max - x_min) * self.px_per_unit))
        self.bev_h = int(round((y_max - y_min) * self.px_per_unit))
        self.DEFAULT_CORNERS = DEFAULT_CORNERS # left
        self.side = side
        if self.side == 'left' and os.path.exists("H_final_left.npy"):
            self.H = np.load("H_final_left.npy")
        elif self.side == 'right' and os.path.exists("H_final_right.npy"):
            self.H = np.load("H_final_right.npy")
        else:
            self.H = None

        self.dst_pts_bev_px = [self.world_to_bev_px(float(xw), float(yw)) for xw, yw in
                               self.dst_world.reshape(-1, 2)]
        # create a window and bind mouse function
        if f_cals:
            self.window = "test manual"
            cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)  # WINDOW_NORMAL is more tolerant than AUTOSIZE
            cv2.setMouseCallback(self.window, self._on_mouse)

            rbev_bgr0 = cv2.imread("BEV/RBEV_1400.png", cv2.IMREAD_COLOR)
            self.rbev_view = rbev_bgr0.copy()

            # initial show
            cv2.imshow(self.window, self.view_bgr)
            cv2.imshow('rbev', self.rbev_view)
            cv2.waitKey(1)

    def _on_mouse(self, event, x, y, flags, userdata=None):
        if event != cv2.EVENT_LBUTTONDOWN or self.H is None:
            return
        # draw marker on the MAIN view (the camera image)
        cv2.drawMarker(self.view_bgr, (x, y), (255, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=8,
                       thickness=2, line_type=cv2.LINE_AA)
        cv2.imshow(self.window, self.view_bgr)

        p = np.array([x, y, 1.0], dtype=np.float64)
        q = self.H @ p
        u = q[0] / q[2]
        v = q[1] / q[2]
        u_i = int(round(u))
        v_i = int(round(v))
        marker_uv = (u_i, v_i)
        # Project BEV marker to RBEV

        r_pt = self._project_bev_to_rbev(marker_uv)
        xr, yr = r_pt
        # next draw xr, yr in the RBEV figure
        cv2.drawMarker(self.rbev_view, (xr, yr), (255, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=8,
                       thickness=2, line_type=cv2.LINE_AA)
        cv2.imshow('rbev', self.rbev_view)
        cv2.waitKey(1)

    def calc_bev_pts(self, x, y):
        p = np.array([x, y, 1.0], dtype=np.float64)
        q = self.H @ p
        u = q[0] / q[2]
        v = q[1] / q[2]
        u_i = int(round(u))
        v_i = int(round(v))
        marker_uv = (u_i, v_i)
        # Project BEV marker to RBEV

        r_pt = self._project_bev_to_rbev(marker_uv)
        xr, yr = r_pt
        return xr, yr

    def calc_distance_meters(self, pt1_rbev, pt2_rbev):
        """Calculate real-world distance in meters between two RBEV coordinate points.

        Uses the court dimensions and RBEV template scale to convert pixel distance
        to meters. This is more accurate than using a fixed pixels-per-meter constant.

        Args:
            pt1_rbev: (x, y) tuple of first point in RBEV coordinates
            pt2_rbev: (x, y) tuple of second point in RBEV coordinates

        Returns:
            Distance in meters
        """
        # Scale factor: meters per RBEV pixel (at baseline level)
        meters_per_pixel = self.COURT_WIDTH_METERS / self.RBEV_BASELINE_WIDTH_PX

        dx = pt2_rbev[0] - pt1_rbev[0]
        dy = pt2_rbev[1] - pt1_rbev[1]
        dist_px = np.sqrt(dx * dx + dy * dy)

        return dist_px * meters_per_pixel

    def calc_H(self):
        corners = select_critical_zone_corners(self.img, self.DEFAULT_CORNERS, image_path=self.image_path)
        print("Selected corners:", corners)
        corners_npy = np.array(corners)
        src_pts_xy = [corners[0], corners[1], corners[2], corners[3]]
        src = np.array(src_pts_xy, dtype=np.float32)
        H_src2world = cv2.getPerspectiveTransform(src, self.dst_world)
        # 图像y向下： u=(X - x_min)*s, v=(Y - y_min)*s
        x_min, x_max, y_min, y_max = self.world_bbox
        A_world2pix = np.array([
            [ self.px_per_unit, 0.0,           -x_min*self.px_per_unit],
            [ 0.0,         self.px_per_unit,   -y_min*self.px_per_unit],
            [ 0.0,         0.0,            1.0              ]
        ], dtype=np.float64)

        self.H = A_world2pix @ H_src2world  # 源像素 → 世界 → 像素

        if self.side == 'left':
            np.save("H_final_left.npy", self.H)
            np.save("corners_left.npy", corners_npy)
        elif self.side == 'right':
            np.save("H_final_right.npy", self.H)
            np.save("corners_right.npy", corners_npy)

    def _project_bev_to_rbev(self, uv):
        """Project BEV pixel (u,v) to RBEV pixel (xr,yr) using H_RBEV; return None on failure."""
        RBEV_TEMPLATE_PTS = ((80, 0),(1320, 0), (925, 580), (475, 580))

        bev_template_pts = (
            self.dst_pts_bev_px[0],
            self.dst_pts_bev_px[1],
            self.dst_pts_bev_px[2],
            self.dst_pts_bev_px[3]
        )

        x_bev_array = (bev_template_pts[0][0], bev_template_pts[3][0], bev_template_pts[2][0], bev_template_pts[1][0])
        x_rbev_array = (RBEV_TEMPLATE_PTS[0][0], RBEV_TEMPLATE_PTS[3][0], RBEV_TEMPLATE_PTS[2][0], RBEV_TEMPLATE_PTS[1][0])

        y_bev_array = (bev_template_pts[0][1], bev_template_pts[2][1])
        y_rbev_array = (RBEV_TEMPLATE_PTS[0][1], RBEV_TEMPLATE_PTS[2][1])


        fx = interp1d(x_bev_array, x_rbev_array, kind='linear', fill_value='extrapolate')
        fy = interp1d(y_bev_array, y_rbev_array, kind='linear', fill_value='extrapolate')

        xr = fx(uv[0])
        yr = fy(uv[1])

        # if xr, yr is out of image and the bev is not class 0 , then do a clip on rbev
        hR, wR = 1400, 1400
        if not (0 <= xr < wR):
            xr = min(max(xr, 0), wR - 1)
        if not (0 <= yr < hR):
            yr = min(max(yr, 0), hR - 1)

        return int(xr), int(yr)

    def world_to_bev_px(self, xw, yw):
        x_min, x_max, y_min, y_max = self.world_bbox
        xpx = (xw - x_min) * self.px_per_unit
        ypx = (yw - y_min) * self.px_per_unit
        return _clip_xy(xpx, ypx, self.bev_w, self.bev_h)

    def run(self):
        print("[help] Left-click on TOP image to set BEV & RBEV markers; 'r' reset; 's' save; 'q'/Esc quit.")
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (27, ord('q')):  # Esc or 'q'
                break
            elif key == ord('r'):
                cv2.imshow(self.window, self.img)
            elif key == ord('s'):
                out = "triple_view_vertical.png"
                # cv2.imwrite(out, self.canvas_bgr)
                print(f"[saved] {out}")
        cv2.destroyAllWindows()

if __name__ == "__main__":

    cap_left = jutils.videoSource(
        "/dev/video0", options={'width': 1200, 'height': 1920, 'framerate': 60})
    cap_right = jutils.videoSource(
        "/dev/video1", options={'width': 1200, 'height': 1920, 'framerate': 60})

    DEFAULT_CORNERS = [(518, 470), (892, 530), (1218, 476), (848, 435)]  # left
    left_np = cudaToNumpy(cap_left.Capture()).astype(np.uint8)  # HWC, uint8
    left_np = left_np[..., ::-1]
    f_cals = False
    side = 'left'
    manual_bev_left = MANUAL_BEV(left_np, DEFAULT_CORNERS, f_cals, side)
    # calculate homography and
    manual_bev_left.calc_H()

    DEFAULT_CORNERS = [(960, 448), (584, 497), (909, 541), (1283, 472)]  # right
    right_np = cudaToNumpy(cap_right.Capture()).astype(np.uint8)  # HWC, uint8
    right_np = right_np[..., ::-1]
    f_cals = False
    side = 'right'
    manual_bev_right = MANUAL_BEV(right_np, DEFAULT_CORNERS, f_cals, side)
    # calculate homography and
    manual_bev_right.calc_H()

