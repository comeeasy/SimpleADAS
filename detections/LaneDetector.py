import cv2
import numpy as np
# from sklearn import linear_model


class SpeedDetector:
    def __init__(self) -> None:
        self.MAX_TOLERANCE = 3
        self.frame_count = 0
        self.tolerance = 0
        
        self.coef_of_equ = 3600 * 5 / 33

    def calc_velocity(self, frame_count):
        if frame_count == 0:
            return 0
        
        v = self.coef_of_equ / frame_count
        if v > 100:
            return 0
        else:
            return v 

    def line_frame_count(self, out):
        for i in range(LaneDetector.BEV_WIDTH-LaneDetector.window_width, LaneDetector.BEV_WIDTH):
            
            if out[LaneDetector.BEV_HEIGHT-10][i] > 150:
                # error condition 
                if self.frame_count > 300000:
                    self.frame_count = 0
                
                self.frame_count += 1
                # velocity is not yet decided
                return False, 0
        else:
            if self.tolerance < self.MAX_TOLERANCE:
                self.tolerance += 1
                return False, 0
            else:
                self.tolerance = 0
                frame_count = self.frame_count
                
                if frame_count != 0:
                    self.frame_count = 0
                    return True, self.calc_velocity(frame_count)
                else:
                    return False, 0
            
            
    def __call__(self, out):
        is_calculated, velocity = self.line_frame_count(out)
        if is_calculated:
            return velocity
        
 

class LaneDetector:
    WARPAFFINE_WIDTH = 256
    WARPAFFINE_HEIGHT = 128

    BEV_WIDTH = 32
    BEV_HEIGHT = 128

    N_WINDOWS = 16
    window_height = BEV_HEIGHT // N_WINDOWS
    window_width = 8

    ROI_WIDTH = 2

    BEV_POINTS = [
        [0, 0],
        [0, BEV_HEIGHT],
        [BEV_WIDTH, 0],
        [BEV_WIDTH, BEV_HEIGHT]
    ]

    def __init__(self, video_name: str):
        # (x, y)순임. (y, x)순 아님
        # LU -> LB -> RU -> RB
        # 256 x 128 (W, H) image 기준
        if video_name.endswith("highway_D6_Trim.mp4"):
            self.LANE_ROI_POINTS = [
                [self.WARPAFFINE_WIDTH // 2 - 25, 30],
                [35, self.WARPAFFINE_HEIGHT],
                [self.WARPAFFINE_WIDTH // 2 + 40, 30],
                [self.WARPAFFINE_WIDTH - 20, self.WARPAFFINE_HEIGHT],
            ]
        elif video_name.endswith("highway_D5_Trim.mp4"):
            self.LANE_ROI_POINTS = [
                [self.WARPAFFINE_WIDTH // 2 - 25, 30],
                [30, self.WARPAFFINE_HEIGHT],
                [self.WARPAFFINE_WIDTH // 2 + 25, 30],
                [self.WARPAFFINE_WIDTH - 30, self.WARPAFFINE_HEIGHT],
            ]

        # self.lr = linear_model.RANSACRegressor()
        self.M = cv2.getPerspectiveTransform(
            np.array(self.LANE_ROI_POINTS, dtype=np.float32),
            np.array(self.BEV_POINTS, dtype=np.float32)
        )
        self.inv_M = np.linalg.inv(self.M)
        self.BEV2TEMPLATE_LOOKUPTBL = []
        # HSV yello
        self.low_yellow = np.array([0, 50, 0])
        self.upper_yellow = np.array([120, 255, 255])

        for x in range(self.BEV_WIDTH+1):
            self.BEV2TEMPLATE_LOOKUPTBL.append([])
            for y in range(self.BEV_HEIGHT+1):
                bef_coor = np.array([x, y, 1])
                aft_coor = np.matmul(self.inv_M, np.transpose(bef_coor))

                self.BEV2TEMPLATE_LOOKUPTBL[x].append(
                        [int(aft_coor[0] * (1 / aft_coor[2])),
                         int(aft_coor[1] * (1 / aft_coor[2]))]
                        )
                    

        self.BEV_color = np.zeros((self.BEV_HEIGHT, self.BEV_WIDTH, 3))
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.roi_to_BEV_table = []
        for row in range(self.N_WINDOWS+1):
            _ = list()
            _row = 8 * row
            for l_idx in range(0, self.BEV_WIDTH // 2):
                _.append((2 * l_idx, _row))
            for r_idx in range(self.BEV_WIDTH // 2, self.BEV_WIDTH+1):
                _.append((2 * r_idx + 1, _row))
            self.roi_to_BEV_table.append(_)

        self.previous_roi_result = None
        self.previous_out = None

        self.num_pix_of_short_line = 37

        # length of white short line is 5m
        self.meter_per_pixel = 5 / 37

        self.previous_optical_line = self.BEV_HEIGHT // 2
        self.previous_speed = 60

        # self.speedMeter = SpeedMeter(speed=/60, threshold=10)
        self.speedDetector = SpeedDetector()
        self.cur_speed = 0
        self.acc_frame_count = 0

    def roi_lane_detect(self, roi_points, template=None):
        roi_result = np.zeros_like(roi_points)
        l_x, r_x = 0, 0
        l_pts, r_pts = [], []

        # num of rois in a window
        n_rois = self.window_width // self.ROI_WIDTH
        # total num of roi in a row
        roi_width = self.BEV_WIDTH // self.ROI_WIDTH

        # find most inner lane
        for i in range(self.N_WINDOWS): # 16: N_WINDOW
            for l_idx in range(n_rois - 1, -1, -1): # 3, 2, 1, 0
                if roi_points[i][l_idx] > 0:
                    roi_result[i][l_idx] = 1
                    x_bev, y_bev = self.roi_to_BEV_table[i][l_idx]
                    l_pts.append(self.BEV2TEMPLATE_LOOKUPTBL[x_bev][y_bev])
                    cv2.circle(self.BEV_color, self.roi_to_BEV_table[i][l_idx], 2, (0, 255, 255), -1)
                    # last idx is left x
                    l_x = l_idx
                    break
            
            for r_idx in range(roi_width - n_rois, roi_width): # 13, 12, 11, 10:
                if roi_points[i][r_idx] > 0:
                    roi_result[i][r_idx] = 1
                    x_bev, y_bev = self.roi_to_BEV_table[i][r_idx]
                    r_pts.append(self.BEV2TEMPLATE_LOOKUPTBL[x_bev][y_bev])

                    cv2.circle(self.BEV_color, self.roi_to_BEV_table[i][r_idx], 2, (0, 255, 255), -1)
                    # last idx is right x
                    r_x = r_idx
                    break

        self.previous_roi_result = roi_result

        # two short lane has length of 3points (two short lane is a line)
        if len(l_pts) > 4:
            x_bev, y_bev = self.roi_to_BEV_table[self.N_WINDOWS][l_x]
            pts = self.BEV2TEMPLATE_LOOKUPTBL[x_bev][y_bev]
            l_pts.append(pts)
            l_x = pts[0]
        else:
            l_x = None

        if len(r_pts) > 4:
            x_bev, y_bev = self.roi_to_BEV_table[self.N_WINDOWS][r_x]
            pts = self.BEV2TEMPLATE_LOOKUPTBL[x_bev][y_bev]
            r_pts.append(pts)
            r_x = pts[0]
        else:
            r_x = None

        if not template is None:
            cv2.polylines(template, [np.array(l_pts)], False, (255, 0 ,255), 2)
            cv2.polylines(template, [np.array(r_pts)], False, (255, 0 ,255), 2)

        return l_x, r_x

    def __call__(self, gray, hsv, template=None):
        self.BEV_color = np.zeros_like(self.BEV_color)

        # center line (yellow line)
        center_line_mask = cv2.inRange(hsv, self.low_yellow, self.upper_yellow)

        # white line image
        ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # combine yellow, white line image
        out = cv2.add(thresh, center_line_mask)

        # BEV image (256, 1024) (W, H)
        out = cv2.warpPerspective(out, self.M, (self.BEV_WIDTH, self.BEV_HEIGHT))
        out = cv2.normalize(out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)               

        cv2.imshow("BEV", out)

        # roi points temp 
        roi_points = cv2.resize(out, (self.BEV_WIDTH // self.ROI_WIDTH, self.N_WINDOWS))
        
        l_x, r_x = self.roi_lane_detect(roi_points, template)
        speed = self.speedDetector(out)
        if speed:
            v_delta = speed - self.cur_speed
            acceleration = v_delta / (0.033 * self.acc_frame_count)
            
            self.cur_speed = speed
            self.acc_frame_count = 0
        else:
            acceleration = 0
            self.acc_frame_count += 1

        return l_x, r_x, self.cur_speed, acceleration        

    def show_BEV(self):
        cv2.imshow("BEV", self.BEV_color)
