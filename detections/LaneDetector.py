import cv2
import numpy as np
# from sklearn import linear_model
from tqdm import tqdm

from detections.Ultra.utils.config import Config
from detections.Ultra.data.constant import culane_row_anchor

import torch
import torch.nn as nn
import os

from torch2trt import TRTModule

class LaneDetectorAI:
    
    def __init__(self):
        self.cfg = Config.fromfile('/home/r320/ComputerVisionADASProject/detections/Ultra/configs/culane.py')
        self.cfg.test_model = f"/home/r320/ComputerVisionADASProject/detections/Ultra/weights/culane_18_fp16.pth"
        torch.backends.cudnn.benchmark = True

        assert self.cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

        if self.cfg.dataset == 'CULane':
            self.cls_num_per_lane = 18
            self.row_anchor = culane_row_anchor
        elif self.cfg.dataset == 'Tusimple':
            self.cls_num_per_lane = 56
        else:
            raise NotImplementedError
            
        #addition 12/26
        self.sm = nn.Softmax(dim=0)
        # we add for fix segementation fault
        self.idx = torch.tensor(list(range(self.cfg.griding_num)), device='cuda', dtype=torch.float16) + 1
        # origin
        # self.idx = torch.arange(self.cfg.griding_num).type(torch.HalfTensor).cuda() + 1
        self.idx = self.idx.reshape(-1, 1, 1)

        # tensorrt model
        self.net_trt = TRTModule()
        self.net_trt.load_state_dict(torch.load(self.cfg.test_model))
        self.net_trt.eval()
        
        self.col_sample = np.linspace(0, 800 - 1, self.cfg.griding_num)
        self.col_sample_w = self.col_sample[1] - self.col_sample[0]
        self.img_h = 288

    def __call__(self, img):
        frame_lane = cv2.resize(img, (800, 288))

        img = torch.tensor(frame_lane, device=torch.device("cuda")).permute(2, 0, 1)
        img = img.view(1, 3, 288, 800)
        img = torch.div(img, 255.)

        with torch.no_grad():
            out = self.net_trt(img)

        ### Lane: calculate out_j
        out_j = out.squeeze()
        prob = self.sm(out_j[:-1, :, :])
        
        loc = torch.sum(prob * self.idx, axis=0)
        
        out_j = torch.argmax(out_j, axis=0)
        out_j = out_j%self.cfg.griding_num
        out_j = out_j.bool().int()
        loc = loc * out_j

        out_j = loc.detach().cpu().numpy()
        out_j = out_j[::-1,:]

        ### Lane: calculate ppp
        line = []
        lane_loc_list = []
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * self.col_sample_w) - 1, int(self.img_h * (self.row_anchor[self.cls_num_per_lane-1-k]/288)) - 1 )
                        line.append(ppp)
                lane_loc_list.append(line)
                line = []
        
        for line in lane_loc_list:
            for locate in line:
                if locate[0] > 0:
                    cv2.circle(frame_lane, tuple(np.int32(locate)), 3, (0, 255, 0), 3)
        
        cv2.imshow("lane", frame_lane)



class SpeedMeter:
    def __init__(self, speed=60, threshold=10) -> None:
        self._speed = speed
        self.threshold = threshold

    def update(self, speed):
        # if abs(speed - self._speed) < self.threshold:
        self._speed = (self._speed + speed) / 2
    
    @property
    def speed(self):
        return self._speed

class SpeedDetector:
    def __init__(self) -> None:
        self.num_pix_of_short_line = 37

        # length of white short line is 5m
        self.meter_per_pixel = 5 / 37

        self.previous_optical_line = LaneDetector.BEV_HEIGHT // 2
        self.previous_speed = 60
        self.previous_out = None

        self.speedMeter = SpeedMeter(speed=60, threshold=10)

    def find_optical_line(self, optical):
        for i in range(LaneDetector.BEV_HEIGHT-1, -1, -1):
            for x in range(LaneDetector.BEV_WIDTH-LaneDetector.window_width, LaneDetector.BEV_WIDTH):
                if optical[i][x] > 0:
                    return i
        else:
            return 0

    def __call__(self, out):
        # optical image
        if self.previous_out is not None:
            optical = cv2.bitwise_and(out, self.previous_out)
            cv2.imshow("back and", optical)

            # search fitst line contating any white point
            optical_line = self.find_optical_line(optical)

            moving_pix_per_30ms = optical_line - self.previous_optical_line
            current_speed_m_per_sec = moving_pix_per_30ms * self.meter_per_pixel * (100/3)

            self.speedMeter.update((3600 / 1000) * current_speed_m_per_sec)
            self.previous_optical_line = optical_line

        self.previous_out = cv2.dilate(out.copy(), np.ones((5, 5)), iterations=3)
        
        return self.speedMeter.speed
 

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
                [self.WARPAFFINE_WIDTH // 2 - 35, 30],
                [25, self.WARPAFFINE_HEIGHT],
                [self.WARPAFFINE_WIDTH // 2 + 40, 30],
                [self.WARPAFFINE_WIDTH - 20, self.WARPAFFINE_HEIGHT],
            ]
        elif video_name.endswith("highway_D5_Trim.mp4"):
            self.LANE_ROI_POINTS = [
                [self.WARPAFFINE_WIDTH // 2 - 35, 30],
                [20, self.WARPAFFINE_HEIGHT],
                [self.WARPAFFINE_WIDTH // 2 + 30, 30],
                [self.WARPAFFINE_WIDTH - 20, self.WARPAFFINE_HEIGHT],
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

        for x in tqdm(range(self.BEV_WIDTH+1)):
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

        # roi points temp 
        roi_points = cv2.resize(out, (self.BEV_WIDTH // self.ROI_WIDTH, self.N_WINDOWS))
        l_x, r_x = self.roi_lane_detect(roi_points, template)
        speed = self.speedDetector(out)
        print(f"speed: {speed:.2f}km/h")

        # cv2.imshow("lane middle result", out)

        return l_x, r_x, speed        

    def show_BEV(self):
        cv2.imshow("BEV", self.BEV_color)
