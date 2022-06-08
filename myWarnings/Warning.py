import cv2
import numpy as np


class Warning:

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

    def __init__(self, lane_warn_threshold, video_name) -> None:
        self.center_of_template = 128
        self.maximum_of_distance_mul = self.center_of_template * self.center_of_template    
        self.lane_warn_threshold = lane_warn_threshold

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

        self.TEMPLATE2BEV_LOOKUPTBL = []
        for x in range(self.BEV_WIDTH+1):
            self.TEMPLATE2BEV_LOOKUPTBL.append([])
            for y in range(self.BEV_HEIGHT+1):
                bef_coor = np.array([x, y, 1])
                aft_coor = np.matmul(self.M, np.transpose(bef_coor))

                self.TEMPLATE2BEV_LOOKUPTBL[x].append(
                        [int(aft_coor[0] * (1 / aft_coor[2])),
                         int(aft_coor[1] * (1 / aft_coor[2]))]
                        )

        self.collision_front_danger_distance = 0.5 * self.BEV_HEIGHT
        self.collision_side_margin = 10


    # @staticmethod
    def is_lane_departure(self, l_x, r_x):
        if not l_x is None and not r_x is None:
            left_dist  = abs(l_x - self.center_of_template)
            right_dist = abs(r_x - self.center_of_template)

            if left_dist > right_dist:
                ratio = right_dist / left_dist
            else:
                ratio = left_dist / right_dist
            
            if ratio < self.lane_warn_threshold:
                return True

        else:
            return False

    # @staticmethod
    def is_collision(self, detections):
        for detection in detections:
            try:
                x, y = int(detection.Center[0]), int(detection.Center[1])
                w_half, h_half = int(detection.Width / 2), int(detection.Height / 2)

                left_p, right_p = (x-w_half, y-h_half), (x+w_half, y-h_half)

                left_p_BEV  = self.TEMPLATE2BEV_LOOKUPTBL[left_p[0], left_p[1]]
                right_p_BEV = self.TEMPLATE2BEV_LOOKUPTBL[right_p[0], right_p[1]]

                x, y = left_p_BEV[0], left_p_BEV[1]
                if self.collision_side_margin < x < self.BEV_WID + self.collision_side_margin \
                            and -20 < y < self.collision_front_danger_distance:
                    return True
                else:
                    False
                
                x, y = right_p_BEV[0], right_p_BEV[1]
                if self.collision_side_margin < x < self.BEV_WID + self.collision_side_margin \
                            and -20 < y < self.collision_front_danger_distance:
                    return True
                else:
                    False
            except:
                continue
        else:
            return  False
        

        

        