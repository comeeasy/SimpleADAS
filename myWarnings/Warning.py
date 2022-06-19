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
        for x in range(256 + 1):
            self.TEMPLATE2BEV_LOOKUPTBL.append([])
            for y in range(128 + 1):
                bef_coor = np.array([x, y, 1])
                aft_coor = np.matmul(self.M, np.transpose(bef_coor))

                self.TEMPLATE2BEV_LOOKUPTBL[x].append(
                        [int(aft_coor[0] * (1 / aft_coor[2])),
                         int(aft_coor[1] * (1 / aft_coor[2]))]
                        )

        self.collision_front_danger_distance = 0.5 * self.BEV_HEIGHT
        self.collision_side_margin = 10
        
        self.MAX_RAPID_TOLERANCE = 3
        self.rapid_start_tolerance = 0
        self.rapid_stop_tolerance = 0


    # @staticmethod
    def is_lane_departure(self, l_x, r_x, template):
        if not l_x is None and not r_x is None:
            left_dist  = abs(l_x - self.center_of_template)
            right_dist = abs(r_x - self.center_of_template)

            if left_dist > right_dist:
                ratio = right_dist / left_dist
            else:
                ratio = left_dist / right_dist
            
            if ratio < self.lane_warn_threshold:
                cv2.putText(
                    template, 
                    "Lane!", (170, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                return True

        else:
            return False

    # @staticmethod
    def is_collision(self, detections, template):
        for x, y, w, h in detections:
            try:
                x, y, w, h = int(x), int(y), int(w), int(h)

                left_p, right_p = (x, y+h), (x+w, y+h)

                left_p_BEV  = self.TEMPLATE2BEV_LOOKUPTBL[left_p[0]][left_p[1]]
                right_p_BEV = self.TEMPLATE2BEV_LOOKUPTBL[right_p[0]][right_p[1]]

                x, y = left_p_BEV[0], left_p_BEV[1]
                if self.collision_side_margin < x < self.BEV_WIDTH - self.collision_side_margin \
                            and self.collision_front_danger_distance < y:
                    cv2.putText(
                        template, 
                        "Warn!", (170, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    return True
                else:
                    False
                
                x, y = right_p_BEV[0], right_p_BEV[1]
                if self.collision_side_margin < x < self.BEV_WIDTH + self.collision_side_margin \
                            and self.collision_front_danger_distance < y:
                    cv2.putText(
                        template, 
                        "Warn!", (170, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    return True
                else:
                    False
            except Exception as e:
                print(e)
                continue
        else:
            return  False
        
    def is_rapid_stop_or_start(self, speed, acceleration, template):
        
        # 국토교통부 위험운전행동 기준
        
        # 5km/h 이하 속도에서 출발하여 초당 6km/h 이상 가속 운행하는 경우
        if acceleration >= 5 and speed <= 5:
            self.rapid_stop_tolerance = 0
            self.rapid_start_tolerance += 1
            
            if self.rapid_start_tolerance >= self.MAX_RAPID_TOLERANCE:
                print("Warn : Rapid Accelerate!")
                cv2.putText(
                    template, 
                    "RapStart", (170, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)
        
        # 초당 8km/h 이상 감속 운행하고 속도가 6km/h 이상인경우
        if acceleration <= -8 and speed >= 6:
            self.rapid_start_tolerance = 0
            self.rapid_stop_tolerance += 1

            if self.rapid_stop_tolerance >= self.MAX_RAPID_TOLERANCE:
                print("Warn! : Rapid Stop!")
                cv2.putText(
                    template, 
                    "RapStop", (170, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)
        

        

        