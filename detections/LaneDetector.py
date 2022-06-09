import cv2
import numpy as np
from sklearn import linear_model
from tqdm import tqdm


class LaneDetector:
    WARPAFFINE_WIDTH = 256
    WARPAFFINE_HEIGHT = 128

    BEV_WIDTH = 32
    BEV_HEIGHT = 128

    N_WINDOWS = 32
    window_height = BEV_HEIGHT // N_WINDOWS
    window_width = 10

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
                [self.WARPAFFINE_WIDTH // 2 - 25, 20],
                [35, self.WARPAFFINE_HEIGHT],
                [self.WARPAFFINE_WIDTH // 2 + 30, 20],
                [self.WARPAFFINE_WIDTH - 30, self.WARPAFFINE_HEIGHT],
            ]
        elif video_name.endswith("highway_D5_Trim.mp4"):
            self.LANE_ROI_POINTS = [
                [self.WARPAFFINE_WIDTH // 2 - 30, 20],
                [30, self.WARPAFFINE_HEIGHT],
                [self.WARPAFFINE_WIDTH // 2 + 30, 20],
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
        self.low_yellow = np.array([10, 100, 100])
        self.upper_yellow = np.array([52, 255, 255])

        for x in tqdm(range(self.BEV_WIDTH)):
            self.BEV2TEMPLATE_LOOKUPTBL.append([])
            for y in range(self.BEV_HEIGHT):
                bef_coor = np.array([x, y, 1])
                aft_coor = np.matmul(self.inv_M, np.transpose(bef_coor))

                self.BEV2TEMPLATE_LOOKUPTBL[x].append(
                    [
                        np.array([int(aft_coor[0] * (1 / aft_coor[2])),
                                  int(aft_coor[1] * (1 / aft_coor[2]))])
                    ])

        self.BEV_color = np.zeros((self.BEV_HEIGHT, self.BEV_WIDTH, 3))
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def right_line_detect(self, img, template):
        X, Y = [], []
        for i in range(self.N_WINDOWS):
            right_window = img[i*(self.window_height):(i+1)*self.window_height, self.BEV_WIDTH-self.window_width:]

            moments = cv2.moments(right_window)
            try:
                cX, cY = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
                X.append(cX + self.BEV_WIDTH - self.window_width)
                Y.append(cY + i * (self.window_height))
                cv2.circle(self.BEV_color, (cX + self.BEV_WIDTH - self.window_width, cY + i * (self.window_height)), 2, (0, 150, 0), -1)
            except:
                pass

        if len(X) > 1:
            # self.lr.fit(np.array(X).reshape(-1, 1), np.array(Y))
            try:
                z = np.polyfit(X, Y, 1)
            except:
                return
            p = np.poly1d(z)
            print(p)

            if abs(p.c[0]) < 20:
                return

            right_points = []
            for i in range(self.BEV_WIDTH//2, self.BEV_WIDTH):
                # x, y = i, int(self.lr.predict(np.array(i).reshape(-1, 1)))
                x, y = i, int(p(i))
                if not (0 < y < self.BEV_HEIGHT):
                    continue

                right_points.append(self.BEV2TEMPLATE_LOOKUPTBL[x][y])
                cv2.circle(self.BEV_color, (x, y), 5, (0, 255, 255), -1)

            x, y = int((self.BEV_HEIGHT - p.c[1]) / p.c[0]), self.BEV_HEIGHT-1
            print(x, y)
            if self.BEV_WIDTH // 2 < x < self.BEV_WIDTH:
                right_points.append(self.BEV2TEMPLATE_LOOKUPTBL[x][y])

            if not template is None:
                cv2.polylines(template, [np.array(right_points)], False, (0, 255, 0), 4)

    def left_line_detect(self, img, template):
        X, Y = [], []
        for i in range(self.N_WINDOWS):
            left_window = img[i*(self.window_height):(i+1)*self.window_height, :self.window_width]

            moments = cv2.moments(left_window)
            try:
                cX, cY = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
                X.append(cX)
                Y.append(cY + i * (self.window_height))
                cv2.circle(self.BEV_color, (cX, cY + i * (self.window_height)), 2, (0, 150, 0), -1)
            except:
                pass

        if len(X) > 1:
            # self.lr.fit(np.array(X).reshape(-1, 1), np.array(Y))
            try:
                z = np.polyfit(X, Y, 1)
            except:
                return
            p = np.poly1d(z)
            print(p)

            if abs(p.c[0]) < 20:
                return

            left_points = []
            for i in range(self.BEV_WIDTH//2):
                # x, y = i, int(self.lr.predict(np.array(i).reshape(-1, 1)))
                x, y = i, int(p(i))
                if not (0 < y < self.BEV_HEIGHT):
                    continue

                left_points.append(self.BEV2TEMPLATE_LOOKUPTBL[x][y])
                cv2.circle(self.BEV_color, (x, y), 5, (0, 255, 255), -1)

            x, y = int((self.BEV_HEIGHT - p.c[1]) / p.c[0]), self.BEV_HEIGHT - 1
            print(x, y)
            if 0 <= x < self.BEV_WIDTH // 2:
                left_points.append(self.BEV2TEMPLATE_LOOKUPTBL[x][y])

            if not template is None:
                cv2.polylines(template, [np.array(left_points)], False, (0, 255, 0), 4)


    def __call__(self, gray, hsv, template=None):
        self.BEV_color = np.zeros_like(self.BEV_color)

        edges = cv2.GaussianBlur(gray, (15, 15), sigmaX=3, sigmaY=3)
        edges = cv2.Canny(edges, 50, 70)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel=self.closing_kernel, iterations=3)

        # center line (yellow line)
        center_line_mask = cv2.inRange(hsv, self.low_yellow, self.upper_yellow)

        # white line image
        ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # combine yellow, white line image
        # out = cv2.addWeighted(thresh, 0.5, center_line_mask, 0.5, 0)
        out = cv2.add(thresh, center_line_mask)
        out = cv2.add(out, edges)


        # BEV image (256, 1024) (W, H)
        out = cv2.warpPerspective(out, self.M, (self.BEV_WIDTH, self.BEV_HEIGHT))
        out = cv2.normalize(out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

        cv2.imshow("lane middle result", out)


        self.right_line_detect(out, template)
        self.left_line_detect(out, template)

    def show_BEV(self):
        cv2.imshow("BEV", self.BEV_color)