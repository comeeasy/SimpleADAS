import cv2
import numpy as np
import random
from sklearn import linear_model
from tqdm import tqdm



WARPAFFINE_WIDTH = 512
WARPAFFINE_HEIGHT = 256

# (x, y)순임. (y, x)순 아님
# LU -> LB -> RU -> RB
# 256 x 512 (H, W) image 기준
LANE_ROI_POINTS = [
    [WARPAFFINE_WIDTH//2 - 50, 50],
    [50, WARPAFFINE_HEIGHT],
    [WARPAFFINE_WIDTH//2 + 50, 50],
    [WARPAFFINE_WIDTH - 50, WARPAFFINE_HEIGHT],
]
# rectangle
BEV_WIDTH = 256
BEV_HEIGHT = 1024
BEV_POINTS = [
    [0, 0],
    [0, BEV_HEIGHT],
    [BEV_WIDTH, 0],
    [BEV_WIDTH, BEV_HEIGHT]
]

M = cv2.getPerspectiveTransform(
    np.array(LANE_ROI_POINTS, dtype=np.float32),
    np.array(BEV_POINTS, dtype=np.float32)
)

BEV2TEMPLATE_LOOKUPTBL = []

print(M)

lr = linear_model.RANSACRegressor()

# HSV yello
low_yellow = np.array([20, 100, 100])
upper_yellow = np.array([42, 255, 255])

def main():
    # create lookup table
    inv_M = np.linalg.inv(M)

    for x in tqdm(range(BEV_WIDTH)):
        BEV2TEMPLATE_LOOKUPTBL.append([])
        for y in range(BEV_HEIGHT):
            bef_coor = np.array([x, y, 1])
            aft_coor = np.matmul(inv_M, np.transpose(bef_coor))

            BEV2TEMPLATE_LOOKUPTBL[x].append(np.array([int(aft_coor[0] * (1/aft_coor[2])), int(aft_coor[1] * (1/aft_coor[2]))]))


    cap = cv2.VideoCapture("/Users/joono/Desktop/joono/ComputerVisionADAS/videos/highway_D5_Trim.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(w, h)

    # BGR image
    template = np.zeros((round(h*(2/3)), w, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        template = frame[round(h*(1/3)):, :, :]
        BEV_color = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3))

        cv2.imshow("half image", template)

        template = cv2.resize(template, (512, 256))
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        center_line_mask = cv2.inRange(hsv, low_yellow, upper_yellow)

        cv2.imshow("center line", center_line_mask)

        ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        out = cv2.addWeighted(thresh, 0.5, center_line_mask, 0.5, 0)

        # BEV
        out = cv2.warpPerspective(out, M, (BEV_WIDTH, BEV_HEIGHT))
        result = cv2.normalize(out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

        cv2.imshow("thresh hsv", result)

        # find centerPoints in sliding window
        N_WINDOWS = 128
        window_height = BEV_HEIGHT // N_WINDOWS
        window_width  = 60

        X, Y = [], []
        for i in range(N_WINDOWS):
            cv2.rectangle(out, (BEV_WIDTH-window_width, i * (window_height)), (BEV_WIDTH, (i + 1) * window_height), (128, 128, 128), 2)
            right_window = result[i*(window_height):(i+1)*window_height, BEV_WIDTH-window_width:]

            moments = cv2.moments(right_window)
            try:
                cX, cY = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
                X.append(cX + BEV_WIDTH - window_width)
                Y.append(cY + i * (window_height))
                cv2.circle(BEV_color, (cX + BEV_WIDTH - window_width, cY + i * (window_height)), 5, (0, 150, 0), -1)
            except:
                cX, cY = -1, -1


        if len(X) > 1:
            lr.fit(np.array(X).reshape(-1, 1), np.array(Y))

        right_points = []
        for i in range(BEV_WIDTH-window_width, BEV_WIDTH):
            x, y = i, int(lr.predict(np.array(i).reshape(-1, 1)))
            if not (0 < y < BEV_HEIGHT):
                continue

            right_points.append(BEV2TEMPLATE_LOOKUPTBL[x][y])
            cv2.circle(BEV_color, (x, y), 5, (0, 255, 255), -1)



        cv2.polylines(template, [np.array(right_points)], False, (0, 255, 0), 4)

        X, Y = [], []
        for i in range(N_WINDOWS):
            # (x, y)
            cv2.rectangle(out, (0, i*(window_height)), (window_width, (i+1)*window_height), (128, 128, 128), 2)

            left_window = result[i*(window_height):(i+1)*window_height, :window_width]
            moments = cv2.moments(left_window)
            try:
                cX, cY = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
                X.append(cX)
                Y.append(cY + i*(window_height))
                cv2.circle(BEV_color, (cX, cY + i * (window_height)), 5, (255, 0, 0), -1)
            except:
                cX, cY = -1, -1



        if len(X) > 1:
            lr.fit(np.array(X).reshape(-1, 1), np.array(Y))

        left_points = []
        for i in range(0, 100):
            x, y = i, int(lr.predict(np.array(i).reshape(-1, 1)))
            if not (0 < y < BEV_HEIGHT):
                continue

            left_points.append(BEV2TEMPLATE_LOOKUPTBL[x][y])
            cv2.circle(BEV_color, (x, y), 5, (0, 255, 255), -1)

        cv2.polylines(template, [np.array(left_points)], False, (0, 255, 0), 4)

        cv2.circle(template, BEV2TEMPLATE_LOOKUPTBL[10][512], 5, (255, 0, 0), -1)
        cv2.circle(template, BEV2TEMPLATE_LOOKUPTBL[10][1023], 5, (0, 255, 0), -1)
        cv2.circle(template, BEV2TEMPLATE_LOOKUPTBL[246][512], 5, (0, 0, 255), -1)
        cv2.circle(template, BEV2TEMPLATE_LOOKUPTBL[246][1023], 5, (0, 255, 255), -1)

        cv2.imshow("result", out)
        cv2.imshow("template", template)
        cv2.imshow("BEV color", BEV_color)

        k = cv2.waitKey(0)
        if 27 == k:
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
