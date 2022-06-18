
import cv2

from detections.LaneDetector import LaneDetector, SpeedDetector
from detections.ObjectDetector import ObjectDetectionHaar
from myWarnings.Warning import Warning

import time
import numpy as np

def main():
    video_name = "/Users/joono/Desktop/joono/ComputerVisionADASProject/videos/highway_D6_Trim.mp4"
    fps = 30

    cap = cv2.VideoCapture(video_name)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w, h = 1920, 1080

    laneDetector = LaneDetector(video_name)
    objectDetector = ObjectDetectionHaar()
    warner = Warning(lane_warn_threshold=0.5, video_name=video_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tt = time.perf_counter()
        
        template = frame[round(h*(1/3)):, :, :]
        template = cv2.resize(template, (256, 128))
        gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)

        l_x, r_x, speed, acceleration = laneDetector(gray, hsv, template)
        detections = objectDetector(gray, template)

        warner.is_lane_departure(l_x, r_x, template)
        warner.is_collision(detections, template)
        warner.is_rapid_stop_or_start(acceleration, template)

        tt2 = time.perf_counter()
        fps = 1 / (tt2-tt)

        cv2.putText(template, f"{speed:.2f}km", (100, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 2)
        cv2.putText(template, f"FPS:{fps:.2f}", (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 2)    
        cv2.imshow(f"ADAS", template)

        k = cv2.waitKey(30)
        if 27 == k:
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
