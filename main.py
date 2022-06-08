import jetson.inference
import jetson.utils

import cv2

from detections.LaneDetector import LaneDetector, LaneDetectorAI
from detections.ObjectDetector import ObjectDetector
from myWarnings.Warning import Warning

import time
import numpy as np

def main():
    video_name = "/home/r320/ComputerVisionADASProject/videos/highway_D5_Trim.mp4"
    fps = 30

    capfile = f'filesrc location={video_name} ! qtdemux ! queue \
                            ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx,width=720,height=380 \
                            ! videorate ! video/x-raw,framerate={fps}/1 !queue ! videoconvert ! queue ! video/x-raw, format=BGR \
                            ! appsink'

    capfile2 = f'filesrc location={video_name} ! qtdemux ! queue \
                            ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! queue \
                            ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink'

    # cap = cv2.VideoCapture(capfile2, cv2.CAP_GSTREAMER)

    cap = cv2.VideoCapture(video_name)
    # cap2 = jetson.utils.videoSource(video_name, argv=sys.argv) 
    # output = jetson.utils.videoOutput("", argv=sys.argv)


    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w, h = 1920, 1080


    print(w, h)


    laneDetector = LaneDetector(video_name)
    # laneDetectorai = LaneDetectorAI()
    # objectDetector = ObjectDetector(network="ssd-inception-v2")
    warner = Warning(lane_warn_threshold=0.5, video_name=video_name)

    while True:
        ret, frame = cap.read()
        # img = cap2.Capture()
        if not ret:
            break

        tt = time.perf_counter()

        # frame = jetson.utils.cudaToNumpy(img)
        template = frame[round(h*(1/3)):, :, :]
        
        # cv2.imshow("half image", template)

        template = cv2.resize(template, (256, 128))
        gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)

        l_x, r_x = laneDetector(gray, hsv, template)
        # laneDetector.show_BEV()

        # self.previous_out = out

        # detections = objectDetector(jetson.utils.cudaFromNumpy(template), template)

        # lane_departure = warner.is_lane_departure(l_x, r_x)
        # collision_warn = warner.is_collision(detections)

        # print(lane_departure, collision_warn)

        tt2 = time.perf_counter()
        print(f"FPS {1000 * (tt2-tt):.3f}ms")

        cv2.imshow("template", template)

        k = cv2.waitKey(0)
        if 27 == k:
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
