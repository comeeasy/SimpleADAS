import cv2
from detections.LaneDetector import LaneDetector



def main():
    video_name = "/Users/joono/Desktop/joono/ComputerVisionADASProject/videos/highway_D5_Trim.mp4"
    cap = cv2.VideoCapture(video_name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(w, h)

    laneDetector = LaneDetector(video_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        template = frame[round(h*(1/3)):, :, :]
        cv2.imshow("half image", template)

        template = cv2.resize(template, (512, 256))
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

        laneDetector(gray, hsv, template)
        laneDetector.show_BEV()

        cv2.imshow("template", template)

        k = cv2.waitKey(30)
        if 27 == k:
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
