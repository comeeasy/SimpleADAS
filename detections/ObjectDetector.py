from tkinter import W
import jetson.inference
import jetson.utils

import argparse
import sys
import cv2

class ObjectDetector:

    def __init__(self, network="ssd-mobilenet-v2", threshold=0.5, overlay="") -> None:
        self.net = jetson.inference.detectNet(network, sys.argv, threshold)
        self.overlay = overlay

    def __call__(self, cudaImg, template):
        detections = self.net.Detect(cudaImg, overlay=self.overlay)

        print("detected {:d} objects in image".format(len(detections)))

        for detection in detections:
            x, y = int(detection.Center[0]), int(detection.Center[1])
            w_half, h_half = int(detection.Width / 2), int(detection.Height / 2)

            pts1 = x - w_half, y - h_half
            pts2 = x + w_half, y + h_half

            cv2.rectangle(template, pts1, pts2, (255, 255, 0), 2)

        return detections


class ObjectDetectionHaar:

    def __init__(self) -> None:
        self.haar_xml = "/home/r320/ComputerVisionADASProject/detections/cars.xml"
        self.car_cascade = cv2.CascadeClassifier(self.haar_xml)

    def __call__(self, gray, template):

        cars = self.car_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x,y,w,h) in cars:
            cv2.rectangle(template, (x,y),(x+w,y+h),(0,0,255),2)

        return cars        

        