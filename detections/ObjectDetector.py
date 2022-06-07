import jetson.inference
import jetson.utils

import argparse
import sys


class ObjectDetector:
    
    def __init__(self, network="ssd-mobilenet-v2", threshold=0.5, overlay="box,conf") -> None:
        self.net = jetson.inference.detectNet(network, sys.argv, threshold)
        self.overlay = overlay

    def __call__(self, img):
        detections = self.net.Detect(img, overlay=self.overlay)

        # print("detected {:d} objects in image".format(len(detections)))

        # for detection in detections:
        #     print(detection)

        # output.Render(img)