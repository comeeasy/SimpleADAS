import cv2

class ObjectDetectionHaar:

    def __init__(self) -> None:
        self.haar_xml = "/Users/joono/Desktop/joono/ComputerVisionADASProject/detections/cars.xml"
        self.car_cascade = cv2.CascadeClassifier(self.haar_xml)

    def __call__(self, gray, template):

        cars = self.car_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x,y,w,h) in cars:
            cv2.rectangle(template, (x,y),(x+w,y+h),(0,0,255),2)

        return cars        

        