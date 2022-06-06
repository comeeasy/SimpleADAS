class LaneDetector:
    def __init__(self, model):
        self.model = model

    def __call__(self, img):
