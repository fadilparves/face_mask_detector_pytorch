from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from pathlib import Path
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from face_detector import FaceDetector
from trainer import MaskDetectorTrainer

@torch.no_grad()
def tagMask(outputPath=None):

    print("[INFO] Loading model")
    fd_model = FaceDetector(
            prototype='./checkpoints/deploy.prototxt.txt',
            model='./checkpoints/res10_300x300_ssd_iter_140000.caffemodel',
        )

    model = MaskDetectorTrainer()
    model.load_state_dict(torch.load('./checkpoints/weights-v0.ckpt')['state_dict'], strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print("[INFO] Starting stream")
    vs = VideoStream(src=0).start()
    time.sleep(2)

    transformations = Compose([
        ToPILImage(),
        Resize((100,100)),
        ToTensor(),
    ])

    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ['No Mask', 'Mask']
    labelColor = [(10,0,255), (10,255,0)]

    while True:
        frame = vs.read()
        # frame = imutils.resize(frame, width=400)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = fd_model.detect(frame)
        for face in faces:
            print(face)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('main', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    tagMask()