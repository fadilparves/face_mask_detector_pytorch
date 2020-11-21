from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

print("[INFO] Loading model")
net = cv2.dnn.readNetFromCaffe('./checkpoints/deploy.prototxt.txt', './checkpoints/res10_300x300_ssd_iter_140000.caffemodel')

print("[INFO] Starting stream")
vs = VideoStream(src=0).start()
time.sleep(2)