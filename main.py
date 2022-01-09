##################################
# IMPORT LIBRARIES
##################################
import os, sys
import cv2
import numpy as np
import pickle
import argparse
from helpfuncs import *


##################################
# SET UP PARSER
##################################
parser = argparse.ArgumentParser()
parser.add_argument('--capture', type=bool, default=False,
                    help="Save face detection or not")
parser.add_argument('--name', type=str, default=None,
                    help="Label name to save")
args = parser.parse_args()


##################################
# DEFAULT DIR, COLORS & CONSTANTS
##################################
DATA_DIR = "model_train/NEW_DATA"
CLASSIFIER = "model_train/knn.pkl"
LABELS = "model_train/labels.pkl"

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

IMG_WIDTH, IMG_HEIGHT = 416, 416
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


##################################
# MAIN PROCESS
##################################

# Load YOLO model
net = load_yolo_model()

# Load pre-trained KNN model & labels
knn_file = open(CLASSIFIER, "rb")
model = pickle.load(knn_file)

labels_file = open(LABELS, "rb")
labels = pickle.load(labels_file)

def main_process():
    window_name = "Face Detection using YOLOv3"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)

        # Foward the frame to YOLO
        outs = yolo_forward(frame, net)
        # Find bounding boxes after NMS applied
        boxes = find_bounding_boxes(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        # Run capture mode
        if args.capture == True:
            crop_bounding_boxes(frame, args.name, boxes)

        # Show results if there are faces
        if len(boxes) != 0:
            show_results(frame, boxes, model, labels)
        
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print("[!] --> Interrupted by user")
            break

    # When done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_process()
