# IMPORT LIBRARIES
import os
import sys
import cv2
import numpy as np
from datetime import datetime

# DEFAULT COLORS
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

# Constants
IMG_WIDTH, IMG_HEIGHT = 416, 416

##################################
# DEFINE FUNCTIONS TO USE

def load_yolo_model():
    '''
    Load the pre-trained YOLO model
    '''
    MODEL = '/Users/dongth/Documents/github/wp_face_detection/yolo/yolov3-face.cfg'
    WEIGHT = '/Users/dongth/Documents/github/wp_face_detection/yolo/yolov3-wider_16000.weights'

    net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net



def yolo_forward(frame):
    '''
    Pass each captured frame into the pretrained yolo model
    '''
    # Makeing blob object from the original image
    blob = cv2.dnn.blobFromImage(frame,
                                1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0,0,0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run prediction
    outs = net.forward(output_layers)
    return outs



def find_bounding_boxes(frame, outs, conf_thresh, nms_thresh):
    '''
    Scan through all the bounding boxes output from the network and keep only
    the ones with high confidence scores. Assign the box's class label as the
    class with the highest score.'''
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    confidences = []
    boxes = []
    final_boxes = []

    # Each frame produces 3 outs correspoding to 3 output layers
    for out in outs:
        # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)

                # Find the top left point of the bounding box
                topleft_x = int(center_x - width//2)
                topleft_y = int(center_y - height//2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        confidence = confidences[i]
        final_boxes.append((box, confidence))

    return final_boxes



def crop_bounding_boxes(frame, boxes, dir):    
    for box, confidence in boxes:
        # Extract position data
        topleft_x = box[0]
        topleft_y = box[1]
        width = box[2]
        height = box[3]
        bottomright_x = topleft_x + width
        bottomright_y = topleft_y + height

        # Crop frame
        crop_img = frame[(topleft_y-15):(topleft_y+height+15),(topleft_x-10):(topleft_x+width+10)]
        now = datetime.now().strftime("%H%M%S_%f")
        img_name = os.path.join(dir,f'{now}.jpg')
        cv2.imwrite(img_name, crop_img) # Save cropped image

##################################
# CROPPING PROCESS

DATA_DIR = "model_train/DATASET"

# Directories for each class
dong_dir = 'model_train/DATASET/Dong'
vy_dir = 'model_train/DATASET/Vy'
thao_dir = 'model_train/DATASET/Thao'
lili_dir = 'model_train/DATASET/Lili'
hiep_dir = 'model_train/DATASET/Hiep'
all_dir = [dong_dir, vy_dir, thao_dir, lili_dir, hiep_dir]

# Creating directory list of images from train folder of each class
dong_fnames = os.listdir(dong_dir)
vy_fnames = os.listdir(vy_dir)
thao_fnames = os.listdir(thao_dir)
lili_fnames = os.listdir(lili_dir)
hiep_fnames = os.listdir(hiep_dir)
all_fnames = [dong_fnames,vy_fnames,thao_fnames,lili_fnames,hiep_fnames]

# Constants
conf_thresh = 0.5
nms_thresh = 0.4

# LOOP over each image in each directory
for i, fnames in enumerate(all_fnames):
    for c, fname in enumerate(fnames):
        img_path = os.path.join(all_dir[i], fname)
        label = img_path.split("/")[-2]
        dir = os.path.join(DATA_DIR, label)
        if not os.path.exists(dir):
            os.mkdir(dir)
        img = cv2.imread(img_path)

        # Find the bounding box
        net = load_yolo_model() # Load the yolo model
        outs = yolo_forward(img) # Pass the image to the model
        final_boxes = find_bounding_boxes(img, outs, conf_thresh, nms_thresh) # Bounding boxes after NMS
        crop_bounding_boxes(img, final_boxes, dir) # Crop and save bounding box(es) image with labels
        if (c+1) % 50 == 0:
            print(f"Successfully saved {c+1}")