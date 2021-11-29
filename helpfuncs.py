##################################
# IMPORT LIBRARIES
##################################
import os
import sys
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import optimizers

# DEFAULT COLORS
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

# Constants
IMG_WIDTH, IMG_HEIGHT = 416, 416
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

##################################
# DEFINE HELP FUNCTIONS TO USE
##################################

def load_yolo_model():
    '''
    Load the pre-trained YOLO model
    '''
    MODEL = 'yolo/yolov3-face.cfg'
    WEIGHT = 'yolo/yolov3-wider_16000.weights'

    net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net



def yolo_forward(frame, net):
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



def crop_bounding_boxes(frame, name, boxes):
    DATA_DIR = "model_train/NEW_DATA"
    dir = os.path.join(DATA_DIR, name)

    if not os.path.exists(dir):
        os.mkdir(dir)

    for box, confidence in boxes:
        try:
        # Extract position data
            topleft_x = box[0]
            topleft_y = box[1]
            width = box[2]
            height = box[3]

            # Crop frame
            crop_img = frame[(topleft_y-15):(topleft_y+height+15),(topleft_x-10):(topleft_x+width+10)]
            now = datetime.now().strftime("%H%M%S_%f")
            img_name = os.path.join(dir,f'{now}.jpg')
            cv2.imwrite(img_name, crop_img) # Save cropped image
        except:
            pass



def load_compile_model(path):
    pretrained_model = load_model(path)
    pretrained_model.compile(optimizer="adam", 
                            loss = "categorical_crossentropy", 
                            metrics=["accuracy"])
    return pretrained_model




def show_results(frame, boxes, model, label_dict):
    for box, confidence in boxes:
        # Extract position data
        topleft_x, topleft_y, width, height = box

        bottomright_x = topleft_x + width
        bottomright_y = topleft_y + height

        # Crop frame & predict
        try:
            crop_img = frame[(topleft_y-15):(topleft_y+height+15),(topleft_x-10):(topleft_x+width+10)]
            crop_img = cv2.resize(crop_img, (300,300), interpolation = cv2.INTER_AREA)
            crop_img = crop_img/255.
            crop_img = crop_img.flatten()

            prediction = model.predict([crop_img])[0]
            name = label_dict[prediction.argmax()]
        except:
            name = "Detecting..."
            print("Out of frame")

        # Draw bouding box with the above measurements & display text
        cv2.rectangle(frame, (topleft_x,topleft_y), (bottomright_x,bottomright_y), COLOR_GREEN, 2)
        text = f'{name}_{(confidence*100):.2f}%'
        cv2.putText(frame, text, (topleft_x, topleft_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 2)

    # Display text about number of detected faces on topleft corner
    text_total = f'Number of faces detected: {len(boxes)}'
    print(text_total)
    cv2.putText(frame, text_total, (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)



def show_results_resnet(frame, boxes, model, label_dict):
    for box, confidence in boxes:
        # Extract position data
        topleft_x, topleft_y, width, height = box

        bottomright_x = topleft_x + width
        bottomright_y = topleft_y + height

        # Crop frame & predict
        try:
            crop_img = frame[(topleft_y-15):(topleft_y+height+15),(topleft_x-10):(topleft_x+width+10)]
            crop_img = cv2.resize(crop_img, (224,224), interpolation = cv2.INTER_AREA)
            crop_img_array = np.expand_dims(crop_img, axis=0)

            prediction = model.predict([crop_img_array])
            a = np.argmax(prediction, axis=1)
            name = label_dict[int(a)]
            # name = label_dict[int(prediction.argmax())]
        except:
            name = "Detecting..."
            print("Out of frame")

        # Draw bouding box with the above measurements & display text
        cv2.rectangle(frame, (topleft_x,topleft_y), (bottomright_x,bottomright_y), COLOR_GREEN, 2)
        text = f'{name}_{(confidence*100):.2f}%'
        cv2.putText(frame, text, (topleft_x, topleft_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 2)

    # Display text about number of detected faces on topleft corner
    text_total = f'Number of faces detected: {len(boxes)}'
    print(text_total)
    cv2.putText(frame, text_total, (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)