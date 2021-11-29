# IMPORT LIBRARIES
import os
import cv2
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score

# DATA DIR
DATA_DIR = "model_train/UPDATE_CROP_DATASET"

# CONSTANTS
IMG_HEIGHT, IMG_WIDTH = 300, 300

# DEFINE TRAINING PROCESS
def training():
    # Load - Preprocess - Label
    data_dir = Path(DATA_DIR)
    all_paths = list(data_dir.glob('*/*'))
    all_labels = list(map(lambda x: str(x).split('/')[-2], all_paths))
    
    label_dict = {}
    for i, name in enumerate(set(all_labels)):
        label_dict[i] = name
    labels = label_binarize(all_labels, classes=list(set(all_labels)))

    def image_to_feature_vector(path, size=(IMG_HEIGHT,IMG_WIDTH)):
        path = str(path)
        image = cv2.imread(path)
        image = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
        image = image/255.
        image = image.flatten()

        return image

    # Apply to the whole dataset
    images = np.array(list(map(image_to_feature_vector, all_paths)))

    # Training model
    knn = KNC(n_neighbors=2)
    knn.fit(images,labels)
    y_pred = knn.predict(images)
    print(accuracy_score(labels, y_pred))

    # Save model
    knn_file = open("knn.pkl", "wb")
    pickle.dump(knn, knn_file)
    knn_file.close()

    # Save label
    label_file = open("labels.pkl", "wb")
    pickle.dump(label_dict, label_file)
    label_file.close()

if __name__ == '__main__':
    training()