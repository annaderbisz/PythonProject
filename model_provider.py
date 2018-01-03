from imutils import paths as p
import os
import numpy as np
import cv2
from histogram import extract_color_histogram
from sklearn.model_selection import train_test_split
from data_model import  DataModel

def get_features(trainfolderpath):
    imagePaths = list(p.list_images(trainfolderpath))
    features = []

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        hist = extract_color_histogram(image)
        features.append(hist)
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    return np.array(features)

def get_data_from_folder(trainfolderpath):
    imagePaths = list(p.list_images(trainfolderpath))
    features = []
    labels = []

    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = extract_color_histogram(image)
        features.append(hist)
        labels.append(label)

        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    # zmiana listy na tablicę
    features = np.array(features)
    labels = np.array(labels)

    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        features, labels, test_size=0.25, random_state=42)

    dataModel = DataModel()
    dataModel.setTrainFeat(trainFeat)
    dataModel.setTrainLabels(trainLabels)
    dataModel.setTestFeat(testFeat)
    dataModel.setTestLabels(testLabels)

    return dataModel
