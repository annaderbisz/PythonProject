import os
import cv2
from sklearn.cluster import KMeans
from model_provider import get_features
from imutils import paths

def classify(features, clusers_number):
    classifier = KMeans(n_clusters=clusers_number)
    classifier.fit(features)
    classes = classifier.predict(features)
    return classes

def save_images(imagePaths, classes):
    root_directory = "KMeans"
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        className = classes[i]
        class_directory = "{}\{}".format(root_directory, className)
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)
        image_name = imagePath.split(os.path.sep)[-1].split(".")[1]
        image_path_to_save = "{}\{}.{}".format(class_directory,image_name, "jpg")
        cv2.imwrite(image_path_to_save, image)

def kmeans_algoritm(trainfolderpath, clusters_number):
    features = get_features(trainfolderpath)
    classes = classify(features, clusters_number)
    imagePaths = list(paths.list_images(trainfolderpath))
    save_images(imagePaths, classes)