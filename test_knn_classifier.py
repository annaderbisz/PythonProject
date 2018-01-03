import unittest
from knn_classifier import classify
from model_provider import get_data_from_folder
import numpy as np

class knnClassifier(unittest.TestCase):
    def testKNNClassifier(self):
        dataModel = get_data_from_folder("C:/Users/Anna/PycharmProjects/Projekt/train2")
        model = classify(dataModel.trainFeat, dataModel.trainLabels)
        assert np.array_equal(model.classes_, ['cat', 'dog']), "Błąd klasyfikatora"

if __name__ == "__main__":
    unittest.main()