import unittest
from KMeans_classifier import classify
from model_provider import get_features
import numpy as np
class knnClassifier(unittest.TestCase):
    def testKNNClassifier(self):
        features = get_features("C:/Users/Anna/PycharmProjects/Projekt/train2")
        classes = classify(features, 2)
        assert len(np.unique(classes)) == 2, "Błąd klasyfikatora"

if __name__ == "__main__":
    unittest.main()