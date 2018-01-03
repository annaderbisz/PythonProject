import unittest
from classifier_chooser import  chooseClassifier

class kNN_classifierTest(unittest.TestCase):
    def testKNNClassifier(self):
        acc = chooseClassifier("C:/Users/Anna/PycharmProjects/Projekt/train2")
        assert acc>0, "Błąd klasyfikacji"

if __name__ == "__main__":
    unittest.main()