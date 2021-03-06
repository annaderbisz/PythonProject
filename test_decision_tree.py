import unittest
from decision_tree_classifier import classify
from model_provider import get_data_from_folder

class decisionTreeClassifier(unittest.TestCase):
    def testDecisionTreeClassifier(self):
        dataModel = get_data_from_folder("C:/Users/Anna/PycharmProjects/Projekt/train2")
        model = classify(dataModel.trainFeat, dataModel.trainLabels)
        assert model.tree_.node_count > 0, "Błąd klasyfikatora"

if __name__ == "__main__":
    unittest.main()