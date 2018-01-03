from sklearn.tree import DecisionTreeClassifier
from model_provider import get_data_from_folder
from helper import getAccuracy, saveModel

def classify(trainFeat, trainLabels):
    model = DecisionTreeClassifier()
    model.fit(trainFeat, trainLabels)
    return model

def decision_tree_maker_algoritm(trainfolderpath):
    dataModel = get_data_from_folder(trainfolderpath)
    decisionTreeModel = classify(dataModel.trainFeat, dataModel.trainLabels)
    accDecitionTree = getAccuracy(decisionTreeModel, dataModel.testFeat, dataModel.testLabels)
    print("DecisionTreeClassifier accuracy: {:.2f}%".format(accDecitionTree * 100))
    saveModel(decisionTreeModel)
