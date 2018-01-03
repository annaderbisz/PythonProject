from decision_tree_classifier import  classify as dtc
from knn_classifier import classify as kc
from model_provider import get_data_from_folder
from helper import getAccuracy, saveModel

def chooseClassifier(trainfolderpath):
    dataModel = get_data_from_folder(trainfolderpath)
    decisionTreeModel = dtc(dataModel.trainFeat, dataModel.trainLabels)
    knnModel = kc(dataModel.trainFeat, dataModel.trainLabels)
    accDecitionTree = getAccuracy(decisionTreeModel, dataModel.testFeat, dataModel.testLabels)
    accKnnModel = getAccuracy(knnModel, dataModel.testFeat, dataModel.testLabels)
    print("DecisionTreeClassifier accuracy: {:.2f}%".format(accDecitionTree * 100))
    print("KNNClassifier accuracy: {:.2f}%".format(accKnnModel * 100))
    acc = accDecitionTree
    if accKnnModel > acc:
        acc = accKnnModel
        print("Knn is saved")
        saveModel(knnModel)
    else:
        print("Decition tree is saved")
        saveModel(decisionTreeModel)

    return acc