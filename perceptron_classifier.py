from sklearn.linear_model import Perceptron
from model_provider import get_data_from_folder
from helper import getAccuracy, saveModel

def classify(trainFeat, trainLabels):
    model = Perceptron()
    reshapedTrainFeat = trainFeat
    reshapedTrainLabels = trainLabels
    model.fit(reshapedTrainFeat, reshapedTrainLabels)
    return model

def perception_maker_algoritm(trainfolderpath):
    dataModel = get_data_from_folder(trainfolderpath)
    model = classify(dataModel.trainFeat, dataModel.trainLabels)
    accmodel = getAccuracy(model, dataModel.testFeat, dataModel.testLabels)
    print("Perceptron Classifier accuracy: {:.2f}%".format(accmodel * 100))
    saveModel(model)
