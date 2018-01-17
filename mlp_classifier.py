from sklearn.neural_network import MLPClassifier
from model_provider import  get_data_from_folder
from helper import  getAccuracy, saveModel

def classify(trainFeat, trainLabels):
    model = MLPClassifier()
    model.fit(trainFeat, trainLabels)
    return model

def mlp_maker_algoritm(trainfolderpath):
    dataModel = get_data_from_folder(trainfolderpath)
    mlpModel = classify(dataModel.trainFeat, dataModel.trainLabels)
    accmlpModel = getAccuracy(mlpModel, dataModel.testFeat, dataModel.testLabels)
    print("MLP Classifier accuracy: {:.2f}%".format(accmlpModel * 100))
    saveModel(mlpModel)

