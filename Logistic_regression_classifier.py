from model_provider import  get_data_from_folder
from helper import  getAccuracy, saveModel
from sklearn.linear_model import LogisticRegression

def classify(trainFeat, trainLabels):
    model = LogisticRegression()
    model.fit(trainFeat, trainLabels)
    return model

def logistic_regression_maker_algoritm(trainfolderpath):
    dataModel = get_data_from_folder(trainfolderpath)
    model = classify(dataModel.trainFeat, dataModel.trainLabels)
    acc = getAccuracy(model, dataModel.testFeat, dataModel.testLabels)
    print("Logistic Regression accuracy: {:.2f}%".format(acc * 100))
    saveModel(model)

