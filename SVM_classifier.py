from model_provider import  get_data_from_folder
from helper import  getAccuracy, saveModel
from sklearn.svm import SVC

def classify(trainFeat, trainLabels):
    model = SVC()
    model.fit(trainFeat, trainLabels)
    return model

def svm_maker_algoritm(trainfolderpath):
    dataModel = get_data_from_folder(trainfolderpath)
    svmModel = classify(dataModel.trainFeat, dataModel.trainLabels)
    accSVMModel = getAccuracy(svmModel, dataModel.testFeat, dataModel.testLabels)
    print("SVMClassifier accuracy: {:.2f}%".format(accSVMModel * 100))
    saveModel(svmModel)

