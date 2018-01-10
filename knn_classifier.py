from sklearn.neighbors import KNeighborsClassifier
from model_provider import  get_data_from_folder
from helper import  getAccuracy, saveModel

def classify(trainFeat, trainLabels):
    model = KNeighborsClassifier(n_neighbors=1,
                                 n_jobs=1)
    model.fit(trainFeat, trainLabels)
    return model

def knn_maker_algoritm(trainfolderpath):
    dataModel = get_data_from_folder(trainfolderpath)
    knnModel = classify(dataModel.trainFeat, dataModel.trainLabels)
    accKnnModel = getAccuracy(knnModel, dataModel.testFeat, dataModel.testLabels)
    print("KNNClassifier accuracy: {:.2f}%".format(accKnnModel * 100))
    saveModel(knnModel)


    #svm
    #random forest
    #regresja logistyczna
    #perceptron
    #mlp
    # wszytsko na jednym oknie
    # wyswietlać pierwsze pliki

