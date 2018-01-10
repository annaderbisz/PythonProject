from model_provider import  get_data_from_folder
from helper import  getAccuracy, saveModel
from sklearn.ensemble import RandomForestClassifier

def classify(trainFeat, trainLabels):
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(trainFeat, trainLabels)
    return model

def random_forest_maker_algoritm(trainfolderpath):
    dataModel = get_data_from_folder(trainfolderpath)
    random_forestModel = classify(dataModel.trainFeat, dataModel.trainLabels)
    accModel = getAccuracy(random_forestModel, dataModel.testFeat, dataModel.testLabels)
    print("Random Forest Classifier accuracy: {:.2f}%".format(accModel * 100))
    saveModel(random_forestModel)

