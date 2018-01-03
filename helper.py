import pickle

def saveModel(model):
    f = open("model.cpickle", "wb")
    f.write(pickle.dumps(model))
    f.close()

def getModel():
    return pickle.loads(open("model.cpickle", "rb").read())

def getAccuracy(model, testFeat, testLabels):
    return model.score(testFeat, testLabels)
