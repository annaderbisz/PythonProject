import numpy as np

class DataModel:
    def __init__(self):
        self.trainFeat = np.array([])
        self.testFeat = np.array([])
        self.trainLabels = np.array([])
        self.testLabels = np.array([])

    def setTrainFeat(self, value):
        self.trainFeat = value

    def setTestFeat(self, value):
        self.testFeat = value

    def setTrainLabels(self, value):
        self.trainLabels = value

    def setTestLabels(self, value):
        self.testLabels = value