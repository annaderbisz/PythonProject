import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import numpy as np
from matplotlib import pyplot as plt
import cv2
from histogram import extract_color_histogram
from classifier_chooser import chooseClassifier
from imutils import paths
from tkinter import messagebox
from KMeans_classifier import kmeans_algoritm
from helper import getModel
from knn_classifier import knn_maker_algoritm
from decision_tree_classifier import decision_tree_maker_algoritm
from SVM_classifier import svm_maker_algoritm
from RandomForest_classifier import random_forest_maker_algoritm
from Logistic_regression_classifier import logistic_regression_maker_algoritm

class MainWindow(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.traning_button = tk.Button(self, text="Create trening window",
                                        command=self.create_traning_window)
        self.traning_button.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        self.testing_button = tk.Button(self, text="Create testing window",
                                 command=self.create_testing_window)
        self.testing_button.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        self.image = np.array([])
        self.train_folder = ""
    def create_traning_window(self):
        t = tk.Toplevel(self)
        t.wm_title("Traning pannel")
        chooseTrainingDataButton = tk.Button(t, text="Choose training data", command=self.chooseTrainingData)
        chooseTrainingDataButton.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        checkTrainingDataButton = tk.Button(t, text="Check data", command=self.checkTrainingData)
        checkTrainingDataButton.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        trainKnnButton = tk.Button(t, text="Train using knn classifier", command=self.train_knn)
        trainKnnButton.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        trainDecisionTreeButton = tk.Button(t, text="Train using decision tree classifier", command=self.train_decision_tree)
        trainDecisionTreeButton.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        trainSVMButton = tk.Button(t, text="Train using SVM classifier", command=self.train_svm)
        trainSVMButton.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        trainRandForestButton = tk.Button(t, text="Train using Random Forest classifier", command=self.train_randForest)
        trainRandForestButton.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        trainLogisticRegressionButton = tk.Button(t, text="Train using Logistic Regression classifier", command=self.train_logistic_regression)
        trainLogisticRegressionButton.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        trainButton = tk.Button(t, text="Train and save the best classifier", command=self.train)
        trainButton.pack(side="top", fill="both", expand=True, padx=50, pady=50)
        kMeansButton = tk.Button(t, text="K-Means", command=self.kmeans)
        kMeansButton.pack(side="top", fill="both", expand=True, padx=50, pady=50)

    def create_testing_window(self):
        t = tk.Toplevel(self)
        t.wm_title("Testing pannel")
        w = tk.Label(t, text="abc")
        w.pack(side="bottom", fill="both", expand=True, padx=100, pady=100)
        chooseFileButton = tk.Button(t, text='Choose file', command=lambda: self.chooseFile(w))
        chooseFileButton.pack(side="top", fill="both", expand=True, padx=100, pady=100)
        showHistButton = tk.Button(t, text="Show histogram", command=self.showColorHist)
        showHistButton.pack(side="top", fill="both", expand=True, padx=100, pady=100)

    def train(self):
        chooseClassifier(self.train_folder)
        print("END")
    def train_knn(self):
        knn_maker_algoritm(self.train_folder)
        print("END")
    def train_logistic_regression(self):
        logistic_regression_maker_algoritm(self.train_folder)
        print("END")
    def train_decision_tree(self):
        decision_tree_maker_algoritm(self.train_folder)
        print("END")
    def kmeans(self):
        kmeans_algoritm(self.train_folder, 2)
        print("END")
    def train_randForest(self):
        random_forest_maker_algoritm(self.train_folder)
        print("END")
    def train_svm(self):
        svm_maker_algoritm(self.train_folder)
        print("END")
    def chooseTrainingData(self):
        self.train_folder = askdirectory()  # show an "Open" dialog box and return the path to the selected file

    def checkTrainingData(self):
        try:
            imagePaths = list(paths.list_images(self.train_folder))

            if(len(imagePaths) == 0):
                raise Exception("Brak poprawnych danych")
            messagebox.showinfo('Poprawne dane',
                                 'Poprawne dane treningowe. Możesz przystąpić do treningu.')
        except:
            messagebox.showerror('Błędne dane', '"Niepoprawne dane treningowe. Sprawdź rozszerzenie (jpg) i nazwy plików.')

    def chooseFile(self, label):
        filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
        img = cv2.imread(filename)
        model = getModel()
        h = extract_color_histogram(img)
        test = h.reshape(1, -1)
        if(model):
            result = model.predict(test)
            print(result[0])
            label.config(text=result[0])
        if(img.size != 0):
            self.image=img

    def showHistGreyScale(img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()

    def showColorHist(self):
        if (self.image.size != 0):
            color = ('b', 'g', 'r')
            for channel, col in enumerate(color):
                histr = cv2.calcHist([self.image], [0], None, [256], [0, 256], False)
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.title('Histogram for color scale picture')
            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    main = MainWindow(root)
    main.pack(side="top", fill="both", expand=True)
    root.mainloop()