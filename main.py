import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import numpy as np
from matplotlib import pyplot as plt
import cv2
from histogram import extract_color_histogram
from imutils import paths
from tkinter import messagebox
from KMeans_classifier import kmeans_algoritm
from helper import getModel
from knn_classifier import knn_maker_algoritm
from decision_tree_classifier import decision_tree_maker_algoritm
from SVM_classifier import svm_maker_algoritm
from RandomForest_classifier import random_forest_maker_algoritm
from perceptron_classifier import  perception_maker_algoritm
from Logistic_regression_classifier import logistic_regression_maker_algoritm
from mlp_classifier import  mlp_maker_algoritm
from PIL import Image, ImageTk
from tkinter import Label


class MainWindow(tk.Frame):

    def train(self, isCheckDecisionTree, isCheckedKnn, isCheckedMLP, isCheckedRandForest, isCheckedRegression, isCheckedSVM, isCheckedPerception):
        if isCheckDecisionTree == 1:
            decision_tree_maker_algoritm(self.train_folder)
        if isCheckedKnn == 1:
            knn_maker_algoritm(self.train_folder)
        if isCheckedSVM == 1:
            svm_maker_algoritm(self.train_folder)
        if isCheckedMLP == 1:
            mlp_maker_algoritm(self.train_folder)
        if isCheckedRandForest == 1:
            random_forest_maker_algoritm(self.train_folder)
        if isCheckedRegression == 1:
            logistic_regression_maker_algoritm(self.train_folder)
        if isCheckedPerception == 1:
            perception_maker_algoritm(self.train_folder)

        print("END")
    def kmeans(self):
        kmeans_algoritm(self.train_folder, 2)
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

    def chooseFile(self, label, imageLabel):
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
            image2 = Image.open(
                filename)  # This is the correct location and spelling for my image location
            photo = ImageTk.PhotoImage(image2)
            imageLabel.configure(image=photo)
            imageLabel.image = photo


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

    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        self.image = np.array([])
        self.train_folder = ""

        chooseTrainingDataButton = tk.Button(self, text="Choose training data", command=self.chooseTrainingData, width = 30)
        chooseTrainingDataButton.grid(row=0, column=0)
        checkTrainingDataButton = tk.Button(self, text="Check data", width = 30, command = self.checkTrainingData)
        checkTrainingDataButton.grid(row=0, column=1)

        kMeansButton = tk.Button(self, text="K-Means", command=self.kmeans, width=30)
        kMeansButton.grid(row=0, column=2)

        is_checkedKnn = tk.IntVar()
        trainKnnButton = tk.Checkbutton(self, text="Train using knn classifier", width = 30, variable=is_checkedKnn)
        trainKnnButton.grid(row=2, column=0)
        is_checkedDecisionTree = tk.IntVar()
        trainDecisionTreeButton = tk.Checkbutton(self, text="Train using decision tree classifier", width = 30, variable=is_checkedDecisionTree)
        trainDecisionTreeButton.grid(row=2, column=1)

        is_checkedSVM = tk.IntVar()
        trainSVMButton = tk.Checkbutton(self, text="Train using SVM classifier", width = 30, variable = is_checkedSVM)
        trainSVMButton.grid(row=2, column=2)
        is_checkedRandForest = tk.IntVar()
        trainRandForestButton = tk.Checkbutton(self, text="Train using Random Forest classifier", width = 30, variable = is_checkedRandForest)
        trainRandForestButton.grid(row=4, column=0)

        is_checkedRegression = tk.IntVar()
        trainLogisticRegressionButton = tk.Checkbutton(self, text="Train using Logistic Regression classifier", width = 30, variable = is_checkedRegression)
        trainLogisticRegressionButton.grid(row=4, column=1)

        is_checkedMLP = tk.IntVar()
        trainMLPButton = tk.Checkbutton(self, text="Train using MLP classifier", width = 30, variable = is_checkedMLP)
        trainMLPButton.grid(row=4, column=2)

        is_checkedPerception = tk.IntVar()
        trainPerceptronButton = tk.Checkbutton(self, text="Train using Perception classifier", width = 30, variable = is_checkedPerception)
        trainPerceptronButton.grid(row=5, column=0)

        trainButton = tk.Button(self, text="Train and save the best classifier", command =lambda: self.train(is_checkedDecisionTree.get(), is_checkedKnn.get(), is_checkedMLP.get(), is_checkedRandForest.get(), is_checkedRegression.get(), is_checkedSVM.get(), is_checkedPerception.get()), width = 30)
        trainButton.grid(row=6, column=0)

        image = Image.open(
            "C:/Users/Anna/PycharmProjects/Projekt/train2/cat.3.jpg")  # This is the correct location and spelling for my image location

        photo = ImageTk.PhotoImage(image)
        label = Label(image=photo)
        label.image = photo
        label.pack()

        chooseFileButton = tk.Button(self, text='Choose file', command=lambda: self.chooseFile(w, label), width = 30)
        chooseFileButton.grid(row=9, column=0)
        showHistButton = tk.Button(self, text="Show histogram", command=self.showColorHist, width = 30)
        showHistButton.grid(row=9, column=1)

        w = tk.Label(self, text="Wynik", width = 30)
        w.config(font=("Courier", 44))
        w.grid(row=12, column=0,  columnspan=3, rowspan=3)



if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(width=600, height=600)
    main = MainWindow(root)
    main.pack(side="top", fill="both", expand=True)
    root.mainloop()