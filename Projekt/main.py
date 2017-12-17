import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

class MainWindow(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.traning_button = tk.Button(self, text="Create trening window",
                                command=self.create_trening_window)
        self.traning_button.pack(side="top")
        # self.testing_button = tk.Button(self, text="Create testing window",
        #                         command=self.create_testing_window)
        # self.testing_button.pack(side="top")
        self.image = np.array([])

    def create_trening_window(self):
        t = tk.Toplevel(self)
        t.wm_title("Trening pannel")
        chooseTreiningDataButton = tk.Button(t, text="Choose training data", command=self.chooseTrainingData)
        chooseTreiningDataButton.pack(side="top", fill="both", expand=True, padx=100, pady=100)
        checkTreiningDataButton = tk.Button(t, text="Check data", command=self.checkTreiningData)
        checkTreiningDataButton.pack(side="top", fill="both", expand=True, padx=100, pady=100)
        trainButton = tk.Button(t, text="Train", command=self.train)
        trainButton.pack(side="top", fill="both", expand=True, padx=100, pady=100)

    def train(self):
        print('Not implemented')

    def chooseTrainingData(self):
        print('Not implemented')

    def checkTreiningData(self):
        print('Not implemented')

if __name__ == "__main__":
    root = tk.Tk()
    main = MainWindow(root)
    main.pack(side="top", fill="both", expand=True)
    root.mainloop()