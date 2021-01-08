#####################################################################################################################
#   CS 6375 - Assignment 1, Linear Regression using Gradient Descent
#   This is a simple starter code in Python 3.6 for linear regression using the notation shown in class.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a file. Of course, you will need to include the file to make sure the code runs.
#         - you can assume the last column will the label column
#   test - test dataset - can be a link to a URL or a file. Of course, you will need to include the file to make sure the code runs.
#         - you can assume the last column will the label column
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import csv
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt




class LinearRegression:
    def __init__(self, train):
        np.random.seed(1)
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        df = pd.read_csv(train)
        df.insert(0, 'X0', 1)
        self.nrows, self.ncols = df.shape[0], df.shape[1]
        self.X =  df.iloc[:, 0:(self.ncols -1)].values.reshape(self.nrows, self.ncols-1)
        self.y = df.iloc[:, (self.ncols-1)].values.reshape(self.nrows, 1)
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)

    # TODO: Perform pre-processing for your dataset. It may include doing the following:
    #   - getting rid of null values
    #   - converting categorical to numerical values
    #   - scaling and standardizing attributes
    #   - anything else that you think could increase model performance
    # Below is the pre-process function
    def preProcess(self):
       self.df.dropna()
        
        #dataset = read_csv('auto-mpg.csv', header=None)
        #print(dataset.head(10))
        #dataset.isnull().sum()
        #modifiedDataset = dataset.fillna("")
        #modifiedDataset.isnull().sum()
        #modifiedDataset.to_csv("modified-mpg.csv", index=False)
        #print("check one")

    # Below is the training function
    def train(self, epochs = 10, learning_rate = 0.05):
        # code epochs and learning rate
        # Perform Gradient Descent
        for i in range(epochs):
            # Make prediction with current weights H=XW
            h = np.dot(self.X, self.W)
            # Find error E=H-Y
            error = h - self.y
            self.W = self.W - (1 / self.nrows) * learning_rate * np.dot(self.X.T, error)
            

        return self.W, error

    # predict on test dataset
    def predict(self, test):
        testDF = pd.read_csv(test)
        testDF.insert(0, "X0", 1)
        nrows, ncols = testDF.shape[0], testDF.shape[1]
        testX = testDF.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        testY = testDF.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        pred = np.dot(testX, self.W)
        error = pred - testY
        mse = (1/(2*nrows)) * np.dot(error.T, error)
        return mse


if __name__ == "__main__":
    print("main")
    model = LinearRegression("train.csv")
    model.preProcess()
    W, e = model.train()
    mse = model.predict("test.csv")
    print(mse)



