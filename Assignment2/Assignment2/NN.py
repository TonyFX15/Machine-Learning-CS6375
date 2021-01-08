# CS 6375 - Assignment 2, Neural Network 

import numpy as np
import pandas as pd
from random import seed
from random import randrange
from random import random
from random import randint
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

class NN:

     # pre-process the dataset
    def preprocess(self):
        data = pd.read_csv("iris.csv", header = None, names = ['X1','X2','X3','X4','Y'])
        encoder = LabelEncoder()
        code = encoder.fit_transform(data['Y'].values)
        code = np.array([code]).T
        onehot = OneHotEncoder(categories='auto')
        code = onehot.fit_transform(code)
        code = code.toarray()
        data = pd.concat([data,pd.DataFrame(code)],axis=1)
        data = data.drop(['Y'],axis=1)
        data = shuffle(data)
        length=len(data)
        testlen=int(length*0.2)
        test=data[0:testlen]
        train=data[testlen:length]
        pd.DataFrame.to_csv(test,"test.csv")
        pd.DataFrame.to_csv(train,"train.csv")
        data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) 

        return data


    def __init__(self, train, header = True, h1 = 4, h2 = 3):
        np.random.seed(1)
      
        train_dataset = pd.read_csv(train, index_col=0)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -3)].values.reshape(nrows, ncols-3)
        self.y = train_dataset.iloc[:, (ncols-3):(ncols)].values.reshape(nrows, 3)
       
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    
   

    # sigmoid activation function
    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # tanh activation function
    def __activation(self, x, activation="tanh"):
        if activation == "tanh":
            self.__tanh(self, x)
    def __activation_derivative(self, x, activation="tanh"):
        if activation == "tanh":
            self.__tanh_derivative(self, x)
    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    def __tanh_derivative(self, x):
        return 1 - x * x

    # ReLu activation function
    def __activation(self, x, activation="ReLu"):
        if activation == "ReLu":
            self.__ReLu(self, x)
    def __activation_derivative(self, x, activation="ReLu"):
        if activation == "ReLu":
            self.__ReLu_derivative(self, x)
    def __ReLu(self, x):
        return np.maximum(0, x)
    def __ReLu_derivative(self, x):
        return np.where(x <= 0, 0, 1)
    
    # training function
    def train(self, activation, max_iterations = 5000, learning_rate = 0.011):
        for iteration in range(max_iterations):
            if activation == "sigmoid":
                out = self.forward_pass(activation="sigmoid")
                error = 0.5 * np.power((out - self.y), 2)
                self.backward_pass(out, activation="sigmoid")
            elif activation == "tanh":
                out = self.forward_pass(activation="tanh")
                error = 0.5 * np.power((out - self.y), 2)
                self.backward_pass(out, activation="tanh")
            elif activation == "ReLu":
                out = self.forward_pass(activation="ReLu")
                error = 0.5 * np.power((out - self.y), 2)
                self.backward_pass(out, activation="ReLu")
            
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        # dsiplay the results of the error
        print("The final weight vectors from input to output layer: ")
        print(self.w01)
        print(self.w12)
        print(self.w23)
        print("The total error is " + str(np.sum(error)))

    def forward_pass(self, activation):
        # pass the inputs through neural network
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        elif activation == "tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        elif activation == "ReLu":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__ReLu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__ReLu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__ReLu(in3)
        
        return out


    def backward_pass(self, out, activation):
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    
    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "ReLu":
            delta_output = (self.y - out) * (self.__ReLu_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__ReLu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__ReLu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    def compute_input_layer_delta(self, activation):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "ReLu":
            delta_input_layer = np.multiply(self.__ReLu_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # Output the test error 
    def predict(self, test, activation, header = True, h1 = 4, h2 = 3):
        test_dataset = pd.read_csv(test, index_col=0)
        ncols = len(test_dataset.columns)
        nrows = len(test_dataset.index)
        testX = test_dataset.iloc[:, 0:(ncols -3)].values.reshape(nrows, ncols-3)
        testy = test_dataset.iloc[:, (ncols-3):(ncols)].values.reshape(nrows, 3)
        
        X_test01 = testX
        X_test12 = np.zeros((len(testX), h1))
        X_test23 = np.zeros((len(testX), h2))

        if activation == "sigmoid":
            in1 = np.dot(testX, self.w01 )
            X_test12 = self.__sigmoid(in1)
            in2 = np.dot(X_test12, self.w12)
            X_test23 = self.__sigmoid(in2)
            in3 = np.dot(X_test23, self.w23)
            y_predict = self.__sigmoid(in3)

        elif activation == "tanh":
            in1 = np.dot(testX, self.w01 )
            X_test12 = self.__tanh(in1)
            in2 = np.dot(X_test12, self.w12)
            X_test23 = self.__tanh(in2)
            in3 = np.dot(X_test23, self.w23)
            y_predict = self.__tanh(in3)

        elif activation == "ReLu":
            in1 = np.dot(testX, self.w01 )
            X_test12 = self.__ReLu(in1)
            in2 = np.dot(X_test12, self.w12)
            X_test23 = self.__ReLu(in2)
            in3 = np.dot(X_test23, self.w23)
            y_predict = self.__ReLu(in3)

        Error = y_predict - testy
        Error = np.sum(np.abs(Error), axis=1)
        mse = 1/(2*nrows) * np.dot(Error.T, Error)

        return mse
            

if __name__ == "__main__":
   
    dataset = NN("iris.csv")
    dataset.preprocess()

    # print sigmoid result
    print("\n" + "  Sigmoid activation function:")
    neural_network = NN("train.csv")
    neural_network.train(activation="sigmoid")
    testError = neural_network.predict(test="test.csv", activation="sigmoid")
    print("The testError is: " + str(testError))

    # print tanh result
    print("\n" + "  tanh activation function:")
    neural_network = NN("train.csv")
    neural_network.train(activation="tanh")
    testError = neural_network.predict(test="test.csv", activation="tanh")
    print("The testError is: " + str(testError))

    # print Relu result
    print("\n" + "  ReLu activation function:")
    neural_network = NN("train.csv")
    neural_network.train(activation="ReLu")
    testError = neural_network.predict(test="test.csv", activation="ReLu")
    print("The testError is: "+ str(testError))