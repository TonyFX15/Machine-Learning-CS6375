#####################################################################################################################
#   CS 6375.003 - Assignment 1, Linear Regression using Gradient Descent
#   This is a simple starter code in Python 3.6 for linear regression using the notation shown in class.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   test - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# imoort sgd library
class SgdLibraryLinearRegression:
    def __init__(self, data):
        self.df = pd.read_csv(data)
        self.regressor = SGDRegressor(max_iter=40, tol=1e-5, learning_rate='constant', eta0=0.06)

    def preprocess(self):
        # Removing Null values
        self.df.dropna()

        # Removing Duplicates
        self.df.drop_duplicates()
        
        # Checking the type of input data
        self.df.dtypes

        # Since horsepower is of object type we want to determine the nature of the attribute
        self.df['horsepower'].unique()

        # We are able to see ? in between numerical values so we are disregarding those instances
        self.df = self.df[self.df.horsepower != '?']

        # We are then casting the object to float for further processing
        self.df['horsepower'] = self.df['horsepower'].astype('float')

        # We are removing the car name attribute since that does not correlate with the mpg of the car
        self.df.drop(['car name'], axis=1, inplace=True)

        # Attributes are starting from column 1
        self.X = self.df.iloc[:, 1:].values
        self.Y = self.df.iloc[:, 0].values

        # Scaling the input attributes
        self.X = StandardScaler().fit_transform(self.X)

        # Splitting the data into training and test data set of the proportion 70:30
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=1)

    def train(self, epoch_count=40, learning_rate=0.06):
        self.regressor = SGDRegressor(max_iter=epoch_count, tol=1e-5, learning_rate='constant', eta0=learning_rate)
        # Running the training by calling the library method
        self.regressor.fit(self.X_train, self.Y_train)

    def predictTrain(self):
        # Predicting the values based on the test data
        self.Y_pred = self.regressor.predict(self.X_train)

        # Getting the accuracy from the library method
        self.accuracy_score = self.regressor.score(self.X_train, self.Y_train)
        # print("accuracy score ", accuracy_score)

        # Getting the mean squared error by comparing the predicted value with the actual test value
        self.calculated_mse = mean_squared_error(self.Y_train, self.Y_pred)
        # print("mean square error ",calculated_mse)

        # Getting the r2 score by comparing the predicted value with the actual test value
        self.r2_scor = r2_score(self.Y_train, self.Y_pred)
        # print("r2_score ", r2_scor)

        return self.calculated_mse

    def print(self):
        print("accuracy score ", self.accuracy_score)
        print("mean square error ", self.calculated_mse)
        print("r2_score ", self.r2_scor)

    def predictTest(self):
        # Predicting the values based on the test data
        self.Y_pred = self.regressor.predict(self.X_test)

        # Getting the accuracy from the library method
        self.accuracy_score = self.regressor.score(self.X_test, self.Y_test)
        # print("accuracy score ", accuracy_score)

        # Getting the mean squared error by comparing the predicted value with the actual test value
        self.calculated_mse = mean_squared_error(self.Y_test, self.Y_pred)
        # print("mean square error ", calculated_mse)

        # Getting the r2 score by comparing the predicted value with the actual test value
        self.r2_scor = r2_score(self.Y_test, self.Y_pred)
        # print("r2_score ", r2_scor)

        return self.calculated_mse

    def plotLearningRate(self, epoch_count, min, max, step, color):
        mse_error = list()
        step_size = max
        x_scale = list()
        label = "epoch = "
        label += str(epoch_count)
        while(step_size >= min):
            self.train(epoch_count, step_size)
            mse_error.append(self.predictTest())
            x_scale.append(step_size)
            step_size = step_size - step
        return plt.scatter(x_scale, mse_error, color=color)


class LinearRegression:
    def __init__(self, data):
        self.data = data

    # TODO: Perform pre-processing for your dataset. It may include doing the following:
    #   - getting rid of null values
    #   - converting categorical to numerical values
    #   - scaling and standardizing attributes
    #   - anything else that you think could increase model performance
    # Below is the pre-process function

    def printDataRelatedInformation(self):
        self.df = pd.read_csv(self.data)
        print('\n--------------Correlation between input attributes and the output(mpg)----------------')
        print(self.df.corr().mpg)

        print('\nType of data in the dataset')

        # Checking the type of input data
        print(self.df.dtypes)

        print('\nUnique values in horsepower since it is of object type')

        # Since horsepower is of object type we want to determine the nature of the attribute
        print(self.df['horsepower'].unique())

        print('\nRemove the \'?\' instances of horsepower')

    def preProcess(self):

        df = pd.read_csv(self.data)
        df.dropna()
        df.drop_duplicates()
        df.dtypes
        df['horsepower'].unique()
        df = df[df.horsepower != '?']
        df['cylinders'] = df['cylinders'].astype(float)
        df['displacement'] = df['displacement'].astype(float)
        df['horsepower'] = df['horsepower'].astype(float)
        df['weight'] = df['weight'].astype(float)
        df['acceleration'] = df['acceleration'].astype(float)
        df['origin'] = df['origin'].astype(float)
        df['model year'] = df['model year'].astype(float)
        df['mpg'] = df['mpg'].astype(float)

        # Bias
        df.insert(1, 'X0', 1)
        self.nrows, self.ncols = df.shape[0], df.shape[1]
        self.X = df.iloc[:, 1:9].values.reshape(self.nrows, 8)
        self.y = df.iloc[:, 0].values.reshape(self.nrows, 1)
        self.W = np.random.rand(8).reshape(8, 1)
        self.X[:, 1:] = StandardScaler().fit_transform(self.X[:, 1:])
        self.X_train, self.testX, self.y_train, self.testY = train_test_split(self.X, self.y, test_size=0.3,
                                                                              random_state=1)
        self.nrows = self.X_train.shape[0]

    # Below is the training function
    def train(self, epochs=100, learning_rate=0.06):
        for i in range(epochs):
            # Make prediction with current weights
            h = np.dot(self.X_train, self.W)
            # Find error
            error = h - self.y_train
            self.W = self.W - (1 / self.nrows) * learning_rate * np.dot(self.X_train.T, error)

        return self.W, error

    # predict on test dataset
    def predictTest(self):
        pred = np.dot(self.testX, self.W)
        error = pred - self.testY

        self.mean_square_error = 1 / (2 * (self.testY.shape[0])) * np.dot(error.T, error)

        self.r2_scor = r2_score(self.testY, pred)

        return self.mean_square_error

    def print(self):
        print("mean square error ", self.mean_square_error)
        print("r2_score ", self.r2_scor)

    def predictTrain(self):
        pred = np.dot(self.X_train, self.W)
        error = pred - self.y_train

        self.mean_square_error = 1 / (2 * (self.y_train.shape[0])) * np.dot(error.T, error)

        self.r2_scor = r2_score(self.y_train, pred)

        return self.mean_square_error

    def plotEpoch(self, min, max, step):
        mse_error = list()
        for epoch_count in range(min, max, step):
            self.train(epoch_count, 0.06)
            mse_error.append(self.predictTest())
        plt.scatter(range(min, max), mse_error)
        plt.show()

    def plotLearningRate(self, epoch_count, min, max, step, color):
        mse_error = list()
        step_size = max
        x_scale = list()
        label = "epoch = "
        label += str(epoch_count)
        while (step_size >= min):
            x_scale.append(step_size)
            self.train(epoch_count, step_size)
            step_size = step_size - step
            mse_error.append(self.predictTest())
        return plt.scatter(x_scale, mse_error, color=color)


if __name__ == "__main__":
    # Part 1. Training our model using the method given by professor

    # We have loaded the mpg dataset by calling the constructor of the class given by professor
    linearRegressionModel = LinearRegression('auto-mpg.csv')

    linearRegressionModel.printDataRelatedInformation()

    # We are removing the null values and cleaning the data so that it can be readily used for training
    print('--------------------------Part1---------------------------------------------')
    linearRegressionModel.preProcess()

    # We are going to find the best number of epochs and learning rate by looping the training and test over the range of both the values
    model_epoch20 = linearRegressionModel.plotLearningRate(20, 0.01, 0.1, 0.01, 'g')
    model_epoch40 = linearRegressionModel.plotLearningRate(40, 0.01, 0.1, 0.01, 'r')
    model_epoch60 = linearRegressionModel.plotLearningRate(60, 0.01, 0.1, 0.01, 'b')

    plt.legend((model_epoch20, model_epoch40, model_epoch60),
               ('Epoch 20', 'Epoch 40', 'Epoch 60'))

    # Plotting the result of the run
    plt.xlabel('Learning rate')
    plt.ylabel('MSE')
    plt.suptitle('Learning rate vs MSE for various epochs For Part-1')
    plt.savefig('linear_regression_various_learningRate_epochs_graph_part_1.png')
    plt.show()

    # Based on the above graph we have determined the most accurate model occurs for epoch = 40 and learning rate = 0.06
    linearRegressionModel.train(40, 0.06)

    # Lets print the values of training dataset performance
    print("Training dataset error:")
    linearRegressionModel.predictTrain()
    linearRegressionModel.print()

    # Lets print the values of training dataset performance
    print("Test dataset error:")
    linearRegressionModel.predictTest()
    linearRegressionModel.print()

    # Library model Part 2
    print('\n\n\n--------------------------Part2---------------------------------------------')

    sgdLibraryModel = SgdLibraryLinearRegression('auto-mpg.csv')
    sgdLibraryModel.preprocess()

    sgdLibraryModelepoch20 = sgdLibraryModel.plotLearningRate(20, 0.01, 0.1, 0.01, 'r')
    sgdLibraryModelepoch40 = sgdLibraryModel.plotLearningRate(40, 0.01, 0.1, 0.01, 'g')
    sgdLibraryModelepoch60 = sgdLibraryModel.plotLearningRate(60, 0.01, 0.1, 0.01, 'b')

    plt.legend((sgdLibraryModelepoch20, sgdLibraryModelepoch40, sgdLibraryModelepoch60),
               ('Epochs = 20', 'Epoch = 40', 'Epoch = 60'))
    # Plotting the result of the run
    plt.xlabel('Learning rate')
    plt.ylabel('MSE')
    plt.suptitle('Learning rate vs MSE for various epochs For Part-2')
    plt.savefig('linear_regression_various_learningRate_epochs_graph_part_2.png')
    plt.show()

    # From the plot we found that MSE is minimum for epoch = 40 and learning rate = 0.06
    sgdLibraryModel.train(epoch_count=40, learning_rate=0.06)
    sgdLibraryModel.predictTrain()
    print("Results For Training data set")
    sgdLibraryModel.print()
    print("Results For Test data set")
    sgdLibraryModel.predictTest()
    sgdLibraryModel.print()
