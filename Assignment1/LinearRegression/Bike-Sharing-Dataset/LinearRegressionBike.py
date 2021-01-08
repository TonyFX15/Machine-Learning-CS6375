import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    def __init__(self, dataFile):
        self.data = dataFile

    # TODO: Perform pre-processing for your dataset. It may include doing the following:
    #   - getting rid of null values
    #   - converting categorical to numerical values
    #   - scaling and standardizing attributes
    #   - anything else that you think could increase model performance
    # Below is the pre-process function
    def preProcess(self):
        #No header in this data set
        dataframe = pd.read_csv(self.data)

        #Drop duplicate values
        dataframe = dataframe.drop_duplicates()

        #Drop 'instant' column (not important)
        dataframe = dataframe.drop(['instant'], axis=1)

        #Drop 'dteday' column because all relevant
        #information is already repeated
        dataframe = dataframe.drop(['dteday'], axis=1)

        #Drop 'casual' and 'registered' columns because the final 'cnt'
        #incorporates both
        dataframe = dataframe.drop(['casual', 'registered'], axis=1)
    
        print(dataframe)

        #Professor's code from init
        np.random.seed(1)
        dataframe.insert(0, 'X0', 1)
        self.nrows, self.ncols = dataframe.shape[0], dataframe.shape[1]
        self.X =  dataframe.iloc[:, 0:(self.ncols -1)].values.reshape(self.nrows, self.ncols-1)
        self.y = dataframe.iloc[:, (self.ncols-1)].values.reshape(self.nrows, 1)
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)

        self.X[:, 1:] = StandardScaler().fit_transform(self.X[:, 1:])
        self.y[:, 1:] = StandardScaler().fit_transform(self.y[:, 1:])
        self.X_train, self.testX, self.y_train, self.testY = train_test_split(self.X, self.y, test_size=0.2,
                                                                              random_state=0)
        self.nrows = self.X_train.shape[0]
        

    # Below is the training function
    def train(self, epochs = 10000, learning_rate = .1):
        # Perform Gradient Descent
        for i in range(epochs):
            # Make prediction with current weights
            h = np.dot(self.X_train, self.W)
            # Find error
            error = h - self.y_train
            self.W = self.W - (1 / self.nrows) * learning_rate * np.dot(self.X_train.T, error)

        return self.W, error

    # predict on test dataset
    def predict(self):
        #testDF = pd.read_csv(test)
        #testDF.insert(0, "X0", 1)
        #nrows, ncols = testDF.shape[0], testDF.shape[1]
        testX = self.testX
        testY = self.testY
        pred = np.dot(testX, self.W)
        error = pred - testY
        mse = (1/(2*self.nrows)) * np.dot(error.T, error)
        return mse

    # Plot
    def plot(self):
        regr = linear_model.LinearRegression()
        regr.fit(self.X_train, self.y_train)
        pred_Y = regr.predict(self.testX)

        #Mean Squared Error
        print('Mean squared error: %.2f'
	        % mean_squared_error(self.testY, pred_Y))
        
        plt.scatter(self.testX, self.testY, color='black')
        plt.plot(self.testX, pred_Y, color='blue', linewidth=3) 

        plt.xticks(())
        plt.yticks(())

        plt.show()


if __name__ == "__main__":
    model = LinearRegression("bikes.csv")
    model.preProcess()
    W, e = model.train()
    mse = model.predict()
    print (mse)
    model.plot()