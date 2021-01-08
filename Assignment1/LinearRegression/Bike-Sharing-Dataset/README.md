Linear-Regression

Linear Regression of UCI data set

In the command prompt in the folder space, type the following commands: "python LinearRegression.py"

Details on the dataset: Number of Instances: 398 Number of Attributes: 9 including the class attribute Attribute Information:

1. mpg:           continuous
2. cylinders:     multi-valued discrete
3. displacement:  continuous
4. horsepower:    continuous
5. weight:        continuous
6. acceleration:  continuous
7. model year:    multi-valued discrete
8. origin:        multi-valued discrete
9. car name:      string (unique for each instance)
Missing Attribute Values: horsepower has 6 missing values. All attributes were converted to float data type before processing.

Question 1: Are you satisfied that the package has found the best solution? How can you check? Explain.

We observed the value of Mean Squared Error(MSE) for different values of epochs and learning rate. On comparison of MSE for different values of epochs and learning rate for our model in part-1, we found that MSE was minimum at learning rate = 0.06 and number of epochs = 40. Increasing the number of epochs and decreasing the learning rate after this point did not yield better MSE.

We found that all the input attributes(except the attribute car name) had correlation with the output attribute(mpg).

We found the correlation between the attribute and the output as below:


We inferred this by the fact that the Mean-Square-Error value was the least(5.89) for the test dataset, when the model was trained with all these attributes namely cylinders, displacement, horsepower, weight, acceleration, model year and origin. Removing one of the attributes also increased the MSE so we decided to keep all the attributes to get the best MSE.

Hence the above results made us feel satisfied that we had found the best model.

Question 2 : Are you satisfied that the package has found the best solution? We found that the MSE found using the Scikit package produced good results and we confirmed it by running the algorithm for various epochs and learning rates. We were able to see minute differences in the MSE value calculated during each iteration as the data set is split into test and training datasets in random fashion and the difference was not much.

Dataset Sources: https://archive.ics.uci.edu/ml/datasets/auto+mpg – Auto-mpg data set UC Irvine ML Repository. https://www.kaggle.com/uciml/autompg-dataset - Auto-mpg kaggle dataset. Since the data was not properly formatted in the UCI repository link. We used the same dataset which was better formatted here.

Package Sources: https://scikit-learn.org/stable/documentation.html - Documentation of libraries from Scikit-learn.