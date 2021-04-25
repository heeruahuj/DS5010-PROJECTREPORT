# This module demonstrates a possible use case for using the Logistic Regression tool in binomial classification.

# Numpy and pandas are used for easy loading of dataset and efficient dataset cleaning and operations.
# sklearn libraries are used only for splitting dataset into random samples of training and testing.
# LabelEncoder has been used to convert the string classes of output into binary classes to feed into the network.

from LogisticRegression import LogisticRegressionClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# For this example, we are using the Breast Cancer Wisconsin Dataset from Kaggle
# (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
data = pd.read_csv("data.csv")

# The columns other than the first column are all features we will use to predict the class labels.
X = data.iloc[:,2:-1]
Y = data.iloc[:,1:2]

# The first column contains the labels - M and B for Malignant and Benign. These are encoded into 1 and 0.
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y.values.ravel())

#Splitting dataset into training and testing data to check how well it performs on new data after training.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# After experimenting with different values of learning rate, 0.0000001 seems to be the most effective for this case.
algorithm = LogisticRegressionClassifier(learning_rate = 0.0000001, iterations = 1000)

algorithm.fit(x_train, y_train, verbose = 0)
# Setting verbose to 1 will print loss at every iteration of training. Any other value of verbose  will not print.

# For example, if verbose = 1, It will print the iterations as:
# Iteration 1 ----------- Loss: 295.272-----
# Iteration 2 ----------- Loss: 290.963-----
# Iteration 3 ----------- Loss: 289.915-----

pred_y = algorithm.predict(x_test)

def accuracy(y, pred_y):
    return np.sum(y == pred_y) / len(y)

# For this example, 
# -> 10 iterations gives an accuracy of ~ 37% 
# -> 100 iterations gives an accuracy of ~ 78%
# -> 1000 iterations gives an accuracy of ~ 90%
# -> 10000 iterations gives an accuracy of ~ 93%
# On the test dataset.

print(accuracy(y_test, pred_y))