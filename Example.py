from Classifier import LogisticRegressionClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data.csv")

X = data.iloc[:,2:-1]
Y = data.iloc[:,1:2]
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y.values.ravel())
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
algorithm = LogisticRegressionClassifier(learning_rate = 0.0000001, iterations = 1000)

algorithm.fit(x_train, y_train, verbose = 1)

pred_y = algorithm.predict(x_test)

def accuracy(y, pred_y):
    return np.sum(y == pred_y) / len(y)

print(accuracy(y_test, pred_y))