from NaiveBayes import NaiveBayesClassifier
#importing the class

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

#Calculating accuracy --(actual-prediction)/(total no of samples)
def accuracy(y_actual, yhat):
    accuracy = np.sum(y_actual == yhat) / len(y_actual)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
#Dividing the dataset into train and test -80-20 division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)#fitting the model
predictions = nb.predict(X_test)#predicting on the test set

print("Accuracy", accuracy(y_test, predictions))#calculating the accuracy
