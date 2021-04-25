import numpy as np


class LogisticRegressionClassifier:

    def __init__(self, learning_rate = 0.1, iterations = 500):
        self.weight = None
        self.iterations = iterations
        self.learning_rate = learning_rate

    def sigmoid(self, Z):

        s_z = 1/(1+np.exp(-Z))

        return s_z

    def loss(self, pred_y, y):
        eps = 1e-5  
        loss_num = (-y*np.log(pred_y + eps))
        num2 = ((1-y)*np.log(1-pred_y + eps))
        cost = loss_num - num2
        return np.sum(cost)

    def fit(self, x, y, verbose = 0):

        self.weight = np.zeros(x.shape[1])

        for it in range(self.iterations):
            pred_y = self.sigmoid(np.dot(x,self.weight))
            if verbose == 1:
                print("Iteration %d ----------- Loss: %.3f-----"%((it+1), self.loss(pred_y, y)))
            
            gradient = (np.dot(x.T, (pred_y - y)))/x.shape[1]
            #print(self.weight)
            self.weight -= self.learning_rate * gradient
            #print(self.weight)
        

    def predict(self, x):
        
        pred_y = self.sigmoid(np.dot(x,self.weight))
        predictions = [1 if probability > 0.5 else 0 for probability in pred_y]

        return np.array(predictions)


