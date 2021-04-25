import numpy as np
# Using numpy for fast and efficient vector calculations.

class LogisticRegressionClassifier:

    def __init__(self, learning_rate = 0.1, iterations = 500):
        '''
        Initializes values for max iterations and learning rate for gradient descent.
        '''
        self.weight = None
        self.iterations = iterations
        self.learning_rate = learning_rate

    def sigmoid(self, Z):
        '''
        Returns the sigmoid probability estimate of vector
        param Z: Vector containing input.
        returns: Vector with estimated sigmoid probabilities.
        '''
        # Calculation of Sigmoid(z) = 1/(1+exp(-z)) 
        s_z = 1/(1+np.exp(-Z))

        return s_z

    def loss(self, pred_y, y):
        '''
        Calculates the cost or loss between predicted and expected class.
        param pred_y: contains predicted classes
        param y: vector with actual expected classes
        returns: Vectorized log probability loss
        '''
        # Calculating and returning log probability loss according to y value being 1 or 0.
        # added 0.00001 for avoiding case of log(0)
        return np.sum((-y*np.log(pred_y + 0.00001)) - ((1-y)*np.log(1-pred_y + 0.00001)))

    def fit(self, x, y, verbose = 0):
        '''
        Calculates new weights using Gradient Descent algorithm. Weights are initialized to zero
        and gradient is calculated for features based on training labels.
        param x: Training features
        param y: Training labels
        param verbose: parameter which decides whether to print loss for every iteration.
        returns: None
        '''
        if self.iterations == 0:
            return None
        # Initializing all weights to zero for first training.
        self.weight = np.zeros(x.shape[1])

        # Training weights for max iterations.
        for it in range(self.iterations):
            # Calculation of estimated probabilities.
            pred_y = self.sigmoid(np.dot(x,self.weight))

            # Printing loss at every iteration if verbose = 1.
            if verbose == 1:
                print("Iteration %d ----------- Loss: %.3f-----"%((it+1), self.loss(pred_y, y)))
            # Calculating gradient for updating weights.
            gradient = (np.dot(x.T, (pred_y - y)))/x.shape[1]

            # Updating  weights after every iteration according to gradient.
            self.weight -= self.learning_rate * gradient
        
        return None
        

    def predict(self, x):
        '''
        Predicts labels for input features
        param x: input features
        returns: predicted classes on input features
        '''
        # Checkling if model has been trained.
        if self.weight is None:
            print("Not trained")
            return None
        # Predicting labels for features with weights with 0.5 as threshold.
        pred_y = self.sigmoid(np.dot(x,self.weight))
        predictions = [1 if probability > 0.5 else 0 for probability in pred_y]

        return np.array(predictions)

