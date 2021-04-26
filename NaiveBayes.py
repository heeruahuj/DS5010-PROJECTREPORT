import numpy as np
# Using numpy for fast and efficient vector calculations.

class NaiveBayesClassifier:

    def __init__(self):
        self.mean = None
        self.variance = None
        self.prior_probs = None
        self.in_classes = None

    def fit(self, X, y):
        """
        Used to fit training data and training labels
        We calculate the priors(Bayes Theorem)
        We also need to calculate the conditional probabilities in the future which require mean(μ) and variance(σ2).
        Input :X -  numpy ndarray
                    It contains the samples and features
        Output:y - 1d Vector

        """
        #
        sam, feat = X.shape
        self.in_classes = np.unique(y) #Getting the unique classes
        classes = len(self.in_classes)#number of classes

        self.mean = np.zeros((classes, feat), dtype=np.float64)#for each class, we calculate the mean of each feature
        self.variance = np.zeros((classes, feat), dtype=np.float64)#or each class, we calculate the variance of each feature
        self.prior_probs =  np.zeros(classes, dtype=np.float64)#for each class, we need a prior

        for index, c in enumerate(self.in_classes):#enumerating through the the classes
            X_class = X[y==c] #filtering samples for chosen class - c
            self.mean[index, :] = X_class.mean(axis=0) #calc the mean of each class and appending to self.mean
            self.variance[index, :] = X_class.var(axis=0) #Calc the variance and appending to self.variance
            self.prior_probs[index] = X_class.shape[0] / float(sam)#calculating the prior probabilities
            #prior prob is the frequency of class c occuring in training samples X #(no of samples with c label)/(total samples)

        return None
    def predict(self, X):
        """
        Used to predict the test Labels
        Input: Training Data and Training samples
        Output: prediction-yhat or y_prediction

        """
        if self.prior_probs == None:
            return None
        yhat = [self._predict(x) for x in X]
        return np.array(yhat)

    def _predict(self, x):

        """
        Helper method for predict()
        It does the prediction for one samples
        The original method predict() uses _predict to  it for all training samples
        we need to calculate :
            y = argmax y log(P(x1|y)) + log(P(x2|y)) +....+ log(P(xn|y)) + log(P(y))
        So, we need the prior probabilities, the posterior probabilities(Class Conditional Probabilities)

        Input: test n_samples
        Output: returns the class with the maximum posterior probability
        """
        posterior_probabilities = []
        for index, c in enumerate(self.in_classes):
            prior = np.log(self.prior_probs[index]) #log(P(y))
            #Applying the gaussian function to calculate the conditional probability
            #Using the helper function prob_den_func
            posterior = np.sum(np.log(self.prob_den_func(index, x)))
            posterior = prior + posterior
            posterior_probabilities.append(posterior)
        return self.in_classes[np.argmax(posterior_probabilities)]

    def prob_den_func(self, class_index, x):

        """
        Calculates the class conditional probability for each class(hence takes class_idx as input)
        , using the gaussian gaussian_function
        P(xi|y) = (1/(2σ2))* exp(-((xi-μ)^2)/2σ2) ---Better shown in readme.md
        It uses the mean and variance calculated in fit()

        Input: individual samples(x), class indexes(class_index)
        Output: Class conditional probability
        """
        mean = self.mean[class_index]
        var = self.variance[class_index]
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
