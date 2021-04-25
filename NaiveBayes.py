import numpy as np

class NaiveBayesClassifier:
    def fit(self, X, y):
        sam, feat = X.shape
        self.in_classes = np.unique(y)
        classes = len(self.in_classes)

        self.mean = np.zeros((classes, feat), dtype=np.float64)
        self.variance = np.zeros((classes, feat), dtype=np.float64)
        self.prior_probs =  np.zeros(classes, dtype=np.float64)

        for index, c in enumerate(self.in_classes):
            X_class = X[y==c]
            self.mean[index, :] = X_class.mean(axis=0)
            self.variance[index, :] = X_class.var(axis=0)
            self.prior_probs[index] = X_class.shape[0] / float(sam)
    def predict(self, X):
        yhat = [self._predict(x) for x in X]
        return np.array(yhat)

    def _predict(self, x):
        posterior_probabilities = []
        for index, c in enumerate(self.in_classes):
            prior = np.log(self.prior_probs[index])
            posterior = np.sum(np.log(self.prob_den_func(index, x)))
            posterior = prior + posterior
            posterior_probabilities.append(posterior)
        return self.in_classes[np.argmax(posterior_probabilities)]

    def prob_den_func(self, class_index, x):
        mean = self.mean[class_index]
        var = self.variance[class_index]
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
