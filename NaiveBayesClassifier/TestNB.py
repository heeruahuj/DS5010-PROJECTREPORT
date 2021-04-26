import unittest
import numpy as np
from NaiveBayes import NaiveBayesClassifier

class Test(unittest.TestCase):

    def test_prob_density_function(self):
        print("test_prob_density_function")
        # Creating object of classifier for unit testing
        nb = NaiveBayesClassifier()
        nb.mean = [1]
        nb.variance = [3]
        # Testing probability calc with known calculation.
        self.assertAlmostEqual(nb.prob_den_func(0, 3), 0.11825507)
        nb.mean = [1,2]
        nb.variance = [3,1]
        self.assertAlmostEqual(nb.prob_den_func(1, 0.1), 0.06561581)
    
    def test_pred(self):
        print("test_pred")
        nb = NaiveBayesClassifier()
        nb.mean = [1,2]
        nb.variance = [3,1]
        # testing condition where it is not trained
        self.assertEqual(nb.predict([1.4,12,3,9]), None)
        self.assertEqual(nb.predict([2,3,4,5]), None)
        self.assertEqual(nb.predict([1]), None)
        self.assertEqual(nb.predict([2,-3,4,-5,-7,-7]), None)
        nb.mean = [1,1]
        nb.variance = [1,5]
        nb.in_classes = [1,0]
        nb.prior_probs = [0.3,0.2]
        pred = nb.predict([5,3,4, 7])
        # Testing condition simulating trained model.
        self.assertEqual(pred[0], [0])
        self.assertEqual(pred[1], [0])
        self.assertEqual(pred[2], [0])
        self.assertEqual(pred[3], [0])

    def test_fit(self):
        print("test_fit")
        nb = NaiveBayesClassifier()
        Xis = np.array([[3, 4], [2, 3]])
        yis = [0,1]
        nb.prior_probs = np.zeros(2, dtype=np.float64)
        # Testing value updation with known calculation.
        self.assertEqual(nb.prior_probs[0], 0)
        self.assertEqual(nb.prior_probs[1], 0)
        self.assertEqual(nb.fit(Xis,yis), None)
        self.assertEqual(nb.prior_probs[0], 0.5)
        self.assertEqual(nb.prior_probs[1], 0.5)



if __name__ == '__main__':
    unittest.main()