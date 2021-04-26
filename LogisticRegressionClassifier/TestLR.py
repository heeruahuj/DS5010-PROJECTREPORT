import unittest
import numpy as np
from LogisticRegression import LogisticRegressionClassifier

class Test(unittest.TestCase):

    def test_sigmoid(self):
        # Creating object of classifier for unit testing
        lrs = LogisticRegressionClassifier()
        print("test_sigmoid")
        # Checking if values for sigmoid are between 0 and 1
        # testing with positive and negative numbers which are varied. 
        self.assertEqual(lrs.sigmoid(100), 1)
        self.assertEqual(lrs.sigmoid(1), 0.7310585786300049)
        self.assertEqual(lrs.sigmoid(2), 0.8807970779778823)
        self.assertEqual(lrs.sigmoid(3), 0.9525741268224334)
        self.assertEqual(lrs.sigmoid(0.1), 0.52497918747894)
        self.assertEqual(lrs.sigmoid(-1), 0.2689414213699951)

    def test_loss(self):

        lrs = LogisticRegressionClassifier()
        print("test_loss")
        # Testing for correct log loss output from cost function.
        self.assertAlmostEqual(lrs.loss(0,0), -0.0000099999)
        self.assertAlmostEqual(lrs.loss(1,1), -0.0000099999)
        self.assertAlmostEqual(lrs.loss(0.5,1), 0.69312718)
        self.assertAlmostEqual(lrs.loss(0.5,0), 0.69312718)

    def test_prediction(self):
        lrs = LogisticRegressionClassifier()
        # Testing prediction when classifier is not trained
        print("test_prediction when not trained")
        self.assertEqual(lrs.predict(0.222), None)
        lrs.weight = np.zeros(1)
        print("test_prediction when trained")
        # Testing prediction when classifier has some weight
        self.assertEqual(lrs.predict(0.222), 0)
        # simulating trained weights.
        lrs.weight += 0.3
        self.assertEqual(lrs.predict(0.222), 1)
    
    def test_gradientD(self):
        lrs = LogisticRegressionClassifier()
        print("test_gradient")
        # Test case for when max iterations are 0.
        lrs.iterations = 0
        self.assertEqual(lrs.fit(np.array([0]),1), None)
        lrs.iterations = 10
        # Checking correct update of weights for network.
        lrs.fit(np.array([[0.4, 0.5],[0.2,0.3]]),np.array([1,0]))
        self.assertAlmostEqual(lrs.weight[0],0.04872931)
        self.assertAlmostEqual(lrs.weight[1],0.04834258)


if __name__ == '__main__':
    unittest.main()