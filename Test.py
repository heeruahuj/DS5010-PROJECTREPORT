import unittest
import numpy as np
from LogisticRegression import LogisticRegressionClassifier

class Test(unittest.TestCase):

    def test_sigmoid(self):
        lrs = LogisticRegressionClassifier()
        print("test_sigmoid")
        self.assertEqual(lrs.sigmoid(100), 1)
        self.assertEqual(lrs.sigmoid(1), 0.7310585786300049)
        self.assertEqual(lrs.sigmoid(2), 0.8807970779778823)
        self.assertEqual(lrs.sigmoid(3), 0.9525741268224334)
        self.assertEqual(lrs.sigmoid(0.1), 0.52497918747894)
        self.assertEqual(lrs.sigmoid(-1), 0.2689414213699951)

    def test_loss(self):

        lrs = LogisticRegressionClassifier()
        print("test_loss")
        self.assertAlmostEqual(lrs.loss(0,0), -0.0000099999)
        self.assertAlmostEqual(lrs.loss(1,1), -0.0000099999)
        self.assertAlmostEqual(lrs.loss(0.5,1), 0.69312718)
        self.assertAlmostEqual(lrs.loss(0.5,0), 0.69312718)

    def test_prediction(self):
        lrs = LogisticRegressionClassifier()
        print("test_prediction when not trained")
        self.assertEqual(lrs.predict(0.222), None)
        lrs.weight = np.zeros(1)
        print("test_prediction when trained")
        self.assertEqual(lrs.predict(0.222), 0)
        lrs.weight += 0.3
        self.assertEqual(lrs.predict(0.222), 1)
    
    def test_gradientD(self):
        lrs = LogisticRegressionClassifier()
        print("test_gradient")
        lrs.iterations = 0
        self.assertEqual(lrs.fit(np.array([0]),1), None)
        lrs.iterations = 10
        lrs.fit(np.array([[0.4, 0.5],[0.2,0.3]]),np.array([1,0]))
        self.assertAlmostEqual(lrs.weight[0],0.04872931)
        self.assertAlmostEqual(lrs.weight[1],0.04834258)


if __name__ == '__main__':
    unittest.main()