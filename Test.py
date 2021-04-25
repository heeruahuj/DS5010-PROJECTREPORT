import unittest
import numpy as np
from numpy.lib.function_base import _percentile_dispatcher
import numpy.testing as npt
from Classifier import LogisticRegressionClassifier

class Test(unittest.TestCase):

    '''def test_sigmoid(self):
        lrs = LogisticRegressionClassifier()
        print("Testing sigmoid function values")
        npt.assert_allclose(lrs.sigmoid(np.array([100,3,2,1])), np.array([1,0.95257413, 0.88079708, 0.73105858]))
    '''
    def test_sigmoid(self):
        lrs = LogisticRegressionClassifier()
        print("Testing sigmoid function values")
        self.assertEqual(lrs.sigmoid(100), 1)
        self.assertEqual(lrs.sigmoid(1), 0.7310585786300049)
        self.assertEqual(lrs.sigmoid(2), 0.8807970779778823)
        self.assertEqual(lrs.sigmoid(3), 0.9525741268224334)
        self.assertEqual(lrs.sigmoid(0.1), 0.52497918747894)
        self.assertEqual(lrs.sigmoid(-1), 0.2689414213699951)

    def test_loss(self):

        lrs = LogisticRegressionClassifier()
        print("Testing for cost function")
        self.assertAlmostEqual(lrs.loss(0,0), -0.0000099999)
        self.assertAlmostEqual(lrs.loss(1,1), -0.0000099999)
        self.assertAlmostEqual(lrs.loss(0.5,1), 0.69312718)
        self.assertAlmostEqual(lrs.loss(0.5,0), 0.69312718)

    def test_prediction(self):
        lrs = LogisticRegressionClassifier()
        print("Testing class prediction when not trained")
        self.assertEqual(lrs.predict(0.222), None)
        lrs.weight = np.zeros(1)
        print("Testing class prediction when trained")
        self.assertEqual(lrs.predict(0.222), 0)
        lrs.weight += 0.3
        self.assertEqual(lrs.predict(0.222), 1)
if __name__ == '__main__':
    unittest.main()