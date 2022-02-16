import sys
sys.path.append('../src')

import unittest
from preprocessing import *
import numpy as np


def dummy_image():
    return np.random.randint(0, 255, size=(1024, 1024, 3))


class TestModels(unittest.TestCase):
        
    def test_data(self):
        output = preprocess_image(dummy_image())
        
        self.assertTrue(output.shape == (1, 112, 112, 1))   ## right shape

        self.assertTrue(output.min() >= 0.)         ## right normalization

        self.assertTrue(output.max() <= 1.)         ## right normalization
    


if __name__ == "__main__":
    unittest.main()