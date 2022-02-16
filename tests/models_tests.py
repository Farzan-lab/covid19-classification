import sys
sys.path.append('../src')

import unittest
from models import EfficientNet, MobileNet
from models import *
import numpy as np


def dummy_image(i=1):
    return np.random.randint(0, 1, size=(i, 112, 112, 1))


class TestModels(unittest.TestCase):
        
    def test_mobile(self):
        model_mobile = MobileNet(model_path='../weights/mobilenet.hdf5',
                               input_shape=(112, 112, 1),
                               num_classes=4)

        i = 1
        output = model_mobile.predict(dummy_image(i))
        self.assertEqual(output.shape, (i, 4))
        
        i = 5
        output = model_mobile.predict(dummy_image(i))
        self.assertEqual(output.shape, (i, 4))
    
    def test_efficient(self):
        model_eff = EfficientNet(model_path='../weights/efficientnet.hdf5',
                               input_shape=(112, 112, 1),
                               num_classes=4)

        i = 1
        output = model_eff.predict(dummy_image(i))
        self.assertTrue(output.shape == (i, 4))
        
        i = 5
        output = model_eff.predict(dummy_image(i))
        self.assertTrue(output.shape == (i, 4))


if __name__ == "__main__":
    unittest.main()