import unittest

import numpy as np

from . import nputil


# pylint: disable=no-self-use
class ShrinkTest(unittest.TestCase):
    def test_shrink_axis(self):
        x = np.zeros((100, 3, 5))
        x[:7,] = 1
        nputil.shrink_axis(x, 7)
        np.testing.assert_array_equal(x, np.ones((7, 3, 5)))
