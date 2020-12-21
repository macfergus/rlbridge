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


class SmoothTest(unittest.TestCase):
    def test_smooth(self):
        x = np.array(100 * [10, 0])
        x_smooth = nputil.smooth(x, 5)
        self.assertEqual(x.shape, x_smooth.shape)
        # Smoothed signal should have approximately the same mean, but
        # much lower variance
        self.assertAlmostEqual(np.mean(x), np.mean(x_smooth), delta=0.1)
        self.assertLess(np.std(x_smooth), np.std(x))
