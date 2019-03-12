import unittest

from .auction import Denomination


class DenominationTest(unittest.TestCase):
    def test_ordering(self):
        self.assertLess(Denomination.clubs(), Denomination.diamonds())
        self.assertLess(Denomination.diamonds(), Denomination.spades())
        self.assertLess(Denomination.clubs(), Denomination.notrump())
        self.assertFalse(Denomination.diamonds() < Denomination.clubs())
        self.assertFalse(Denomination.spades() < Denomination.diamonds())
        self.assertFalse(Denomination.notrump() < Denomination.clubs())
