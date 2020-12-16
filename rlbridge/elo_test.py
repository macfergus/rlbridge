import unittest

from .elo import Match, calculate_ratings


class EloTest(unittest.TestCase):
    def test_calculate_ratings(self):
        matches = (
            24 * [Match('a', 'b')] +
            76 * [Match('b', 'a')] +
            24 * [Match('b', 'c')] +
            76 * [Match('c', 'b')] +
            9 * [Match('a', 'c')] +
            81 * [Match('c', 'a')]
        )

        ratings = calculate_ratings(matches, anchor='a')

        self.assertAlmostEqual(1000, ratings['a'])
        self.assertAlmostEqual(1200, ratings['b'], delta=10)
        self.assertAlmostEqual(1400, ratings['c'], delta=10)
