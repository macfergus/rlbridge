import unittest

from .player import Player


class PlayerTest(unittest.TestCase):
    def test_is_teammate(self):
        self.assertTrue(Player.north.is_teammate(Player.south))
        self.assertFalse(Player.north.is_teammate(Player.east))
