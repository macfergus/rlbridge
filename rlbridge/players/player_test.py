import unittest

from .player import Player


class PlayerTest(unittest.TestCase):
    def test_is_teammate(self):
        self.assertTrue(Player.north.is_teammate(Player.south))
        self.assertFalse(Player.north.is_teammate(Player.east))

    def test_is_opponent(self):
        self.assertTrue(Player.north.is_opponent(Player.east))
        self.assertTrue(Player.north.is_opponent(Player.west))
        self.assertFalse(Player.north.is_opponent(Player.south))
        self.assertFalse(Player.north.is_opponent(Player.north))

    def test_rotate(self):
        self.assertEqual(Player.east, Player.north.rotate())
        self.assertEqual(Player.south, Player.east.rotate())
        self.assertEqual(Player.west, Player.south.rotate())
        self.assertEqual(Player.north, Player.west.rotate())
