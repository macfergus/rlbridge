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

    def test_lho(self):
        self.assertEqual(Player.east, Player.north.lho())
        self.assertEqual(Player.south, Player.east.lho())
        self.assertEqual(Player.west, Player.south.lho())
        self.assertEqual(Player.north, Player.west.lho())

    def test_partner(self):
        self.assertEqual(Player.south, Player.north.partner)
        self.assertEqual(Player.west, Player.east.partner)
        self.assertEqual(Player.north, Player.south.partner)
        self.assertEqual(Player.east, Player.west.partner)
