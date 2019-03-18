import unittest

from ..cards import Card, Deal
from .auction import Bid, Contract, Scale
from .play import Play, PlayState
from .player import Player

EXAMPLE_DEAL = Deal.from_dict({
    Player.north: map(Card.of, [
        "KS", "7S", "5S", "4S",
        "AH", "QH", "6H", "3H", "2H",
        "AD",
        "KC", "QC", "2C",
    ]),
    Player.east: map(Card.of, [
        "JS", "9S", "8S", "6S",
        "8H", "5H", "4H",
        "10D", "5D",
        "8C", "7C", "4C", "3C",
    ]),
    Player.south: map(Card.of, [
        "AS", "QS", "3S", "2S",
        "7D", "6D", "4D", "2D",
        "AC", "JC", "10C", "9C", "5C",
    ]),
    Player.west: map(Card.of, [
        "10S",
        "KH", "JH", "10H", "9H", "7H",
        "KD", "QD", "JD", "9D", "8D", "3D",
        "5C",
    ])
})


def start_play(declarer):
    return PlayState.open_play(
        auction_result=Contract(
            declarer=declarer,
            bid=Bid.of("1S"),  # arbitrary
            scale=Scale.undoubled  # arbitrary
        ),
        deal=EXAMPLE_DEAL)


class IsLegalTest(unittest.TestCase):
    def test_can_open_any_card(self):
        game = start_play(declarer=Player.north)
        self.assertTrue(game.is_legal(Play.of("10D")))

    def test_cannot_play_card_not_in_hand(self):
        game = start_play(declarer=Player.north)
        self.assertFalse(game.is_legal(Play.of("AC")))
