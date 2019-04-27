import unittest

from ..cards import Card, Deal
from ..players import Player
from .auction import Bid, Contract, Scale
from .play import Play, PlayState

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
        "6C",
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

    def test_must_follow_suit_if_possible(self):
        game = start_play(declarer=Player.north) \
            .apply(Play.of("10D"))
        self.assertTrue(game.is_legal(Play.of("7D")))
        self.assertFalse(game.is_legal(Play.of("AC")))

    def test_can_change_suit_if_void(self):
        game = start_play(declarer=Player.north) \
            .apply(Play.of("8H"))
        self.assertTrue(game.is_legal(Play.of("AC")))

    def test_winner_leads_next_trick(self):
        game = (
            start_play(declarer=Player.north)
                .apply(Play.of("3C"))  # east
                .apply(Play.of("5C"))  # south
                .apply(Play.of("6C"))  # west
                .apply(Play.of("KC"))  # north
        )
        self.assertEqual(Player.north, game.next_player)

    def test_off_suit_cannot_win_trick(self):
        game = (
            start_play(declarer=Player.north)
                .apply(Play.of("4H"))  # east
                .apply(Play.of("AC"))  # south
                .apply(Play.of("7H"))  # west
                .apply(Play.of("2H"))  # north
        )
        self.assertEqual(Player.west, game.next_player)

    def test_trump_wins_trick(self):
        game = (
            start_play(declarer=Player.north)
                .apply(Play.of("4H"))  # east
                .apply(Play.of("2S"))  # south
                .apply(Play.of("7H"))  # west
                .apply(Play.of("2H"))  # north
        )
        self.assertEqual(Player.south, game.next_player)

    def test_cannot_play_card_twice(self):
        game = (
            start_play(declarer=Player.north)
                .apply(Play.of("3C"))  # east
                .apply(Play.of("5C"))  # south
                .apply(Play.of("6C"))  # west
                .apply(Play.of("KC"))  # north
        )
        self.assertFalse(game.is_legal(Play.of("KC")))


class VisibleCardsTest(unittest.TestCase):
    def test_dummy_not_visible_before_open(self):
        game = start_play(declarer=Player.north)
        visible_cards = game.visible_cards(Player.east)
        # Before the first play, you can't see the dummy's holding.
        self.assertCountEqual([Player.east], visible_cards.keys())

    def test_dummy_visible_after_open(self):
        game = (
            start_play(declarer=Player.north)
                .apply(Play.of("3C"))
        )
        visible_cards = game.visible_cards(Player.east)
        self.assertCountEqual(
            [Player.east, Player.south], visible_cards.keys())
