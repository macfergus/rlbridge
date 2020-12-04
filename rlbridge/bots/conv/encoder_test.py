import unittest

import numpy as np
from numpy.testing import assert_array_equal

from ...cards import Card, Deal
from ...game import Call, GameState
from ...players import Player
from .encoder import Encoder


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


# helpers for extracting parts of a state representation
def visible_cards(row):
    card_start = Encoder.VISIBLE_CARD_START
    return row[card_start:card_start + 4 * 53]


def call_action(row):
    start_idx = Encoder.AUCTION_START
    return row[start_idx:start_idx + Encoder.DIM_AUCTION]


def play_action(row):
    start_idx = Encoder.PLAY_START
    return row[start_idx:start_idx + Encoder.DIM_PLAY]


class ConvEncoderTest(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()

    def test_new_game(self):
        state = GameState.new_deal(EXAMPLE_DEAL, Player.north, False, False)

        encoded = self.encoder.encode_full_game(state, Player.north)
        first = encoded[0]
        rest = encoded[1:]
        self.assertFalse(np.any(rest))

        card_start = self.encoder.VISIBLE_CARD_START
        my_cards = first[card_start:card_start + 53]
        # Check the visible sentinel
        self.assertEqual(0, my_cards[0])
        # Check for a random card in the hand
        card_idx = self.encoder.encode_card(Card.of('7S'))
        self.assertEqual(1, my_cards[card_idx + 1])
        for other_player in range(3):
            offset = self.encoder.VISIBLE_CARD_START + 53 * (other_player + 1)
            other_player_cards = first[offset:offset + 53]
            # Check for not visible sentinel
            self.assertEqual(1, other_player_cards[0])
            # All other cards should be masked
            self.assertFalse(np.any(other_player_cards[1:]))

    def test_later_game(self):
        state = (
            GameState.new_deal(EXAMPLE_DEAL, Player.north, False, False)
            .apply_call(Call.of('1H'))
        )

        encoded = self.encoder.encode_full_game(state, Player.east)
        first = encoded[0]
        second = encoded[1]
        # Visible cards are the same throughout auction
        assert_array_equal(visible_cards(first), visible_cards(second))
        # First action: north player bids 1H
        # It's now east's turn, so north is RHO
        self.assertFalse(np.any(play_action(first)))
        call_idx = self.encoder.encode_call(Call.of('1H'))
        rho_offset = 3 * 38
        self.assertEqual(1, call_action(first)[rho_offset + call_idx])
        # Second action: hasn't happened yet
        self.assertFalse(np.any(call_action(second)))
        self.assertFalse(np.any(play_action(second)))
        # Rest of game: hasn't happened yet
        rest = encoded[2:]
        self.assertFalse(np.any(rest))
