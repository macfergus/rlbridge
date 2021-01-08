import unittest

import numpy as np
from numpy.testing import assert_array_equal

from ...cards import Card, Deal
from ...game import Bid, Call, GameState, Play
from ...players import Player
from .encoder2d import Encoder2D

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
def visible_cards(row, player=None):
    all_cards = row[:, :16]
    if player == 'self':
        return all_cards[:, :4]
    if player == 'lho':
        return all_cards[:, 4:8]
    if player == 'partner':
        return all_cards[:, 8:12]
    if player == 'rho':
        return all_cards[:, 12:16]
    assert player is None
    return all_cards


def call_action(row):
    start_idx = Encoder2D.CALL_BEGIN
    end_idx = start_idx + 4 * 8
    return row[:, start_idx:end_idx]


def play_action(row):
    return row[:, Encoder2D.PLAY_BEGIN:]


def are_hidden(array):
    return not np.any(array)


class ConvEncoder2DTest(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder2D()

    def test_new_game(self):
        state = GameState.new_deal(EXAMPLE_DEAL, Player.north, False, False)

        encoded = self.encoder.encode_full_game(state, Player.north)
        first = encoded[:, 0, :]
        rest = encoded[:, 1:, :]
        self.assertFalse(np.any(rest))

        # my cards: channels 0 - 3
        my_cards = first[:, :4]
        # Check for a random card in the hand
        # 7 of spades: column 5, channel 3
        self.assertEqual(1, my_cards[5, 3])

        for other_player in range(3):
            offset = 4 * (other_player + 1)
            other_player_cards = first[:, offset:offset + 4]
            # All other cards should be masked
            self.assertFalse(np.any(other_player_cards))

    def test_later_game(self):
        state = (
            GameState.new_deal(EXAMPLE_DEAL, Player.north, False, False)
            .apply_call(Call.of('1H'))
        )

        encoded = self.encoder.encode_full_game(state, Player.east)
        first = encoded[:, 0, :]
        second = encoded[:, 1, :]
        # Visible cards are the same throughout auction
        assert_array_equal(visible_cards(first), visible_cards(second))
        # First action: north player bids 1H
        # It's now east's turn, so north is RHO
        rho_offset = 3 * 8
        self.assertFalse(np.any(play_action(first)))
        self.assertEqual(1, call_action(first)[0, rho_offset + 2])
        # Second action: hasn't happened yet
        self.assertFalse(np.any(call_action(second)))
        self.assertFalse(np.any(play_action(second)))
        # Rest of game: hasn't happened yet
        rest = encoded[:, 2:, :]
        self.assertFalse(np.any(rest))

    def test_card_play(self):
        state = (
            GameState.new_deal(EXAMPLE_DEAL, Player.north, False, False)
            .apply_call(Call.of('1H'))
            .apply_call(Call.of('pass'))
            .apply_call(Call.of('pass'))
            .apply_call(Call.of('pass'))
            .apply_play(Play.of('10D'))
        )

        # Array should have 6 filled rows:
        # 0 - N bids 1H
        # 1 - E passes
        # 2 - S passes
        # 3 - W passes
        # 4 - E opens 10D
        # 5 - dummy exposed, N chooses
        encoded = self.encoder.encode_full_game(state, Player.north)
        # auction_1 = encoded[0]
        # auction_2 = encoded[1]
        # auction_3 = encoded[2]
        # auction_4 = encoded[3]
        play_1 = encoded[:, 4, :]
        play_2 = encoded[:, 5, :]
        rest = encoded[:, 6:, :]
        self.assertFalse(np.any(rest))

        # Opening: dummy is not visible yet
        self.assertTrue(are_hidden(visible_cards(play_1, 'lho')))
        self.assertTrue(are_hidden(visible_cards(play_1, 'rho')))
        self.assertTrue(are_hidden(visible_cards(play_1, 'partner')))
        self.assertFalse(are_hidden(visible_cards(play_1, 'self')))

        # After opening: dummy is visible
        self.assertTrue(are_hidden(visible_cards(play_2, 'lho')))
        self.assertTrue(are_hidden(visible_cards(play_2, 'rho')))
        self.assertFalse(are_hidden(visible_cards(play_2, 'partner')))
        self.assertFalse(are_hidden(visible_cards(play_2, 'self')))

        # E (lho) plays 10 of diamonds
        # 10 -> column 8
        # diamonds -> channel 1
        plays = play_1[:, Encoder2D.PLAY_BEGIN + 4:Encoder2D.PLAY_BEGIN + 8]
        self.assertEqual(1, plays[8, 1])

    def test_encode_contract_no_contract(self):
        assert_array_equal(np.zeros(5), self.encoder.encode_contract(None))

    def test_encode_contract(self):
        assert_array_equal(
            [5 / 7, 0, 0, 0, 0],
            self.encoder.encode_contract(Bid.of('5C'))
        )
