import unittest

from . import encoder
from ...cards import new_deal
from ...game import Call, GameState
from ...players import Player


class EncoderTest(unittest.TestCase):
    def test_encode_no_contract_game(self):
        game = (
            GameState.new_deal(new_deal(), Player.north, False, False)
            .apply_call(Call.of('pass'))
            .apply_call(Call.of('pass'))
            .apply_call(Call.of('pass'))
            .apply_call(Call.of('pass'))
        )
        assert game.is_over()

        enc = encoder.Encoder()
        enc.encode_full_game(game, Player.north)
