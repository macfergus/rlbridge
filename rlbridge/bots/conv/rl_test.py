import unittest

import numpy as np
from numpy.testing import assert_array_equal

from ...game import Action, Call, Play
from ...players import Player
from ...simulate import GameRecord
from .bot import ConvBot, prepare_training_data


# Test RL functionality on the bot
class RLTest(unittest.TestCase):
    def setUp(self):
        self.dummy_model = object()
        self.bot = ConvBot(self.dummy_model, metadata={})
        self.zero_state = np.zeros((self.bot.encoder.GAME_LENGTH, self.bot.encoder.DIM))

    def _state(self, x):
        return x * np.ones_like(self.zero_state)

    def test_encode_episode_made_contract(self):
        game_result = GameRecord(
            game=None,
            points_ns=1,
            points_ew=0,
            declarer=Player.north,
            contract_made=True,
            contract_level=2
        )

        decisions = [
            {
                'state': self._state(1),
                'action': Action.make(Call.of('2S')),
                'expected_value': 0
            },
            {
                'state': self._state(2),
                'action': Action.make(Play.of('AS')),
                'expected_value': 1
            },
        ]

        episode = self.bot.encode_episode(
            game_result,
            Player.north,
            decisions,
            reward='contracts'
        )

        # Check:
        # states: just concat all the input states
        self.assertEqual(2, episode['states'].shape[0])
        assert_array_equal(self._state(1), episode['states'][0])
        assert_array_equal(self._state(2), episode['states'][1])
        # call_actions, calls_made
        assert_array_equal([1, 0], episode['calls_made'])
        # play_actions, plays_made
        assert_array_equal([0, 1], episode['plays_made'])
        # advantages
        assert_array_equal([2, 1], episode['advantages'])
        # rewards: all 2 in contract scoring method
        assert_array_equal([2, 2], episode['rewards'])

    def test_prepare_training_data(self):
        episode = {
            'states': np.array([
                [2, 2, 2, 2],
                [3, 3, 3, 3],
            ]),
            'call_actions': np.array([
                [1, 0, 0],
                [0, 0, 1],
            ]),
            'calls_made': np.array([1, 0]),
            'play_actions': np.array([
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ]),
            'plays_made': np.array([0, 1]),
            'rewards': [-2, -2],
            'advantages': [0, 0],
        }
        X_state, y_call, y_play, y_value = prepare_training_data(
            [episode],
            reinforce_only=False,
            use_advantage=False
        )

        assert_array_equal(episode['states'], X_state)
        # Call output:
        # First row, we should reinforce the made decision
        assert_array_equal([-2, 0, 0], y_call[0])
        # Second row is the "not my turn sentinel" -- does not get weighted
        assert_array_equal([0, 0, 1], y_call[1])

        # Play output:
        # First row is "not my turn" sentinel
        assert_array_equal([0, 0, 0, 1], y_play[0])
        # Second row should be reinforced according to reward
        assert_array_equal([-2, 0, 0, 0], y_play[1])

        # Value output: just the rewards, but reshaped
        assert_array_equal(np.array([[-2], [-2]]), y_value)
