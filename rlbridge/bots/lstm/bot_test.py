import unittest

import numpy as np

from .bot import prepare_training_data
from ...rl import Episode


class PrepareTrainingTest(unittest.TestCase):
    def test_prepare_training_data(self):
        ep1 = Episode(
            states=np.ones((2, 3, 5)),
            call_actions=np.array([
                [1, 0, 0],
                [0, 0, 1],
            ]),
            play_actions=np.array([
                [0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
            ]),
            calls_made=np.array([1, 0]),
            plays_made=np.array([0, 1]),
            rewards=np.array([5, 5]),
        )
        ep2 = Episode(
            states=np.ones((3, 3, 5)),
            call_actions=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            play_actions=np.array([
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
            ]),
            calls_made=np.array([1, 1, 0]),
            plays_made=np.array([0, 0, 1]),
            rewards=np.array([-3, -3, -3]),
        )

        x_state, y_call, y_play = prepare_training_data([ep1, ep2])

        np.testing.assert_array_equal(
            x_state,
            np.ones((5, 3, 5))
        )
        np.testing.assert_array_equal(
            y_call,
            np.array([
                [ 5,  0, 0],
                [ 0,  0, 1],
                [-3,  0, 0],
                [ 0, -3, 0],
                [ 0,  0, 1],
            ])
        )
        np.testing.assert_array_equal(
            y_play,
            np.array([
                [0, 0, 0,  0, 1],
                [0, 5, 0,  0, 0],
                [0, 0, 0,  0, 1],
                [0, 0, 0,  0, 1],
                [0, 0, 0, -3, 0],
            ])
        )
