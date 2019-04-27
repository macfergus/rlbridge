import numpy as np

from ...game import Action, Phase
from .encoder import Encoder
from .model import construct_model

__all__ = [
    'LSTMBot',
]


def sample(p, temperature):
    min_temp = 0.001
    if temperature < min_temp:
        return np.argsort(p)[::-1]
    p = np.power(p, 1.0 / temperature)
    p /= np.sum(p)
    n = p.shape[0]
    return np.random.choice(n, size=n, replace=False, p=p)


class LSTMBot:
    def __init__(self):
        self.encoder = Encoder()
        self.model = construct_model(self.encoder.input_shape())
        self.temperature = 1.0

    def select_action(self, state):
        game_record = self.encoder.encode_full_game(state, state.next_player)
        # This will actually select moves for every move of the game up
        # to this point. Just grab the last timestep.
        # Shape is (timestamp, batch_index, action)
        calls, plays = self.model.predict(game_record)
        if state.phase == Phase.auction:
            call_p = calls[-1].reshape((-1,))[1:]
            for call_index in sample(call_p, self.temperature):
                call = self.encoder.decode_call_index(call_index)
                if state.auction.is_legal(call):
                    chosen_action = Action.make_call(call)
                    break
        else:
            # play
            play_p = plays[-1].reshape((-1,))[1:]
            for play_index in sample(play_p, self.temperature):
                play = self.encoder.decode_play_index(play_index)
                if state.playstate.is_legal(play):
                    chosen_action = Action.make_play(play)
                    break
        return chosen_action
