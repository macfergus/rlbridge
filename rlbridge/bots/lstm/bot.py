import numpy as np

from ...game import Action, Phase
from ...players import Player
from ...rl import Episode
from ..base import Bot
from .encoder import Encoder

__all__ = [
    'LSTMBot',
]


# The longest possible auction has 319 calls. But that is very
# unrealistic. Here we impose an arbitrary cap of 60 calls.
MAX_GAME = 1 + 60 + 52


def sample(p, temperature):
    min_temp = 0.001
    if temperature < min_temp:
        return np.argsort(p)[::-1]
    p = np.power(p, 1.0 / temperature)
    p /= np.sum(p)
    n = p.shape[0]
    return np.random.choice(n, size=n, replace=False, p=p)


def replay_game(state):
    states = []
    actions = []
    while state is not None:
        states.append(state)
        if state.prev_action is not None:
            actions.append(state.prev_action)
        state = state.prev_state
    states.reverse()
    actions.reverse()
    return zip(states[:-1], actions)


class LSTMBot(Bot):
    def __init__(self, model, metadata):
        super().__init__(metadata)
        self.encoder = Encoder()
        self.model = model
        self.temperature = 1.0

    def select_action(self, state):
        game_record = self.encoder.encode_full_game(state, state.next_player)
        # This will actually select moves for every move of the game up
        # to this point. Just grab the last timestep.
        # Shape is (timestep, batch_index, action)
        calls, plays = self.model.predict(game_record)
        chosen_call = None
        chosen_play = None
        if state.phase == Phase.auction:
            call_p = calls[-1].reshape((-1,))[1:]
            for call_index in sample(call_p, self.temperature):
                call = self.encoder.decode_call_index(call_index)
                if state.auction.is_legal(call):
                    chosen_call = call
                    break
        else:
            # play
            play_p = plays[-1].reshape((-1,))[1:]
            for play_index in sample(play_p, self.temperature):
                play = self.encoder.decode_play_index(play_index)
                if state.playstate.is_legal(play):
                    chosen_play = play
                    break
        if chosen_call is not None:
            chosen_action = Action.make_call(chosen_call)
        else:
            chosen_action = Action.make_play(chosen_play)
        return chosen_action

    def encode_episode(self, game_result, perspective):
        if perspective in (Player.north, Player.south):
            reward = game_result.points_ns - game_result.points_ew
        else:
            reward = game_result.points_ew - game_result.points_ns
        reward /= 100

        game = game_result.game
        states = self.encoder.encode_full_game(game, perspective)
        n_states = states.shape[0]
        assert n_states < MAX_GAME
        padded_size = (MAX_GAME,) + states.shape[1:]
        states.resize(padded_size)
        calls = np.zeros((MAX_GAME, self.encoder.DIM_CALL_ACTION))
        plays = np.zeros((MAX_GAME, self.encoder.DIM_PLAY_ACTION))
        # Index 0 is reserved for the new game sentinel. No action
        # follows.
        calls[0] = self.encoder.encode_call_action(None)
        plays[0] = self.encoder.encode_play_action(None)
        for game_i, (state, action) in enumerate(replay_game(game)):
            i = game_i + 1
            if state.next_decider == perspective:
                if action.is_call:
                    calls[i] = self.encoder.encode_call_action(action.call)
                    plays[i] = self.encoder.encode_play_action(None)
                else:
                    plays[i] = self.encoder.encode_play_action(action.play)
                    calls[i] = self.encoder.encode_call_action(None)
            else:
                # This turn belongs to a different player.
                calls[i] = self.encoder.encode_call_action(None)
                plays[i] = self.encoder.encode_play_action(None)
        # Need to fill in the rest of the softmax outputs.
        while i < MAX_GAME:
            calls[i] = self.encoder.encode_call_action(None)
            plays[i] = self.encoder.encode_play_action(None)
            i += 1
        return Episode(
            states=states,
            call_actions=calls,
            play_actions=plays,
            reward=reward
        )
