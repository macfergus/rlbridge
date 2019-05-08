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

        self._prefix = None

    def reset(self):
        self.model.reset_states()
        self._prefix = None

    def _match_prefix(self, array):
        if self._prefix is None:
            return array
        n_prefix = self._prefix.shape[0]
        n_array = array.shape[0]
        if n_prefix >= n_array:
            # If the stored state is longer, it can't be a prefix.
            return array
        if np.array_equal(self._prefix, array[:n_prefix]):
            return array[n_prefix:]
        return array

    def select_action(self, state):
        game_record = self.encoder.encode_full_game(state, state.next_player)
        # If this agent has been playing the game, the model's internal
        # state should match some point in this game record. In that
        # case, we can just feed the remainder and save some computation.
        suffix = self._match_prefix(game_record)
        n_orig = game_record.shape[0]
        n_new = suffix.shape[0]
        if n_new == n_orig:
            # This is not a continuation of the previous game, so we
            # should reset the inner state.
            self.reset()
        for i in range(n_new):
            tmp = suffix[i].reshape((1, 1) + suffix[i].shape)
            calls, plays = self.model.predict(tmp)
        self._prefix = game_record
        chosen_call = None
        chosen_play = None
        if state.phase == Phase.auction:
            call_p = calls.reshape((-1,))[1:]
            for call_index in sample(call_p, self.temperature):
                call = self.encoder.decode_call_index(call_index)
                if state.auction.is_legal(call):
                    chosen_call = call
                    break
        else:
            # play
            play_p = plays.reshape((-1,))[1:]
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
        if reward == 0:
            # This can only happen if there is no contract. Impose a
            # small penalty to get out of the equilibrium where no one
            # tries to bid.
            reward = -500
        reward /= 200

        game = game_result.game
        states = self.encoder.encode_full_game(game, perspective)
        n_states = states.shape[0]
        calls = np.zeros((n_states, self.encoder.DIM_CALL_ACTION))
        plays = np.zeros((n_states, self.encoder.DIM_PLAY_ACTION))
        calls_made = np.zeros(n_states)
        plays_made = np.zeros(n_states)
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
                    calls_made[i] = 1
                else:
                    plays[i] = self.encoder.encode_play_action(action.play)
                    calls[i] = self.encoder.encode_call_action(None)
                    plays_made[i] = 1
            else:
                # This turn belongs to a different player.
                calls[i] = self.encoder.encode_call_action(None)
                plays[i] = self.encoder.encode_play_action(None)
        return Episode(
            states=states,
            call_actions=calls,
            play_actions=plays,
            calls_made=calls_made,
            plays_made=plays_made,
            reward=reward
        )

    def train_episode(self, episode):
        n_states = episode['states'].shape[0]
        states = np.array(episode['states'])
        call_actions = np.array(episode['call_actions'])
        play_actions = np.array(episode['play_actions'])
        reward = episode['reward']
        # On steps where the agent made a decision, we want to reinforce
        # or deinforce the action according to the reward. But on other
        # steps, we can just target the "not my turn" sentinel directly.
        call_cols = call_actions.shape[-1]
        call_mask = np.repeat(np.reshape(episode['calls_made'], (-1, 1)), call_cols, axis=1)
        call_actions = np.where(call_mask, reward * call_actions, call_actions)
        play_cols = play_actions.shape[-1]
        play_mask = np.repeat(np.reshape(episode['plays_made'], (-1, 1)), play_cols, axis=1)
        play_actions = np.where(play_mask, reward * play_actions, play_actions)

        self.reset()
        for i in range(n_states):
            x_state = states[i].reshape((1,1) + states[i].shape)
            y_call = call_actions[i].reshape((1,) + call_actions[i].shape)
            y_play = play_actions[i].reshape((1,) + play_actions[i].shape)
            self.model.train_on_batch(x_state, [y_call, y_play])
