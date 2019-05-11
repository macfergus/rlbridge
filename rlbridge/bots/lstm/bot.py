import numpy as np

from ... import nputil
from ...game import Action, Phase
from ...players import Player
from ...rl import Episode, concat_episodes
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


def prepare_training_data(episodes):
    experience = concat_episodes(episodes)
    states = experience['states']
    call_actions = experience['call_actions']
    play_actions = experience['play_actions']
    rewards = experience['rewards'].reshape((-1, 1))

    calls_made = experience['calls_made'].reshape((-1, 1))
    plays_made = experience['plays_made'].reshape((-1, 1))

    # On steps where the agent made a decision, we want to reinforce
    # or deinforce the action according to the reward. But on other
    # steps, we can just target the "not my turn" sentinel directly.
    call_cols = call_actions.shape[-1]
    call_mask = np.repeat(calls_made, call_cols, axis=1)
    call_actions = np.where(call_mask, rewards * call_actions, call_actions)
    play_cols = play_actions.shape[-1]
    play_mask = np.repeat(plays_made, play_cols, axis=1)
    play_actions = np.where(play_mask, rewards * play_actions, play_actions)

    return states, call_actions, play_actions


class LSTMBot(Bot):
    def __init__(self, model, metadata):
        super().__init__(metadata)
        self.encoder = Encoder()
        self.model = model
        self.temperature = 1.0

    def select_action(self, state):
        game_record = self.encoder.encode_full_game(state, state.next_player)
        n = game_record.shape[0]
        states = np.zeros((1, MAX_GAME, self.encoder.DIM))
        states[0, MAX_GAME - n:] = game_record
        calls, plays = self.model.predict(states)
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
        n = game.num_states
        states = np.zeros((n, MAX_GAME, self.encoder.DIM))
        calls = np.zeros((n, self.encoder.DIM_CALL_ACTION))
        plays = np.zeros((n, self.encoder.DIM_PLAY_ACTION))
        calls_made = np.zeros(n)
        plays_made = np.zeros(n)
        rewards = reward * np.ones(n)

        full_state = self.encoder.encode_full_game(game, perspective)
        i = 0
        for game_i, (state, action) in enumerate(replay_game(game)):
            if state.next_decider != perspective:
                continue
            seq_len = game_i + 2
            states[i, MAX_GAME - seq_len:] = full_state[:seq_len]
            if action.is_call:
                calls[i] = self.encoder.encode_call_action(action.call)
                plays[i] = self.encoder.encode_play_action(None)
                calls_made[i] = 1
            else:
                plays[i] = self.encoder.encode_play_action(action.play)
                calls[i] = self.encoder.encode_call_action(None)
                plays_made[i] = 1
            i += 1
        nputil.shrink_axis(states, i)
        nputil.shrink_axis(calls, i)
        nputil.shrink_axis(plays, i)
        nputil.shrink_axis(calls_made, i)
        nputil.shrink_axis(plays_made, i)
        nputil.shrink_axis(rewards, i)
        return Episode(
            states=states,
            call_actions=calls,
            play_actions=plays,
            calls_made=calls_made,
            plays_made=plays_made,
            rewards=rewards
        )

    def encode_pretraining(self, game_record, perspective):
        game = game_record.game
        full_state = self.encoder.encode_full_game(game, perspective)
        n = game.num_states
        states = np.zeros((n, MAX_GAME, self.encoder.DIM))
        calls = np.zeros((n, self.encoder.DIM_CALL_ACTION))
        plays = np.zeros((n, self.encoder.DIM_PLAY_ACTION))
        for i, (state, _) in enumerate(replay_game(game)):
            seq_len = i + 2
            states[i, MAX_GAME - seq_len:] = full_state[:seq_len]
            if state.next_decider == perspective:
                if state.phase == Phase.auction:
                    calls[i] = self.encoder.encode_legal_calls(state)
                    calls[i] /= np.sum(calls[i])
                    plays[i] = self.encoder.encode_play_action(None)
                else:
                    plays[i] = self.encoder.encode_legal_plays(state)
                    plays[i] /= np.sum(plays[i])
                    calls[i] = self.encoder.encode_call_action(None)
            else:
                # This turn belongs to a different player.
                calls[i] = self.encoder.encode_call_action(None)
                plays[i] = self.encoder.encode_play_action(None)
        return states, calls, plays

    def pretrain(self, x_state, y_call, y_play):
        self.model.fit(x_state, [y_call, y_play])

    def train(self, episodes):
        x_state, y_call, y_play = prepare_training_data(episodes)
        self.model.fit(
            x_state,
            [y_call, y_play]
        )
