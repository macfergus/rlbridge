import numpy as np
from keras.optimizers import SGD

from ...game import Action, Phase
from ...players import Player
from ...rl import Decision, Episode, concat_episodes
from ..base import Bot, UnrecognizedOptionError
from .encoder import Encoder
from .losses import policy_loss

__all__ = [
    'ConvBot',
]


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


def get_reward(game_result, perspective):
    if perspective in (Player.north, Player.south):
        reward = game_result.points_ns - game_result.points_ew
    else:
        reward = game_result.points_ew - game_result.points_ns
    if reward == 0:
        # This can only happen if there is no contract. Impose a
        # small penalty to get out of the equilibrium where no one
        # tries to bid.
        reward = -100
    reward /= 1000
    return reward


def prepare_training_data(episodes):
    experience = concat_episodes(episodes)
    states = experience['states']
    call_actions = experience['call_actions']
    play_actions = experience['play_actions']
    rewards = experience['rewards']
    advantages = np.clip(experience['advantages'].reshape((-1, 1)), -2, 2)

    calls_made = experience['calls_made'].reshape((-1, 1))
    plays_made = experience['plays_made'].reshape((-1, 1))

    # On steps where the agent made a decision, we want to reinforce
    # or deinforce the action according to the reward. But on other
    # steps, we can just target the "not my turn" sentinel directly.
    call_cols = call_actions.shape[-1]
    call_mask = np.repeat(calls_made, call_cols, axis=1)
    call_actions = np.where(call_mask, advantages * call_actions, call_actions)
    play_cols = play_actions.shape[-1]
    play_mask = np.repeat(plays_made, play_cols, axis=1)
    play_actions = np.where(play_mask, advantages * play_actions, play_actions)

    return states, call_actions, play_actions, rewards


class ConvBot(Bot):
    def __init__(self, model, metadata):
        super().__init__(metadata)
        self.encoder = Encoder()
        self.model = model
        self.temperature = 1.0

        self._max_contract = 7

    def identify(self):
        return '{}_{:07d}'.format(
            self.name(),
            self.metadata.get('num_games', 0)
        )

    def set_option(self, key, value):
        if key == 'max_contract':
            self._max_contract = int(value)
        elif key == 'temperature':
            self.temperature = float(value)
        else:
            raise UnrecognizedOptionError(key)

    def add_games(self, num_games):
        if 'num_games' not in self.metadata:
            self.metadata['num_games'] = 0
        self.metadata['num_games'] += num_games

    def select_action(self, state, recorder=None):
        game_record = self.encoder.encode_full_game(state, state.next_player)
        X = game_record.reshape((-1,) + self.encoder.input_shape())
        calls, plays, values = self.model.predict(X)
        chosen_call = None
        chosen_play = None
        if state.phase == Phase.auction:
            call_p = calls.reshape((-1,))[1:]
            for call_index in sample(call_p, self.temperature):
                call = self.encoder.decode_call_index(call_index)
                if call.is_bid and call.bid.tricks > self._max_contract:
                    continue
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
        if recorder is not None:
            recorder.record_decision(
                Decision(
                    state=game_record,
                    action=chosen_action,
                    expected_value=values[0]
                ),
                state.next_player
            )
        return chosen_action

    def encode_episode(self, game_result, perspective, decisions):
        reward = get_reward(game_result, perspective)
        n = len(decisions)
        states = np.zeros((n, self.encoder.GAME_LENGTH, self.encoder.DIM))
        calls = np.zeros((n, self.encoder.DIM_CALL_ACTION))
        plays = np.zeros((n, self.encoder.DIM_PLAY_ACTION))
        calls_made = np.zeros(n)
        plays_made = np.zeros(n)
        rewards = reward * np.ones(n)
        advantages = np.zeros(n)

        for i, decision in enumerate(decisions):
            states[i] = decision['state']
            action = decision['action']
            if action.is_call:
                calls[i] = self.encoder.encode_call_action(action.call)
                plays[i] = self.encoder.encode_play_action(None)
                calls_made[i] = 1
            else:
                plays[i] = self.encoder.encode_play_action(action.play)
                calls[i] = self.encoder.encode_call_action(None)
                plays_made[i] = 1
            advantages[i] = reward - decision['expected_value']
        return Episode(
            states=states,
            call_actions=calls,
            play_actions=plays,
            calls_made=calls_made,
            plays_made=plays_made,
            advantages=advantages,
            rewards=rewards
        )

    def encode_pretraining(self, game_record, perspective):
        reward = get_reward(game_record, perspective)
        game = game_record.game
        full_state = self.encoder.encode_full_game(game, perspective)
        n = game.num_states
        states = np.zeros((n, self.encoder.GAME_LENGTH, self.encoder.DIM))
        calls = np.zeros((n, self.encoder.DIM_CALL_ACTION))
        plays = np.zeros((n, self.encoder.DIM_PLAY_ACTION))
        values = np.zeros(n)
        i = 0
        for state, _ in replay_game(game):
            if state.next_decider == perspective:
                states[i] = self.encoder.encode_full_game(state, perspective)
                values[i] = reward
                if state.phase == Phase.auction:
                    calls[i] = self.encoder.encode_legal_calls(state)
                    calls[i] /= np.sum(calls[i])
                    plays[i] = self.encoder.encode_play_action(None)
                else:
                    plays[i] = self.encoder.encode_legal_plays(state)
                    plays[i] /= np.sum(plays[i])
                    calls[i] = self.encoder.encode_call_action(None)
                i += 1
        # Discount the rewards
        values = values[:i].copy()
        seq = i - np.arange(i)
        values = values * (1. / seq)
        return (
            states[:i].copy(),
            calls[:i].copy(),
            plays[:i].copy(),
            values
        )

    def pretrain(self, x_state, y_call, y_play, y_value, callback=None):
        self.model.fit(
            x_state,
            [y_call, y_play, y_value],
            verbose=0,
            batch_size=1024,
            callbacks=[callback]
        )

    def train(self, episodes):
        self.model.compile(
            optimizer=SGD(lr=0.01, clipnorm=0.5),
            loss=[
                policy_loss,
                policy_loss,
                'mse'
            ],
            loss_weights=[
                1.0,
                1.0,
                0.05,
            ]
        )
        x_state, y_call, y_play, y_value = prepare_training_data(episodes)
        history = self.model.fit(
            x_state,
            [y_call, y_play, y_value],
            epochs=1,
            verbose=0
        )
        return {
            'loss': history.history['loss'][0],
            'call_loss': history.history['dense_2_loss'][0],
            'play_loss': history.history['dense_4_loss'][0],
            'value_loss': history.history['dense_6_loss'][0],
        }
