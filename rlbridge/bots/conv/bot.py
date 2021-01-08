import numpy as np
from keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy

from ...game import Action, Bid, Call, Phase
from ...players import Player
from ...rl import Decision, Episode, concat_episodes
from ..base import Bot, UnrecognizedOptionError
from .encoder import Encoder

__all__ = [
    'ConvBot',
]


def softmax(x):
    x = np.clip(x, -20, 20)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def sample(logits, temperature):
    p = softmax(logits)
    eps = 1e-4
    min_temp = 0.001
    if temperature < min_temp:
        return np.argsort(p)[::-1]
    p = np.clip(p, eps, 1 - eps)
    p = np.power(p, 1.0 / temperature)
    p /= np.sum(p)
    p = np.clip(p, eps, 1 - eps)
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


def get_reward_points(game_result, perspective):
    if perspective in (Player.north, Player.south):
        reward = game_result.points_ns - game_result.points_ew
    else:
        reward = game_result.points_ew - game_result.points_ns
    if reward == 0:
        # This can only happen if there is no contract. Impose a
        # giant penalty to get out of the equilibrium where no one
        # tries to bid.
        reward = -100
    return reward


def get_reward_tricks(game_result, perspective):
    if perspective in (Player.north, Player.south):
        reward = 50.0 * game_result.tricks_ns
    else:
        reward = 50.0 * game_result.tricks_ew
    return reward


def get_reward_contracts(game_result, perspective):
    contract_made = game_result.contract_made
    is_declarer = (
        game_result.declarer in [perspective, perspective.partner]
    )
    is_defender = (
        game_result.declarer in [perspective.lho(), perspective.rho()]
    )
    if contract_made and is_declarer:
        # Big reward for making contracts
        return float(game_result.contract.tricks)
    # Nothing for going down, or for no contract
    return 0.0


def prepare_training_data(episodes, reinforce_only=False, use_advantage=True):
    experience = concat_episodes(episodes)
    states = experience['states']
    call_actions = experience['call_actions']
    play_actions = experience['play_actions']
    rewards = experience['rewards'].reshape((-1, 1))
    advantages = experience['advantages'].reshape((-1, 1))

    calls_made = experience['calls_made'].reshape((-1, 1))
    plays_made = experience['plays_made'].reshape((-1, 1))
    contracts = experience['contracts']
    tricks_won = experience['tricks_won'].reshape((-1, 1))
    contract_made = experience['contract_made'].reshape((-1, 1))

    weight = advantages if use_advantage else rewards

    if not reinforce_only:
        # On steps where the agent made a decision, we want to reinforce
        # or deinforce the action according to the reward. But on other
        # steps, we can just target the "not my turn" sentinel directly.
        call_cols = call_actions.shape[-1]
        call_mask = np.repeat(calls_made, call_cols, axis=1)
        call_actions = np.where(
            call_mask, weight * call_actions, call_actions)
        play_cols = play_actions.shape[-1]
        play_mask = np.repeat(plays_made, play_cols, axis=1)
        play_actions = np.where(
            play_mask, weight * play_actions, play_actions)

    return {
        'X': states,
        'y_call': call_actions,
        'y_play': play_actions,
        'y_value': rewards,
        'y_contract': contracts,
        'y_tricks': tricks_won,
        'y_contract_made': contract_made,
    }


class ConvBot(Bot):
    def __init__(self, encoder, model, metadata):
        super().__init__(metadata)
        self.encoder = encoder
        self.model = model
        self.temperature = 1.0

        self._max_contract = 7

        self._compiled_lr = -1
        self._compiled_for_pretraining = False

        self._force_contract = None

        self.last_output = {}

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
        elif key == 'force_contract':
            self._force_contract = value
        else:
            raise UnrecognizedOptionError(key)

    def add_games(self, num_games):
        if 'num_games' not in self.metadata:
            self.metadata['num_games'] = 0
        self.metadata['num_games'] += num_games

    def select_action(self, state, recorder=None):
        game_record = self.encoder.encode_full_game(state, state.next_player)
        X = game_record.reshape((-1,) + self.encoder.input_shape())
        outputs = self.model.predict(X)
        all_outputs = {}
        for name, output_val in zip(self.model.output_names, outputs):
            all_outputs[name] = output_val[0]
        self.last_outputs = all_outputs
        calls, plays, values = outputs[:3]
        chosen_call = None
        chosen_play = None
        if state.phase == Phase.auction:
            if self._force_contract is not None:
                tricks, denom, declarer = self._force_contract
                if state.next_decider == declarer:
                    chosen_call = Call.make_bid(Bid(denom, tricks))
                else:
                    chosen_call = Call.pass_turn()
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

    def encode_episode(
            self, game_result, perspective, decisions, contract_bonus=0,
            trick_weight=0.0,
            reward_scale='linear'
    ):
        reward_amt = (
            trick_weight * get_reward_tricks(game_result, perspective) +
            (1 - trick_weight) * get_reward_points(game_result, perspective) +
            contract_bonus * get_reward_contracts(
                game_result, perspective
            )
        )
        if reward_scale == 'log':
            sign = np.sign(reward_amt)
            reward_amt = sign * np.log(np.abs(reward_amt) + 1)
        elif reward_scale == 'linear':
            reward_amt = reward_amt / 200.0
        else:
            raise ValueError(reward_scale)

        n = len(decisions)
        states = np.zeros((n,) + self.encoder.input_shape())
        calls = np.zeros((n, self.encoder.DIM_CALL_ACTION))
        plays = np.zeros((n, self.encoder.DIM_PLAY_ACTION))
        contracts = np.tile(
            self.encoder.encode_contract(game_result.contract),
            (n, 1)
        )
        calls_made = np.zeros(n)
        plays_made = np.zeros(n)
        rewards = reward_amt * np.ones(n)
        num_tricks_made = (
            game_result.tricks_ns if perspective in (Player.north, Player.south)
            else game_result.tricks_ew
        )
        tricks_won = (num_tricks_made / 13.0) * np.ones(n)
        contract_made = float(game_result.contract_made) * np.ones(n)
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
            advantages[i] = reward_amt - decision['expected_value']
        return Episode(
            states=states,
            call_actions=calls,
            play_actions=plays,
            calls_made=calls_made,
            plays_made=plays_made,
            advantages=advantages,
            rewards=rewards,
            contracts=contracts,
            tricks_won=tricks_won,
            contract_made=contract_made
        )

    def encode_pretraining(self, game_record, perspective):
        reward = get_reward_points(game_record, perspective)
        game = game_record.game
        n = game.num_states
        states = np.zeros((n,) + self.encoder.input_shape())
        calls = np.zeros((n, self.encoder.DIM_CALL_ACTION))
        plays = np.zeros((n, self.encoder.DIM_PLAY_ACTION))
        values = np.zeros(n)
        i = 0
        for state, action in replay_game(game):
            if state.next_decider == perspective:
                states[i] = self.encoder.encode_full_game(state, perspective)
                values[i] = reward
                if state.phase == Phase.auction:
                    calls[i] = self.encoder.encode_call_action(action.call)
                    plays[i] = self.encoder.encode_play_action(None)
                else:
                    plays[i] = self.encoder.encode_play_action(action.play)
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
        kwargs = {}
        if callback is not None:
            kwargs['callbacks'] = [callback]
        if not self._compiled_for_pretraining:
            self.model.compile(
                optimizer='adam',
                loss=[
                    CategoricalCrossentropy(from_logits=True),
                    CategoricalCrossentropy(from_logits=True),
                    'mse'
                ],
                loss_weights=[
                    1.0,
                    1.0,
                    0.1,
                ]
            )
            self._compiled_for_pretraining = True
        return self.model.fit(
            x_state,
            [y_call, y_play, y_value],
            verbose=0,
            batch_size=256,
            **kwargs
        )

    def train(
            self,
            episodes,
            lr=0.1,
            call_weight=1.0,
            play_weight=1.0,
            value_weight=0.1,
            reinforce_only=False,
            use_advantage=True
    ):
        has_contract_output = 'contract_output' in self.model.output_names
        has_tricks_output = 'tricks_output' in self.model.output_names
        has_contract_made_output = (
            'contract_made_output' in self.model.output_names
        )

        losses = {
            'call_output': CategoricalCrossentropy(from_logits=True),
            'play_output': CategoricalCrossentropy(from_logits=True),
            'value_output': 'mse',
        }
        loss_weights = {
            'call_output': call_weight,
            'play_output': play_weight,
            'value_output': value_weight,
        }
        if has_contract_output:
            losses['contract_output'] = 'mse'
            loss_weights['contract_output'] = 0.5
        if has_tricks_output:
            losses['tricks_output'] = 'mse'
            loss_weights['tricks_output'] = 0.5
        if has_contract_made_output:
            losses['contract_made_output'] = 'binary_crossentropy'
            loss_weights['contract_made_output'] = 0.5
        self.model.compile(
            optimizer=SGD(lr=lr),
            loss=losses,
            loss_weights=loss_weights
        )

        data = prepare_training_data(
            episodes,
            reinforce_only=reinforce_only,
            use_advantage=use_advantage
        )
        y = {
            'call_output': data['y_call'],
            'play_output': data['y_play'],
            'value_output': data['y_value'],
        }
        if has_contract_output:
            y['contract_output'] = data['y_contract']
        if has_tricks_output:
            y['tricks_output'] = data['y_tricks']
        if has_contract_made_output:
            y['contract_made_output'] = data['y_contract_made']
        history = self.model.fit(
            data['X'],
            y,
            batch_size=256,
            epochs=1,
            verbose=0
        )
        res = {
            'loss': history.history['loss'][0],
            'call_loss': history.history['call_output_loss'][0],
            'play_loss': history.history['play_output_loss'][0],
            'value_loss': history.history['value_output_loss'][0],
        }
        if has_contract_output:
            res['contract_loss'] = history.history['contract_output_loss'][0]
        if has_tricks_output:
            res['tricks_loss'] = history.history['tricks_output_loss'][0]
        if has_contract_made_output:
            res['contract_made_loss'] = (
                history.history['contract_made_output_loss'][0]
            )
        return res
