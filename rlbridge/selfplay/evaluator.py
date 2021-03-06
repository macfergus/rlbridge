import os
import random
import time

import numpy as np

from .. import bots, elo, kerasutil
from ..evalstore import Match
from ..mputil import Loopable, LoopingProcess
from ..players import Player
from ..simulate import simulate_game


class NotEnoughBots(Exception):
    pass


class EvaluatorImpl(Loopable):
    def __init__(self, workspace, logger, config):
        self._workspace = workspace
        self._logger = logger
        self._config = config
        kerasutil.set_tf_options(disable_gpu=True)

        self._game_queue = []

    def _save_result(
            self, bot1, bot2, num_hands, bot1_points, bot2_points,
            bot1_contracts, bot2_contracts
    ):
        self._workspace.eval_store.store_match(Match(
            bot1=bot1.identify(),
            bot2=bot2.identify(),
            num_hands=int(num_hands),
            bot1_points=int(bot1_points),
            bot2_points=int(bot2_points),
            bot1_contracts=int(bot1_contracts),
            bot2_contracts=int(bot2_contracts)
        ))

    def _get_bot_weights(self, bot_names):
        weights = {bot_name: 0 for bot_name in bot_names}
        matches = self._workspace.eval_store.get_eval_matches()
        for match in matches:
            weights[match.bot1] += 1
            weights[match.bot2] += 1
        return weights

    def _get_elo(self):
        return self._workspace.eval_store.get_elo_ratings()

    def _extend_queue_by_elo(self, bot_names):
        weights = self._get_bot_weights(bot_names)
        weight_array = np.array([weights[name] for name in bot_names])
        idx = np.argmin(weight_array)

        ratings = self._get_elo()
        if not ratings:
            bot1, bot2 = random.sample(bot_names, 2)
            self._game_queue.append((bot1, bot2))
            self._logger.log(f'Evaluating {bot1} vs {bot2}')
            return

        base_rating = ratings.pop(bot_names[idx], 1000)
        self._logger.log(f'Evaluating {bot_names[idx]} elo {base_rating}')
        elo_diff = [
            (abs(rating - base_rating), bot)
            for bot, rating in ratings.items()
            if bot != bot_names[idx]
        ]
        elo_diff.sort()
        diff_weights = []
        for i, (_diff, bot) in enumerate(elo_diff):
            diff_weights.append(1. / (i + 1))
        diff_weights = np.array(diff_weights)
        diff_weights /= np.sum(diff_weights)

        if len(elo_diff) > 10:
            opponents = np.random.choice(
                len(elo_diff),
                size=10,
                replace=False,
                p=diff_weights
            )
        else:
            opponents = list(range(len(elo_diff)))
        for j in opponents:
            self._game_queue.append((bot_names[idx], elo_diff[j][1]))

    def _select_bots(self):
        if not self._game_queue:
            bot_names = []
            for fname in os.listdir(self._workspace.eval_dir):
                bot_names.append(fname)

            if len(bot_names) < 2:
                raise NotEnoughBots()

            if len(bot_names) == 2:
                self._game_queue.append((bot_names[0], bot_names[1]))
            else:
                self._extend_queue_by_elo(bot_names)

        bot1_fname, bot2_fname = self._game_queue.pop(0)
        path1 = os.path.join(self._workspace.eval_dir, bot1_fname)
        path2 = os.path.join(self._workspace.eval_dir, bot2_fname)
        return bots.load_bot(path1), bots.load_bot(path2)

    def run_once(self):
        try:
            bot1, bot2 = self._select_bots()
        except NotEnoughBots:
            # Wait for more to get trained
            time.sleep(2)
            return
        bot1.set_option('temperature', self._config['temperature'])
        bot2.set_option('temperature', self._config['temperature'])

        bot1_points = 0
        bot2_points = 0
        bot1_contracts = 0
        bot2_contracts = 0
        bot1_side = 'ns'
        num_hands = 0
        discarded_hands = 0
        while num_hands < self._config['num_hands_per_match']:
            if bot1_side == 'ns':
                ns_bot = bot1
                ew_bot = bot2
            else:
                ew_bot = bot1
                ns_bot = bot2
            result = simulate_game(ns_bot, ew_bot)
            if result.declarer is None:
                discarded_hands += 1
                continue
            num_hands += 1
            bot1_declared = (
                (
                    bot1_side == 'ns' and
                    result.declarer in (Player.north, Player.south)
                ) or (
                    bot1_side == 'ew' and
                    result.declarer in (Player.east, Player.west)
                )
            )
            bot2_declared = (
                (
                    bot1_side == 'ew' and
                    result.declarer in (Player.north, Player.south)
                ) or (
                    bot1_side == 'ns' and
                    result.declarer in (Player.east, Player.west)
                )
            )
            if bot1_side == 'ns':
                bot1_points += result.points_ns
                bot2_points += result.points_ew
            else:
                bot1_points += result.points_ew
                bot2_points += result.points_ns
            if bot1_declared and result.contract_made:
                bot1_contracts += 1
            if bot2_declared and result.contract_made:
                bot2_contracts += 1
            bot1_side = 'ew' if bot1_side == 'ns' else 'ew'
        if discarded_hands > 0:
            self._logger.log(f'Discarded {discarded_hands} hands')

        self._save_result(
            bot1, bot2,
            num_hands=num_hands,
            bot1_points=bot1_points,
            bot2_points=bot2_points,
            bot1_contracts=bot1_contracts,
            bot2_contracts=bot2_contracts
        )


class Evaluator:
    def __init__(self, workspace, config, logger):
        self._workspace = workspace
        self._config = config
        self._logger = logger
        self._proc = LoopingProcess(
            'evaluator',
            EvaluatorImpl,
            kwargs={
                'workspace': self._workspace,
                'logger': self._logger,
                'config': self._config['evaluation'],
            },
            restart=True
        )

    def start(self):
        self._proc.start()

    def stop(self):
        self._proc.stop()

    def maintain(self):
        self._proc.maintain()
