import os
import sqlite3
import time

import numpy as np

from .. import bots
from .. import kerasutil
from ..mputil import Loopable, LoopingProcess
from ..players import Player
from ..simulate import simulate_game


class NotEnoughBots(Exception):
    pass


class EvaluatorImpl(Loopable):
    def __init__(self, out_dir, logger, config):
        self._logger = logger
        self._config = config
        kerasutil.set_tf_options(disable_gpu=True)

        self._bot_dir = os.path.join(out_dir, 'bots')
        self._db_file = os.path.join(out_dir, 'evaluation.db')
        self._conn = sqlite3.connect(self._db_file)

        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                bot1 TEXT,
                bot2 TEXT,
                num_hands INTEGER,
                bot1_points INTEGER,
                bot2_points INTEGER,
                bot1_contracts INTEGER,
                bot2_contracts INTEGER
            )
        ''')
        self._conn.commit()

    def _save_result(
            self, bot1, bot2, num_hands, bot1_points, bot2_points,
            bot1_contracts, bot2_contracts
    ):
        self._conn.execute('''
            INSERT INTO matches (
                bot1,
                bot2,
                num_hands,
                bot1_points,
                bot2_points,
                bot1_contracts,
                bot2_contracts
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            bot1.identify(),
            bot2.identify(),
            int(num_hands),
            int(bot1_points),
            int(bot2_points),
            int(bot1_contracts),
            int(bot2_contracts)
        ))
        self._conn.commit()

    def _get_bot_weights(self, bot_names):
        weights = {bot_name: 0 for bot_name in bot_names}
        cursor = self._conn.execute('''
            SELECT bot1, COUNT(*)
            FROM matches GROUP BY bot1
        ''')
        for row in cursor:
            bot_name, count = row
            if bot_name in weights:
                weights[bot_name] += count
        cursor = self._conn.execute('''
            SELECT bot2, COUNT(*)
            FROM matches GROUP BY bot2
        ''')
        for row in cursor:
            bot_name, count = row
            if bot_name in weights:
                weights[bot_name] += count
        return weights

    def _select_bots(self):
        bot_names = []
        for fname in os.listdir(self._bot_dir):
            bot_names.append(fname)

        if len(bot_names) < 2:
            raise NotEnoughBots()

        weights = self._get_bot_weights(bot_names)
        weight_array = np.array([weights[name] for name in bot_names])
        probs = 1. / (weight_array + np.ones_like(weight_array))
        probs = probs / np.sum(probs)
        n_bots = probs.shape[0]

        idx1, idx2 = np.random.choice(n_bots, size=2, p=probs, replace=False)
        bot1_fname = bot_names[idx1]
        bot2_fname = bot_names[idx2]
        path1 = os.path.join(self._bot_dir, bot1_fname)
        path2 = os.path.join(self._bot_dir, bot2_fname)
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
        self._logger.log(f'Evaluating {bot1.identify()} vs {bot2.identify()}')

        bot1_points = 0
        bot2_points = 0
        bot1_contracts = 0
        bot2_contracts = 0
        bot1_side = 'ns'
        num_hands = 0
        while num_hands < self._config['num_hands_per_match']:
            if bot1_side == 'ns':
                ns_bot = bot1
                ew_bot = bot2
            else:
                ew_bot = bot1
                ns_bot = bot2
            result = simulate_game(ns_bot, ew_bot)
            if result.declarer is None:
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

        self._save_result(
            bot1, bot2,
            num_hands=num_hands,
            bot1_points=bot1_points,
            bot2_points=bot2_points,
            bot1_contracts=bot1_contracts,
            bot2_contracts=bot2_contracts
        )


class Evaluator:
    def __init__(self, out_dir, config, logger):
        self._out_dir = out_dir
        self._config = config
        self._logger = logger

    def start(self):
        self._proc = LoopingProcess(
            'evaluator',
            EvaluatorImpl,
            kwargs={
                'out_dir': self._out_dir,
                'logger': self._logger,
                'config': self._config['evaluation'],
            },
            restart=True
        )
        self._proc.start()

    def stop(self):
        self._proc.stop()

    def maintain(self):
        self._proc.maintain()
