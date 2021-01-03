import copy
import json
import multiprocessing
import queue
import random
import time
from collections import namedtuple

import numpy as np

from .. import kerasutil
from ..bots import load_bot
from ..game import ALL_DENOMINATIONS
from ..mputil import disable_sigint
from ..players import Player
from ..rl import ExperienceRecorder
from ..simulate import simulate_game

__all__ = [
    'ExperienceGenerator',
]


Worker = namedtuple('Worker', 'name proc ctl_q')


class BotPool:
    def __init__(self, fname, logger):
        self._fname = fname
        self._ref_bot_names = None
        self._ref_bots = []
        self._ref_weights = []
        self._learn_bot_name = None
        self._learn_bot = None
        self.logger = logger

    def refresh(self):
        # prevent all the workers from hitting the files at once
        time.sleep(0.1 * random.random())
        new_learner = False
        data = json.load(open(self._fname))
        if self._ref_bot_names != data['ref']:
            self._ref_bot_names = copy.copy(data['ref'])
            self._ref_bots = []
            self._ref_weights = []
            for i, bot_file in enumerate(self._ref_bot_names):
                ref_bot = load_bot(bot_file)
                self._ref_bots.append(ref_bot)
                self._ref_weights.append(i + 1)
            self._ref_weights = (
                np.array(self._ref_weights) / np.sum(self._ref_weights)
            )
        if self._learn_bot_name != data['learn']:
            new_learner = True
            self._learn_bot_name = copy.copy(data['learn'])
            self._learn_bot = load_bot(self._learn_bot_name)
        return new_learner

    def select_ref_bot(self):
        bot_idx = np.random.choice(len(self._ref_bots), p=self._ref_weights)
        return self._ref_bots[bot_idx]

    def get_learn_bot(self):
        return self._learn_bot


def generate_games(
        ctl_q, exp_q, stat_q, workspace, state_fname, logger, config
):
    disable_sigint()
    kerasutil.set_tf_options(disable_gpu=True)

    bot_pool = BotPool(state_fname, logger)

    count = 0
    while True:
        try:
            ctl_q.get_nowait()
            return
        except queue.Empty:
            pass

        bot_pool.refresh()
        learn_bot = bot_pool.get_learn_bot()
        ref_bot = bot_pool.select_ref_bot()

        learn_temp = config.get('learn_temperature', config.get('temperature'))
        ref_temp = config.get('ref_temperature', config.get('temperature'))
        if learn_temp is None:
            raise ValueError(f'Must set learn_temperature or temperature')
        if ref_temp is None:
            raise ValueError(f'Must set ref_temperature or temperature')
        learn_bot.set_option('temperature', learn_temp)
        ref_bot.set_option('temperature', ref_temp)

        max_contract = workspace.params.get_int('max_contract')
        mode = config['contract_limiting']['mode']
        if mode == 'off':
            max_contract = 7
        elif mode == 'spread':
            max_contract = random.randint(max_contract, 7)
        learn_bot.set_option('max_contract', max_contract)
        ref_bot.set_option('max_contract', max_contract)

        force_pct = config.get('force_contract', {}).get('pct', 0.0)
        force_fade = config.get('force_contract', {}).get('fadeout', 1)
        age = learn_bot.metadata.get('num_games', 0)
        fade = 1.0 - float(age) / float(force_fade)
        force_contract_pct = max(0.0, force_pct * fade)
        should_force_contract = random.random() < force_contract_pct
        if should_force_contract:
            tricks = random.randint(1, 7)
            denom = random.choice(ALL_DENOMINATIONS)
            declarer = random.choice([
                Player.north, Player.east, Player.south, Player.west
            ])
            learn_bot.set_option('force_contract', (tricks, denom, declarer))
            ref_bot.set_option('force_contract', (tricks, denom, declarer))
        else:
            learn_bot.set_option('force_contract', None)
            ref_bot.set_option('force_contract', None)

        recorder = ExperienceRecorder()
        learn_side = random.choice(['ns', 'ew'])
        made_contract = 0
        n_games = learn_bot.metadata.get('num_games', 0)

        contract_bonus = 0
        if 'contract_bonus' in config:
            contract_bonus = float(config['contract_bonus'])
            strength = 1.0
            if 'contract_bonus_fadeout' in config:
                fadeout = float(config['contract_bonus_fadeout'])
                strength = 1.0 - float(n_games) / fadeout
                strength = max(strength, 0.0)
            contract_bonus = strength * contract_bonus
        reward_scale = config.get('reward_scale', 'linear')

        trick_weight_fadeout = config.get('trick_weight_fadeout', 1.0)
        trick_weight = 1.0 - float(n_games) / trick_weight_fadeout
        trick_weight = max(trick_weight, 0.0)

        if learn_side == 'ns':
            game_result = simulate_game(
                learn_bot, ref_bot, ns_recorder=recorder
            )
            if game_result.declarer is None:
                logger.log('No bids, continue')
                continue
            if (
                    game_result.contract_made and
                    game_result.declarer in (Player.north, Player.south)
            ):
                made_contract = 1
            episode1 = learn_bot.encode_episode(
                game_result,
                Player.north,
                recorder.get_decisions(Player.north),
                contract_bonus=contract_bonus,
                reward_scale=reward_scale,
                trick_weight=trick_weight
            )
            episode2 = learn_bot.encode_episode(
                game_result,
                Player.south,
                recorder.get_decisions(Player.south),
                contract_bonus=contract_bonus,
                reward_scale=reward_scale,
                trick_weight=trick_weight
            )
        else:
            game_result = simulate_game(
                ref_bot, learn_bot, ew_recorder=recorder
            )
            if game_result.declarer is None:
                continue
            if (
                    game_result.contract_made and
                    game_result.declarer in (Player.east, Player.west)
            ):
                made_contract = 1
            episode1 = learn_bot.encode_episode(
                game_result,
                Player.east,
                recorder.get_decisions(Player.east),
                contract_bonus=contract_bonus,
                reward_scale=reward_scale,
                trick_weight=trick_weight
            )
            episode2 = learn_bot.encode_episode(
                game_result,
                Player.west,
                recorder.get_decisions(Player.west),
                contract_bonus=contract_bonus,
                reward_scale=reward_scale,
                trick_weight=trick_weight
            )
        stat_q.put(made_contract)
        exp_q.put(episode1)
        exp_q.put(episode2)

        count += 1
        if count >= config['max_games_per_worker']:
            logger.log(f'Shutting down after {count} games')
            return


class ExperienceGenerator:
    def __init__(self, exp_q, workspace, config, logger):
        self.recv_queue = exp_q
        self._stat_queue = multiprocessing.Queue()
        self._workspace = workspace
        self._logger = logger
        self._config = config['self_play']
        self._worker_idx = 0
        self._contract_history = []
        self._last_recv = 0.0

        self._workers = {}
        for _ in range(config['self_play']['num_workers']):
            self._new_worker()

        if not self._workspace.params.has_key('max_contract'):
            self._logger.log('Initialize max_contract to 1')
            self._workspace.params.set_int('max_contract', 1)

    def _new_worker(self):
        self._worker_idx += 1
        name = f'worker-{self._worker_idx}'
        ctl_q = multiprocessing.Queue()
        worker = Worker(
            name=name,
            ctl_q=ctl_q,
            proc=multiprocessing.Process(
                name=f'worker-{self._worker_idx}',
                target=generate_games,
                args=(
                    ctl_q,
                    self.recv_queue,
                    self._stat_queue,
                    self._workspace,
                    self._workspace.state_file,
                    self._logger,
                    self._config
                )
            )
        )
        self._workers[name] = worker
        return worker

    def start(self):
        self._last_recv = time.time()
        for worker in self._workers.values():
            worker.proc.start()

    def maintain(self):
        # Adjust contract limits if needed. If we are making "too many"
        # contracts, we should relax contract limits (to present more
        # challenging contracts to the agents)
        # This will take effect whenever workers get recycled
        while True:
            try:
                made = self._stat_queue.get(block=False)
                self._last_recv = time.time()
                self._contract_history.append(made)
            except queue.Empty:
                break
        made = np.sum(self._contract_history)
        n_hands = len(self._contract_history)
        if n_hands >= 1000:
            self._logger.log(f'Made {made} contracts over {n_hands} hands')
            pct_made = np.mean(self._contract_history)
            self._adjust_contract_limits(pct_made)
            self._contract_history = []

        # Manage the worker pool
        # Look for stuck workers, and reap any that shut themselves down
        # Then bring the pool back up to size
        now = time.time()
        if now - self._last_recv > 90.0:
            self._logger.log('Have not received a game in 90 seconds.')
            self._logger.log('Replacing ALL workers')
            for k in list(self._workers.keys()):
                self._stop_worker(k)

        # Replace any dead processes
        for k in list(self._workers.keys()):
            if not self._workers[k].proc.is_alive():
                self._stop_worker(k)

        while len(self._workers) < self._config['num_workers']:
            self._logger.log('Launching new worker')
            self._new_worker().proc.start()

    def _adjust_contract_limits(self, pct_made):
        max_contract = self._workspace.params.get_int('max_contract', 1)
        upper = self._config['contract_limiting'].get('target_upper', 1.0)
        lower = self._config['contract_limiting'].get('target_lower', 0.0)
        if pct_made >= upper and max_contract < 7:
            self._logger.log(f'Raising max contract to {max_contract + 1}')
            self._workspace.params.set_int('max_contract', max_contract + 1)
        elif pct_made < lower and max_contract > 1:
            self._logger.log(f'Dropping max contract to {max_contract - 1}')
            self._workspace.params.set_int('max_contract', max_contract - 1)

    def _stop_worker(self, k):
        stop_time = time.time()
        self._logger.log(f'Stopping {k}')
        worker = self._workers.pop(k)
        if worker.proc.is_alive():
            worker.ctl_q.put(None, timeout=1)
        worker.proc.join(timeout=0.001)
        while worker.proc.is_alive():
            if time.time() - stop_time > 15:
                self._logger.log('Worker is still around after 15s!')
                self._logger.log('Sending TERM')
                worker.proc.terminate()
                worker.proc.join(timeout=0.01)
                break
            worker.proc.join(timeout=1)
            # drain queues to prevent deadlock
            try:
                self.recv_queue.get(block=False)
            except queue.Empty:
                pass
            try:
                self._stat_queue.get(block=False)
            except queue.Empty:
                pass

    def stop(self):
        for k in list(self._workers.keys()):
            self._stop_worker(k)
