import copy
import json
import multiprocessing
import queue
import random

import numpy as np

from .. import kerasutil
from ..bots import load_bot, save_bot
from ..mputil import disable_sigint
from ..players import Player
from ..rl import ExperienceRecorder
from ..simulate import simulate_game

__all__ = [
    'ExperienceGenerator',
]


class BotPool:
    def __init__(self, fname, logger):
        self._fname = fname
        self._ref_bot_names = None
        self._learn_bot_name = None
        self.logger = logger

    def refresh(self):
        new_learner = False
        data = json.load(open(self._fname))
        if self._ref_bot_names != data['ref']:
            #self.logger.log('Updating reference bot pool')
            self._ref_bot_names = copy.copy(data['ref'])
            self._ref_bots = []
            self._ref_weights = []
            for i, bot_file in enumerate(self._ref_bot_names):
                ref_bot = load_bot(bot_file)
                self._ref_bots.append(ref_bot)
                self._ref_weights.append(i + 1)
                #self.logger.log('=> Loaded {} with weight {}'.format(
                #    ref_bot.identify(),
                #    i + 1
                #))
            self._ref_weights = (
                np.array(self._ref_weights) / np.sum(self._ref_weights)
            )
        if self._learn_bot_name != data['learn']:
            new_learner = True
            self.logger.log('Updating learn bot')
            self._learn_bot_name = copy.copy(data['learn'])
            self._learn_bot = load_bot(self._learn_bot_name)
            self.logger.log('=> Loaded {} as new learner'.format(
                self._learn_bot.identify()
            ))
        return new_learner

    def select_ref_bot(self):
        bot_idx = np.random.choice(len(self._ref_bots), p=self._ref_weights)
        return self._ref_bots[bot_idx]

    def get_learn_bot(self):
        return self._learn_bot


def generate_games(
    ctl_q, exp_q, stat_q, max_contract, state_fname, logger, config
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

        learn_bot.set_option('max_contract', max_contract)
        learn_bot.set_option('temperature', config['temperature'])
        ref_bot.set_option('max_contract', max_contract)
        ref_bot.set_option('temperature', config['temperature'])

        recorder = ExperienceRecorder()
        learn_side = random.choice(['ns', 'ew'])
        if learn_side == 'ns':
            game_result = simulate_game(
                learn_bot, ref_bot, ns_recorder=recorder
            )
            if (
                game_result.contract_made and 
                game_result.declarer in (Player.north, Player.south)
            ):
                stat_q.put(1)
            else:
                stat_q.put(0)
            episode1 = learn_bot.encode_episode(
                game_result,
                Player.north,
                recorder.get_decisions(Player.north),
                reward=config['reward']
            )
            episode2 = learn_bot.encode_episode(
                game_result,
                Player.south,
                recorder.get_decisions(Player.south),
                reward=config['reward']
            )
        else:
            game_result = simulate_game(
                ref_bot, learn_bot, ew_recorder=recorder
            )
            if (
                game_result.contract_made and 
                game_result.declarer in (Player.east, Player.west)
            ):
                stat_q.put(1)
            else:
                stat_q.put(0)
            episode1 = learn_bot.encode_episode(
                game_result,
                Player.east,
                recorder.get_decisions(Player.east),
                reward=config['reward']
            )
            episode2 = learn_bot.encode_episode(
                game_result,
                Player.west,
                recorder.get_decisions(Player.west),
                reward=config['reward']
            )
        if True or np.sum(episode1['rewards']) > 0:
            exp_q.put(episode1)
        if True or np.sum(episode2['rewards']) > 0:
            exp_q.put(episode2)

        count += 1
        if count >= config['max_games_per_worker']:
            logger.log(f'Shutting down after {count} games')
            return


class ExperienceGenerator:
    def __init__(self, exp_q, state_path, logger, config):
        self.recv_queue = exp_q
        self._stat_queue = multiprocessing.Queue()
        self.ctrl_queues = []
        self._bot_fname = state_path
        self._logger = logger
        self._config = config['self_play']
        self._processes = []
        self._worker_idx = 0
        self._max_contract = 1
        self._contract_history = []
        for _ in range(config['self_play']['num_workers']):
            self._new_worker()

    def _new_worker(self):
        self._worker_idx += 1
        ctrl_q = multiprocessing.Queue()
        proc = multiprocessing.Process(
            name=f'worker-{self._worker_idx}',
            target=generate_games,
            args=(
                ctrl_q,
                self.recv_queue,
                self._stat_queue,
                self._max_contract,
                self._bot_fname,
                self._logger,
                self._config

            )
        )
        self._processes.append(proc)
        self.ctrl_queues.append(ctrl_q)
        return proc

    def start(self):
        for proc in self._processes:
            proc.start()

    def maintain(self):
        # Adjust contract limits if needed. If we are making "too many"
        # contracts, we should relax contract limits (to present more
        # challenging contracts to the agents)
        # This will take effect whenever workers get recycled
        while True:
            try:
                made = self._stat_queue.get(block=False)
                self._contract_history.append(made)
            except queue.Empty:
                break
        made = np.sum(self._contract_history)
        n_hands = len(self._contract_history)
        if n_hands >= 500:
            self._logger.log(f'Made {made} contracts over {n_hands} hands')
            pct_made = np.mean(self._contract_history)
            if (
                pct_made >= self._config['target_contracts_made'] and
                self._max_contract < 7
            ):
                self._max_contract += 1
                self._logger.log(
                    f'Raising max contract to {self._max_contract}'
                )
            self._contract_history = []

        # Replace any dead processes
        to_clean = []
        for i in range(len(self._processes)):
            proc = self._processes[i]
            proc.join(timeout=0.001)
            if not proc.is_alive():
                to_clean.append(i)
                break
        for i in to_clean:
            del self._processes[i]
            del self.ctrl_queues[i]
        for _ in range(len(to_clean)):
            self._logger.log('Recycling worker')
            self._new_worker().start()

    def stop(self):
        for q in self.ctrl_queues:
            q.put(None)
        for proc in self._processes:
            while proc.is_alive():
                proc.join(timeout=1)
                while True:
                    try:
                        self.recv_queue.get(block=False)
                    except queue.Empty:
                        break
                while True:
                    try:
                        self._stat_queue.get(block=False)
                    except queue.Empty:
                        break
