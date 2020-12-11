import multiprocessing
import queue
import random
from collections import namedtuple

import numpy as np

__all__ = [
    'GameGenerator',
]


STOP = object()


TrainingGame = namedtuple('TrainingGame', 'X y_call y_play y_value')


def generate_games(out_q, ctrl_q, bot_fname, config):
    from .. import kerasutil
    kerasutil.set_tf_options(disable_gpu=True)

    from ..bots import load_bot
    from .simulate import simulate_game

    sample_defeated = config['self_play']['sample_defeated']
    sample_lost = config['self_play']['sample_lost']

    bot = load_bot(bot_fname)
    count = 0
    while True:
        try:
            ctrl_q.get_nowait()
            return
        except queue.Empty:
            pass

        bot.set_option('max_contract', random.randint(3, 7))
        bot.set_option('temperature', config['self_play']['temperature'])
        game_result = simulate_game(bot, bot)
        if game_result.declarer is None:
            continue

        declarers = [game_result.declarer, game_result.declarer.partner]
        defenders = [game_result.declarer.lho(), game_result.declarer.rho()]

        if game_result.contract_made:
            # Skip most low-value contracts
            if game_result.contract_level < 3 and np.random.random() >= 0.9:
                continue
            if np.random.random() < sample_lost:
                p = random.choice(defenders)
            else:
                p = random.choice(declarers)
        else:
            if np.random.random() > sample_defeated:
                continue
            if np.random.random() < sample_lost:
                p = random.choice(declarers)
            else:
                p = random.choice(defenders)
        X, y_call, y_play, y_value = bot.encode_pretraining(game_result, p)
        out_q.put(TrainingGame(
            X=X,
            y_call=y_call,
            y_play=y_play,
            y_value=y_value
        ))
        count += 1
        if count >= config['self_play']['max_games_per_worker']:
            return


class GameGenerator:
    def __init__(self, bot_fname, config):
        self.recv_queue = multiprocessing.Queue()
        self.ctrl_queues = []
        self._bot_fname = bot_fname
        self._config = config
        self._processes = []
        for _ in range(config['self_play']['num_workers']):
            self._new_worker()

    def _new_worker(self):
        ctrl_q = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=generate_games,
            args=(
                self.recv_queue,
                ctrl_q,
                self._bot_fname,
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
            self._new_worker().start()

    def stop(self):
        for q in self.ctrl_queues:
            q.put(STOP)
        for proc in self._processes:
            while proc.is_alive():
                proc.join(timeout=1)
                while True:
                    try:
                        self.recv_queue.get(block=False)
                    except queue.Empty:
                        break

