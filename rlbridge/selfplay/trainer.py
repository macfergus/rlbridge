import copy
import json
import multiprocessing
import os
import queue
import time
from collections import namedtuple

from .. import bots
from ..mputil import disable_sigint


Worker = namedtuple('Worker', 'ctl_q proc')


class WriteableBotPool:
    def __init__(self, pool_fname, out_dir, logger):
        self.pool_fname = pool_fname
        self.out_dir = out_dir
        init = json.load(open(pool_fname))
        self.ref_fnames = copy.copy(init['ref'])
        self.learn_fname = copy.copy(init['learn'])

        self.learn_bot = bots.load_bot(self.learn_fname)

        self.logger = logger

    def get_learn_bot(self):
        return self.learn_bot

    def _save_bot(self, bot):
        out_fname = os.path.join(self.out_dir, bot.identify())
        out_fname = out_fname.replace(' ', '_')
        bots.save_bot(bot, out_fname)
        return out_fname

    def promote(self, new_best_bot):
        new_bot_fname = self._save_bot(new_best_bot)
        # Keep the last 5 promoted bots
        self.ref_fnames.append(new_bot_fname)
        self.ref_fnames = self.ref_fnames[-5:]
        self.logger.log(f'Ref bots are: {self.ref_fnames}')
        self.learn_fname = new_bot_fname

        tmpfname = self.pool_fname + '.tmp'
        with open(tmpfname, 'w') as outf:
            outf.write(json.dumps({
                'ref': self.ref_fnames,
                'learn': self.learn_fname,
            }))
        os.rename(tmpfname, self.pool_fname)


def do_training(ctl_q, q, state_fname, out_dir, logger, config):
    disable_sigint()

    bot_pool = WriteableBotPool(state_fname, out_dir, logger)
    bot = bot_pool.get_learn_bot()

    total_games = 0
    num_games = 0
    experience_size = 0
    experience = []
    last_log = time.time()
    while True:
        try:
            ctl_q.get_nowait()
            return
        except queue.Empty:
            pass

        try:
            episode = q.get(timeout=1)
        except queue.Empty:
            continue

        total_games += 1
        num_games += 1

        experience.append(episode)
        experience_size += episode['states'].shape[0]
        now = time.time()
        if now - last_log > 60.0:
            logger.log(f'{total_games} total games received so far')
            last_log = now
        if experience_size < config['chunk_size']:
            continue

        # When the chunk is big enough, train the current bot
        logger.log(
            f'Training on {experience_size} examples from {num_games} games'
        )
        hist = bot.train(experience, use_advantage=config['use_advantage'])
        logger.log(
            f'call_loss {hist["call_loss"]} '
            f'play_loss {hist["play_loss"]} '
            f'value_loss {hist["value_loss"]}'
        )
        bot.add_games(num_games)
        bot_pool.promote(bot)
        num_games = 0
        experience = []
        experience_size = 0


class Trainer:
    def __init__(self, exp_q, state_path, out_dir, logger, config):
        self._exp_q = exp_q
        self._state_path = state_path
        self._out_dir = out_dir
        self._logger = logger
        self._config = config

        self._worker = self._new_worker()

    def _new_worker(self):
        ctl_q = multiprocessing.Queue()
        proc = multiprocessing.Process(
            name='trainer',
            target=do_training,
            args=(
                ctl_q,
                self._exp_q,
                self._state_path,
                self._out_dir,
                self._logger,
                self._config['training']
            )
        )
        return Worker(ctl_q=ctl_q, proc=proc)

    def start(self):
        self._worker.proc.start()

    def stop(self):
        if self._worker.proc.is_alive():
            self._worker.ctl_q.put(None)
        self._worker.proc.join()

    def maintain(self):
        # Restart the trainer process if it died
        if not self._worker.proc.is_alive():
            self._logger.log('Restarting trainer')
            self._worker = self._new_worker()
            self._worker.proc.start()
