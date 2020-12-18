import copy
import json
import os
import queue
import time

import numpy as np
from .. import bots
from ..mputil import Loopable, LoopingProcess


class WriteableBotPool:
    def __init__(self, pool_fname, out_dir, bots_to_keep, logger):
        self.pool_fname = pool_fname
        self.out_dir = out_dir
        self.bot_dir = os.path.join(self.out_dir, 'bots')
        if not os.path.exists(self.bot_dir):
            os.mkdir(self.bot_dir)

        self._bots_to_keep = int(bots_to_keep)

        init = json.load(open(pool_fname))
        self.ref_fnames = copy.copy(init['ref'])
        self.learn_fname = copy.copy(init['learn'])

        self.learn_bot = bots.load_bot(self.learn_fname)

        self.logger = logger

    def get_learn_bot(self):
        return self.learn_bot

    def _save_bot(self, bot):
        out_fname = os.path.join(self.bot_dir, bot.identify())
        out_fname = out_fname.replace(' ', '_')
        bots.save_bot(bot, out_fname)
        return out_fname

    def promote(self, new_best_bot):
        new_bot_fname = self._save_bot(new_best_bot)
        # Keep the last 5 promoted bots
        self.ref_fnames.append(new_bot_fname)
        self.ref_fnames = self.ref_fnames[-self._bots_to_keep:]
        self.logger.log(f'Ref bots are: {self.ref_fnames}')
        self.learn_fname = new_bot_fname

        tmpfname = self.pool_fname + '.tmp'
        with open(tmpfname, 'w') as outf:
            outf.write(json.dumps({
                'ref': self.ref_fnames,
                'learn': self.learn_fname,
            }))
        os.rename(tmpfname, self.pool_fname)


class TrainerImpl(Loopable):
    def __init__(self, q, state_fname, out_dir, logger, config):
        self._out_dir = out_dir
        self._bot_pool = WriteableBotPool(
            state_fname, out_dir, config['training']['bots_to_keep'], logger
        )
        self._bot = self._bot_pool.get_learn_bot()
        self._logger = logger
        self._config = config['training']

        self._q = q

        self._total_games = 0
        self._num_games = 0
        self._experience_size = 0
        self._experience = []
        self._last_log = time.time()

        self._chunks_done = 0

    def run_once(self):
        try:
            episode = self._q.get(timeout=1)
        except queue.Empty:
            return

        self._total_games += 1
        self._num_games += 1

        self._experience.append(episode)
        self._experience_size += episode['states'].shape[0]
        now = time.time()
        if now - self._last_log > 60.0:
            self._logger.log(f'{self._total_games} total games received so far')
            self._last_log = now
        if self._experience_size < self._config['chunk_size']:
            return

        # When the chunk is big enough, train the current bot
        self._logger.log(
            f'Training on {self._experience_size} examples from '
            f'{self._num_games} games'
        )
        hist = self._bot.train(
            self._experience,
            lr=self._config['lr'],
            use_advantage=self._config['use_advantage']
        )
        self._logger.log(
            f'call_loss {hist["call_loss"]} '
            f'play_loss {hist["play_loss"]} '
            f'value_loss {hist["value_loss"]}'
        )
        self._bot.add_games(self._num_games)
        self._chunks_done += 1
        if self._chunks_done >= self._config['chunks_per_promote']:
            self._logger.log('Promoting!')
            self._chunks_done = 0
            self._bot_pool.promote(self._bot)
            if np.random.random() < self._config['eval_frac']:
                self._logger.log('and marking for evaluation')
                self._save_bot_for_eval(self._bot)

        self._num_games = 0
        self._experience = []
        self._experience_size = 0

    def _save_bot_for_eval(self, bot):
        eval_dir = os.path.join(self._out_dir, 'eval')
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        out_fname = os.path.join(eval_dir, bot.identify())
        out_fname = out_fname.replace(' ', '_')
        bots.save_bot(bot, out_fname)


class Trainer:
    def __init__(self, exp_q, state_path, out_dir, logger, config):
        self._exp_q = exp_q
        self._proc = LoopingProcess(
            'trainer',
            TrainerImpl,
            kwargs={
                'q': self._exp_q,
                'state_fname': state_path,
                'out_dir': out_dir,
                'logger': logger,
                'config': config,
            },
            restart=True
        )

    def start(self):
        self._proc.start()

    def stop(self):
        self._proc.stop()

    def maintain(self):
        self._proc.maintain()
