import copy
import json
import os
import queue
import random

import numpy as np

from .. import bots
from ..simulate import simulate_game

__all__ = [
    'TrainEvalLoop',
]


def estimate_ci(values, min_pct, max_pct, n_bootstrap=1000):
    """Estimate a confidence interval by bootstrapping."""
    values = np.array(values)
    n = values.shape[0]
    vals = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resampled = np.random.choice(values, size=n, replace=True)
        vals[i] = np.mean(resampled)
    lower = np.quantile(vals, min_pct)
    upper = np.quantile(vals, max_pct)
    return lower, upper


class WriteableBotPool:
    def __init__(self, pool_fname, out_dir, logger):
        self.pool_fname = pool_fname
        self.out_dir = out_dir
        init = json.load(open(pool_fname))
        self.ref_fnames = copy.copy(init['ref'])
        self.learn_fname = copy.copy(init['learn'])

        self.learn_bot = bots.load_bot(self.learn_fname)
        self.gating_bot = bots.load_bot(self.ref_fnames[-1])

        self.logger = logger

    def get_learn_bot(self):
        return self.learn_bot

    def get_gating_bot(self):
        return self.gating_bot

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

        # After promotion, this is the new gating bot.
        self.gating_bot = bots.load_bot(self.learn_fname)

        tmpfname = self.pool_fname + '.tmp'
        with open(tmpfname, 'w') as outf:
            outf.write(json.dumps({
                'ref': self.ref_fnames,
                'learn': self.learn_fname,
            }))
        os.rename(tmpfname, self.pool_fname)


class TrainEvalLoop:
    def __init__(
            self, episode_q, pool_fname, out_dir, logger,
            gate=True,
            max_contract=7,
            max_games=50000,
            episodes_per_train=200,
            eval_games=200,
            eval_chunk=20,
            eval_threshold=0.05):
        self.episode_q = episode_q
        self.bot_pool = WriteableBotPool(pool_fname, out_dir, logger)
        self.logger = logger
        self.episode_buffer = []
        self.should_continue = True
        self.total_games = 0

        self._gate = gate
        self._max_contract = max_contract
        self._max_games = max_games
        self._episodes_per_train = episodes_per_train
        self._eval_games = eval_games
        self._eval_chunk = eval_chunk
        self._eval_threshold = eval_threshold

        self.training_bot = self.bot_pool.get_learn_bot()

    def run(self):
        while self.should_continue:
            self.wait_for_episodes()
            if len(self.episode_buffer) < self._episodes_per_train:
                continue

            work = self.episode_buffer
            self.episode_buffer = []
            self.logger.log('Training on {} episodes'.format(len(work)))
            stats = self.training_bot.train(
                work,
                reinforce_only=True,
                use_advantage=False
            )
            self.logger.log('Loss: {:.3f} '.format(stats['loss']))
            self.logger.log(
                'Call: {call_loss:.3f} '
                'Play: {play_loss:.3f} '
                'Value: {value_loss:.3f}'.format(**stats)
            )
            self.training_bot.add_games(len(work))
            self.total_games += len(work)
            work = []

            should_promote = True
            if self._gate:
                should_promote = self.evaluate_bot()
            if should_promote:
                self.logger.log('Promoting!')
                self.bot_pool.promote(self.training_bot)
            if self.total_games >= self._max_games:
                # Shut the process down to free up memory.
                self.logger.log('Shutting down after {} games'.format(
                    self.total_games
                ))
                break
        self.logger.log('Bye!!')

    def wait_for_episodes(self):
        episode = self.episode_q.get()
        if episode is None:
            self.should_continue = False
        else:
            self.episode_buffer.append(episode)

    def receive(self):
        try:
            episode = self.episode_q.get_nowait()
        except queue.Empty:
            return
        if episode is None:
            self.should_continue = False
        else:
            self.episode_buffer.append(episode)

    def evaluate_bot(self):
        ref_bot = self.bot_pool.get_gating_bot()
        self.logger.log(
            f'Evaluating {self.training_bot.identify()} '
            f'against {ref_bot.identify()}'
        )
        # For evaluation, lower the temperature to bias toward
        # stronger actions.
        ref_bot.set_option('temperature', 0.2)
        ref_bot.set_option('max_contract', self._max_contract)
        self.training_bot.set_option('temperature', 0.2)
        self.training_bot.set_option('max_contract', self._max_contract)

        num_games = 0
        lower_p = self._eval_threshold
        upper_p = 1 - self._eval_threshold
        while num_games < self._eval_games:
            margins = []
            for _ in range(self._eval_chunk):
                self.receive()
                if random.choice([0, 1]) == 0:
                    ns_bot = self.training_bot
                    ew_bot = ref_bot
                else:
                    ns_bot = ref_bot
                    ew_bot = self.training_bot
                result = simulate_game(ns_bot, ew_bot)
                if ns_bot is self.training_bot:
                    margins.append(result.points_ns - result.points_ew)
                else:
                    margins.append(result.points_ew - result.points_ns)
                num_games += 1
            mean = np.mean(margins)
            lower, upper = estimate_ci(margins, lower_p, upper_p)
            self.logger.log('eval {} games {:.1f}, {:.1f}'.format(
                num_games,
                lower, upper
            ))
            if lower > 0 or upper < 0:
                break
        self.logger.log(
            'Candidate gained {:.1f} points per hand over {} games '
            '(lower {:.1f} upper {:.1f})'.format(
                mean,
                num_games,
                lower, upper))
        return lower > 0.01
