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


class TrainEvalLoop:
    def __init__(self, episode_q, ref_fname, out_fname, logger):
        self.episode_q = episode_q
        self.ref_fname = ref_fname
        self.out_fname = out_fname
        self.logger = logger
        self.episode_buffer = []
        self.eval_games = 200
        self.should_continue = True
        self.total_games = 0

        self.training_bot = self.load_ref_bot()

    def run(self):
        while self.should_continue:
            self.ensure_episodes()
            if not self.episode_buffer:
                continue

            work = self.episode_buffer
            self.episode_buffer = []
            random.shuffle(work)
            self.logger.log('Training on {} episodes'.format(len(work)))
            for ep in work:
                self.training_bot.train_episode(ep)
                self.total_games += 1
                self.receive()

            if self.evaluate_bot():
                self.promote()
        self.logger.log('Bye!!')

    def ensure_episodes(self):
        if self.episode_buffer:
            return
        ep = self.episode_q.get()
        if ep is None:
            should_continue = False
        else:
            self.episode_buffer.append(ep)

    def receive(self):
        try:
            ep = self.episode_q.get_nowait()
        except queue.Empty:
            return
        if ep is None:
            should_continue = False
        else:
            self.episode_buffer.append(ep)

    def evaluate_bot(self):
        ref_bot = self.load_ref_bot()
        num_games = 0
        while num_games < self.eval_games:
            margins = []
            for i in range(20):
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
            lower, upper = estimate_ci(margins, 0.1, 0.9)
            self.logger.log('eval {} games {:.1f}, {:.1f}'.format(
                num_games,
                lower, upper
            ))
            if lower > 0:
                break
        self.logger.log(
            'Candidate gained {:.1f} points per hand over {} games '
            '(lower {:.1f} upper {:.1f})'.format(
                mean,
                num_games,
                lower, upper))
        return lower > 0

    def promote(self):
        out_fname = '{}_{:06d}'.format(self.out_fname, self.total_games)
        self.logger.log('Saving as {} and promoting'.format(out_fname))
        bots.save_bot(self.training_bot, out_fname)

        tmp_path = self.ref_fname + '.tmp'
        with open(tmp_path, 'w') as ref_outf:
            ref_outf.write(out_fname)
        os.rename(tmp_path, self.ref_fname)

    def load_ref_bot(self):
        ref_path = open(self.ref_fname).read().strip()
        return bots.load_bot(ref_path)
