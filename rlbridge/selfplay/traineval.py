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
    def __init__(
            self, episode_q, ref_fname, learn_fname, out_dir, logger,
            gate=True,
            max_games=10000,
            episodes_per_train=200,
            eval_games=200,
            eval_chunk=20,
            eval_threshold=0.05):
        self.episode_q = episode_q
        self._ref_fname = ref_fname
        self._learn_fname = learn_fname
        self.out_dir = out_dir
        self.logger = logger
        self.episode_buffer = []
        self.should_continue = True
        self.total_games = 0

        self._gate = gate
        self._max_games = max_games
        self._episodes_per_train = episodes_per_train
        self._eval_games = eval_games
        self._eval_chunk = eval_chunk
        self._eval_threshold = eval_threshold

        self.training_bot = self.load_learn_bot()

    def run(self):
        while self.should_continue:
            self.wait_for_episodes()
            if len(self.episode_buffer) < self._episodes_per_train:
                continue

            work = self.episode_buffer
            self.episode_buffer = []
            self.logger.log('Training on {} episodes'.format(len(work)))
            stats = self.training_bot.train(work)
            self.logger.log('Loss: {:.3f} '.format(stats['loss']))
            self.logger.log(
                'Call: {call_loss:.3f} '
                'Play: {play_loss:.3f} '
                'Value: {value_loss:.3f}'.format(**stats)
            )
            self.training_bot.add_games(len(work))
            self.total_games += len(work)
            work = []

            promote = True
            if self._gate:
                self.logger.log('Evaluating...')
                promote = self.evaluate_bot()
            if promote:
                self.promote()
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
        ref_bot = self.load_ref_bot()
        # For evaluation, make both bots choose their strongest actions.
        ref_bot.set_option('temperature', 0.0)
        self.training_bot.set_option('temperature', 0.0)
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
        return lower > 0

    def promote(self):
        out_fname = os.path.join(self.out_dir, self.training_bot.identify())
        out_fname = out_fname.replace(' ', '_')
        self.logger.log('Saving as {}'.format(out_fname))
        bots.save_bot(self.training_bot, out_fname)
        self.logger.log('Promoting {}'.format(out_fname))
        tmp_path = self._ref_fname + '.tmp'
        with open(tmp_path, 'w') as ref_outf:
            ref_outf.write(out_fname)
        os.rename(tmp_path, self._ref_fname)

    def update_learn_bot(self):
        self.logger.log('Update learner')
        out_fname = os.path.join(self.out_dir, 'learner')
        bots.save_bot(self.training_bot, out_fname)
        tmp_path = self._learn_fname + '.tmp'
        with open(tmp_path, 'w') as learn_outf:
            learn_outf.write(out_fname)
        os.rename(tmp_path, self._learn_fname)

    def load_ref_bot(self):
        ref_path = open(self._ref_fname).read().strip()
        return bots.load_bot(ref_path)

    def load_learn_bot(self):
        learn_path = open(self._learn_fname).read().strip()
        return bots.load_bot(learn_path)
