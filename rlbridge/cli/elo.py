import sqlite3
from collections import namedtuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from .. import elo
from .. import nputil
from ..workspace import open_workspace
from .command import Command


Match = namedtuple('Match', 'winner loser')


def plot_ratings(ratings, color='b', label=None):
    pairs = []
    for bot_name, rating in ratings.items():
        n_games = int(bot_name.split('_')[-1])
        pairs.append((n_games, rating))
    pairs.sort()
    xs = np.array([n_games for n_games, _ in pairs])
    ys = np.array([rating for _, rating in pairs])
    smooth_ys = nputil.smooth(ys, 7, width=2)
    plt.plot(xs, ys, color=color, alpha=0.2)
    plt.plot(xs, smooth_ys, color=color, label=label)


def plot_all_ratings(all_ratings, out_fname):
    run_ids = sorted(all_ratings.keys())
    n_runs = len(run_ids)
    palette = sns.color_palette('husl', n_runs)
    plt.figure()
    for i, run_id in enumerate(run_ids):
        color = palette[i]
        ratings = all_ratings[run_id]
        plot_ratings(ratings, color, label=run_id)
    plt.xlabel('num games')
    plt.ylabel('elo (1000 = random)')
    plt.legend()
    plt.grid()
    plt.savefig(out_fname, dpi=100)
    


class Elo(Command):
    def register_arguments(self, parser):
        parser.add_argument('--goal', default='points')
        parser.add_argument('--plot')
        parser.add_argument('run_id', nargs='+')

    def run(self, args):
        all_ratings = {}
        for run_id in args.run_id:
            workspace = open_workspace(run_id)
            ratings = workspace.eval_store.get_elo_ratings()
            all_ratings[run_id] = ratings
    
            sorted_ratings = sorted(
                [(rating, bot) for bot, rating in ratings.items()],
                reverse=True
            )
            print(f'{run_id} Elo ratings with goal {args.goal}')
            for rating, bot in sorted_ratings:
                print(bot, int(rating))

        if args.plot:
            plot_all_ratings(all_ratings, args.plot)
