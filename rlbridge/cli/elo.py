import sqlite3
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

from .. import elo
from .. import nputil
from ..workspace import open_workspace
from .command import Command


Match = namedtuple('Match', 'winner loser')


def plot_ratings(ratings, out_fname):
    pairs = []
    for bot_name, rating in ratings.items():
        n_games = int(bot_name.split('_')[-1])
        pairs.append((n_games, rating))
    pairs.sort()
    xs = np.array([n_games for n_games, _ in pairs])
    ys = np.array([rating for _, rating in pairs])
    smooth_ys = nputil.smooth(ys, 7, width=2)
    plt.xlabel('num games')
    plt.ylabel('elo (1000 = random)')
    plt.plot(xs, ys, color='b', alpha=0.2)
    plt.plot(xs, smooth_ys, color='b')
    plt.grid()
    plt.savefig(out_fname, dpi=100)


class Elo(Command):
    def register_arguments(self, parser):
        parser.add_argument('--goal', default='points')
        parser.add_argument('--plot')
        parser.add_argument('run_id')

    def run(self, args):
        workspace = open_workspace(args.run_id)
        conn = sqlite3.connect(workspace.eval_db_file)
        cursor = conn.execute('''
            SELECT
                bot1, bot2,
                bot1_points, bot2_points,
                bot1_contracts, bot2_contracts
            FROM matches
        ''')
        bots = set()
        matches = []
        for row in cursor:
            (
                bot1, bot2,
                bot1_points, bot2_points,
                bot1_contracts, bot2_contracts
            ) = row
            bots.add(bot1)
            bots.add(bot2)
            if args.goal == 'points':
                if bot1_points > bot2_points:
                    matches.append(elo.Match(winner=bot1, loser=bot2))
                if bot2_points > bot1_points:
                    matches.append(elo.Match(winner=bot2, loser=bot1))
            elif args.goal == 'contracts':
                if bot1_contracts > bot2_contracts:
                    matches.append(elo.Match(winner=bot1, loser=bot2))
                if bot2_contracts > bot1_contracts:
                    matches.append(elo.Match(winner=bot2, loser=bot1))

        first_bot = sorted(bots)[0]
        ratings = elo.calculate_ratings(matches, anchor=first_bot)

        sorted_ratings = sorted(
            [(rating, bot) for bot, rating in ratings.items()],
            reverse=True
        )
        print(f'Elo ratings with goal {args.goal}')
        for rating, bot in sorted_ratings:
            print(bot, int(rating))

        if args.plot:
            plot_ratings(ratings, args.plot)
