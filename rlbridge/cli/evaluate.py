import random

import numpy as np
from tqdm import tqdm

from .. import bots
from ..io import parse_options
from ..players import Player
from ..simulate import simulate_game
from .command import Command


def estimate_ci(values, min_pct, max_pct, n_bootstrap):
    """Estimate a confidence interval by bootstrapping."""
    n = values.shape[0]
    vals = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resampled = np.random.choice(values, size=n, replace=True)
        vals[i] = np.mean(resampled)
    lower = np.quantile(vals, min_pct)
    upper = np.quantile(vals, max_pct)
    return lower, upper


class Evaluate(Command):
    def register_arguments(self, parser):
        parser.add_argument('--options')
        parser.add_argument('--num-games', type=int, default=1)
        parser.add_argument('bot1')
        parser.add_argument('bot2')

    def run(self, args):
        bot1 = bots.load_bot(args.bot1)
        bot2 = bots.load_bot(args.bot2)
        if args.options:
            opts = parse_options(args.options)
        for key, value in opts.items():
            bot1.set_option(key, value)
            bot2.set_option(key, value)

        margins = []
        bot1_wins = 0
        bot2_wins = 0
        bot1_contracts = 0
        bot2_contracts = 0
        for _ in tqdm(range(args.num_games)):
            if random.choice([0, 1]) == 0:
                ns_bot = bot1
                ew_bot = bot2
            else:
                ns_bot = bot2
                ew_bot = bot1
            try:
                result = simulate_game(ns_bot, ew_bot)
            except ValueError:
                tqdm.write("oops :(")
                continue
            bot1_declared = (
                (
                    ns_bot is bot1 and
                    result.declarer in (Player.north, Player.south)
                ) or (
                    ew_bot is bot1 and
                    result.declarer in (Player.east, Player.west)
                )
            )
            bot2_declared = (
                (
                    ns_bot is bot2 and
                    result.declarer in (Player.north, Player.south)
                ) or (
                    ew_bot is bot2 and
                    result.declarer in (Player.east, Player.west)
                )
            )
            if bot1_declared and result.contract_made:
                bot1_contracts += 1
            if bot2_declared and result.contract_made:
                bot2_contracts += 1
            if ns_bot is bot1:
                margins.append(result.points_ns - result.points_ew)
                if result.points_ns > result.points_ew:
                    bot1_wins += 1
                else:
                    bot2_wins += 1
            else:
                margins.append(result.points_ew - result.points_ns)
                if result.points_ew > result.points_ns:
                    bot1_wins += 1
                else:
                    bot2_wins += 1
        margins = np.array(margins)
        mean_margin = np.mean(margins)
        lower, upper = estimate_ci(margins, 0.05, 0.95, n_bootstrap=5000)
        if mean_margin > 0:
            winner = bot1
            loser = bot2
        else:
            winner = bot2
            loser = bot1
            mean_margin = -1 * mean_margin
            lower = -1 * lower
            upper = -1 * upper
        print('{} made {} contracts, {} made {}'.format(
            bot1.identify(),
            bot1_contracts,
            bot2.identify(),
            bot2_contracts
        ))
        print('{} won {}, {} won {}'.format(
            bot1.identify(),
            bot1_wins,
            bot2.identify(),
            bot2_wins
        ))
        print('{} beats {} by {:.1f} points per game (over {} games)'.format(
            winner.identify(),
            loser.identify(),
            mean_margin,
            args.num_games
        ))
        print('Confidence interval ({:.1f}, {:.1f})'.format(lower, upper))
