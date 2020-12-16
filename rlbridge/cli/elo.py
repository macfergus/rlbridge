import sqlite3
from collections import namedtuple

from .. import elo
from .command import Command


Match = namedtuple('Match', 'winner loser')


class Elo(Command):
    def register_arguments(self, parser):
        parser.add_argument('eval_db_file')

    def run(self, args):
        conn = sqlite3.connect(args.eval_db_file)
        cursor = conn.execute('''
            SELECT bot1, bot2, bot1_points, bot2_points FROM matches
        ''')
        bots = set()
        matches = []
        for bot1, bot2, bot1_points, bot2_points in cursor:
            bots.add(bot1)
            bots.add(bot2)
            if bot1_points > bot2_points:
                matches.append(elo.Match(winner=bot1, loser=bot2))
            if bot2_points > bot1_points:
                matches.append(elo.Match(winner=bot2, loser=bot1))

        first_bot = sorted(bots)[0]
        ratings = elo.calculate_ratings(matches, anchor=first_bot)

        sorted_ratings = sorted(
            [(rating, bot) for bot, rating in ratings.items()],
            reverse=True
        )
        for rating, bot in sorted_ratings:
            print(bot, rating)