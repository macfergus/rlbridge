import time

from tqdm import trange

from ..bots import load_bot
from ..simulate import simulate_game
from .command import Command


class Benchmark(Command):
    def register_arguments(self, parser):
        parser.add_argument('--num-games', type=int, default=1)
        parser.add_argument('bot')

    def run(self, args):
        bot = load_bot(args.bot)
        start = time.time()
        for _ in trange(args.num_games):
            simulate_game(bot, bot)
        end = time.time()
        elapsed_hours = (end - start) / 3600
        print('{:.1f} games per hour'.format(args.num_games / elapsed_hours))
