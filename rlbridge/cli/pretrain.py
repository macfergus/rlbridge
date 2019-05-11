import random

from tqdm import tqdm

from ..bots import init_bot, load_bot, save_bot
from ..nputil import concat_inplace
from ..players import Player
from ..simulate import simulate_game
from .command import Command


class Pretrain(Command):
    def register_arguments(self, parser):
        parser.add_argument('--num-games', type=int, default=1)
        parser.add_argument('bot_in')
        parser.add_argument('bot_out')

    def run(self, args):
        bot = load_bot(args.bot_in)

        simulate_bot = init_bot('randombot', {}, {})
        X, y_call, y_play = None, None, None
        for _ in tqdm(range(args.num_games)):
            game_result = simulate_game(simulate_bot, simulate_bot)
            p = random.choice([
                Player.north, Player.east, Player.west, Player.south
            ])
            x1, y1, y2 = bot.encode_pretraining(game_result, p)
            if X is None:
                X = x1
                y_call = y1
                y_play = y2
            else:
                concat_inplace(X, x1)
                concat_inplace(y_call, y1)
                concat_inplace(y_play, y2)
            if X.shape[0] > 50000:
                print('Stop and train')
                bot.pretrain(X, y_call, y_play)
                save_bot(bot, args.bot_out)
                X, y_call, y_play = None, None, None
