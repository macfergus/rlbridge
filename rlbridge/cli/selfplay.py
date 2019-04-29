import random

import numpy as np
from tqdm import tqdm

from .. import bots
from ..players import Player
from ..rl import ExperienceSaver
from ..simulate import simulate_game
from .command import Command


class SelfPlay(Command):
    def register_arguments(self, parser):
        parser.add_argument('--num-games', type=int, default=1)
        parser.add_argument('bot')
        parser.add_argument('experience_out')

    def run(self, args):
        bot = bots.load_bot(args.bot)
        with ExperienceSaver(args.experience_out) as exp_sink:
            for _ in tqdm(range(args.num_games)):
                game_result = simulate_game(bot, bot)
                # One game makes 4 episodes (from each player's perspective)
                exp_sink.record_episode(bot.encode_episode(
                    game_result, Player.north
                ))
                exp_sink.record_episode(bot.encode_episode(
                    game_result, Player.south
                ))
                exp_sink.record_episode(bot.encode_episode(
                    game_result, Player.east
                ))
                exp_sink.record_episode(bot.encode_episode(
                    game_result, Player.west
                ))
