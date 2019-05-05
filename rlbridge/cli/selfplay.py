from multiprocessing import Process, Queue

from tqdm import tqdm

from .command import Command


def train_and_evaluate(q, bot_infname, bot_outfname):
    print("TRAINER THREAD!!!")
    from ..import kerasutil
    kerasutil.set_tf_options(gpu_frac=0.4)

    print('Load that bot')
    from .. import bots
    bot = bots.load_bot(bot_infname)
    while True:
        episode = q.get()
        if episode is None:
            break
        bot.train_episode(episode)
    bots.save_bot(bot, bot_outfname)


def do_selfplay(q, num_games, bot_infname):
    print("PLAYER THREAD!!!")
    from ..import kerasutil
    kerasutil.set_tf_options(gpu_frac=0.4)

    from .. import bots
    from ..players import Player
    from ..simulate import simulate_game
    try:
        bot = bots.load_bot(bot_infname)
        for _ in tqdm(range(num_games)):
            game_result = simulate_game(bot, bot)
            # One game makes 4 episodes (from each player's perspective)
            q.put(bot.encode_episode(game_result, Player.north))
            q.put(bot.encode_episode(game_result, Player.east))
            q.put(bot.encode_episode(game_result, Player.south))
            q.put(bot.encode_episode(game_result, Player.west))
    finally:
        q.put(None)


class SelfPlay(Command):
    def register_arguments(self, parser):
        parser.add_argument('--num-games', type=int, default=2)
        parser.add_argument('bot')
        parser.add_argument('checkpoint_out')

    def run(self, args):
        q = Queue()
        train_proc = Process(
            target=train_and_evaluate,
            args=(q, args.bot, args.checkpoint_out))
        train_proc.start()
        play_proc = Process(
            target=do_selfplay,
            args=(q, args.num_games, args.bot))
        play_proc.start()
        play_proc.join()
        train_proc.join()
