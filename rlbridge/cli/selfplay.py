import datetime
import os
from multiprocessing import Process, Queue

from tqdm import tqdm

from .command import Command


class QLogger:
    def __init__(self, log_q, src):
        self.q = log_q
        self.src = src

    def log(self, msg):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.q.put('{} ({}) {}'.format(ts, self.src, msg))


def train_and_evaluate(q, ref_fname, out_patt, logger):
    from ..import kerasutil
    kerasutil.set_tf_options(gpu_frac=0.4)

    from ..rl import TrainEvalLoop
    worker = TrainEvalLoop(q, ref_fname, out_patt, logger)
    worker.run()


def do_selfplay(q, logger, ref_fname):
    from ..import kerasutil
    kerasutil.set_tf_options(gpu_frac=0.4)

    from .. import bots
    from ..players import Player
    from ..simulate import simulate_game
    cur_bot = None
    try:
        while True:
            ref_path = open(ref_fname).read().strip()
            if ref_path != cur_bot:
                logger.log('Starting self-play with {}'.format(ref_path))
                cur_bot = ref_path
                bot = bots.load_bot(ref_path)
            game_result = simulate_game(bot, bot)
            # One game makes 4 episodes (from each player's perspective)
            q.put(bot.encode_episode(game_result, Player.north))
            q.put(bot.encode_episode(game_result, Player.east))
            q.put(bot.encode_episode(game_result, Player.south))
            q.put(bot.encode_episode(game_result, Player.west))
    finally:
        q.put(None)


def show_log(log_q):
    while True:
        msg = log_q.get()
        if msg is None:
            break
        print(msg)


class SelfPlay(Command):
    def register_arguments(self, parser):
        parser.add_argument('bot')
        parser.add_argument('checkpoint_out')

    def run(self, args):
        ref_fname = os.path.join(args.checkpoint_out, 'ref')
        with open(ref_fname, 'w') as outf:
            outf.write(args.bot)
        checkpoint_patt = os.path.join(args.checkpoint_out, 'checkpoint')

        q = Queue()
        log_q = Queue()
        logger_proc = Process(
            target=show_log,
            args=(log_q,)
        )
        logger_proc.start()
        train_proc = Process(
            target=train_and_evaluate,
            args=(
                q,
                ref_fname,
                checkpoint_patt,
                QLogger(log_q, 'trainer')
            )
        )
        train_proc.start()
        play_proc = Process(
            target=do_selfplay,
            args=(q, QLogger(log_q, 'selfplay'), ref_fname)
        )
        play_proc.start()
        play_proc.join()
        train_proc.join()
        log_q.put(None)
        logger_proc.join()
