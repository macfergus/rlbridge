import datetime
import os
import random
from multiprocessing import Process, Queue

from .command import Command


class QLogger:
    def __init__(self, log_q, src):
        self.q = log_q
        self.src = src

    def log(self, msg):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.q.put('{} ({}) {}'.format(ts, self.src, msg))


def train_and_evaluate(
        q, ref_fname, learn_fname, out_dir,
        gate,
        max_games, episodes_per_train,
        eval_games, eval_chunk, eval_threshold,
        logger):
    logger.log('Running in PID {}'.format(os.getpid()))
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    from ..selfplay import TrainEvalLoop
    worker = TrainEvalLoop(
        q, ref_fname, learn_fname, out_dir, logger,
        episodes_per_train=episodes_per_train,
        gate=gate,
        max_games=max_games,
        eval_games=eval_games,
        eval_chunk=eval_chunk,
        eval_threshold=eval_threshold
    )
    worker.run()


def do_selfplay(q, logger, bot_dir, max_contract):
    logger.log('Running in PID {}'.format(os.getpid()))
    #from ..import kerasutil
    #kerasutil.set_tf_options(gpu_frac=0.4)

    from .. import bots
    from ..players import Player
    from ..rl import ExperienceRecorder
    from ..simulate import simulate_game

    ref_fname = os.path.join(bot_dir, 'ref')
    learn_fname = os.path.join(bot_dir, 'learn')

    cur_ref_bot = None
    cur_learn_bot = None
    try:
        while True:
            ref_path = open(ref_fname).read().strip()
            if ref_path != cur_ref_bot:
                cur_ref_bot = ref_path
                ref_bot = bots.load_bot(ref_path)
                logger.log('Setting ref bot to {}'.format(ref_bot.identify()))
                logger.log('Setting max contract to {}'.format(max_contract))
                ref_bot.set_option('max_contract', max_contract)
                ref_bot.temperature = 0
                learn_bot = bots.load_bot(ref_path)
                logger.log('Setting learn bot to {}'.format(
                    learn_bot.identify()
                ))
                logger.log('Setting max contract to {}'.format(max_contract))
                learn_bot.temperature = 1.5
                num_games = 0

            recorder = ExperienceRecorder()
            learn_side = random.choice(['ns', 'ew'])
            if learn_side == 'ns':
                game_result = simulate_game(
                    learn_bot, ref_bot, ns_recorder=recorder)
            else:
                game_result = simulate_game(
                    ref_bot, learn_bot, ew_recorder=recorder)
            num_games += 1
            if num_games % 20 == 0:
                logger.log('Completed {} games with {}'.format(
                    num_games,
                    learn_bot.identify()
                ))
            # One game makes 2 episodes (from each player's perspective)
            if learn_side == 'ns':
                q.put(learn_bot.encode_episode(
                    game_result,
                    Player.north,
                    recorder.get_decisions(Player.north)
                ))
                q.put(learn_bot.encode_episode(
                    game_result,
                    Player.south,
                    recorder.get_decisions(Player.south)
                ))
            if learn_side == 'ew':
                q.put(learn_bot.encode_episode(
                    game_result,
                    Player.east,
                    recorder.get_decisions(Player.east)
                ))
                q.put(learn_bot.encode_episode(
                    game_result,
                    Player.west,
                    recorder.get_decisions(Player.west)
                ))
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
        parser.add_argument(
            '--max-games', type=int, default=10000,
            help='Restart the trainer process after this many games.'
        )
        parser.add_argument('--max-contract', type=int, default=7)
        parser.add_argument('--gate', dest='gate', action='store_true')
        parser.add_argument('--no-gate', dest='gate', action='store_false')
        parser.set_defaults(gate=True)
        parser.add_argument('--episodes-per-train', type=int, default=200)
        parser.add_argument('--eval-games', type=int, default=200)
        parser.add_argument('--eval-chunk', type=int, default=20)
        parser.add_argument('--eval-threshold', type=float, default=0.05)
        parser.add_argument('bot')
        parser.add_argument('checkpoint_out')

    def run(self, args):
        ref_fname = os.path.join(args.checkpoint_out, 'ref')
        learn_fname = os.path.join(args.checkpoint_out, 'learn')
        with open(ref_fname, 'w') as outf:
            outf.write(args.bot)
        with open(learn_fname, 'w') as outf:
            outf.write(args.bot)

        q = Queue()
        log_q = Queue()
        logger_proc = Process(
            target=show_log,
            args=(log_q,)
        )
        logger_proc.start()
        play_proc = Process(
            target=do_selfplay,
            args=(
                q,
                QLogger(log_q, 'selfplay'),
                args.checkpoint_out,
                args.max_contract
            )
        )
        play_proc.start()
        try:
            while True:
                # To prevent running out of memory, the training process
                # will shut itself down periodically. This loop restarts
                # it.
                train_proc = Process(
                    target=train_and_evaluate,
                    args=(
                        q,
                        ref_fname,
                        learn_fname,
                        args.checkpoint_out,
                        args.gate,
                        args.max_games,
                        args.episodes_per_train,
                        args.eval_games,
                        args.eval_chunk,
                        args.eval_threshold,
                        QLogger(log_q, 'trainer')
                    )
                )
                train_proc.start()
                train_proc.join()
        finally:
            play_proc.join()
            train_proc.join()
            log_q.put(None)
            logger_proc.join()
