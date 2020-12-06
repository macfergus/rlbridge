import copy
import datetime
import json
import os
import random
from multiprocessing import Process, Queue

import numpy as np

from .command import Command


class QLogger:
    def __init__(self, log_q, src):
        self.q = log_q
        self.src = src

    def log(self, msg):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.q.put('{} ({}) {}'.format(ts, self.src, msg))


def train_and_evaluate(
        q, pool_fname, out_dir,
        gate,
        max_games, episodes_per_train,
        max_contract,
        eval_games, eval_chunk, eval_threshold,
        logger):
    logger.log('Running in PID {}'.format(os.getpid()))
    from .. import kerasutil
    kerasutil.set_tf_options(limit_memory=True)

    from ..selfplay import TrainEvalLoop
    worker = TrainEvalLoop(
        q, pool_fname, out_dir, logger,
        episodes_per_train=episodes_per_train,
        max_contract=max_contract,
        gate=gate,
        max_games=max_games,
        eval_games=eval_games,
        eval_chunk=eval_chunk,
        eval_threshold=eval_threshold
    )
    worker.run()


class BotPool:
    def __init__(self, fname, logger):
        self._fname = fname
        self._ref_bot_names = None
        self._learn_bot_name = None
        self.logger = logger

    def refresh(self):
        from .. import bots
        new_learner = False
        data = json.load(open(self._fname))
        if self._ref_bot_names != data['ref']:
            self.logger.log('Updating reference bot pool')
            self._ref_bot_names = copy.copy(data['ref'])
            self._ref_bots = []
            self._ref_weights = []
            for i, bot_file in enumerate(self._ref_bot_names):
                ref_bot = bots.load_bot(bot_file)
                self._ref_bots.append(ref_bot)
                self._ref_weights.append(i + 1)
                self.logger.log('=> Loaded {} with weight {}'.format(
                    ref_bot.identify(),
                    i + 1
                ))
            self._ref_weights = (
                np.array(self._ref_weights) / np.sum(self._ref_weights)
            )
        if self._learn_bot_name != data['learn']:
            new_learner = True
            self.logger.log('Updating learn bot')
            self._learn_bot_name = copy.copy(data['learn'])
            self._learn_bot = bots.load_bot(self._learn_bot_name)
            self.logger.log('=> Loaded {} as new learner'.format(
                self._learn_bot.identify()
            ))
        return new_learner

    def select_ref_bot(self):
        bot_idx = np.random.choice(len(self._ref_bots), p=self._ref_weights)
        return self._ref_bots[bot_idx]

    def get_learn_bot(self):
        return self._learn_bot


def do_selfplay(q, logger, bot_dir, max_contract):
    logger.log('Running in PID {}'.format(os.getpid()))
    from ..import kerasutil
    kerasutil.set_tf_options(limit_memory=True)

    from .. import bots
    from ..players import Player
    from ..rl import ExperienceRecorder
    from ..simulate import simulate_game

    pool_fname = os.path.join(bot_dir, 'bot_status')

    bot_pool = BotPool(pool_fname, logger)
    num_games = 0

    try:
        while True:
            if bot_pool.refresh():
                num_games = 0
                discarded = 0

            learn_bot = bot_pool.get_learn_bot()
            ref_bot = bot_pool.select_ref_bot()

            ref_bot.set_option('max_contract', max_contract)
            ref_bot.temperature = 0
            learn_bot.set_option('max_contract', max_contract)
            learn_bot.temperature = 1.5

            recorder = ExperienceRecorder()
            learn_side = random.choice(['ns', 'ew'])
            if learn_side == 'ns':
                game_result = simulate_game(
                    learn_bot, ref_bot, ns_recorder=recorder)
            else:
                game_result = simulate_game(
                    ref_bot, learn_bot, ew_recorder=recorder)
            # Oversample made contracts
            learner_declared = (
                (
                    learn_side == 'ns' and
                    game_result.declarer in (Player.north, Player.south)
                ) or (
                    learn_side == 'ew' and
                    game_result.declarer in (Player.east, Player.west)
                )
            )
            contract_made = game_result.contract_made
            p_keep = 0.0
            if learner_declared and contract_made:
                p_keep = 1.0
            elif learner_declared and (not contract_made):
                p_keep = 0.2
            elif (not learner_declared) and contract_made:
                # learner defended and lost
                p_keep = 0.04
            elif (not learner_declared) and (not contract_made):
                # learner defended and won
                p_keep = 0.2
            if np.random.random() >= p_keep:
                discarded += 1
                continue
            num_games += 1
            if num_games % 20 == 0:
                logger.log('Completed {} games with {}; discarded {}'.format(
                    num_games,
                    learn_bot.identify(),
                    discarded
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
        pool_fname = os.path.join(args.checkpoint_out, 'bot_status')
        if not os.path.exists(pool_fname):
            with open(pool_fname, 'w') as outf:
                outf.write(json.dumps({
                    'ref': [args.bot],
                    'learn': args.bot,
                }))

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
                        pool_fname,
                        args.checkpoint_out,
                        args.gate,
                        args.max_games,
                        args.episodes_per_train,
                        args.max_contract,
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
