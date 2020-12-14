import queue
import time

import yaml
from tqdm import tqdm

from ..nputil import concat_inplace
from ..simulate import GameGenerator
from .command import Command


class SimpleTrain(Command):
    def register_arguments(self, parser):
        parser.add_argument('--config', '-c', required=True)
        parser.add_argument('--num-games', type=int, default=1)
        parser.add_argument('bot_in')
        parser.add_argument('bot_out')

    def run(self, args):
        config = yaml.safe_load(open(args.config))

        gen = GameGenerator(args.bot_in, config)
        q = gen.recv_queue
        X, y_call, y_play, y_value = None, None, None, None

        gen.start()

        from ..bots import load_bot, save_bot
        bot = load_bot(args.bot_in)

        i = 0
        total = 0
        last_recv = time.time()
        with tqdm(total=args.num_games) as t:
            while total < args.num_games:
                gen.maintain()
                try:
                    episode = q.get(timeout=0.5)
                    last_recv = time.time()
                except queue.Empty:
                    time_since_recv = time.time() - last_recv
                    if time_since_recv > 10:
                        tqdm.write(
                            'Have not received an episode in 10 seconds; '
                            'shutting down'
                        )
                        break
                    continue
                i += 1
                total += 1
                t.update()

                if X is None:
                    X = episode.X.copy()
                    y_call = episode.y_call.copy()
                    y_play = episode.y_play.copy()
                    y_value = episode.y_value.copy()
                else:
                    concat_inplace(X, episode.X)
                    concat_inplace(y_call, episode.y_call)
                    concat_inplace(y_play, episode.y_play)
                    concat_inplace(y_value, episode.y_value)

                if (
                        X.shape[0] >= config['training']['chunk_size'] or
                        total + 1 >= args.num_games
                ):
                    tqdm.write(f'Training on {X.shape[0]} examples')
                    hist = bot.pretrain(
                        X, y_call, y_play, y_value,
                    )
                    call_loss = hist.history['call_output_loss'][0]
                    play_loss = hist.history['play_output_loss'][0]
                    value_loss = hist.history['value_output_loss'][0]
                    tqdm.write(
                        f'after {total} games: ' +
                        f'call {call_loss:.3f} ' +
                        f'play {play_loss:.3f} ' +
                        f'value {value_loss:.3f}'
                    )
                    bot.add_games(i)
                    i = 0
                    save_bot(bot, args.bot_out)
                    X, y_call, y_play, y_value = None, None, None, None
        gen.stop()
