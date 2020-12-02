import random

from keras.callbacks import Callback
from tqdm import tqdm

from ..bots import init_bot, load_bot, save_bot
from ..nputil import concat_inplace
from ..players import Player
from ..simulate import simulate_game
from .command import Command


class Accumulator:
    def __init__(self):
        self.history = []

    def append(self, value):
        self.history.append(value)
        self.history = self.history[-20:]

    def avg(self):
        return sum(self.history) / len(self.history)


class TQDMCallback(Callback):
    def __init__(self, pbar, n_examples):
        self.pbar = pbar
        self.n_examples = n_examples
        self._set_total = False
        self.call_loss = Accumulator()
        self.play_loss = Accumulator()
        self.value_loss = Accumulator()

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            return
        if not self._set_total:
            if 'size' in logs:
                self.pbar.total = self.n_examples // logs['size']
                self._set_total = True
        self.pbar.update()
        self.call_loss.append(logs.get('dense_1_loss', 0))
        self.play_loss.append(logs.get('dense_2_loss', 0))
        self.value_loss.append(logs.get('dense_3_loss', 0))
        self.pbar.set_postfix(
            call=self.call_loss.avg(),
            play=self.play_loss.avg(),
            value=self.value_loss.avg(),
        )


class Pretrain(Command):
    def register_arguments(self, parser):
        parser.add_argument('--num-games', type=int, default=1)
        parser.add_argument('bot_in')
        parser.add_argument('bot_out')

    def run(self, args):
        bot = load_bot(args.bot_in)

        simulate_bot = init_bot('randombot', {}, {})
        X, y_call, y_play, y_value = None, None, None, None
        for _ in tqdm(range(args.num_games)):
            # Make sure the training data includes a wide range of
            # contracts. Without this limit, it will tend to land on
            # very high contracts.
            simulate_bot.set_option('max_contract', random.randint(1, 7))
            game_result = simulate_game(simulate_bot, simulate_bot)
            p = random.choice([
                Player.north, Player.east, Player.west, Player.south
            ])
            x1, y1, y2, y3 = bot.encode_pretraining(game_result, p)
            if X is None:
                X = x1
                y_call = y1
                y_play = y2
                y_value = y3
            else:
                concat_inplace(X, x1)
                concat_inplace(y_call, y1)
                concat_inplace(y_play, y2)
                concat_inplace(y_value, y3)
            if X.shape[0] > 50000:
                with tqdm() as trainbar:
                    bot.pretrain(
                        X, y_call, y_play, y_value,
                        callback=TQDMCallback(trainbar, X.shape[0])
                    )
                save_bot(bot, args.bot_out)
                X, y_call, y_play, y_value = None, None, None, None
