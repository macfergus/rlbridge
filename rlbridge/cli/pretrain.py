import random

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


class Pretrain(Command):
    def register_arguments(self, parser):
        parser.add_argument('--num-games', type=int, default=1)
        parser.add_argument('--simulate-bot')
        parser.add_argument('bot_in')
        parser.add_argument('bot_out')

    def run(self, args):
        bot = load_bot(args.bot_in)

        simulate_bot = init_bot('randombot', {}, {})
        X, y_call, y_play, y_value = None, None, None, None
        made = 0
        defended = 0
        for i in tqdm(range(args.num_games)):
            # Make sure the training data includes a wide range of
            # contracts. Without this limit, it will tend to land on
            # very high contracts.
            simulate_bot.set_option('max_contract', random.randint(1, 7))
            game_result = simulate_game(simulate_bot, simulate_bot)
            if game_result.declarer is None:
                continue
            if game_result.contract_made:
                made += 1
                p = random.choice([
                    game_result.declarer, game_result.declarer.partner
                ])
            else:
                # down-sample defended contracts
                if random.random() > 0.1:
                    continue
                defended += 1
                # Prefer the partnership that earned points, but not
                # 100% of the time
                if random.random() > 0.1:
                    p = random.choice([
                        game_result.declarer.lho(), game_result.declarer.rho(),
                    ])
                else:
                    p = random.choice([
                        game_result.declarer, game_result.declarer.partner,
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
            if X.shape[0] >= 15000:
                tqdm.write(f'Training: made {made} defended {defended}')
                made = 0
                defended = 0
                hist = bot.pretrain(
                    X, y_call, y_play, y_value,
                )
                call_loss = hist.history['call_output_loss'][0]
                play_loss = hist.history['play_output_loss'][0]
                value_loss = hist.history['value_output_loss'][0]
                tqdm.write(
                    f'after {i} games: ' +
                    f'call {call_loss:.3f} ' +
                    f'play {play_loss:.3f} ' +
                    f'value {value_loss:.3f}'
                )
                save_bot(bot, args.bot_out)
                X, y_call, y_play, y_value = None, None, None, None
