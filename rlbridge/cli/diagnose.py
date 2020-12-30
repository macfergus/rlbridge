import random
from collections import namedtuple
from collections import Counter

import pandas as pd
from tqdm import tqdm

from ..bots import load_bot
from .. import cards
from ..game import GameState
from ..players import Player
from .command import Command


def high_card_points(hand):
    hcp = 0
    for card in hand:
        if card.rank == 11:
            hcp += 1
        elif card.rank == 12:
            hcp += 2
        elif card.rank == 13:
            hcp += 3
        elif card.rank == 14:
            hcp += 4
    return hcp


def longest_suit(hand):
    counter = Counter([card.suit for card in hand])
    return max(counter.values())


class Diagnose(Command):
    def register_arguments(self, parser):
        parser.add_argument('bot', nargs='+')
        parser.add_argument('--out', '-o')

    def run(self, args):
        results = []
        for bot_name in tqdm(args.bot):
            bot = load_bot(bot_name)
            tqdm.write(bot.identify())
            bot.set_option('temperature', 0.0)

            for _ in tqdm(range(1000), leave=False):
                hand = GameState.new_deal(
                    cards.new_deal(),
                    dealer=Player.north,
                    northsouth_vulnerable=random.choice([True, False]),
                    eastwest_vulnerable=random.choice([True, False])
                )
                h = hand.deal.initial_hands[hand.next_decider]
                action = bot.select_action(hand)

                did_open = action.call.is_bid
                hcp = high_card_points(h)
                length = longest_suit(h)

                results.append({
                    'bot': bot.identify(),
                    'did_open': int(did_open),
                    'hcp': hcp,
                    'longest_suit': length,
                })
        df = pd.DataFrame(results)
        df.to_csv(args.out)
