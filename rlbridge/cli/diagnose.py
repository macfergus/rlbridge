import random
from collections import Counter, namedtuple

import pandas as pd
from tqdm import tqdm

from .. import cards
from ..bots import load_bot
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
    return counter.most_common()[0]


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

            for _ in tqdm(range(750), leave=False):
                hand = GameState.new_deal(
                    cards.new_deal(),
                    dealer=Player.north,
                    northsouth_vulnerable=random.choice([True, False]),
                    eastwest_vulnerable=random.choice([True, False])
                )
                h = hand.deal.initial_hands[hand.next_decider]
                action = bot.select_action(hand)
                out = bot.last_outputs
                expected_value = out['value_output'][0]
                expected_tricks = out['tricks_output'][0]
                expected_contract = out['contract_made_output'][0]

                did_open = action.call.is_bid
                open_suit = None
                open_tricks = None
                if did_open:
                    open_suit = str(action.call.bid.denomination)
                    open_tricks = action.call.bid.tricks
                hcp = high_card_points(h)
                suit, length = longest_suit(h)

                results.append({
                    'bot': bot.identify(),
                    'did_open': int(did_open),
                    'open_suit': open_suit,
                    'open_tricks': open_tricks,
                    'hcp': hcp,
                    'longest_suit': suit,
                    'longest_length': length,
                    'ex_value': expected_value,
                    'ex_tricks': expected_tricks,
                    'ex_contract': expected_contract,
                })
        df = pd.DataFrame(results)
        df.to_csv(args.out)
