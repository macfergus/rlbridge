import random

from .. import cards
from ..bots import load_bot
from ..game import GameState
from ..io import GamePrinter
from ..players import Player
from ..scoring import score_hand
from .command import Command


class DemoGame(Command):
    def register_arguments(self, parser):
        parser.add_argument('northsouth_bot')
        parser.add_argument('eastwest_bot')

    def run(self, args):
        agents = {
            Player.north: load_bot(args.northsouth_bot),
            Player.east: load_bot(args.eastwest_bot),
            Player.south: load_bot(args.northsouth_bot),
            Player.west: load_bot(args.eastwest_bot),
        }
        print('{} vs {}'.format(
            agents[Player.north].name(),
            agents[Player.east].name()
        ))
        hand = GameState.new_deal(
            cards.new_deal(),
            dealer=Player.north,
            northsouth_vulnerable=random.choice([True, False]),
            eastwest_vulnerable=random.choice([True, False])
        )
        p = GamePrinter()
        while not hand.is_over():
            next_player = hand.next_player
            next_decider = hand.next_decider
            agent = agents[next_decider]
            action = agent.select_action(hand)
            hand = hand.apply(action)
        p.show_game(hand)
        result = score_hand(hand)
        print(result)
