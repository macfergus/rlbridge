import random

from .. import cards
from ..bots import load_bot
from ..game import GameState
from ..io import GamePrinter, parse_options
from ..players import Player
from ..scoring import score_hand
from .command import Command


class DemoGame(Command):
    def register_arguments(self, parser):
        parser.add_argument('--diagnostics', action='store_true')
        parser.add_argument('--options')
        parser.add_argument('northsouth_bot')
        parser.add_argument('eastwest_bot')

    def run(self, args):
        opts = {}
        if args.options:
            opts = parse_options(args.options)
        ns_bot = load_bot(args.northsouth_bot)
        ew_bot = load_bot(args.eastwest_bot)
        for key, value in opts.items():
            ns_bot.set_option(key, value)
            ew_bot.set_option(key, value)
        agents = {
            Player.north: ns_bot,
            Player.east: ew_bot,
            Player.south: ns_bot,
            Player.west: ew_bot,
        }
        print('{} vs {}'.format(
            agents[Player.north].identify(),
            agents[Player.east].identify()
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
            if args.diagnostics:
                diagnostics = agent.get_diagnostics()
                print('Player: {}'.format(next_decider))
                for key, value in diagnostics.items():
                    print(' * {}: {}'.format(key, value))
                print('Chose: {}'.format(action))
                print()
            hand = hand.apply(action)
        p.show_game(hand)
        result = score_hand(hand)
        print(result)
