import random

from .. import bots, cards
from ..game import GameState
from ..io import GamePrinter
from ..players import Player
from ..scoring import score_hand
from .command import Command


class DemoGame(Command):
    def run(self, args):
        agents = {
            Player.north: bots.lstm.LSTMBot(),
            Player.east: bots.randombot.RandomBot(),
            Player.south: bots.lstm.LSTMBot(),
            Player.west: bots.randombot.RandomBot(),
        }
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
