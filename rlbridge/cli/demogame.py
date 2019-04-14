from .. import bots, cards
from ..game import GameState
from ..players import Player
from ..scoring import score_hand
from .command import Command


class DemoGame(Command):
    def run(self, args):
        agents = {
            Player.north: bots.randombot.RandomBot(),
            Player.east: bots.randombot.RandomBot(),
            Player.south: bots.randombot.RandomBot(),
            Player.west: bots.randombot.RandomBot(),
        }
        hand = GameState.new_deal(cards.new_deal(), dealer=Player.north)
        while not hand.is_over():
            next_player = hand.next_player
            next_decider = hand.next_decider
            agent = agents[next_decider]
            action = agent.select_action(hand.perspective(next_decider))
            hand = hand.apply(action)
        result = score_hand(hand)
        print(result)
