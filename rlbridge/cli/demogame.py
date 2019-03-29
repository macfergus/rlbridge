from .. import bots, cards
from ..game import GameState
from ..players import Player
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
            agent = agents[next_player]
            action = agent.select_action(hand.perspective(next_player))
            print('{}: {}'.format(next_player, action))
            hand = hand.apply(action)
