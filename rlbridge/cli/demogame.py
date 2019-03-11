from ..game import GameState
from .command import Command


class DemoGame(Command):
    def run(self, args):
        print('Game!')
