import random

from ..game import Action, Phase

__all__ = [
    'RandomBot',
]


class RandomBot:
    def select_action(self, state):
        if state.phase == Phase.auction:
            return Action.make_call(random.choice(state.legal_calls()))
