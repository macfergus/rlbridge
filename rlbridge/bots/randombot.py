import random

from ..game import Action, Phase

__all__ = [
    'RandomBot',
]


class RandomBot:
    def select_action(self, state):
        return random.choice(state.legal_actions())
