import random

from .base import Bot

__all__ = [
    'init',
    'load',
    'save',
]


class RandomBot(Bot):
    def select_action(self, state):
        return random.choice(state.legal_actions())


def init(options, metadata):
    return RandomBot(metadata)


def save(_bot, _h5group):
    return


def load(_bot, metadata):
    return RandomBot(metadata)
