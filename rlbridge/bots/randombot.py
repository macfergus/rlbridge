import random

from .base import Bot
from ..game import Action, Phase

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


def save(bot, h5group):
    return


def load(h5group, metadata):
    return RandomBot(metadata)
