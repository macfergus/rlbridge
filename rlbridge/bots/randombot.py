import random

from .base import Bot, UnrecognizedOptionError

__all__ = [
    'init',
    'load',
    'save',
]


class RandomBot(Bot):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._max_contract = 7

    def set_option(self, key, value):
        if key == 'max_contract':
            self._max_contract = int(value)
        else:
            raise UnrecognizedOptionError(key)

    def select_action(self, state, recorder=None):
        while True:
            action = random.choice(state.legal_actions())
            if not action.is_call:
                break
            if not action.call.is_bid:
                break
            if action.call.bid.tricks <= self._max_contract:
                break
        return action


def init(options, metadata):
    return RandomBot(metadata)


def save(_bot, _h5group):
    return


def load(_bot, metadata):
    return RandomBot(metadata)
