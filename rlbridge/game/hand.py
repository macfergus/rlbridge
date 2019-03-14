import enum

from .auction import *

__all__ = [
    'Action',
    'GameState',
    'Phase',
]


class Phase(enum.Enum):
    auction = 1


class Action:
    def __init__(self, call=None):
        self.call = call
        self.is_call = call is not None

    @classmethod
    def make_call(cls, call):
        return Action(call=call)

    def __str__(self):
        if self.is_call:
            return str(self.call)
        assert False


class Perspective:
    def __init__(self, phase, auction):
        self.phase = phase
        self.auction = auction

    def legal_actions(self):
        if self.phase == Phase.auction:
            return [Action.make_call(call)
                    for call in self.auction.legal_calls()]
        return []


class GameState:
    def __init__(self, auction):
        self.phase = Phase.auction
        self.auction = auction

    @property
    def next_player(self):
        if self.phase == Phase.auction:
            return self.auction.next_player
        return None

    @classmethod
    def new_hand(cls, deal, dealer):
        return GameState(
            auction=Auction.new_auction(dealer),
        )

    def is_over(self):
        return self.auction.is_over()

    def perspective(self, player):
        return Perspective(
            phase=self.phase,
            auction=self.auction,
        )

    def apply(self, action):
        if self.phase == Phase.auction:
            return self.apply_call(action.call)
        raise ValueError(action)

    def apply_call(self, call):
        assert self.phase == Phase.auction
        return GameState(
            auction=self.auction.apply(call),
        )
