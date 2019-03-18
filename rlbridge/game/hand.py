import enum

from .auction import *
from .play import *

__all__ = [
    'Action',
    'GameState',
    'Phase',
]


class Phase(enum.Enum):
    auction = 1
    # Leading the first trick, before dummy is revealed
    opening = 2
    # Trick-taking after dummy is revealed
    play = 3


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
    def __init__(self, phase, auction, playstate, player):
        self.phase = phase
        self.auction = auction
        self.playstate = playstate
        self.player = player

    def legal_actions(self):
        if self.phase == Phase.auction:
            return [Action.make_call(call)
                    for call in self.auction.legal_calls()]
        return [Action.make_play(play)
                for play in self.playstate.legal_plays()]


class GameState:
    def __init__(self, deal, phase, auction, playstate):
        self.deal = deal
        self.phase = phase
        self.auction = auction
        self.playstate = playstate

    @property
    def next_player(self):
        if self.phase == Phase.auction:
            return self.auction.next_player
        return self.playstate.next_player

    @classmethod
    def new_deal(cls, deal, dealer):
        return GameState(
            deal=deal,
            phase=Phase.auction,
            auction=Auction.new_auction(dealer),
            playstate=None,
        )

    def is_over(self):
        return self.phase == Phase.play

    def perspective(self, player):
        return Perspective(
            phase=self.phase,
            auction=self.auction,
            playstate=self.playstate,
            player=player,
        )

    def apply(self, action):
        if self.phase == Phase.auction:
            return self.apply_call(action.call)
        raise ValueError(action)

    def apply_call(self, call):
        assert self.phase == Phase.auction
        next_phase = self.phase
        next_auction = self.auction.apply(call)
        playstate = None
        if next_auction.is_over():
            next_phase = Phase.opening
            playstate = PlayState.open_play(next_auction.result())
        return GameState(
            deal=self.deal,
            phase=next_phase,
            auction=next_auction,
            playstate=playstate
        )
