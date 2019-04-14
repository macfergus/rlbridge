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
    play = 2


class Action:
    def __init__(self, call=None, play=None):
        assert (call is not None) ^ (play is not None)
        self.call = call
        self.play = play
        self.is_call = call is not None
        self.is_play = play is not None

    @classmethod
    def make_call(cls, call):
        return Action(call=call)

    @classmethod
    def make_play(cls, play):
        return Action(play=play)

    def __str__(self):
        if self.is_call:
            return str(self.call)
        if self.is_play:
            return str(self.play)
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
    def __init__(self, deal, northsouth_vulnerable, eastwest_vulnerable,
                 phase, auction, playstate):
        self.deal = deal
        self.northsouth_vulnerable = northsouth_vulnerable
        self.eastwest_vulnerable = eastwest_vulnerable
        self.phase = phase
        self.auction = auction
        self.playstate = playstate

    @property
    def next_player(self):
        if self.phase == Phase.auction:
            return self.auction.next_player
        return self.playstate.next_player

    @property
    def next_decider(self):
        if self.phase == Phase.auction:
            return self.auction.next_player
        auction_result = self.auction.result()
        dummy = auction_result.declarer.partner
        if self.playstate.next_player == dummy:
            return auction_result.declarer
        return self.playstate.next_player

    @classmethod
    def new_deal(cls, deal, dealer,
                 northsouth_vulnerable, eastwest_vulnerable):
        return GameState(
            deal=deal,
            northsouth_vulnerable=northsouth_vulnerable,
            eastwest_vulnerable=eastwest_vulnerable,
            phase=Phase.auction,
            auction=Auction.new_auction(dealer),
            playstate=None,
        )

    def is_over(self):
        return self.phase == Phase.play and self.playstate.is_over()

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
        if self.phase == Phase.play:
            return self.apply_play(action.play)
        raise ValueError(action)

    def apply_call(self, call):
        assert self.phase == Phase.auction
        next_phase = self.phase
        next_auction = self.auction.apply(call)
        playstate = None
        if next_auction.is_over():
            next_phase = Phase.play
            playstate = PlayState.open_play(next_auction.result(), self.deal)
        return GameState(
            deal=self.deal,
            northsouth_vulnerable=self.northsouth_vulnerable,
            eastwest_vulnerable=self.eastwest_vulnerable,
            phase=next_phase,
            auction=next_auction,
            playstate=playstate
        )

    def apply_play(self, play):
        assert self.phase == Phase.play
        next_playstate = self.playstate.apply(play)
        return GameState(
            deal=self.deal,
            northsouth_vulnerable=self.northsouth_vulnerable,
            eastwest_vulnerable=self.eastwest_vulnerable,
            phase=self.phase,
            auction=self.auction,
            playstate=next_playstate
        )
