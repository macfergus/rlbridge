import sys

from ..cards import Suit
from ..game import Scale
from ..players import Player

__all__ = [
    'GamePrinter',
    'format_hand',
]


def format_scale(scale):
    if scale == Scale.doubled:
        return 'X'
    if scale == Scale.redoubled:
        return 'XX'
    return ''


def pad(s, n):
    padding = max(0, n - len(s))
    return s + padding * ' '


def format_rank(rank):
    if rank == 11:
        return 'J'
    if rank == 12:
        return 'Q'
    if rank == 13:
        return 'K'
    if rank == 14:
        return 'A'
    return str(rank)


def format_cards(suit, cards):
    """Format a set of cards that are the same suit."""
    ranks = [c.rank for c in cards]
    ranks.sort(reverse=True)
    return str(suit) + ' ' + ' '.join(format_rank(r) for r in ranks)


def format_hand(hand):
    clubs = [c for c in hand if c.suit == Suit.clubs]
    diamonds = [c for c in hand if c.suit == Suit.diamonds]
    hearts = [c for c in hand if c.suit == Suit.hearts]
    spades = [c for c in hand if c.suit == Suit.spades]
    return [
        format_cards(Suit.spades, spades),
        format_cards(Suit.hearts, hearts),
        format_cards(Suit.diamonds, diamonds),
        format_cards(Suit.clubs, clubs),
    ]


class GamePrinter:
    def __init__(self, outf=None):
        if outf is None:
            outf = sys.stdout
        self.outf = outf

    def show_deal(self, deal):
        hands = deal.hands()
        north_str = format_hand(hands[Player.north])
        east_str = format_hand(hands[Player.east])
        south_str = format_hand(hands[Player.south])
        west_str = format_hand(hands[Player.west])

        max_west = max(len(s) for s in west_str)
        max_ns = max(len(s) for s in north_str + south_str)
        l_padding = (max_west + 2) * ' '
        center_padding = (max_ns + 2) * ' '
        for row in north_str:
            self.outf.write(l_padding + row + '\n')
        for i in range(4):
            self.outf.write('{}{}{}\n'.format(
                pad(west_str[i], max_west + 2),
                center_padding,
                east_str[i]
            ))
        for row in south_str:
            self.outf.write(l_padding + row + '\n')

    def show_auction(self, auction):
        player = auction.dealer
        for i, call in enumerate(auction.calls):
            call_str = '{}: {}'.format(player, call)
            self.outf.write(pad(call_str, 9))
            player = player.rotate()
            if i % 4 == 3:
                self.outf.write('\n')
        if len(auction.calls) % 4 != 0:
            self.outf.write('\n')

    def show_trick(self, trick):
        assert trick.is_complete()
        player = trick.next_player
        for card in trick.cards:
            trick_str = '{}: {}'.format(player, card)
            self.outf.write(pad(trick_str, 8))
            player = player.rotate()
        self.outf.write('-> {} wins\n'.format(trick.winner()))

    def show_play(self, playstate):
        for trick in playstate.completed_tricks:
            self.show_trick(trick)

    def show_game(self, state):
        self.show_deal(state.deal)
        self.show_auction(state.auction)
        if not state.auction.has_contract():
            # No one bid, so there's no play to show.
            return
        auction_result = state.auction.result()
        self.outf.write('\n')
        self.outf.write('Contract: {} {}\n'.format(
            auction_result.bid,
            format_scale(auction_result.scale)))
        self.outf.write('Declarer: {}\n'.format(auction_result.declarer))
        self.show_play(state.playstate)
