from ..cards import Suit


__all__ = [
    'Denomination',
    'ALL_DENOMINATIONS',
    'Bid',
    'ALL_BIDS',
    'Call',
    'Auction',
]


class Denomination:
    def __init__(self, trump_suit=None, is_notrump=False):
        assert (trump_suit is None) ^ (is_notrump is False)
        self.trump_suit = trump_suit
        self.is_notrump = is_notrump

    @classmethod
    def suit(cls, suit):
        return Denomination(trump_suit=suit)

    @classmethod
    def clubs(cls):
        return cls.suit(Suit.clubs)

    @classmethod
    def diamonds(cls):
        return cls.suit(Suit.diamonds)

    @classmethod
    def hearts(cls):
        return cls.suit(Suit.hearts)

    @classmethod
    def spades(cls):
        return cls.suit(Suit.spades)

    @classmethod
    def notrump(cls):
        return Denomination(is_notrump=True)

    def __lt__(self, other):
        if self.is_notrump:
            return False
        if other.is_notrump:
            return True
        return self.trump_suit.value < other.trump_suit.value

    def __str__(self):
        if self.trump_suit == Suit.clubs:
            return '♣'
        if self.trump_suit == Suit.diamonds:
            return '♦'
        if self.trump_suit == Suit.hearts:
            return '♥'
        if self.trump_suit == Suit.spades:
            return '♠'
        return 'NT'


ALL_DENOMINATIONS = (
    Denomination.suit(Suit.clubs),
    Denomination.suit(Suit.diamonds),
    Denomination.suit(Suit.hearts),
    Denomination.suit(Suit.spades),
    Denomination.notrump()
)


class Bid:
    def __init__(self, denomination, tricks):
        self.denomination = denomination
        self.tricks = tricks

    def __lt__(self, other_bid):
        if self.tricks < other_bid.tricks:
            return True
        elif other_bid.tricks < self.tricks:
            return False
        return self.denomination < other_bid.denomination

    def __str__(self):
        return '{}{}'.format(self.tricks, self.denomination)


ALL_BIDS = [
    Bid(denomination, tricks)
    for denomination in ALL_DENOMINATIONS
    for tricks in range(1, 8)
]


class Call:
    def __init__(self, bid=None, double=False, redouble=False, is_pass=False):
        assert (bid is not None) ^ double ^ redouble ^ is_pass
        self.bid = bid
        self.is_bid = (bid is not None)
        self.is_double = double
        self.is_redouble = redouble
        self.is_pass = is_pass

    @classmethod
    def bid(cls, the_bid):
        return Call(bid=the_bid)

    @classmethod
    def pass_turn(cls):
        return Call(is_pass=True)

    def __eq__(self, other):
        return self.is_pass == other.is_pass and \
            self.is_redouble == other.is_redouble and \
            self.is_double == other.is_double and \
            self.is_bid == other.is_bid and \
            self.bid == other.bid

    def __str__(self):
        if self.is_bid:
            return str(self.bid)
        if self.is_double:
            return 'X'
        if self.is_redouble:
            return 'XX'
        if self.is_pass:
            return 'pass'
        raise ValueError()


class Auction:
    def __init__(self, dealer, calls=None, last_bid=None, next_player=None):
        self.dealer = dealer
        if calls is None:
            self.calls = []
        else:
            self.calls = list(calls)
        self.last_bid = last_bid
        if next_player is None:
            self.next_player = dealer
        else:
            self.next_player = next_player

    @classmethod
    def new_auction(cls, dealer):
        return Auction(
            dealer=dealer
        )

    def is_over(self):
        pass_call = Call.pass_turn()
        return len(self.calls) >= 3 and \
            self.calls[-1] == pass_call and \
            self.calls[-2] == pass_call and \
            self.calls[-3] == pass_call

    def apply(self, call):
        return Auction(
            dealer=self.dealer,
            calls=self.calls + [call],
            last_bid=call.bid if call.is_bid else self.last_bid,
            next_player=self.next_player.rotate()
        )
