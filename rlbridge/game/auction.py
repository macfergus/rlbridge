import enum

from ..cards import Suit

__all__ = [
    'Denomination',
    'ALL_DENOMINATIONS',
    'Bid',
    'ALL_BIDS',
    'Call',
    'Auction',
    'Scale',
]


class Scale(enum.Enum):
    undoubled = 1
    doubled = 2
    redoubled = 3


class Contract:
    def __init__(self, declarer, bid, scale):
        self.declarer = declarer
        self.bid = bid
        self.scale = scale

    @property
    def trump(self):
        return self.bid.denomination.trump_suit


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

    def is_minor(self):
        return self.trump_suit in (Suit.clubs, Suit.diamonds)

    def is_major(self):
        return self.trump_suit in (Suit.hearts, Suit.spades)

    def __lt__(self, other):
        if self.is_notrump:
            return False
        if other.is_notrump:
            return True
        return self.trump_suit.value < other.trump_suit.value

    def __eq__(self, other):
        return self.trump_suit == other.trump_suit and \
            self.is_notrump == other.is_notrump

    def __hash__(self):
        return hash((self.trump_suit, self.is_notrump))

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

    @classmethod
    def of(cls, denom_str):
        """Helper for testing and such."""
        if denom_str == 'NT':
            return Denomination.notrump()
        return Denomination.suit(Suit.of(denom_str))


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

    @classmethod
    def of(cls, bid_str):
        tricks = int(bid_str[0])
        denom_str = bid_str[1:]
        if denom_str in ('C', 'c', '♣'):
            denomination = Denomination.suit(Suit.clubs)
        elif denom_str in ('D', 'd', '♦'):
            denomination = Denomination.suit(Suit.diamonds)
        elif denom_str in ('H', 'h', '♥'):
            denomination = Denomination.suit(Suit.hearts)
        elif denom_str in ('S', 's', '♠'):
            denomination = Denomination.suit(Suit.spades)
        elif denom_str in ('NT', 'nt'):
            denomination = Denomination.notrump()
        else:
            raise ValueError(denom_str)
        return Bid(denomination, tricks)


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
    def make_bid(cls, the_bid):
        return Call(bid=the_bid)

    @classmethod
    def pass_turn(cls):
        return Call(is_pass=True)

    @classmethod
    def double(cls):
        return Call(double=True)

    @classmethod
    def redouble(cls):
        return Call(redouble=True)

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

    @classmethod
    def of(cls, call_str):
        """Helper for tests."""
        if call_str in ('X', 'x'):
            return Call.double()
        if call_str in ('XX', 'xx'):
            return Call.redouble()
        if call_str == 'pass':
            return Call.pass_turn()
        return Call.make_bid(Bid.of(call_str))


ALL_CALLS = [Call.make_bid(bid) for bid in ALL_BIDS] + \
    [Call.double(), Call.redouble(), Call.pass_turn()]


class Auction:
    def __init__(self, dealer, calls=None,
                 last_bid=None, last_bidder=None, last_scale=Scale.undoubled,
                 next_player=None):
        self.dealer = dealer
        if calls is None:
            self.calls = []
        else:
            self.calls = list(calls)
        self.last_bid = last_bid
        self.last_bidder = last_bidder
        self.last_scale = last_scale
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
        return len(self.calls) > 3 and \
            self.calls[-1] == pass_call and \
            self.calls[-2] == pass_call and \
            self.calls[-3] == pass_call

    def has_contract(self):
        return self.is_over() and self.last_bid is not None

    def result(self):
        """Return the result of the auction.

        If there were no bids, returns None.
        """
        assert self.is_over()
        if self.last_bid is None:
            return None
        # Figure out the declarer. The declarer is the first person to
        # on the winning partnership to name the denomination of the
        # winning bid.
        declarer = None
        winning_denom = self.last_bid.denomination
        player = self.dealer
        for call in self.calls:
            if call.is_bid and call.bid.denomination == winning_denom and \
                    player.is_teammate(self.last_bidder):
                declarer = player
                break
            player = player.rotate()
        assert declarer is not None
        return Contract(
            declarer=declarer,
            bid=self.last_bid,
            scale=self.last_scale
        )

    def apply(self, call):
        assert not self.is_over()
        next_scale = self.last_scale
        if call.is_bid:
            next_scale = Scale.undoubled
        if call.is_double:
            next_scale = Scale.doubled
        if call.is_redouble:
            next_scale = Scale.redoubled
        return Auction(
            dealer=self.dealer,
            calls=self.calls + [call],
            last_bid=call.bid if call.is_bid else self.last_bid,
            last_bidder=self.next_player if call.is_bid else self.last_bidder,
            last_scale=next_scale,
            next_player=self.next_player.rotate()
        )

    def is_legal(self, call):
        """Check if a call is legal in this auction."""
        if self.is_over():
            return False
        if call.is_pass:
            return True
        if call.is_bid:
            return self.last_bid is None or self.last_bid < call.bid
        if call.is_double:
            return self.last_bid is not None and \
                self.last_bidder.is_opponent(self.next_player) and \
                self.last_scale == Scale.undoubled
        if call.is_redouble:
            return self.last_bid is not None and \
                self.last_bidder.is_teammate(self.next_player) and \
                self.last_scale == Scale.doubled
        return False

    def legal_calls(self):
        return [call for call in ALL_CALLS if self.is_legal(call)]
