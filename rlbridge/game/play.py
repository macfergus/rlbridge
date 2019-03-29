from ..cards import Card

__all__ = [
    'Play',
    'PlayState',
]


def beats(candidate_card, ref_card, trump_suit):
    if ref_card is None:
        return True
    if candidate_card.suit == ref_card.suit:
        return candidate_card.rank > ref_card.rank
    if candidate_card.suit == trump_suit:
        return True
    return False


class Trick:
    def __init__(self, next_player, cards, trump_suit):
        self.next_player = next_player
        self.cards = list(cards)
        self.trump_suit = trump_suit

    def apply(self, play):
        assert len(self.cards) < 4
        return Trick(
            next_player=self.next_player.rotate(),
            cards=self.cards + [play.card],
            trump_suit=self.trump_suit
        )

    def has_lead(self):
        return bool(self.cards)

    @property
    def lead(self):
        assert self.has_lead()
        return self.cards[0]

    def is_complete(self):
        return len(self.cards) == 4

    @classmethod
    def begin(self, opener, trump_suit):
        return Trick(next_player=opener, cards=[], trump_suit=trump_suit)

    def winner(self):
        assert self.is_complete()
        leader = None
        leader_i = None
        for i, card in enumerate(self.cards):
            if beats(card, leader, self.trump_suit):
                leader = card
                leader_i = i
        winning_player = self.next_player
        for _ in range(leader_i):
            winning_player = winning_player.rotate()
        return winning_player


class Play:
    def __init__(self, card):
        self.card = card

    @classmethod
    def of(cls, play_str):
        """Helper for tests."""
        return cls(Card.of(play_str))


class PlayState:
    def __init__(self, trump_suit, next_player, hands,
                 completed_tricks, current_trick):
        self.trump_suit = trump_suit
        self.next_player = next_player
        self.hands = hands
        self.completed_tricks = list(completed_tricks)
        self.current_trick = current_trick

    def is_legal(self, play):
        if play.card not in self.hands[self.next_player]:
            return False
        if self.current_trick.has_lead():
            lead_suit = self.current_trick.lead.suit
            if self.hands[self.next_player].has_suit(lead_suit):
                if play.card.suit != lead_suit:
                    return False
        return True

    def legal_plays(self):
        pass

    def apply(self, play):
        assert self.is_legal(play)
        next_trick = self.current_trick.apply(play)
        next_player = self.next_player.rotate()
        completed_tricks = self.completed_tricks
        if next_trick.is_complete():
            completed_tricks = completed_tricks + [next_trick]
            next_player = next_trick.winner()
            next_trick = Trick.begin(next_player, self.trump_suit)
        return PlayState(
            trump_suit=self.trump_suit,
            next_player=next_player,
            hands=self.hands,
            completed_tricks=completed_tricks,
            current_trick=next_trick)

    @classmethod
    def open_play(cls, auction_result, deal):
        next_player = auction_result.declarer.rotate()
        return cls(
            trump_suit=auction_result.trump,
            next_player=next_player,
            hands=deal.hands(),
            completed_tricks=[],
            current_trick=Trick.begin(next_player, auction_result.trump)
        )
