from collections import namedtuple

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


PlayerCard = namedtuple('PlayerCard', 'player card')


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
    def begin(cls, opener, trump_suit):
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

    def opener(self):
        num_cards_played = len(self.cards)
        if num_cards_played == 3:
            return self.next_player.rotate()
        if num_cards_played == 2:
            return self.next_player.rotate().rotate()
        if num_cards_played == 1:
            return self.next_player.rotate().rotate().rotate()
        return self.next_player

    def leader(self):
        leader = None
        lead_player = None
        player = self.opener()
        for card in self.cards:
            if beats(card, leader, self.trump_suit):
                leader = card
                lead_player = player
            player = player.rotate()
        return PlayerCard(player, card)


class Play:
    def __init__(self, card):
        self.card = card

    @classmethod
    def of(cls, play_str):
        """Helper for tests."""
        return cls(Card.of(play_str))

    def __str__(self):
        return str(self.card)


class PlayState:
    def __init__(self, trump_suit, dummy, next_player, hands,
                 completed_tricks, current_trick):
        self.trump_suit = trump_suit
        self.dummy = dummy
        self.next_player = next_player
        self.hands = hands
        self.completed_tricks = list(completed_tricks)
        self.current_trick = current_trick

    def visible_cards(self, player):
        visible = {player: self.hands[player]}
        if self.dummy_is_visible():
            visible[self.dummy] = self.hands[self.dummy]
        return visible

    def dummy_is_visible(self):
        return (
            len(self.completed_tricks) > 0 or
            self.current_trick.has_lead()
        )

    def is_legal(self, play):
        if play.card not in self.hands[self.next_player]:
            return False
        if self.current_trick.has_lead():
            lead_suit = self.current_trick.lead.suit
            if self.hands[self.next_player].has_suit(lead_suit):
                if play.card.suit != lead_suit:
                    return False
        return True

    def is_over(self):
        return self.hands[self.next_player].is_empty()

    def legal_plays(self):
        plays = []
        for card in self.hands[self.next_player]:
            if self.is_legal(Play(card)):
                plays.append(Play(card))
        return plays

    def apply(self, play):
        assert self.is_legal(play)
        next_hands = self.hands.after_removing(self.next_player, play.card)
        next_trick = self.current_trick.apply(play)
        next_player = self.next_player.rotate()
        completed_tricks = self.completed_tricks
        if next_trick.is_complete():
            completed_tricks = completed_tricks + [next_trick]
            next_player = next_trick.winner()
            next_trick = Trick.begin(next_player, self.trump_suit)
        return PlayState(
            trump_suit=self.trump_suit,
            dummy=self.dummy,
            next_player=next_player,
            hands=next_hands,
            completed_tricks=completed_tricks,
            current_trick=next_trick)

    @classmethod
    def open_play(cls, auction_result, deal):
        next_player = auction_result.declarer.rotate()
        dummy = auction_result.declarer.partner
        return cls(
            trump_suit=auction_result.trump,
            dummy=dummy,
            next_player=next_player,
            hands=deal.hands(),
            completed_tricks=[],
            current_trick=Trick.begin(next_player, auction_result.trump)
        )
