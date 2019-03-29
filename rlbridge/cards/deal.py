import copy

__all__ = [
    'Deal',
    'new_deal',
]


class Hand:
    def __init__(self, cards):
        self.cards = frozenset(cards)

    def has_suit(self, suit):
        for card in self.cards:
            if card.suit == suit:
                return True
        return False

    def __contains__(self, card):
        return card in self.cards

    def __str__(self):
        return ' '.join(str(card) for card in self.cards)

    def without(self, card):
        assert card in self.cards
        return Hand(self.cards - {card})


class Hands:
    def __init__(self, hands):
        self.hands = dict(hands)

    def __getitem__(self, player):
        return self.hands[player]

    def after_removing(self, player, card):
        next_hands = copy.copy(self.hands)
        next_hands[player] = self.hands[player].without(card)
        return Hands(next_hands)


class Deal:
    def __init__(self, initial_hands):
        self.initial_hands = initial_hands

    def hands(self):
        return self.initial_hands

    @classmethod
    def from_dict(cls, card_dict):
        return cls(
            Hands({player: Hand(cards) for player, cards in card_dict.items()})
        )


def new_deal():
    pass
