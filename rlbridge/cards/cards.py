from .suits import Suit


class Card:
    def __init__(self, rank, suit):
        assert 2 <= rank <= 14
        self.rank = rank
        self.suit = suit

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    @classmethod
    def of(cls, card_str):
        suit_str = card_str[-1:]
        rank_str = card_str[:-1].upper()
        if rank_str == 'J':
            rank = 11
        elif rank_str == 'Q':
            rank = 12
        elif rank_str == 'K':
            rank = 13
        elif rank_str == 'A':
            rank = 14
        else:
            rank = int(rank_str)
        return Card(rank, Suit.of(suit_str))

    def __str__(self):
        if self.rank == 11:
            rank_str = 'J'
        elif self.rank == 12:
            rank_str = 'Q'
        elif self.rank == 13:
            rank_str = 'K'
        elif self.rank == 14:
            rank_str = 'A'
        else:
            rank_str = str(self.rank)
        return '{}{}'.format(rank_str, str(self.suit))
