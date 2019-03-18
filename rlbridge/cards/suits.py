import enum

__all__ = [
    'Suit',
]


class Suit(enum.Enum):
    clubs = 1
    diamonds = 2
    hearts = 3
    spades = 4

    @classmethod
    def of(cls, suit_str):
        if suit_str in ('C', 'c', '♣'):
            return Suit.clubs
        elif suit_str in ('D', 'd', '♦'):
            return Suit.diamonds
        elif suit_str in ('H', 'h', '♥'):
            return Suit.hearts
        elif suit_str in ('S', 's', '♠'):
            return Suit.spades
        raise ValueError(suit_str)
