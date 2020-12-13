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
        if suit_str in ('D', 'd', '♦'):
            return Suit.diamonds
        if suit_str in ('H', 'h', '♥'):
            return Suit.hearts
        if suit_str in ('S', 's', '♠'):
            return Suit.spades
        raise ValueError(suit_str)

    def __str__(self):
        if self == Suit.clubs:
            return '♣'
        if self == Suit.diamonds:
            return '♦'
        if self == Suit.hearts:
            return '♥'
        if self == Suit.spades:
            return '♠'
        raise ValueError(self)
