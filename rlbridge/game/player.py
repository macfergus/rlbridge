import enum

__all__ = [
    'Player',
]


class Player(enum.Enum):
    north = 1
    east = 2
    south = 3
    west = 4
