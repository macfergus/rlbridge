import enum

__all__ = [
    'Player',
]


class Player(enum.Enum):
    north = 1
    east = 2
    south = 3
    west = 4

    def rotate(self):
        """Return the player who has the next turn after this player."""
        if self == Player.north:
            return Player.east
        if self == Player.east:
            return Player.south
        if self == Player.south:
            return Player.west
        return Player.north

    def is_teammate(self, other):
        if self in (Player.north, Player.south):
            return other in (Player.north, Player.south)
        return other in (Player.east, Player.west)
