import enum

__all__ = [
    'Player',
    'Side',
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
        """Check if the player is on the same team as another player.

        A player counts as its own teammate! So
        Player.north.is_teammate(Player.north)
        returns True.
        """
        if self in (Player.north, Player.south):
            return other in (Player.north, Player.south)
        return other in (Player.east, Player.west)

    def is_opponent(self, other):
        if self in (Player.north, Player.south):
            return other in (Player.east, Player.west)
        return other in (Player.north, Player.south)

    def __str__(self):
        if self == Player.north:
            return 'N'
        if self == Player.east:
            return 'E'
        if self == Player.south:
            return 'S'
        return 'W'


class Side(enum.Enum):
    north_south = 1
    east_west = 2


    def __str__(self):
        if self == Side.north_south:
            return 'NS'
        return 'EW'
