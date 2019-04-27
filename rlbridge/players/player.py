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

    def lho(self):
        """Return the left-hand opponent.

        This is the player who plays after this player."""
        return self.rotate()

    def rho(self):
        """Return the right-hand opponent.

        This is the player who plays before this player."""
        return self.rotate().rotate().rotate()

    def is_teammate(self, other):
        """Check if the player is on the same team as another player.

        A player counts as its own teammate! So
        Player.north.is_teammate(Player.north)
        returns True.
        """
        if self in (Player.north, Player.south):
            return other in (Player.north, Player.south)
        return other in (Player.east, Player.west)

    @property
    def partner(self):
        if self == Player.north:
            return Player.south
        if self == Player.south:
            return Player.north
        if self == Player.east:
            return Player.west
        return Player.east

    def is_opponent(self, other):
        if self in (Player.north, Player.south):
            return other in (Player.east, Player.west)
        return other in (Player.north, Player.south)

    def side(self):
        if self in (Player.north, Player.south):
            return Side.north_south
        return Side.east_west

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

    def opposite(self):
        if self == Side.north_south:
            return Side.east_west
        return Side.north_south
