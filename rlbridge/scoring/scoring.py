from ..players import Player, Side

__all__ = [
    'score_hand',
]


def score_hand(state):
    assert state.is_over()
    tricks_won = {
        Side.north_south: 0,
        Side.east_west: 0,
    }
    for trick in state.playstate.completed_tricks:
        if trick.winner() in (Player.north, Player.south):
            tricks_won[Side.north_south] += 1
        else:
            tricks_won[Side.east_west] += 1
    return tricks_won
