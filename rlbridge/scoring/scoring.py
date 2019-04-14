from ..game import Scale
from ..players import Player, Side

__all__ = [
    'score_hand',
]


class Score:
    def __init__(self, declarer, defender):
        self.declarer = declarer
        self.defender = defender

    def __repr__(self):
        return 'Score({}, {})'.format(self.declarer, self.defender)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return (
            self.declarer == other.declarer and
            self.defender == other.defender
        )


class DealResult:
    def __init__(self, bid, scale, vulnerable, tricks_won):
        self.bid = bid
        self.scale = scale
        self.vulnerable = vulnerable
        self.tricks_won = tricks_won


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


def trick_points(deal_result):
    odd_tricks = deal_result.tricks_won - 6
    if odd_tricks <= 0:
        return 0
    bid_tricks = deal_result.bid.tricks
    overtricks = odd_tricks - bid_tricks
    doubled_ot_bonus = 200 if deal_result.vulnerable else 100
    redoubled_ot_bonus = 400 if deal_result.vulnerable else 200
    if deal_result.bid.denomination.is_minor():
        if deal_result.scale == Scale.undoubled:
            return 20 * odd_tricks
        if deal_result.scale == Scale.doubled:
            return 40 * bid_tricks + doubled_ot_bonus * overtricks
        # else redoubled
        return 80 * bid_tricks + redoubled_ot_bonus * overtricks
    if deal_result.bid.denomination.is_major():
        if deal_result.scale == Scale.undoubled:
            return 30 * odd_tricks
        if deal_result.scale == Scale.doubled:
            return 60 * bid_tricks + doubled_ot_bonus * overtricks
        # else redoubled
        return 120 * bid_tricks + redoubled_ot_bonus * overtricks
    # notrump
    if deal_result.scale == Scale.undoubled:
        return 10 + 30 * odd_tricks
    if deal_result.scale == Scale.doubled:
        return 20 + 60 * bid_tricks + doubled_ot_bonus * overtricks
    # else redoubled
    return 40 + 120 * bid_tricks + redoubled_ot_bonus * overtricks


def is_contract_game(deal_result):
    return trick_points(DealResult(
        bid=deal_result.bid,
        scale=deal_result.scale,
        vulnerable=deal_result.vulnerable,
        tricks_won=deal_result.bid.tricks + 6
    )) >= 100


def game_points(deal_result):
    if deal_result.tricks_won < deal_result.bid.tricks + 6:
        return 0
    if is_contract_game(deal_result):
        if deal_result.vulnerable:
            return 500
        return 300
    return 50  # part score


def slam_points(deal_result):
    if deal_result.tricks_won < deal_result.bid.tricks + 6:
        return 0
    if deal_result.bid.tricks == 6:
        return 750 if deal_result.vulnerable else 500
    if deal_result.bid.tricks == 7:
        return 1500 if deal_result.vulnerable else 1000
    return 0


def undertrick_points(deal_result):
    undertricks = deal_result.bid.tricks + 6 - deal_result.tricks_won
    if undertricks < 1:
        return 0
    if deal_result.scale == Scale.undoubled:
        if deal_result.vulnerable:
            return 100 * undertricks
        return 50 * undertricks
    if deal_result.scale == Scale.doubled:
        if deal_result.vulnerable:
            return (
                200 * undertricks +
                100 * max(undertricks - 1, 0)
            )
        return (
            100 * undertricks +
            100 * max(undertricks - 1, 0) +
            100 * max(undertricks - 3, 0)
        )
    # else redoubled
    if deal_result.vulnerable:
        return (
            400 * undertricks +
            200 * max(undertricks - 1, 0)
        )
    return (
        200 * undertricks +
        200 * max(undertricks - 1, 0) +
        200 * max(undertricks - 3, 0)
    )


def doubled_contract_bonus(deal_result):
    if deal_result.tricks_won < deal_result.bid.tricks + 6:
        return 0
    if deal_result.scale == Scale.doubled:
        return 50
    if deal_result.scale == Scale.redoubled:
        return 100
    return 0


def calculate_score(deal_result):
    return Score(
        declarer=(
            trick_points(deal_result) +
            game_points(deal_result) +
            slam_points(deal_result) +
            doubled_contract_bonus(deal_result)
        ),
        defender=(
            undertrick_points(deal_result)
        )
    )
