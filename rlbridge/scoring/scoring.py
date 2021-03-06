from ..game import Scale
from ..players import Player, Side

__all__ = [
    'get_deal_result',
    'score_hand',
]


class Score:
    def __init__(self, declarer, defender):
        # Only one side can score points on any deal.
        assert (
            (declarer >= 0 and defender == 0) or
            (defender >= 0 and declarer == 0)
        )
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


def made_contract(deal_result):
    return deal_result.tricks_won >= deal_result.bid.tricks + 6


def trick_points(deal_result):
    if not made_contract(deal_result):
        return 0
    odd_tricks = deal_result.tricks_won - 6
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
    if not made_contract(deal_result):
        return 0
    if is_contract_game(deal_result):
        if deal_result.vulnerable:
            return 500
        return 300
    return 50  # part score


def slam_points(deal_result):
    if not made_contract(deal_result):
        return 0
    if deal_result.bid.tricks == 6:
        return 750 if deal_result.vulnerable else 500
    if deal_result.bid.tricks == 7:
        return 1500 if deal_result.vulnerable else 1000
    return 0


def undertrick_points(deal_result):
    if made_contract(deal_result):
        return 0
    undertricks = deal_result.bid.tricks + 6 - deal_result.tricks_won
    assert undertricks >= 1
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
    if not made_contract(deal_result):
        return 0
    if deal_result.scale == Scale.doubled:
        return 50
    if deal_result.scale == Scale.redoubled:
        return 100
    return 0


def calculate_score(deal_result):
    if deal_result.bid is None:
        # No contract was reached, so no points for anyone.
        return Score(declarer=0, defender=0)
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


def get_deal_result(state):
    assert state.is_over()
    if not state.auction.has_contract():
        # No one bid, so no play occurred.
        return DealResult(
            bid=None,
            scale=None,
            vulnerable=None,
            tricks_won=0
        )
    tricks_won = {
        Side.north_south: 0,
        Side.east_west: 0,
    }
    for trick in state.playstate.completed_tricks:
        if trick.winner() in (Player.north, Player.south):
            tricks_won[Side.north_south] += 1
        else:
            tricks_won[Side.east_west] += 1
    auction_result = state.auction.result()
    if auction_result.declarer.side() == Side.north_south:
        vulnerable = state.northsouth_vulnerable
    else:
        vulnerable = state.eastwest_vulnerable
    return DealResult(
        bid=auction_result.bid,
        scale=auction_result.scale,
        vulnerable=vulnerable,
        tricks_won=tricks_won[auction_result.declarer.side()]
    )


def score_hand(state):
    return calculate_score(get_deal_result(state))
