import random
from collections import namedtuple

from .. import cards
from ..game import GameState
from ..players import Player
from ..scoring import get_deal_result, score_hand

__all__ = [
    'GameRecord',
    'simulate_game',
]


GameRecord = namedtuple('GameRecord', [
    'game',
    'points_ns',
    'points_ew',
    'tricks_ns',
    'tricks_ew',
    'declarer',
    'contract_made',
    'contract',
])


def simulate_game(ns_bot, ew_bot, ns_recorder=None, ew_recorder=None):
    agents = {
        Player.north: ns_bot,
        Player.east: ew_bot,
        Player.south: ns_bot,
        Player.west: ew_bot,
    }
    recorders = {
        Player.north: ns_recorder,
        Player.east: ew_recorder,
        Player.south: ns_recorder,
        Player.west: ew_recorder,
    }
    hand = GameState.new_deal(
        cards.new_deal(),
        dealer=Player.north,
        northsouth_vulnerable=random.choice([True, False]),
        eastwest_vulnerable=random.choice([True, False])
    )
    while not hand.is_over():
        next_decider = hand.next_decider
        agent = agents[next_decider]
        action = agent.select_action(hand, recorders[next_decider])
        hand = hand.apply(action)
    deal_result = get_deal_result(hand)
    result = score_hand(hand)
    if not hand.auction.has_contract():
        return GameRecord(
            game=hand,
            points_ns=0,
            points_ew=0,
            tricks_ns=0,
            tricks_ew=0,
            declarer=None,
            contract_made=False,
            contract=None,
        )
    declarer = hand.auction.result().declarer
    if declarer in (Player.north, Player.south):
        points_ns = result.declarer
        points_ew = result.defender
        tricks_ns = deal_result.tricks_won
        tricks_ew = 13 - tricks_ns
    else:
        points_ns = result.defender
        points_ew = result.declarer
        tricks_ew = deal_result.tricks_won
        tricks_ns = 13 - tricks_ew
    return GameRecord(
        game=hand,
        points_ns=points_ns,
        points_ew=points_ew,
        tricks_ns=tricks_ns,
        tricks_ew=tricks_ew,
        declarer=hand.auction.result().declarer,
        contract_made=result.declarer > 0,
        contract=hand.auction.result().bid
    )
