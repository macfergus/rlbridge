import random
from collections import namedtuple

from .. import cards
from ..game import GameState
from ..players import Player
from ..scoring import score_hand

__all__ = [
    'GameRecord',
    'simulate_game',
]


GameRecord = namedtuple('GameRecord', [
    'game',
    'points_ns', 
    'points_ew',
    'declarer',
    'contract_made'
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
    result = score_hand(hand)
    if not hand.auction.has_contract():
        return GameRecord(
            game=hand,
            points_ns=0,
            points_ew=0,
            declarer=None,
            contract_made=False,
        )
    declarer = hand.auction.result().declarer
    if declarer in (Player.north, Player.south):
        points_ns = result.declarer
        points_ew = result.defender
    else:
        points_ns = result.defender
        points_ew = result.declarer
    return GameRecord(
        game=hand,
        points_ns=points_ns,
        points_ew=points_ew,
        declarer=hand.auction.result().declarer,
        contract_made=result.declarer > 0
    )
