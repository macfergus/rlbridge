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


GameRecord = namedtuple('GameRecord', 'game points_ns points_ew')


def simulate_game(ns_bot, ew_bot, recorder=None):
    agents = {
        Player.north: ns_bot,
        Player.east: ew_bot,
        Player.south: ns_bot,
        Player.west: ew_bot,
    }
    hand = GameState.new_deal(
        cards.new_deal(),
        dealer=Player.north,
        northsouth_vulnerable=random.choice([True, False]),
        eastwest_vulnerable=random.choice([True, False])
    )
    while not hand.is_over():
        next_player = hand.next_player
        next_decider = hand.next_decider
        agent = agents[next_decider]
        action = agent.select_action(hand, recorder)
        hand = hand.apply(action)
    result = score_hand(hand)
    if not hand.auction.has_contract():
        return GameRecord(
            game=hand,
            points_ns=0,
            points_ew=0
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
        points_ew=points_ew
    )
