import random

from .. import cards
from ..game import GameState
from ..players import Player
from ..scoring import score_hand

__all__ = [
    'simulate_game',
]


def simulate_game(ns_bot, ew_bot):
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
        action = agent.select_action(hand)
        hand = hand.apply(action)
    result = score_hand(hand)
    declarer = hand.auction.result().declarer
    if declarer in (Player.north, Player.south):
        return result.declarer, result.defender
    return result.defender, result.declarer
