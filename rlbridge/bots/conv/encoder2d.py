from collections import namedtuple

import numpy as np

from ...cards import Card, Suit
from ...game import ALL_DENOMINATIONS, Bid, Call, Denomination, Play

PSA = namedtuple('PSA', 'player state action')


def reverse_states(final_state):
    states = []
    state = final_state
    while state is not None:
        if not state.is_over():
            states.append(state)
        state = state.prev_state
    states.reverse()
    return states


def unwind_states(final_state):
    states = reverse_states(final_state)
    unwound = []
    for i, state in enumerate(states):
        action = None
        if i < len(states) - 1:
            action = states[i + 1].prev_action
        unwound.append(PSA(
            player=state.next_player,
            state=state,
            action=action
        ))
    return unwound


class Encoder2D:
    WIDTH = 13
    CHANNELS = 66
    VISIBLE_BEGIN = 0
    VULN_US = 16
    VULN_THEM = 17
    CALL_BEGIN = 18
    PLAY_BEGIN = 50

    STATE_CHANNELS = 18
    ACTION_CHANNELS = 48

    # These include a "not my turn" sentinel
    DIM_CALL_ACTION = 39
    DIM_PLAY_ACTION = 53

    # 319 is the theoretical longest possible auction, but it's very
    # unlikely
    MAX_AUCTION = 60
    GAME_LENGTH = MAX_AUCTION + 52

    def encode_full_game(self, state, perspective):
        sequence = np.zeros((self.WIDTH, self.GAME_LENGTH, self.CHANNELS))
        for i, psa in enumerate(unwind_states(state)):
            sequence[:, i, :self.CALL_BEGIN] = (
                self.encode_game_state(psa.state, perspective)
            )
            sequence[:, i, self.CALL_BEGIN:] = (
                self.encode_action(psa.action, psa.player, perspective)
            )
        return sequence

    def encode_rank(self, rank):
        return rank - 2

    def encode_suit(self, suit):
        suit_offset = {
            Suit.clubs: 0,
            Suit.diamonds: 1,
            Suit.hearts: 2,
            Suit.spades: 3,
        }
        return suit_offset[suit]

    def encode_card(self, card):
        suit_offset = {
            Suit.clubs: 0,
            Suit.diamonds: 1,
            Suit.hearts: 2,
            Suit.spades: 3,
        }
        return 13 * suit_offset[card.suit] + (card.rank - 2)

    def decode_play_index(self, index):
        rank = (index % 13) + 2
        suit_index = index // 13
        suits = [Suit.clubs, Suit.diamonds, Suit.hearts, Suit.spades]
        return Play(Card(rank, suits[suit_index]))

    def encode_call(self, call):
        if call.is_double:
            return 35
        if call.is_redouble:
            return 36
        if call.is_pass:
            return 37
        denoms = {
            Denomination.clubs(): 0,
            Denomination.diamonds(): 1,
            Denomination.hearts(): 2,
            Denomination.spades(): 3,
            Denomination.notrump(): 4,
        }
        return 5 * (call.bid.tricks - 1) + denoms[call.bid.denomination]

    def decode_call_index(self, index):
        if index == 35:
            return Call.double()
        if index == 36:
            return Call.redouble()
        if index == 37:
            return Call.pass_turn()
        tricks_index = index // 5
        denom_index = index % 5
        denoms = [
            Denomination.clubs(),
            Denomination.diamonds(),
            Denomination.hearts(),
            Denomination.spades(),
            Denomination.notrump(),
        ]
        return Call.make_bid(Bid(denoms[denom_index], tricks_index + 1))

    def encode_legal_calls(self, state):
        calls = np.zeros(self.DIM_CALL_ACTION)
        for action in state.legal_actions():
            if action.is_call:
                calls[self.encode_call(action.call) + 1] = 1
        return calls

    def encode_legal_plays(self, state):
        plays = np.zeros(self.DIM_PLAY_ACTION)
        for action in state.legal_actions():
            if action.is_play:
                plays[self.encode_card(action.play.card) + 1] = 1
        return plays

    def encode_call_action(self, call):
        action = np.zeros(self.DIM_CALL_ACTION)
        if call is None:
            action[0] = 1
        else:
            action[self.encode_call(call) + 1] = 1
        return action

    def encode_play_action(self, play):
        action = np.zeros(self.DIM_PLAY_ACTION)
        if play is None:
            action[0] = 1
        else:
            action[self.encode_card(play.card) + 1] = 1
        return action

    def encode_game_state(self, state, perspective):
        array = np.zeros((self.WIDTH, self.STATE_CHANNELS))
        players = [
            perspective,
            perspective.lho(),
            perspective.partner,
            perspective.rho()
        ]

        # Fill in visible cards
        cards = state.visible_cards(perspective)
        for i, player in enumerate(players):
            offset = 4 * i
            if player in cards:
                for card in cards[player]:
                    array[
                        self.encode_rank(card.rank),
                        offset + self.encode_suit(card.suit)
                    ] = 1

        # Fill in vulnerability bits
        side = perspective.side()
        opposite_side = side.opposite()
        if state.is_vulnerable(side):
            array[:, self.VULN_US] = 1
        if state.is_vulnerable(opposite_side):
            array[:, self.VULN_THEM] = 1
        return array

    def encode_action(self, action, who_did_it, perspective):
        # Channels:
        # 0: clubs
        # 1: diamonds
        # 2: hearts
        # 3: spades
        # 4: no trump
        # 5: double
        # 6: redouble
        # 7: pass

        array = np.zeros((self.WIDTH, self.ACTION_CHANNELS))
        if action is None:
            return array
        players = [
            perspective,
            perspective.lho(),
            perspective.partner,
            perspective.rho()
        ]
        player_offset = {player: i for i, player in enumerate(players)}
        offset = player_offset[who_did_it]
        if action.is_call:
            start_index = 8 * offset
            call = action.call
            if call.is_bid:
                bid = call.bid
                tricks_idx = bid.tricks - 1
                denom_idx = (
                    start_index + ALL_DENOMINATIONS.index(bid.denomination)
                )
                array[tricks_idx, denom_idx] = 1
            elif call.is_double:
                array[:, start_index + 5] = 1
            elif call.is_redouble:
                array[:, start_index + 6] = 1
            elif call.is_pass:
                array[:, start_index + 7] = 1
        if action.is_play:
            start_index = 32 + 4 * offset
            card = action.play.card
            array[
                self.encode_rank(card.rank),
                start_index + self.encode_suit(card.suit)
            ] = 1
        return array

    def encode_contract(self, contract):
        array = np.zeros(5)
        if contract is None:
            return array
        scale = contract.tricks / 7.0
        index = {
            Denomination.suit(Suit.clubs): 0,
            Denomination.suit(Suit.diamonds): 1,
            Denomination.suit(Suit.hearts): 2,
            Denomination.suit(Suit.spades): 3,
            Denomination.notrump(): 4,
        }[contract.denomination]
        array[index] = scale
        return array

    def input_shape(self):
        return (self.WIDTH, self.GAME_LENGTH, self.CHANNELS)
