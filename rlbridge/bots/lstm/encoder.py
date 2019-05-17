import numpy as np

from ...cards import Card, Suit
from ...game import Bid, Call, Denomination, Play


def reverse_states(final_state):
    states = []
    state = final_state
    while state is not None:
        if not state.is_over():
            states.append(state)
        state = state.prev_state
    states.reverse()
    return states


class Encoder:
    DIM_NEW_GAME = 1
    DIM_VISIBLE_CARDS = 4 * 53
    DIM_VULNERABILITY = 2
    DIM_AUCTION = 4 * 38
    DIM_PLAY = 4 * 52
    DIM = (
        DIM_NEW_GAME +
        DIM_VISIBLE_CARDS +
        DIM_VULNERABILITY +
        DIM_AUCTION +
        DIM_PLAY
    )

    # These include a "not my turn" sentinel
    DIM_CALL_ACTION = 39
    DIM_PLAY_ACTION = 53

    VISIBLE_CARD_START = DIM_NEW_GAME
    VULNERABILITY_START = VISIBLE_CARD_START + DIM_VISIBLE_CARDS
    AUCTION_START = VULNERABILITY_START + DIM_VULNERABILITY
    PLAY_START = AUCTION_START + DIM_AUCTION

    def encode_full_game(self, state, perspective):
        num_states = state.num_states
        # Include an extra slot for the new game indicator.
        sequence = np.zeros((num_states + 1, self.DIM))
        sequence[0] = self.encode_new_game()
        for i, s in enumerate(reverse_states(state)):
            sequence[i + 1] = self.encode_game_state(s, perspective)
        return sequence

    def encode_new_game(self):
        new_game_sentinel = np.zeros(self.DIM)
        new_game_sentinel[0] = 1
        return new_game_sentinel

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
        s = np.zeros(self.DIM)
        players = [
            perspective,
            perspective.lho(),
            perspective.partner,
            perspective.rho()
        ]
        player_offset = {player: i for i, player in enumerate(players)}

        # Fill in visible cards
        cards = state.visible_cards(perspective)
        for i, player in enumerate(players):
            start_index = self.VISIBLE_CARD_START + 53 * i
            card_array = np.zeros(53)
            if player in cards:
                for c in cards[player]:
                    card_array[self.encode_card(c) + 1] = 1
            else:
                # This player's cards are not currently visible to the
                # current decider. This lets us distinguish an empty
                # hand from one that we can't see.
                card_array[0] = 1
            s[start_index:start_index + 53] = card_array

        # Fill in vulnerability bits
        side = perspective.side()
        opposite_side = side.opposite()
        if state.is_vulnerable(side):
            s[self.DIM_VULNERABILITY] = 1
        if state.is_vulnerable(opposite_side):
            s[self.DIM_VULNERABILITY + 1] = 1

        # Fill in last action
        if state.prev_action is None:
            return s
        last_actor = state.next_player.rho()
        offset = player_offset[last_actor]
        if state.prev_action.is_call:
            start_index = self.AUCTION_START + 38 * offset
            call_index = self.encode_call(state.prev_action.call)
            s[start_index + call_index] = 1
        if state.prev_action.is_play:
            start_index = self.PLAY_START + 52 * offset
            card_index = self.encode_card(state.prev_action.play.card)
            s[start_index + card_index] = 1
        return s

    def input_shape(self):
        return (self.DIM,)
