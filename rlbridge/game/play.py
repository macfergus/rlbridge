from ..cards import Card

__all__ = [
    'Play',
    'PlayState',
]


class Play:
    def __init__(self, card):
        self.card = card

    @classmethod
    def of(cls, play_str):
        """Helper for tests."""
        return cls(Card.of(play_str))


class PlayState:
    def __init__(self, next_player, hands):
        self.next_player = next_player
        self.hands = hands

    def is_legal(self, play):
        if play.card not in self.hands[self.next_player]:
            return False
        return True

    def legal_plays(self):
        pass

    @classmethod
    def open_play(cls, auction_result, deal):
        return cls(
            next_player=auction_result.declarer.rotate(),
            hands=deal.hands()
        )
