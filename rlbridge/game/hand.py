__all__ = [
    'GameState',
]


class GameState:
    @classmethod
    def new_hand(cls, deal, dealer):
        return GameState()

    def is_over(self):
        return True
