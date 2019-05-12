import numpy as np

__all__ = [
    'Decision',
    'Episode',
    'ExperienceRecorder',
    'concat_episodes',
]


class Decision(dict):
    pass


class Episode(dict):
    pass


class Experience(dict):
    pass


def concat_episodes(episode_list):
    keys = list(episode_list[0].keys())
    result = {}
    for k in keys:
        result[k] = np.concatenate([ep[k] for ep in episode_list], axis=0)
    return Experience(result)


class ExperienceRecorder:
    def __init__(self):
        self._decisions = {}

    def record_decision(self, decision, player):
        if player not in self._decisions:
            self._decisions[player] = []
        self._decisions[player].append(decision)

    def get_decisions(self, player):
        return list(self._decisions[player])
