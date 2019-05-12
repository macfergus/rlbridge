import numpy as np

__all__ = [
    'Episode',
    'ExperienceRecorder',
    'concat_episodes',
]


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
    pass
