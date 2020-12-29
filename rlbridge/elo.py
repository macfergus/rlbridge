from collections import namedtuple

import numpy as np
from scipy.optimize import minimize

__all__ = [
    'Match',
    'calculate_ratings',
]


Match = namedtuple('Match', 'winner loser')


ELO_1000 = 1000.0 / 400.0


def negative_log_likelihood(ratings, wins):
    all_ratings = np.concatenate([ELO_1000 * np.ones(1), ratings])
    exp_ratings = np.power(10.0, all_ratings)
    log_p_win = np.log(exp_ratings / np.add.outer(exp_ratings, exp_ratings))
    log_likelihood = np.sum(log_p_win * wins)

    baseline = ELO_1000 * np.ones_like(all_ratings)
    diff = all_ratings - baseline
    diff2 = diff * diff
    return -1 * log_likelihood + 0.02 * np.sum(diff2)


def calculate_ratings(matches, anchor=None, guess=None):
    all_bots = list(sorted(
        {match.winner for match in matches} |
        {match.loser for match in matches}
    ))
    # Move the anchor to the front
    if anchor in all_bots:
        all_bots.remove(anchor)
        all_bots.insert(0, anchor)

    index = {bot: i for i, bot in enumerate(all_bots)}
    n_bots = len(all_bots)
    wins = np.zeros((n_bots, n_bots))
    for match in matches:
        wins[index[match.loser], index[match.winner]] += 1

    if guess is None:
        guess = {}
    guess_array = np.array([guess.get(bot, 1000) for bot in all_bots[1:]])
    guess_array = guess_array.astype(np.float32) / 400

    result = minimize(
        negative_log_likelihood, guess_array,
        method='Nelder-Mead',
        args=(wins,),
        options={
            'maxiter': 5000000,
            'maxfev': 5000000,
        }
    )
    if not result.success:
        print(result)
        assert result.success

    abstract_ratings = np.concatenate([ELO_1000 * np.ones(1), result.x])
    elo_ratings = 400.0 * abstract_ratings
    if anchor is not None and anchor in index:
        anchor_rating = elo_ratings[index[anchor]]
        elo_ratings += 1000 - anchor_rating
    else:
        min_rating = np.min(elo_ratings)
        # Scale so that the weakest player has rating 0
        elo_ratings -= min_rating

    return {all_bots[i]: elo for i, elo in enumerate(elo_ratings)}
