from collections import namedtuple

import numpy as np
from scipy.optimize import minimize

__all__ = [
    'Match',
    'calculate_ratings',
]


Match = namedtuple('Match', 'winner loser')


def nll_results(ratings, winners, losers):
    all_ratings = np.concatenate([np.ones(1), ratings])
    winner_ratings = all_ratings[winners]
    loser_ratings = all_ratings[losers]
    log_p_wins = np.log(winner_ratings / (winner_ratings + loser_ratings))
    log_likelihood = np.sum(log_p_wins)
    return -1 * log_likelihood


def calculate_ratings(matches, anchor=None):
    all_bots = list(sorted(
        {match.winner for match in matches} |
        {match.loser for match in matches}
    ))
    index = {bot: i for i, bot in enumerate(all_bots)}

    n = len(matches)
    winners = np.zeros(n, dtype=np.int64)
    losers = np.zeros(n, dtype=np.int64)

    for i, match in enumerate(matches):
        winners[i] = index[match.winner]
        losers[i] = index[match.loser]

    n_bot = len(all_bots)
    guess = np.ones(n_bot - 1)
    # Can't let the abstract rating go to 0, since we take its log
    bounds = [(1e-5, None) for _ in guess]
    result = minimize(
        nll_results, guess,
        args=(winners, losers),
        bounds=bounds,
        options={
            'maxiter': 1000000,
        }
    )
    assert result.success

    abstract_ratings = np.concatenate([np.ones(1), result.x])
    elo_ratings = 400.0 * np.log10(abstract_ratings)
    if anchor is not None:
        anchor_rating = elo_ratings[index[anchor]]
        elo_ratings += 1000 - anchor_rating
    else:
        min_rating = np.min(elo_ratings)
        # Scale so that the weakest player has rating 0
        elo_ratings -= min_rating

    return {all_bots[i]: elo for i, elo in enumerate(elo_ratings)}
