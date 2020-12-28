from collections import namedtuple

import numpy as np
from scipy.optimize import minimize

__all__ = [
    'Match',
    'calculate_ratings',
]


Match = namedtuple('Match', 'winner loser')


ELO_1000 = 1000.0 / 400.0


def nll_results(ratings, winners, losers):
    all_ratings = np.concatenate([ELO_1000 * np.ones(1), ratings])
    winner_ratings = np.power(10.0, all_ratings[winners])
    loser_ratings = np.power(10.0, all_ratings[losers])
    log_p_wins = np.log(winner_ratings / (winner_ratings + loser_ratings))
    log_likelihood = np.sum(log_p_wins)

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

    n = len(matches)
    winners = np.zeros(n, dtype=np.int64)
    losers = np.zeros(n, dtype=np.int64)

    for i, match in enumerate(matches):
        winners[i] = index[match.winner]
        losers[i] = index[match.loser]

    n_bot = len(all_bots)
    if guess is None:
        guess = {}
    guess_array = np.array([guess.get(bot, 1000) for bot in all_bots[1:]])
    guess_array = guess_array.astype(np.float32) / 400
    result = minimize(
        nll_results, guess_array,
        args=(winners, losers),
        options={
            'gtol': 1e-4,
            'maxiter': 10000000,
        }
    )
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
