import os
import random
import time

import numpy as np

from .. import bots, elo, kerasutil
from ..mputil import Loopable, LoopingProcess


class EloCalculatorImpl(Loopable):
    def __init__(self, workspace, logger):
        self._workspace = workspace
        self._logger = logger
        kerasutil.set_tf_options(disable_gpu=True)

        self._last_update = 0

    def run_once(self):
        eval_store = self._workspace.eval_store
        eval_matches = eval_store.get_eval_matches()
        if not eval_matches:
            return
        old_ratings = eval_store.get_elo_ratings()

        botset = (
            {eval_match.bot1 for eval_match in eval_matches} |
            {eval_match.bot2 for eval_match in eval_matches}
        )
        first_bot = sorted(botset)[0]

        elo_matches = []
        for eval_match in eval_matches:
            if eval_match.bot1_points > eval_match.bot2_points:
                elo_matches.append(
                    elo.Match(winner=eval_match.bot1, loser=eval_match.bot2)
                )
            elif eval_match.bot2_points > eval_match.bot1_points:
                elo_matches.append(
                    elo.Match(winner=eval_match.bot2, loser=eval_match.bot1)
                )

        self._logger.log('Updating Elo ratings')
        new_ratings = elo.calculate_ratings(
            elo_matches,
            anchor=first_bot,
            guess=old_ratings
        )
        eval_store.store_elo_ratings(new_ratings)
        self._last_update = time.time()


class EloCalculator:
    def __init__(self, workspace, logger):
        self._workspace = workspace
        self._logger = logger
        self._proc = LoopingProcess(
            'elo-calc',
            EloCalculatorImpl,
            kwargs={
                'workspace': self._workspace,
                'logger': self._logger,
            },
            restart=True,
            min_period=120
        )

    def start(self):
        self._proc.start()

    def stop(self):
        self._proc.stop()

    def maintain(self):
        self._proc.maintain()
