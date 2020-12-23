import multiprocessing

from .elocalculator import EloCalculator
from .evaluator import Evaluator
from .experience import ExperienceGenerator
from .trainer import Trainer

__all__ = ['SelfPlayManager']

class SelfPlayManager:
    def __init__(self, workspace, config, logger, evaluate_only=False):
        self.config = config
        self.logger = logger
        self._experience_q = multiprocessing.Queue()
        self._worker_pool = ExperienceGenerator(
            exp_q=self._experience_q,
            workspace=workspace,
            logger=self.logger,
            config=self.config
        )
        self._trainer = Trainer(
            exp_q=self._experience_q,
            workspace=workspace,
            config=config,
            logger=self.logger
        )
        self._elo_calculator = EloCalculator(
            workspace=workspace,
            logger=self.logger
        )
        self._evaluator = Evaluator(
            workspace=workspace,
            config=config,
            logger=self.logger
        )
        self._evaluate_only = evaluate_only

    def start(self):
        self.logger.log('start self-play!')
        if not self._evaluate_only:
            self._trainer.start()
            self._worker_pool.start()
        self._elo_calculator.start()
        self._evaluator.start()

    def maintain(self):
        if not self._evaluate_only:
            self._worker_pool.maintain()
            self._trainer.maintain()
        self._elo_calculator.maintain()
        self._evaluator.maintain()

    def stop(self):
        self.logger.log('stop self-play!')
        self._evaluator.stop()
        self._elo_calculator.stop()
        if not self._evaluate_only:
            self._worker_pool.stop()
            self._trainer.stop()
