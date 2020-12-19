import multiprocessing

from .evaluator import Evaluator
from .experience import ExperienceGenerator
from .trainer import Trainer

__all__ = ['SelfPlayManager']

class SelfPlayManager:
    def __init__(self, workspace, config, logger):
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
        self._evaluator = Evaluator(
            workspace=workspace,
            config=config,
            logger=self.logger
        )

    def start(self):
        self.logger.log('start self-play!')
        self._trainer.start()
        self._worker_pool.start()
        self._evaluator.start()

    def maintain(self):
        self._worker_pool.maintain()
        self._trainer.maintain()
        self._evaluator.maintain()

    def stop(self):
        self.logger.log('stop self-play!')
        self._evaluator.stop()
        self._worker_pool.stop()
        self._trainer.stop()
