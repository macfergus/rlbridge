import multiprocessing

from .experience import ExperienceGenerator
from .trainer import Trainer

__all__ = ['SelfPlayManager']

class SelfPlayManager:
    def __init__(self, state_path, out_dir, config, logger):
        self.config = config
        self.logger = logger
        self._experience_q = multiprocessing.Queue()
        self._worker_pool = ExperienceGenerator(
            exp_q=self._experience_q,
            state_path=state_path,
            logger=self.logger,
            config=self.config
        )
        self._trainer = Trainer(
            exp_q=self._experience_q, 
            state_path=state_path,
            out_dir=out_dir,
            config=config,
            logger=self.logger
        )

    def start(self):
        self.logger.log('start self-play!')
        self._trainer.start()
        self._worker_pool.start()

    def is_healthy(self):
        return self._trainer.is_healthy()

    def maintain(self):
        self._worker_pool.maintain()

    def stop(self):
        self.logger.log('stop self-play!')
        self._worker_pool.stop()
        self._trainer.stop()
