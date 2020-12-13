import multiprocessing
import queue
from collections import namedtuple

from .interrupt import disable_sigint

__all__ = [
    'Loopable',
    'LoopingProcess',
]

Worker = namedtuple('Worker', 'ctrl_q proc')


class Loopable:
    def run_once(self):
        raise NotImplementedError()


def _run_forever(ctrl_q, loopable_ctor, loopable_args, loopable_kwargs):
    disable_sigint()

    loopable = loopable_ctor(*loopable_args, **loopable_kwargs)
    while True:
        try:
            ctrl_q.get(block=False)
            return
        except queue.Empty:
            pass

        loopable.run_once()


class LoopingProcess:
    def __init__(
            self, name, loopable_ctor, args=None, kwargs=None, restart=False
    ):
        self._name = name
        self._restart = restart
        self._loopable_ctor = loopable_ctor
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()
        self._loopable_args = args
        self._loopable_kwargs = kwargs
        self._worker = self._new_worker()

    def start(self):
        self._worker.proc.start()

    def _new_worker(self):
        ctrl_q = multiprocessing.Queue()
        proc = multiprocessing.Process(
            name=self._name,
            target=_run_forever,
            args=(
                ctrl_q,
                self._loopable_ctor,
                self._loopable_args,
                self._loopable_kwargs
            )
        )
        return Worker(ctrl_q=ctrl_q, proc=proc)

    def maintain(self):
        if self._restart:
            if not self._worker.proc.is_alive():
                self._worker = self._new_worker()
                self._worker.proc.start()

    def stop(self):
        if self._worker.proc.is_alive():
            self._worker.ctrl_q.put(None)
        self._worker.proc.join()
