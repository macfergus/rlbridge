import multiprocessing
import queue
import time
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


def _run_forever(
        ctrl_q, min_period, loopable_ctor, loopable_args, loopable_kwargs
):
    disable_sigint()
    last_run = time.time() - min_period

    loopable = loopable_ctor(*loopable_args, **loopable_kwargs)
    while True:
        try:
            ctrl_q.get(block=False)
            return
        except queue.Empty:
            pass
        now = time.time()
        if now - last_run < min_period:
            time.sleep(1)
            continue

        last_run = now
        loopable.run_once()


class LoopingProcess:
    def __init__(
            self, name, loopable_ctor, args=None, kwargs=None,
            restart=False, min_period=0
    ):
        self._name = name
        self._restart = restart
        self._min_period = min_period
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
                self._min_period,
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

    def stop(self, drain=None):
        if self._worker.proc.is_alive():
            self._worker.ctrl_q.put(None)
        self._worker.proc.join(timeout=0.001)
        stop_time = time.time()
        while self._worker.proc.is_alive():
            self._worker.proc.join(timeout=1)
            if time.time() - stop_time > 15:
                # tried to exit cleanly for 15 seconds; give up
                self._worker.proc.terminate()
                self._worker.proc.join(timeout=0.001)
                break
            if drain is not None:
                drain()
