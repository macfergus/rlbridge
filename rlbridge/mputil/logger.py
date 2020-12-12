import datetime
import multiprocessing

from .interrupt import disable_sigint


__all__ = ['MPLogManager']


class MPLogger:
    def __init__(self, log_q):
        self._log_q = log_q

    def log(self, msg):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        proc_name = multiprocessing.current_process().name

        self._log_q.put(f'{ts} ({proc_name}) {msg}')


def print_log(log_q):
    disable_sigint()
    while True:
        msg = log_q.get()
        if msg is None:
            return
        print(msg)


class MPLogManager:
    def __init__(self):
        self._log_q = multiprocessing.Queue()
        self._log_printer = multiprocessing.Process(
            target=print_log,
            name='logger',
            args=(self._log_q,)
        )

    def start(self):
        self._log_printer.start()

    def stop(self):
        self._log_q.put(None)
        self._log_printer.join()

    def get_logger(self):
        return MPLogger(self._log_q)
