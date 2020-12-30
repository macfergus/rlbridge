import multiprocessing
import queue
import time
import unittest

from .looper import Loopable, LoopingProcess


class LooperTestCase(unittest.TestCase):
    def test_looper(self):
        recv_q = multiprocessing.Queue()

        class Counter(Loopable):
            def __init__(self, recv_q):
                self._q = recv_q
                self._count = 0

            def run_once(self):
                self._q.put(self._count)
                self._count += 1
                time.sleep(0.01)

        loop_proc = LoopingProcess('loop', Counter, kwargs={'recv_q': recv_q})
        loop_proc.start()
        tries = 0
        succeeded = False
        while tries < 20:
            try:
                time.sleep(0.01)
                tries += 1
                received = recv_q.get(block=False)
                if received == 5:
                    succeeded = True
                    break
            except queue.Empty:
                break
        loop_proc.stop()

        self.assertTrue(succeeded, 'did not receive 5 values from worker')

    def test_min_period(self):
        recv_q = multiprocessing.Queue()

        class Counter(Loopable):
            def __init__(self, recv_q):
                self._q = recv_q

            def run_once(self):
                self._q.put(1)

        # Run a process with a max period of 10 ms
        loop_proc = LoopingProcess(
            'loop',
            Counter,
            kwargs={'recv_q': recv_q},
            min_period=0.01
        )
        loop_proc.start()

        start = time.time()
        succeeded = False
        received = []
        # Let it run for 100 ms. We should not get more than 10 items 
        # (or maybe 11 with some timing slop)
        while time.time() - start < 0.1:
            try:
                received.append(recv_q.get(block=False))
                time.sleep(0.001)
            except queue.Empty:
                break
        loop_proc.stop()

        self.assertLessEqual(len(received), 11)
