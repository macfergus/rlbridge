import unittest

import yaml

from .lrschedule import Breakpoint, LRSchedule


class LRScheduleTest(unittest.TestCase):
    def test_fixed(self):
        schedule = LRSchedule.fixed(0.1)
        self.assertAlmostEqual(0.1, schedule.lookup(0))
        self.assertAlmostEqual(0.1, schedule.lookup(9999999))

    def test_changing(self):
        schedule = LRSchedule([Breakpoint(endpoint=1000, lr=0.1)], 0.2)
        self.assertAlmostEqual(0.1, schedule.lookup(999))
        self.assertAlmostEqual(0.2, schedule.lookup(1000))
        self.assertAlmostEqual(0.2, schedule.lookup(1001))

    def test_from_dicts(self):
        as_yaml = '''
        schedule:
          - until: 1000
            lr: 0.1
          - until: 2000
            lr: 0.2
          - finally: 0.3
        '''
        schedule = LRSchedule.from_dicts(yaml.safe_load(as_yaml)['schedule'])
        self.assertAlmostEqual(0.1, schedule.lookup(999))
        self.assertAlmostEqual(0.2, schedule.lookup(1500))
        self.assertAlmostEqual(0.3, schedule.lookup(3000))
