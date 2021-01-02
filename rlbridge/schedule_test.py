import unittest

import yaml

from .schedule import Breakpoint, Schedule


class ScheduleTest(unittest.TestCase):
    def test_fixed(self):
        schedule = Schedule.fixed(0.1)
        self.assertAlmostEqual(0.1, schedule.lookup(0))
        self.assertAlmostEqual(0.1, schedule.lookup(9999999))

    def test_changing(self):
        schedule = Schedule([Breakpoint(endpoint=1000, value=0.1)], 0.2)
        self.assertAlmostEqual(0.1, schedule.lookup(999))
        self.assertAlmostEqual(0.2, schedule.lookup(1000))
        self.assertAlmostEqual(0.2, schedule.lookup(1001))

    def test_from_dicts(self):
        as_yaml = '''
        schedule:
          - until: 1000
            value: 0.1
          - until: 2000
            value: 0.2
          - finally: 0.3
        '''
        schedule = Schedule.from_dicts(yaml.safe_load(as_yaml)['schedule'])
        self.assertAlmostEqual(0.1, schedule.lookup(999))
        self.assertAlmostEqual(0.2, schedule.lookup(1500))
        self.assertAlmostEqual(0.3, schedule.lookup(3000))
