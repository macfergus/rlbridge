import re
import unittest

from ..game import Bid, Denomination, Scale
from .scoring import DealResult, Score, calculate_score


def deal_result(contract, tricks, vulnerable=False):
    mo = re.match(r'(\d+)(C|H|S|D|NT)(X{0,2})', contract)
    num_tricks = int(mo.group(1))
    denom = Denomination.of(mo.group(2))
    scale_str = mo.group(3)
    if scale_str == 'XX':
        scale = Scale.redoubled
    elif scale_str == 'X':
        scale = Scale.doubled
    else:
        scale = Scale.undoubled

    return DealResult(
        bid=Bid(denom, num_tricks),
        scale=scale,
        vulnerable=vulnerable,
        tricks_won=tricks
    )


def deal_result_vulnerable(contract, tricks):
    return deal_result(contract, tricks, vulnerable=True)


class CalculateScoreNotVulnerableTest(unittest.TestCase):
    def test_make_minor(self):
        score = calculate_score(deal_result('1C', tricks=7))
        self.assertEqual(Score(70, 0), score)

    def test_make_minor_doubled(self):
        score = calculate_score(deal_result('2CX', tricks=8))
        self.assertEqual(Score(180, 0), score)

    def test_make_minor_redoubled(self):
        score = calculate_score(deal_result('2CXX', tricks=8))
        self.assertEqual(Score(560, 0), score)

    def test_make_major(self):
        score = calculate_score(deal_result('1S', tricks=7))
        self.assertEqual(Score(80, 0), score)

    def test_make_major_doubled(self):
        score = calculate_score(deal_result('1HX', tricks=7))
        self.assertEqual(Score(160, 0), score)

    def test_make_major_redoubled(self):
        score = calculate_score(deal_result('1HXX', tricks=7))
        self.assertEqual(Score(520, 0), score)

    def test_make_notrump(self):
        score = calculate_score(deal_result('1NT', tricks=7))
        self.assertEqual(Score(90, 0), score)

    def test_make_notrump_doubled(self):
        score = calculate_score(deal_result('2NTX', tricks=8))
        self.assertEqual(Score(490, 0), score)

    def test_make_notrump_redoubled(self):
        score = calculate_score(deal_result('2NTXX', tricks=8))
        self.assertEqual(Score(680, 0), score)

    def test_make_minor_game(self):
        score = calculate_score(deal_result('5C', tricks=11))
        self.assertEqual(Score(400, 0), score)

    def test_make_major_game(self):
        score = calculate_score(deal_result('4S', tricks=10))
        self.assertEqual(Score(420, 0), score)

    def test_make_notrump_game(self):
        score = calculate_score(deal_result('3NT', tricks=9))
        self.assertEqual(Score(400, 0), score)

    def test_overtricks(self):
        score = calculate_score(deal_result('1C', tricks=8))
        self.assertEqual(Score(90, 0), score)

    def test_overtricks_doubled(self):
        score = calculate_score(deal_result('1SX', tricks=9))
        self.assertEqual(Score(360, 0), score)

    def test_overtricks_redoubled(self):
        score = calculate_score(deal_result('1SXX', tricks=9))
        self.assertEqual(Score(920, 0), score)

    def test_notrump_overtricks(self):
        score = calculate_score(deal_result('1NT', tricks=8))
        self.assertEqual(Score(120, 0), score)

    def test_notrump_overtricks_doubled(self):
        score = calculate_score(deal_result('1NTX', tricks=8))
        self.assertEqual(Score(280, 0), score)

    def test_notrump_overtricks_redoubled(self):
        score = calculate_score(deal_result('1NTXX', tricks=8))
        self.assertEqual(Score(760, 0), score)

    def test_minor_slam(self):
        score = calculate_score(deal_result('6C', tricks=12))
        self.assertEqual(Score(920, 0), score)

    def test_minor_slam_doubled(self):
        score = calculate_score(deal_result('6CX', tricks=12))
        self.assertEqual(Score(1090, 0), score)

    def test_minor_slam_redoubled(self):
        score = calculate_score(deal_result('6CXX', tricks=12))
        self.assertEqual(Score(1380, 0), score)

    def test_major_slam(self):
        score = calculate_score(deal_result('6S', tricks=12))
        self.assertEqual(Score(980, 0), score)

    def test_major_slam_doubled(self):
        score = calculate_score(deal_result('6SX', tricks=12))
        self.assertEqual(Score(1210, 0), score)

    def test_major_slam_redoubled(self):
        score = calculate_score(deal_result('6SXX', tricks=12))
        self.assertEqual(Score(1620, 0), score)

    def test_grand_slam(self):
        score = calculate_score(deal_result('7H', tricks=13))
        self.assertEqual(Score(1510, 0), score)

    def test_undertricks(self):
        score = calculate_score(deal_result('1C', tricks=6))
        self.assertEqual(Score(0, 50), score)

    def test_undertricks_doubled(self):
        score = calculate_score(deal_result('1CX', tricks=6))
        self.assertEqual(Score(0, 100), score)

    def test_undertricks_redoubled(self):
        score = calculate_score(deal_result('1CXX', tricks=6))
        self.assertEqual(Score(0, 200), score)

    def test_lots_of_undertricks_doubled(self):
        score = calculate_score(deal_result('1CX', tricks=1))
        self.assertEqual(Score(0, 1400), score)

    def test_lots_of_undertricks_redoubled(self):
        score = calculate_score(deal_result('1CXX', tricks=1))
        self.assertEqual(Score(0, 2800), score)


class CalculateScoreVulnerableTest(unittest.TestCase):
    def test_make_minor(self):
        score = calculate_score(deal_result_vulnerable('1C', tricks=7))
        self.assertEqual(Score(70, 0), score)

    def test_make_minor_doubled(self):
        score = calculate_score(deal_result_vulnerable('2CX', tricks=8))
        self.assertEqual(Score(180, 0), score)

    def test_make_minor_redoubled(self):
        score = calculate_score(deal_result_vulnerable('2CXX', tricks=8))
        self.assertEqual(Score(760, 0), score)

    def test_make_major(self):
        score = calculate_score(deal_result_vulnerable('1S', tricks=7))
        self.assertEqual(Score(80, 0), score)

    def test_make_major_doubled(self):
        score = calculate_score(deal_result_vulnerable('1HX', tricks=7))
        self.assertEqual(Score(160, 0), score)

    def test_make_major_redoubled(self):
        score = calculate_score(deal_result_vulnerable('1HXX', tricks=7))
        self.assertEqual(Score(720, 0), score)

    def test_make_notrump(self):
        score = calculate_score(deal_result_vulnerable('1NT', tricks=7))
        self.assertEqual(Score(90, 0), score)

    def test_make_notrump_doubled(self):
        score = calculate_score(deal_result_vulnerable('2NTX', tricks=8))
        self.assertEqual(Score(690, 0), score)

    def test_make_notrump_redoubled(self):
        score = calculate_score(deal_result_vulnerable('2NTXX', tricks=8))
        self.assertEqual(Score(880, 0), score)

    def test_make_minor_game(self):
        score = calculate_score(deal_result_vulnerable('5C', tricks=11))
        self.assertEqual(Score(600, 0), score)

    def test_make_major_game(self):
        score = calculate_score(deal_result_vulnerable('4S', tricks=10))
        self.assertEqual(Score(620, 0), score)

    def test_make_notrump_game(self):
        score = calculate_score(deal_result_vulnerable('3NT', tricks=9))
        self.assertEqual(Score(600, 0), score)

    def test_overtricks(self):
        score = calculate_score(deal_result_vulnerable('1C', tricks=8))
        self.assertEqual(Score(90, 0), score)

    def test_overtricks_doubled(self):
        score = calculate_score(deal_result_vulnerable('1SX', tricks=9))
        self.assertEqual(Score(560, 0), score)

    def test_overtricks_redoubled(self):
        score = calculate_score(deal_result_vulnerable('1SXX', tricks=9))
        self.assertEqual(Score(1520, 0), score)

    def test_notrump_overtricks(self):
        score = calculate_score(deal_result_vulnerable('1NT', tricks=8))
        self.assertEqual(Score(120, 0), score)

    def test_notrump_overtricks_doubled(self):
        score = calculate_score(deal_result_vulnerable('1NTX', tricks=8))
        self.assertEqual(Score(380, 0), score)

    def test_notrump_overtricks_redoubled(self):
        score = calculate_score(deal_result_vulnerable('1NTXX', tricks=8))
        self.assertEqual(Score(1160, 0), score)

    def test_minor_slam(self):
        score = calculate_score(deal_result_vulnerable('6C', tricks=12))
        self.assertEqual(Score(1370, 0), score)

    def test_minor_slam_doubled(self):
        score = calculate_score(deal_result_vulnerable('6CX', tricks=12))
        self.assertEqual(Score(1540, 0), score)

    def test_minor_slam_redoubled(self):
        score = calculate_score(deal_result_vulnerable('6CXX', tricks=12))
        self.assertEqual(Score(1830, 0), score)

    def test_major_slam(self):
        score = calculate_score(deal_result_vulnerable('6S', tricks=12))
        self.assertEqual(Score(1430, 0), score)

    def test_major_slam_doubled(self):
        score = calculate_score(deal_result_vulnerable('6SX', tricks=12))
        self.assertEqual(Score(1660, 0), score)

    def test_major_slam_redoubled(self):
        score = calculate_score(deal_result_vulnerable('6SXX', tricks=12))
        self.assertEqual(Score(2070, 0), score)

    def test_grand_slam(self):
        score = calculate_score(deal_result_vulnerable('7H', tricks=13))
        self.assertEqual(Score(2210, 0), score)

    def test_undertricks(self):
        score = calculate_score(deal_result_vulnerable('1C', tricks=6))
        self.assertEqual(Score(0, 100), score)

    def test_undertricks_doubled(self):
        score = calculate_score(deal_result_vulnerable('1CX', tricks=6))
        self.assertEqual(Score(0, 200), score)

    def test_undertricks_redoubled(self):
        score = calculate_score(deal_result_vulnerable('1CXX', tricks=6))
        self.assertEqual(Score(0, 400), score)

    def test_lots_of_undertricks_doubled(self):
        score = calculate_score(deal_result_vulnerable('1CX', tricks=1))
        self.assertEqual(Score(0, 1700), score)

    def test_lots_of_undertricks_redoubled(self):
        score = calculate_score(deal_result_vulnerable('1CXX', tricks=1))
        self.assertEqual(Score(0, 3400), score)
