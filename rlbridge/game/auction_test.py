import unittest

from .auction import Auction, Call, Denomination
from .player import Player
from ..cards import Suit


class DenominationTest(unittest.TestCase):
    def test_ordering(self):
        self.assertLess(Denomination.clubs(), Denomination.diamonds())
        self.assertLess(Denomination.diamonds(), Denomination.spades())
        self.assertLess(Denomination.clubs(), Denomination.notrump())
        self.assertFalse(Denomination.diamonds() < Denomination.clubs())
        self.assertFalse(Denomination.spades() < Denomination.diamonds())
        self.assertFalse(Denomination.notrump() < Denomination.clubs())


class AuctionTest(unittest.TestCase):
    def test_no_contract(self):
        auction = Auction.new_auction(Player.north)
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        self.assertTrue(auction.is_over())
        contract = auction.result()
        self.assertIsNone(contract)

    def test_bid(self):
        auction = Auction.new_auction(Player.north)
        auction = auction.apply(Call.of('1S'))
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        self.assertTrue(auction.is_over())
        contract = auction.result()
        self.assertEqual(1, contract.bid.tricks)
        self.assertEqual(Suit.spades, contract.trump)
        self.assertEqual(Player.north, contract.declarer)

    def test_bid_and_raise(self):
        auction = Auction.new_auction(Player.north)
        auction = auction.apply(Call.of('1S'))
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.of('2S'))
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        self.assertTrue(auction.is_over())
        contract = auction.result()
        self.assertEqual(2, contract.bid.tricks)
        self.assertEqual(Suit.spades, contract.trump)
        self.assertEqual(Player.north, contract.declarer)

    def test_declarer_is_not_dealer(self):
        auction = Auction.new_auction(Player.north)
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.of('1S'))
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        self.assertTrue(auction.is_over())
        contract = auction.result()
        self.assertEqual(Player.east, contract.declarer)

    def test_declarer_is_not_first_to_name_suit(self):
        # North bids spades, but then east also bids spades. Then east
        # is the declarer, not north (because EW won the auction).
        auction = Auction.new_auction(Player.north)
        auction = auction.apply(Call.of('1S'))
        auction = auction.apply(Call.of('2S'))
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        auction = auction.apply(Call.pass_turn())
        self.assertTrue(auction.is_over())
        contract = auction.result()
        self.assertEqual(2, contract.bid.tricks)
        self.assertEqual(Suit.spades, contract.trump)
        self.assertEqual(Player.east, contract.declarer)
