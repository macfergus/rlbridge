import contextlib
import os
import tempfile
import unittest

from .evalstore import EvalStore, Match


@contextlib.contextmanager
def temp_db_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield os.path.join(tmpdirname, 'test.db')


class EvalStoreTest(unittest.TestCase):
    def test_get_ratings_empty(self):
        with temp_db_file() as fname:
            store = EvalStore(fname)
            ratings = store.get_elo_ratings()
            self.assertEqual({}, ratings)

    def test_save_and_get_ratings(self):
        with temp_db_file() as fname:
            store = EvalStore(fname)
            ratings = {'a': 1000, 'b': 1500}
            store.store_elo_ratings(ratings)
            self.assertEqual(ratings, store.get_elo_ratings())

    def test_replace_ratings(self):
        with temp_db_file() as fname:
            store = EvalStore(fname)
            ratings1 = {'a': 1000, 'b': 1500}
            ratings2 = {'a': 1100, 'b': 1600}
            store.store_elo_ratings(ratings1)
            store.store_elo_ratings(ratings2)
            self.assertEqual(ratings2, store.get_elo_ratings())

    def test_save_and_get_matches(self):
        with temp_db_file() as fname:
            store = EvalStore(fname)
            match = Match(
                bot1='a',
                bot2='b',
                num_hands=10,
                bot1_points=100,
                bot2_points=200,
                bot1_contracts=2,
                bot2_contracts=3
            )
            store.store_match(match)
            matches = store.get_eval_matches()
            self.assertEqual([match], matches)
