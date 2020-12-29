import contextlib
import os
import tempfile
import unittest

from .paramstore import MissingParamError, ParamStore


@contextlib.contextmanager
def temp_db_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield os.path.join(tmpdirname, 'test.db')


class ParamStoreTest(unittest.TestCase):
    def test_store_string(self):
        with temp_db_file() as fname:
            store1 = ParamStore(fname)
            store1.set_string('key', 'value')
            self.assertEqual('value', store1.get_string('key'))

    def test_persist(self):
        with temp_db_file() as fname:
            store1 = ParamStore(fname)
            store1.set_string('key', 'value')

            store2 = ParamStore(fname)
            self.assertEqual('value', store2.get_string('key'))

    def test_has_key(self):
        with temp_db_file() as fname:
            store = ParamStore(fname)
            self.assertFalse(store.has_key('newkey'))
            store.set_string('newkey', 'newvalue')
            self.assertTrue(store.has_key('newkey'))

    def test_get_string_missing(self):
        with temp_db_file() as fname:
            store1 = ParamStore(fname)
            with self.assertRaises(MissingParamError):
                store1.get_string('key')

    def test_get_string_with_default(self):
        with temp_db_file() as fname:
            store1 = ParamStore(fname)
            self.assertEqual('default', store1.get_string('key', 'default'))

    def test_store_float(self):
        with temp_db_file() as fname:
            store = ParamStore(fname)
            store.set_float('key', 0.123456789)
            self.assertAlmostEqual(0.123456789, store.get_float('key'))

    def test_store_list(self):
        with temp_db_file() as fname:
            store = ParamStore(fname)
            store.set_list('key', ['a', 'b', 'c'])
            self.assertEqual(['a', 'b', 'c'], store.get_list('key'))

    def test_store_int(self):
        with temp_db_file() as fname:
            store = ParamStore(fname)
            store.set_int('key', 5)
            self.assertEqual(5, store.get_int('key'))
