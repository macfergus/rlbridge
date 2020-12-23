import json
import sqlite3


class MissingParamError(Exception):
    pass


NO_DEFAULT = object()


class ParamStore:
    def __init__(self, db_fname):
        self._conn = sqlite3.connect(db_fname)
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS params (
                key_name TEXT PRIMARY KEY,
                raw_value TEXT
            )
        ''')
        self._conn.commit()

    def _set_raw(self, key, raw_value):
        cursor = self._conn.cursor()
        cursor.execute('''
            REPLACE INTO params (key_name, raw_value) VALUES (?, ?)
        ''', (key, raw_value))
        self._conn.commit()

    def _get_raw(self, key, decoder=lambda x: x, default=NO_DEFAULT):
        cursor = self._conn.cursor()
        cursor.execute('''
            SELECT raw_value FROM params WHERE key_name=?
        ''', (key,))
        row = cursor.fetchone()
        if row is None:
            if default is NO_DEFAULT:
                raise MissingParamError(key)
            return default
        raw_value, = row
        return decoder(raw_value)

    def has_key(self, key):
        cursor = self._conn.cursor()
        cursor.execute('''
            SELECT 1 FROM params WHERE key_name=?
        ''', (key,))
        row = cursor.fetchone()
        return row is not None

    def set_string(self, key, value):
        self._set_raw(key, value)

    def get_string(self, key, default=NO_DEFAULT):
        return self._get_raw(key, default=default)

    def set_int(self, key, value):
        self._set_raw(key, int(value))

    def get_int(self, key, default=NO_DEFAULT):
        return self._get_raw(key, decoder=int, default=default)

    def set_float(self, key, value):
        self._set_raw(key, str(value))

    def get_float(self, key, default=NO_DEFAULT):
        return self._get_raw(key, decoder=float, default=default)

    def set_list(self, key, list_value):
        self._set_raw(key, json.dumps(list_value))

    def get_list(self, key, default=NO_DEFAULT):
        return self._get_raw(key, decoder=json.loads, default=default)
