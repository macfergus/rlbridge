import collections
import sqlite3


Match = collections.namedtuple('Match', [
    'bot1',
    'bot2',
    'num_hands',
    'bot1_points',
    'bot2_points',
    'bot1_contracts',
    'bot2_contracts'
])


class EvalStore:
    def __init__(self, db_fname):
        self._db_fname = db_fname
        conn = sqlite3.connect(db_fname)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                bot1 TEXT,
                bot2 TEXT,
                num_hands INTEGER,
                bot1_points INTEGER,
                bot2_points INTEGER,
                bot1_contracts INTEGER,
                bot2_contracts INTEGER
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS elo_ratings (
                bot TEXT PRIMARY KEY,
                elo_rating INTEGER
            )
        ''')
        conn.commit()

    def _conn(self):
        return sqlite3.connect(self._db_fname)

    def get_elo_ratings(self):
        cursor = self._conn().cursor()
        cursor.execute('''SELECT bot, elo_rating FROM elo_ratings''')
        return dict(cursor)

    def store_elo_ratings(self, ratings):
        rows = [(bot, int(rating)) for bot, rating in ratings.items()]
        conn = self._conn()
        conn.executemany('''
            INSERT OR REPLACE INTO elo_ratings (bot, elo_rating) VALUES (?, ?)
        ''', rows)
        conn.commit()

    def store_match(self, match):
        conn = self._conn()
        conn.execute('''
            INSERT INTO matches (
                bot1, bot2,
                num_hands,
                bot1_points, bot2_points,
                bot1_contracts, bot2_contracts
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            match.bot1, match.bot2,
            int(match.num_hands),
            int(match.bot1_points), int(match.bot2_points),
            int(match.bot1_contracts), int(match.bot2_contracts)
        ))
        conn.commit()

    def get_eval_matches(self):
        cursor = self._conn().cursor()
        cursor.execute('''
            SELECT
                bot1, bot2,
                num_hands,
                bot1_points, bot2_points,
                bot1_contracts, bot2_contracts
            FROM matches
        ''')
        return [
            Match(
                bot1=row[0], bot2=row[1],
                num_hands=row[2],
                bot1_points=row[3], bot2_points=row[4],
                bot1_contracts=row[5], bot2_contracts=row[6]
            ) for row in cursor
        ]
