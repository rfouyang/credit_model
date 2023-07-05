import sqlite3

import pandas as pd


class SQLite:
    conn = None

    @classmethod
    def get_conn(cls, db_path):
        cls.conn = sqlite3.connect(f"{db_path}")
        return cls.conn

    @classmethod
    def query(cls, sql):
        df = pd.read_sql(sql, cls.conn)
        return df
