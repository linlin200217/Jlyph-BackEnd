#!/usr/bin/python

import sqlite3

conn = sqlite3.connect('./database/flower.db')

def create_table():
    c = conn.cursor()
    c.execute('''
            CREATE TABLE IF NOT EXISTS flower (
                sepal_length    REAL NOT NULL,
                sepal_width     REAL NOT NULL,
                petal_length    REAL NOT NULL,
                petal_width     REAL NOT NULL,
                variety         TEXT,
                ccc             TEXT
            );''')
    conn.commit()
    conn.close()


def insert_flower():
    c = conn.cursor()
    c.execute("INSERT INTO flower (sepal_length, sepal_width, petal_length, petal_width, variety, ccc) VALUES (?, ?, ?, ?, ?, ?)",
              (5.1, 3.5, 1.4, 0.2, "Setosa", "aaa"))
    conn.commit()
    conn.close()
