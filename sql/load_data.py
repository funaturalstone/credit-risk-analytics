"""Load the Lending Club CSV into a SQLite database for the SQL analysis.

Usage (from the repo root):
    python sql/load_data.py data/lending_club_80k.csv
Creates data/lending_club.db with one table, `loans`.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

COLUMNS = [
    "id", "loan_amnt", "term", "int_rate", "grade", "sub_grade", "purpose",
    "annual_inc", "dti", "fico_range_low", "fico_range_high",
    "total_pymnt", "loan_status",
]

csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/lending_club_80k.csv")
db_path = Path("data/lending_club.db")

df = pd.read_csv(csv_path, usecols=COLUMNS)
df["term"] = df["term"].str.strip()          # " 36 months" -> "36 months"

with sqlite3.connect(db_path) as con:
    df.to_sql("loans", con, if_exists="replace", index=False)
    n = con.execute("SELECT COUNT(*) FROM loans").fetchone()[0]

print(f"Loaded {n:,} rows into {db_path} (table: loans)")
