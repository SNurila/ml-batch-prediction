"""
Database setup: creates SQLite database with input_data and predictions tables,
and populates input_data with sample Iris-like features.
"""

import sqlite3
import random
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "pipeline.db")


def setup_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create input_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS input_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL,
            sepal_width  REAL,
            petal_length REAL,
            petal_width  REAL
        )
    """)

    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                   INTEGER PRIMARY KEY,
            prediction           TEXT,
            prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Seed input_data only when empty
    cursor.execute("SELECT COUNT(*) FROM input_data")
    if cursor.fetchone()[0] == 0:
        random.seed(42)
        rows = [
            (
                round(random.uniform(4.3, 7.9), 1),
                round(random.uniform(2.0, 4.4), 1),
                round(random.uniform(1.0, 6.9), 1),
                round(random.uniform(0.1, 2.5), 1),
            )
            for _ in range(50)
        ]
        cursor.executemany(
            "INSERT INTO input_data (sepal_length, sepal_width, petal_length, petal_width) VALUES (?,?,?,?)",
            rows,
        )
        print(f"[setup_db] Inserted {len(rows)} sample rows into input_data.")
    else:
        print("[setup_db] input_data already populated – skipping seed.")

    conn.commit()
    conn.close()
    print(f"[setup_db] Database ready at {DB_PATH}")


if __name__ == "__main__":
    setup_database()