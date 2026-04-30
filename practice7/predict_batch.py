"""
predict_batch.py
Core batch-prediction script:
  1. Connects to the SQLite database
  2. Reads all rows from input_data
  3. Loads the trained ML model
  4. Generates predictions
  5. Writes results into the predictions table
"""

import sqlite3
import pickle
import logging
import os
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DB_PATH    = os.path.join(BASE_DIR, "data",   "pipeline.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_model.pkl")
LOG_PATH   = os.path.join(BASE_DIR, "logs",   "pipeline.log")

# ── logging ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def load_model(path: str):
    """Load the pickled model bundle and return (model, target_names)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_model.py first."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["target_names"]


def fetch_input_data(cursor) -> list[dict]:
    """Return all rows from input_data as a list of dicts."""
    cursor.execute("SELECT id, sepal_length, sepal_width, petal_length, petal_width FROM input_data")
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def run_batch():
    """Main batch-prediction routine."""
    log.info("=== Batch prediction run started ===")
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Connect to database
    log.info(f"Connecting to database: {DB_PATH}")
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 2. Read input data
    rows = fetch_input_data(cursor)
    log.info(f"Fetched {len(rows)} rows from input_data")

    if not rows:
        log.warning("No input data found. Exiting.")
        conn.close()
        return

    # 3. Load model
    log.info(f"Loading model from: {MODEL_PATH}")
    model, target_names = load_model(MODEL_PATH)

    # 4. Generate predictions
    features = [
        [r["sepal_length"], r["sepal_width"], r["petal_length"], r["petal_width"]]
        for r in rows
    ]
    raw_preds = model.predict(features)
    predictions = [target_names[p] for p in raw_preds]
    log.info(f"Generated {len(predictions)} predictions")

    # 5. Write results into predictions table
    cursor.execute("DELETE FROM predictions")   # fresh batch each run
    records = [
        (row["id"], pred, timestamp)
        for row, pred in zip(rows, predictions)
    ]
    cursor.executemany(
        "INSERT INTO predictions (id, prediction, prediction_timestamp) VALUES (?,?,?)",
        records,
    )
    conn.commit()
    log.info(f"Stored {len(records)} predictions in the database")

    # summary
    from collections import Counter
    dist = Counter(predictions)
    log.info(f"Prediction distribution: {dict(dist)}")

    conn.close()
    log.info("=== Batch prediction run complete ===")


if __name__ == "__main__":
    run_batch()