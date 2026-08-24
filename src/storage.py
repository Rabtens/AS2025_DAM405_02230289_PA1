"""
storage.py — Storage / serving stage of the pipeline.

Persists the curated, validated DataFrame to:
  1. SQLite (data/processed/market_prices.db) -> queryable via SQL
  2. Parquet (data/processed/market_prices.parquet) -> efficient columnar
     format for downstream analytics/ML tooling

Writing to both is intentional: SQLite is the "queryable store" required by
the brief and is easy for a teammate to explore with any SQL client;
Parquet is what a downstream ML training job would actually read, since it
preserves dtypes and is far faster to load at scale than CSV.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DB_PATH = PROCESSED_DIR / "market_prices.db"
PARQUET_PATH = PROCESSED_DIR / "market_prices.parquet"
TABLE_NAME = "curated_market_prices"


def save_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH, table_name: str = TABLE_NAME) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    logger.info("Wrote %d rows to SQLite table '%s' at %s", len(df), table_name, db_path)
    return db_path


def save_to_parquet(df: pd.DataFrame, parquet_path: Path = PARQUET_PATH) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    logger.info("Wrote %d rows to Parquet at %s", len(df), parquet_path)
    return parquet_path


def query(sql: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Convenience helper for querying the curated store (used in the report/demo)."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)
