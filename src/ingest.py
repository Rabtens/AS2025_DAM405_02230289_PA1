"""
ingest.py — Ingestion stage of the pipeline.

Reads the two raw source formats:
  1. CSV  -> daily market price records   (data/raw/market_prices.csv)
  2. JSON -> dzongkhag reference metadata (data/raw/dzongkhag_metadata.json)

Design note: `load_market_prices` accepts an optional `source_url`. If given,
it tries to fetch the CSV from that URL first (this is how a real REST/HTTP
source system would be wired in) and transparently falls back to the local
cached copy in data/raw/ if the request fails (e.g. no network access, or the
endpoint is temporarily down). This keeps the pipeline runnable both in
CI/offline environments and against a live source, without changing any
downstream code.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

EXPECTED_CSV_COLUMNS = {
    "date",
    "dzongkhag",
    "commodity",
    "price_nu_per_kg",
    "quantity_kg",
    "market",
}


class IngestionError(Exception):
    """Raised when a raw source cannot be read or does not match the expected schema."""


def load_market_prices(
    csv_path: Optional[Path] = None,
    source_url: Optional[str] = None,
) -> pd.DataFrame:
    """Load the market-prices CSV source.

    Tries `source_url` first (a real HTTP source system) when provided,
    then falls back to the local cached file. Raises IngestionError if
    neither is available or the schema does not match what downstream
    stages expect.
    """
    csv_path = csv_path or (RAW_DIR / "market_prices.csv")
    df = None

    if source_url:
        try:
            logger.info("Attempting to fetch market prices from %s", source_url)
            df = pd.read_csv(source_url)
        except Exception as exc:  # network unavailable, 404, etc.
            logger.warning("Live source unavailable (%s); falling back to cached CSV", exc)

    if df is None:
        if not csv_path.exists():
            raise IngestionError(f"No cached CSV found at {csv_path} and no reachable source_url given")
        logger.info("Loading market prices from cached file %s", csv_path)
        df = pd.read_csv(csv_path)

    missing = EXPECTED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise IngestionError(f"market_prices source is missing expected columns: {missing}")

    logger.info("Ingested %d raw rows from market_prices source", len(df))
    return df


def load_dzongkhag_metadata(json_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the dzongkhag reference metadata (JSON source)."""
    json_path = json_path or (RAW_DIR / "dzongkhag_metadata.json")
    if not json_path.exists():
        raise IngestionError(f"No metadata JSON found at {json_path}")

    with open(json_path) as f:
        payload = json.load(f)

    if "dzongkhags" not in payload:
        raise IngestionError("metadata JSON is missing the 'dzongkhags' key")

    df = pd.DataFrame(payload["dzongkhags"])
    logger.info("Ingested %d dzongkhag reference records from JSON source", len(df))
    return df
