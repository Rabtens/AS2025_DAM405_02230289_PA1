"""
pipeline.py — End-to-end orchestrator.

Single documented command to run the whole pipeline:

    python -m src.pipeline

Stages: ingest -> transform -> quality gate -> store.
If the quality gate fails, the pipeline stops before anything is written
to the curated store (see src/quality.py).
"""

import argparse
import logging
import sys
from pathlib import Path

from src.ingest import load_dzongkhag_metadata, load_market_prices
from src.quality import DataQualityError, run_quality_gate
from src.storage import save_to_parquet, save_to_sqlite
from src.transform import run_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(source_url: str = None) -> int:
    logger.info("=== Bhutan Vegetable Market Prices pipeline: starting ===")

    # 1. Ingest
    prices_raw = load_market_prices(source_url=source_url)
    metadata_raw = load_dzongkhag_metadata()

    # 2. Transform
    curated = run_transform(prices_raw, metadata_raw)

    # 3. Quality gate — raises and stops the pipeline if validation fails
    try:
        report = run_quality_gate(curated, stop_on_fail=True)
    except DataQualityError as exc:
        logger.error("Quality gate FAILED — pipeline stopped, nothing was written.\n%s", exc)
        return 1
    logger.info(report.summary())

    # 4. Store
    db_path = save_to_sqlite(curated)
    try:
        parquet_path = save_to_parquet(curated)
        logger.info("Parquet store: %s", parquet_path)
    except ImportError as exc:
        # pyarrow/fastparquet not installed in this environment — SQLite write
        # above already satisfies the "queryable store" requirement, so this
        # is logged as a warning rather than failing the whole run.
        logger.warning("Skipped Parquet write (%s). Install pyarrow to enable it.", exc)

    logger.info("=== Pipeline finished successfully ===")
    logger.info("Rows curated: %d", len(curated))
    logger.info("SQLite store: %s", db_path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the market-prices data pipeline end to end.")
    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional live CSV endpoint to ingest from instead of the cached file.",
    )
    args = parser.parse_args()
    sys.exit(main(source_url=args.source_url))
