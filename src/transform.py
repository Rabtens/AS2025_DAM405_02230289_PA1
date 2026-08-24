"""
transform.py — Transformation stage of the pipeline.

Cleans and types the raw market-price data, joins it against the
dzongkhag reference metadata, and adds derived features.
Every function here is pure (DataFrame in -> DataFrame out) so it is
straightforward to unit test in isolation from ingestion/storage.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_market_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Type, normalise and de-duplicate the raw market-price records."""
    df = df.copy()

    # normalise text fields (strip whitespace, consistent case)
    df["commodity"] = df["commodity"].astype(str).str.strip().str.title()
    df["dzongkhag"] = df["dzongkhag"].astype(str).str.strip()
    df["market"] = df["market"].astype(str).str.strip()

    # type the numeric/date columns
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price_nu_per_kg"] = pd.to_numeric(df["price_nu_per_kg"], errors="coerce")
    df["quantity_kg"] = pd.to_numeric(df["quantity_kg"], errors="coerce")

    # drop exact duplicate records
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d exact duplicate rows", before - len(df))

    # drop rows with an invalid (non-positive) or missing price/quantity/date —
    # these cannot be sensibly imputed for a price record
    before = len(df)
    df = df[
        df["price_nu_per_kg"].notna()
        & (df["price_nu_per_kg"] > 0)
        & df["quantity_kg"].notna()
        & (df["quantity_kg"] > 0)
        & df["date"].notna()
    ]
    logger.info("Dropped %d rows with invalid/missing price, quantity or date", before - len(df))

    return df.reset_index(drop=True)


def enrich_with_metadata(prices_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Join cleaned price records with dzongkhag reference metadata."""
    merged = prices_df.merge(metadata_df, on="dzongkhag", how="left")
    unmatched = merged["region"].isna().sum()
    if unmatched:
        logger.warning("%d rows had no matching dzongkhag metadata", unmatched)
    return merged


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features used by downstream analysis/reporting."""
    df = df.copy()

    # 1. revenue proxy for the recorded lot
    df["revenue_nu"] = (df["price_nu_per_kg"] * df["quantity_kg"]).round(2)

    # 2. per-commodity price z-score -> flags unusually cheap/expensive lots
    grp = df.groupby("commodity")["price_nu_per_kg"]
    df["price_zscore"] = ((df["price_nu_per_kg"] - grp.transform("mean")) / grp.transform("std")).round(3)
    df["is_price_outlier"] = df["price_zscore"].abs() > 3

    # 3. calendar feature for time-based analysis
    df["day_of_week"] = df["date"].dt.day_name()

    return df


def run_transform(prices_raw: pd.DataFrame, metadata_raw: pd.DataFrame) -> pd.DataFrame:
    """Full transform pipeline: clean -> enrich -> add derived features."""
    cleaned = clean_market_prices(prices_raw)
    enriched = enrich_with_metadata(cleaned, metadata_raw)
    featured = add_derived_features(enriched)
    logger.info("Transform complete: %d rows, %d columns", *featured.shape)
    return featured
