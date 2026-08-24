"""
quality.py — Data-quality gate.

A lightweight, dependency-free "Great-Expectations-style" validation gate:
each check is a self-contained assertion with a human-readable name. If any
check fails, `run_quality_gate` raises a DataQualityError listing every
failure, and the pipeline refuses to write the batch to the store.

(Great Expectations itself was left out of requirements.txt only because it
pulls in a large dependency tree; the same checks below could be expressed
as GE Expectations 1:1 — see README for the mapping — and swapping the
implementation in would not change the pipeline's behaviour or interface.)
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

VALID_DZONGKHAGS = {
    "Thimphu", "Paro", "Punakha", "Wangdue Phodrang", "Chukha", "Haa",
    "Samtse", "Sarpang", "Tsirang", "Trongsa", "Bumthang", "Zhemgang",
    "Mongar", "Trashigang", "Lhuentse",
}

REQUIRED_COLUMNS = {
    "date", "dzongkhag", "commodity", "price_nu_per_kg",
    "quantity_kg", "market", "revenue_nu",
}


@dataclass
class QualityReport:
    passed: bool
    failures: list = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return "PASSED — all data-quality checks succeeded."
        lines = ["FAILED — the following checks did not pass:"]
        lines += [f"  - {f}" for f in self.failures]
        return "\n".join(lines)


class DataQualityError(Exception):
    """Raised by run_quality_gate() when validation fails and stop_on_fail=True."""


def run_quality_gate(df: pd.DataFrame, stop_on_fail: bool = True) -> QualityReport:
    """Run all data-quality checks against the transformed DataFrame.

    Each check appends a message to `failures` if it does not pass.
    If stop_on_fail is True and any check failed, raises DataQualityError
    (this is the gate that stops the pipeline before writing to the store).
    """
    failures = []

    # 1. schema check — required columns must be present
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        failures.append(f"missing required columns: {missing_cols}")

    # 2. completeness — no nulls in critical columns
    for col in ["date", "dzongkhag", "commodity", "price_nu_per_kg", "quantity_kg"]:
        if col in df.columns and df[col].isna().any():
            n = df[col].isna().sum()
            failures.append(f"column '{col}' has {n} null value(s)")

    # 3. value ranges — prices and quantities must be strictly positive
    if "price_nu_per_kg" in df.columns and (df["price_nu_per_kg"] <= 0).any():
        n = (df["price_nu_per_kg"] <= 0).sum()
        failures.append(f"{n} row(s) have non-positive price_nu_per_kg")
    if "quantity_kg" in df.columns and (df["quantity_kg"] <= 0).any():
        n = (df["quantity_kg"] <= 0).sum()
        failures.append(f"{n} row(s) have non-positive quantity_kg")

    # 4. categorical membership — dzongkhag must be one of the 15 known values
    if "dzongkhag" in df.columns:
        bad = set(df["dzongkhag"].unique()) - VALID_DZONGKHAGS
        if bad:
            failures.append(f"unrecognised dzongkhag value(s): {bad}")

    # 5. uniqueness — no fully-duplicated rows should remain post-cleaning
    dupes = df.duplicated().sum()
    if dupes:
        failures.append(f"{dupes} fully-duplicated row(s) remain after cleaning")

    # 6. referential integrity — every row must have matched metadata (region)
    if "region" in df.columns and df["region"].isna().any():
        n = df["region"].isna().sum()
        failures.append(f"{n} row(s) failed to join against dzongkhag metadata")

    report = QualityReport(passed=(len(failures) == 0), failures=failures)
    logger.info(report.summary())

    if stop_on_fail and not report.passed:
        raise DataQualityError(report.summary())

    return report
