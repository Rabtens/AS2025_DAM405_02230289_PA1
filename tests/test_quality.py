import pandas as pd
import pytest

from src.quality import DataQualityError, run_quality_gate

GOOD_ROW = {
    "date": pd.Timestamp("2026-01-01"),
    "dzongkhag": "Thimphu",
    "commodity": "Potato",
    "price_nu_per_kg": 30.0,
    "quantity_kg": 10.0,
    "market": "M1",
    "revenue_nu": 300.0,
    "region": "Western",
}


def make_df(overrides=None, n=3):
    # vary the date across rows so a "good" multi-row batch isn't itself
    # flagged as duplicates by the uniqueness check
    rows = []
    for i in range(n):
        row = dict(GOOD_ROW)
        row["date"] = GOOD_ROW["date"] + pd.Timedelta(days=i)
        rows.append(row)
    if overrides:
        for key, value in overrides.items():
            rows[0][key] = value
    return pd.DataFrame(rows)


def test_valid_batch_passes():
    df = make_df()
    report = run_quality_gate(df, stop_on_fail=True)
    assert report.passed
    assert report.failures == []


def test_negative_price_fails_gate():
    df = make_df({"price_nu_per_kg": -5.0})
    with pytest.raises(DataQualityError):
        run_quality_gate(df, stop_on_fail=True)


def test_null_critical_column_fails_gate():
    df = make_df({"quantity_kg": None})
    with pytest.raises(DataQualityError):
        run_quality_gate(df, stop_on_fail=True)


def test_unrecognised_dzongkhag_fails_gate():
    df = make_df({"dzongkhag": "Narnia"})
    with pytest.raises(DataQualityError):
        run_quality_gate(df, stop_on_fail=True)


def test_duplicate_rows_fail_gate():
    df = pd.DataFrame([GOOD_ROW, GOOD_ROW])  # identical rows
    with pytest.raises(DataQualityError):
        run_quality_gate(df, stop_on_fail=True)


def test_missing_column_fails_gate():
    df = make_df().drop(columns=["revenue_nu"])
    with pytest.raises(DataQualityError):
        run_quality_gate(df, stop_on_fail=True)


def test_stop_on_fail_false_returns_report_without_raising():
    df = make_df({"price_nu_per_kg": -5.0})
    report = run_quality_gate(df, stop_on_fail=False)
    assert not report.passed
    assert len(report.failures) > 0
