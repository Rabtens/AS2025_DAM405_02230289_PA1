import pandas as pd
import pytest

from src.transform import (
    add_derived_features,
    clean_market_prices,
    enrich_with_metadata,
    run_transform,
)


@pytest.fixture
def raw_prices():
    return pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02", "not-a-date", "2026-01-03"],
            "dzongkhag": ["Thimphu", "Thimphu", " Paro ", "Punakha", "Thimphu"],
            "commodity": ["potato", "  POTATO  ", "cabbage", "onion", "potato"],
            "price_nu_per_kg": [30.0, 30.0, 20.0, -5.0, None],
            "quantity_kg": [100.0, 100.0, 50.0, 20.0, 30.0],
            "market": ["M1", "M1", "M2", "M1", "M1"],
        }
    )


@pytest.fixture
def metadata():
    return pd.DataFrame(
        {
            "dzongkhag": ["Thimphu", "Paro", "Punakha"],
            "region": ["Western", "Western", "Western"],
            "altitude_m": [2334, 2200, 1200],
            "avg_annual_rainfall_mm": [650, 780, 900],
        }
    )


class TestCleanMarketPrices:
    def test_drops_exact_duplicates(self, raw_prices):
        cleaned = clean_market_prices(raw_prices)
        # the two identical "Thimphu / potato / 30.0" rows collapse to one
        assert len(cleaned[(cleaned["dzongkhag"] == "Thimphu") & (cleaned["commodity"] == "Potato")]) == 1

    def test_drops_invalid_price_and_date_rows(self, raw_prices):
        cleaned = clean_market_prices(raw_prices)
        # the negative-price row and the unparseable-date row must be gone
        assert (cleaned["price_nu_per_kg"] <= 0).sum() == 0
        assert cleaned["date"].isna().sum() == 0

    def test_normalises_text_fields(self, raw_prices):
        cleaned = clean_market_prices(raw_prices)
        assert set(cleaned["commodity"].unique()) <= {"Potato", "Cabbage", "Onion"}
        assert "Paro" in cleaned["dzongkhag"].unique()
        assert " Paro " not in cleaned["dzongkhag"].unique()

    def test_output_has_no_missing_quantity(self, raw_prices):
        cleaned = clean_market_prices(raw_prices)
        assert cleaned["quantity_kg"].isna().sum() == 0


class TestEnrichWithMetadata:
    def test_join_adds_region_column(self, raw_prices, metadata):
        cleaned = clean_market_prices(raw_prices)
        enriched = enrich_with_metadata(cleaned, metadata)
        assert "region" in enriched.columns
        assert enriched["region"].notna().all()

    def test_unmatched_dzongkhag_yields_null_region(self, metadata):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"]),
                "dzongkhag": ["Unknownplace"],
                "commodity": ["Potato"],
                "price_nu_per_kg": [30.0],
                "quantity_kg": [10.0],
                "market": ["M1"],
            }
        )
        enriched = enrich_with_metadata(df, metadata)
        assert enriched["region"].isna().all()


class TestAddDerivedFeatures:
    def test_revenue_is_price_times_quantity(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "commodity": ["Potato", "Potato"],
                "price_nu_per_kg": [30.0, 40.0],
                "quantity_kg": [10.0, 5.0],
            }
        )
        featured = add_derived_features(df)
        assert list(featured["revenue_nu"]) == [300.0, 200.0]

    def test_day_of_week_is_derived_from_date(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"]),  # a Thursday
                "commodity": ["Potato"],
                "price_nu_per_kg": [30.0],
                "quantity_kg": [10.0],
            }
        )
        featured = add_derived_features(df)
        assert featured.loc[0, "day_of_week"] == "Thursday"

    def test_outlier_flag_identifies_extreme_price(self):
        # 19 normal prices around 30, one wildly high price for the same commodity.
        # (needs >= ~17 points for a single outlier's z-score to be able to
        # exceed 3 at all - with n points the max possible |z| for one point
        # is (n-1)/sqrt(n), which only clears 3 once n >= 17)
        prices = [30.0] * 19 + [3000.0]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"] * 20),
                "commodity": ["Potato"] * 20,
                "price_nu_per_kg": prices,
                "quantity_kg": [10.0] * 20,
            }
        )
        featured = add_derived_features(df)
        assert featured["is_price_outlier"].iloc[-1] == True  # noqa: E712
        assert featured["is_price_outlier"].iloc[:-1].sum() == 0


def test_run_transform_end_to_end(raw_prices, metadata):
    result = run_transform(raw_prices, metadata)
    assert "revenue_nu" in result.columns
    assert "region" in result.columns
    assert (result["price_nu_per_kg"] > 0).all()
    assert result["date"].notna().all()
