"""
generate_raw_data.py
---------------------
Produces the two raw source files used by the pipeline:

  data/raw/market_prices.csv        -> daily vegetable market prices (CSV source)
  data/raw/dzongkhag_metadata.json  -> dzongkhag reference data (JSON / "API" source)

This script is run ONCE to seed data/raw/. It is kept in the repo so the
whole project is reproducible from nothing but this script (no external
downloads required to review or re-mark the assignment).

Note on the dataset: the values are synthetically generated (fixed random
seed => fully reproducible) to model realistic Bhutanese vegetable-market
behaviour, because the assignment needs a dataset that anyone can regenerate
offline without depending on a third-party endpoint staying alive. The
ingestion layer (src/ingest.py) is written so that market_prices.csv could
just as easily be swapped for a live CSV export or metadata.json for a real
API response -- the rest of the pipeline does not care where the file came
from, only that it matches the expected schema.
"""

import json
import random
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RANDOM_SEED = 42

DZONGKHAGS = [
    {"dzongkhag": "Thimphu", "region": "Western", "altitude_m": 2334, "avg_annual_rainfall_mm": 650},
    {"dzongkhag": "Paro", "region": "Western", "altitude_m": 2200, "avg_annual_rainfall_mm": 780},
    {"dzongkhag": "Punakha", "region": "Western", "altitude_m": 1200, "avg_annual_rainfall_mm": 900},
    {"dzongkhag": "Wangdue Phodrang", "region": "Western", "altitude_m": 1350, "avg_annual_rainfall_mm": 870},
    {"dzongkhag": "Chukha", "region": "Western", "altitude_m": 1500, "avg_annual_rainfall_mm": 1600},
    {"dzongkhag": "Haa", "region": "Western", "altitude_m": 2670, "avg_annual_rainfall_mm": 900},
    {"dzongkhag": "Samtse", "region": "Southern", "altitude_m": 300, "avg_annual_rainfall_mm": 3500},
    {"dzongkhag": "Sarpang", "region": "Southern", "altitude_m": 250, "avg_annual_rainfall_mm": 3200},
    {"dzongkhag": "Tsirang", "region": "Southern", "altitude_m": 900, "avg_annual_rainfall_mm": 2200},
    {"dzongkhag": "Trongsa", "region": "Central", "altitude_m": 2200, "avg_annual_rainfall_mm": 850},
    {"dzongkhag": "Bumthang", "region": "Central", "altitude_m": 2800, "avg_annual_rainfall_mm": 650},
    {"dzongkhag": "Zhemgang", "region": "Central", "altitude_m": 1900, "avg_annual_rainfall_mm": 2700},
    {"dzongkhag": "Sarpang", "region": "Southern", "altitude_m": 250, "avg_annual_rainfall_mm": 3200},
    {"dzongkhag": "Mongar", "region": "Eastern", "altitude_m": 1600, "avg_annual_rainfall_mm": 1100},
    {"dzongkhag": "Trashigang", "region": "Eastern", "altitude_m": 1100, "avg_annual_rainfall_mm": 1400},
    {"dzongkhag": "Lhuentse", "region": "Eastern", "altitude_m": 1700, "avg_annual_rainfall_mm": 1300},
]
# de-duplicate any accidental repeats while preserving order
seen = set()
DZONGKHAGS = [d for d in DZONGKHAGS if not (d["dzongkhag"] in seen or seen.add(d["dzongkhag"]))]

COMMODITIES = [
    ("Potato", 25, 45),
    ("Cabbage", 15, 30),
    ("Chilli (dried)", 180, 320),
    ("Chilli (fresh)", 60, 140),
    ("Tomato", 35, 90),
    ("Onion", 40, 70),
    ("Cauliflower", 30, 60),
    ("Spinach", 20, 45),
    ("Radish", 15, 28),
    ("Broccoli", 45, 85),
]

MARKETS = ["Centenary Farmers Market", "Municipal Vegetable Market", "Weekend Roadside Market"]


def generate_market_prices_csv(n_days: int = 120, seed: int = RANDOM_SEED) -> Path:
    rng = random.Random(seed)
    rows = []
    start_date = pd.Timestamp("2026-01-01")

    for day_offset in range(n_days):
        date = start_date + pd.Timedelta(days=day_offset)
        # not every dzongkhag reports every day - realistic missingness
        active_dzongkhags = rng.sample(DZONGKHAGS, k=rng.randint(8, len(DZONGKHAGS)))
        for dz in active_dzongkhags:
            n_commodities = rng.randint(4, len(COMMODITIES))
            for commodity, lo, hi in rng.sample(COMMODITIES, k=n_commodities):
                base_price = rng.uniform(lo, hi)
                seasonal = 1 + 0.15 * rng.uniform(-1, 1)
                price = round(base_price * seasonal, 2)
                quantity = round(rng.uniform(20, 500), 1)
                market = rng.choice(MARKETS)

                # deliberately inject a small amount of "dirty" data so the
                # quality gate and cleaning steps have something real to do
                if rng.random() < 0.015:
                    price = -abs(price)  # invalid negative price
                if rng.random() < 0.01:
                    quantity = None  # missing quantity
                if rng.random() < 0.008:
                    commodity = f"  {commodity.upper()}  "  # inconsistent formatting

                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "dzongkhag": dz["dzongkhag"],
                        "commodity": commodity,
                        "price_nu_per_kg": price,
                        "quantity_kg": quantity,
                        "market": market,
                    }
                )

    df = pd.DataFrame(rows)
    # a few exact duplicate rows, another realistic data-quality issue
    dupes = df.sample(n=15, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    out_path = RAW_DIR / "market_prices.csv"
    df.to_csv(out_path, index=False)
    return out_path


def generate_dzongkhag_metadata_json() -> Path:
    out_path = RAW_DIR / "dzongkhag_metadata.json"
    with open(out_path, "w") as f:
        json.dump({"source": "simulated_NSB_style_reference_data", "dzongkhags": DZONGKHAGS}, f, indent=2)
    return out_path


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = generate_market_prices_csv()
    json_path = generate_dzongkhag_metadata_json()
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
