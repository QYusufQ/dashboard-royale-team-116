import pandas as pd

daily = pd.read_parquet("data/1_prepared_views/prepared_daily_view.parquet")
print(daily.shape)   # (56110, 44): one row per machine per day
print(daily.head())

from pathlib import Path

for f in sorted(Path("data").rglob("*.parquet")):
    if f.parent.name in ("5_optional",):
        continue
    df = pd.read_parquet(f)
    print(f"\n{f.relative_to('data')}  ({len(df):,} rows)")
    print("  " + ", ".join(df.columns))