import pandas as pd

# Load the daily view dataset
df = pd.read_parquet("data/1_prepared_views/prepared_daily_view.parquet")

# Aggregate earnings for active days per machine
bottom_10 = (
    df[df["active"]]
    .groupby("machine_id")
    .agg(
        total_house_win=("house_win_usd", "sum"),
        total_theo=("theo_usd", "sum"),
        total_coin_in=("coin_in_usd", "sum"),
        available_hours=("available_hours", "sum")
    )
    .sort_values("total_house_win")
    .head(10)
)

print(bottom_10)