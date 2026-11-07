import pandas as pd

# Load the daily prepared view dataset
df = pd.read_parquet("data/1_prepared_views/prepared_daily_view.parquet")

# Convert currency and time columns to float to avoid Decimal arithmetic errors
numeric_cols = ["house_win_usd", "theo_usd", "coin_in_usd", "available_hours"]
df[numeric_cols] = df[numeric_cols].astype(float)

# 1. Aggregate active floor performance per machine
m_summary = (
    df[df["active"]]
    .groupby("machine_id")
    .agg(
        house_win=("house_win_usd", "sum"),
        theo=("theo_usd", "sum"),
        coin_in=("coin_in_usd", "sum"),
        available_hours=("available_hours", "sum"),
        days_active=("gaming_date", "nunique")
    )
)

# 2. Add diagnostic rates
m_summary["luck_residual"] = m_summary["house_win"] - m_summary["theo"]
m_summary["theo_per_hour"] = m_summary["theo"] / m_summary["available_hours"]

# Floor-wide baseline demand rate
floor_avg_theo_hr = m_summary["theo_per_hour"].mean()

# 3. Filter out non-machine issues
def diagnose(row):
    if row["available_hours"] < 3500:
        return "FILTER OUT: Maintenance / Downtime Leak"
    elif row["luck_residual"] < -5000 and row["theo_per_hour"] >= (0.75 * floor_avg_theo_hr):
        return "FILTER OUT: Unlucky (Strong Demand)"
    else:
        return "TRULY UNDERPERFORMING"

m_summary["status"] = m_summary.apply(diagnose, axis=1)

# 4. Rank true underperformers: lowest theoretical hourly demand rate with high uptime
true_underperformers = (
    m_summary[m_summary["status"] == "TRULY UNDERPERFORMING"]
    .sort_values("theo_per_hour")
    .head(10)
)

print("--- TRULY UNDERPERFORMING MACHINES ---")
print(true_underperformers[["house_win", "theo", "available_hours", "theo_per_hour", "status"]])