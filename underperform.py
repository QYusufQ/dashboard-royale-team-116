import panda as pd

df = pd.read_parquet("data/1_prepared_views/prepared_daily_view.parquet")

underperform= (
    df[df[active]]
)
