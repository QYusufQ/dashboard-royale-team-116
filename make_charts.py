"""
Generates the new chart HTML files referenced in index.html.
Edit the CONFIG block to match your file and column names, then run:
    python make_charts.py
Each chart is built independently, so if one fails the rest still get written.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ======================= CONFIG: EDIT TO MATCH YOUR DATA =======================
MACHINE_DAY_FILE = "machine_day.csv"   # one row per machine per day
M = dict(
    date="date", machine="machine_id", zone="zone", theme="theme",
    coin_in="coin_in", theo="theo_win", games="games_played",
    hours_avail="hours_available",      # hours the machine was up
    hours_occ="hours_occupied",         # hours someone was playing
    downtime="downtime_hours",
)

PLAYER_FILE = "player_summary.csv"     # one row per player (aggregate first if needed)
P = dict(player="player_id", tier="loyalty_tier", theo="theo_win", comp="comp_cost")

FORECAST_FILE = "forecast_results.csv" # held-out test days
F = dict(date="date", actual="actual", predicted="predicted", baseline="baseline")

# The manager's bottom-10 machine IDs
MANAGER_BOTTOM_10 = []  # e.g. ["M014", "M087", ...]

# Recommendation spend (keep in sync with the cards in index.html)
BUDGET = 100_000
SPEND = {"Move 8 machines": 14_400, "Add technician": 11_520, "Inspect RTP settings": 6_000}
# ===============================================================================

GOLD, RED, CREAM, GREY = "#d4af37", "#b22222", "#f5ecd7", "#8a8a8a"
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=CREAM, size=13),
    margin=dict(l=60, r=30, t=40, b=60),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    legend=dict(orientation="h", y=-0.2),
)

def save(fig, name):
    fig.update_layout(**LAYOUT)
    fig.write_html(name, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": False, "responsive": True})
    print(f"  wrote {name}")

def safe(fn):
    try:
        fn()
    except Exception as e:
        print(f"  SKIPPED {fn.__name__}: {e}")

# ------------------------------------------------------------------ machines
def rank_shift():
    df = pd.read_csv(MACHINE_DAY_FILE)
    g = df.groupby(M["machine"]).agg(coin_in=(M["coin_in"], "sum"),
                                     theo=(M["theo"], "sum"),
                                     occ=(M["hours_occ"], "sum"))
    g["raw_pct"] = g["coin_in"].rank(pct=True) * 100
    # theo per occupied hour removes luck (theo, not actual) and traffic/availability (per occupied hr)
    g["adj_pct"] = (g["theo"] / g["occ"].replace(0, np.nan)).rank(pct=True) * 100
    sub = g.loc[[m for m in MANAGER_BOTTOM_10 if m in g.index]].sort_values("raw_pct")
    if sub.empty:
        raise ValueError("fill MANAGER_BOTTOM_10 with machine IDs")
    fig = go.Figure()
    for mid, r in sub.iterrows():
        fig.add_trace(go.Scatter(x=[r.raw_pct, r.adj_pct], y=[str(mid)] * 2, mode="lines",
                                 line=dict(color=GREY, width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sub.raw_pct, y=sub.index.astype(str), mode="markers",
                             marker=dict(color=RED, size=12), name="Raw coin-in percentile"))
    fig.add_trace(go.Scatter(x=sub.adj_pct, y=sub.index.astype(str), mode="markers",
                             marker=dict(color=GOLD, size=12), name="Adjusted percentile"))
    fig.add_vline(x=10, line_dash="dot", line_color=RED,
                  annotation_text="bottom 10% line", annotation_font_color=RED)
    fig.update_layout(xaxis_title="Percentile across all machines (0 = worst)",
                      xaxis_range=[0, 100])
    save(fig, "rank_shift.html")

# ------------------------------------------------------------------ demand
def demand_decomposition():
    df = pd.read_csv(MACHINE_DAY_FILE)
    z = df.groupby(M["zone"]).agg(occ=(M["hours_occ"], "sum"), avail=(M["hours_avail"], "sum"),
                                  coin=(M["coin_in"], "sum"), games=(M["games"], "sum"))
    z["occupancy"] = z.occ / z.avail
    z["avg_bet"] = z.coin / z.games
    floor_occ = z.occ.sum() / z.avail.sum()
    floor_bet = z.coin.sum() / z.games.sum()
    z["occ_idx"] = z.occupancy / floor_occ * 100
    z["bet_idx"] = z.avg_bet / floor_bet * 100
    z = z.sort_values("occ_idx")
    fig = go.Figure([
        go.Bar(x=z.index, y=z.occ_idx, name="Occupancy index", marker_color=GOLD,
               customdata=z.occupancy, hovertemplate="%{x}<br>Occupancy: %{customdata:.1%}<extra></extra>"),
        go.Bar(x=z.index, y=z.bet_idx, name="Avg bet index", marker_color=RED,
               customdata=z.avg_bet, hovertemplate="%{x}<br>Avg bet: $%{customdata:.2f}<extra></extra>"),
    ])
    fig.add_hline(y=100, line_dash="dot", line_color=CREAM, annotation_text="floor average")
    fig.update_layout(barmode="group", yaxis_title="Index (floor avg = 100)")
    save(fig, "demand_decomposition.html")

# ------------------------------------------------------------------ downtime
def downtime_pareto():
    df = pd.read_csv(MACHINE_DAY_FILE)
    g = df.groupby([M["machine"], M["zone"]]).agg(down=(M["downtime"], "sum"),
                                                  theo=(M["theo"], "sum"),
                                                  avail=(M["hours_avail"], "sum")).reset_index()
    g = g[g.down > 0].sort_values("down", ascending=False)
    # lost theo ~ downtime hours x the machine's theo per available hour
    g["lost_theo"] = g.down * g.theo / g.avail.replace(0, np.nan)
    g["cum_pct"] = g.down.cumsum() / g.down.sum() * 100
    top = g.head(30)
    fig = go.Figure([
        go.Bar(x=top[M["machine"]].astype(str), y=top.down, name="Downtime hours", marker_color=RED,
               customdata=np.stack([top[M["zone"]], top.lost_theo], axis=1),
               hovertemplate="%{x} (%{customdata[0]})<br>%{y:.0f} hrs down<br>"
                             "Est. lost theo: $%{customdata[1]:,.0f}<extra></extra>"),
        go.Scatter(x=top[M["machine"]].astype(str), y=top.cum_pct, name="Cumulative % of all downtime",
                   yaxis="y2", line=dict(color=GOLD, width=3)),
    ])
    fig.update_layout(yaxis_title="Downtime hours",
                      yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%",
                                  showgrid=False))
    save(fig, "downtime_pareto.html")
    n80 = int((g.cum_pct < 80).sum()) + 1
    print(f"    -> {n80} of {len(g)} machines with downtime cause 80% of downtime hours; "
          f"est. lost theo total ${g.lost_theo.sum():,.0f}")

# ------------------------------------------------------------------ players
def player_concentration():
    p = pd.read_csv(PLAYER_FILE).sort_values(P["theo"], ascending=False)
    x = np.arange(1, len(p) + 1) / len(p) * 100
    y = p[P["theo"]].cumsum() / p[P["theo"]].sum() * 100
    fig = go.Figure([
        go.Scatter(x=x, y=y, name="Players", line=dict(color=GOLD, width=3), fill="tozeroy",
                   fillcolor="rgba(212,175,55,0.15)",
                   hovertemplate="Top %{x:.1f}% of players → %{y:.1f}% of theo<extra></extra>"),
        go.Scatter(x=[0, 100], y=[0, 100], name="Perfectly equal", line=dict(color=GREY, dash="dot")),
    ])
    fig.update_layout(xaxis_title="% of players (most valuable first)", yaxis_title="% of total theo",
                      xaxis_ticksuffix="%", yaxis_ticksuffix="%")
    save(fig, "player_concentration.html")
    for k in (1, 10, 20):
        print(f"    -> top {k}% of players = {np.interp(k, x, y):.1f}% of theo")

def comp_by_tier():
    p = pd.read_csv(PLAYER_FILE)
    t = p.groupby(P["tier"]).agg(theo=(P["theo"], "sum"), comp=(P["comp"], "sum"),
                                 n=(P["player"], "count"))
    t["rate"] = t.comp / t.theo * 100
    t = t.sort_values("theo")
    overall = t.comp.sum() / t.theo.sum() * 100
    fig = go.Figure(go.Bar(x=t.index.astype(str), y=t.rate, marker_color=GOLD,
                           customdata=np.stack([t.n, t.comp, t.theo], axis=1),
                           hovertemplate="%{x}: %{y:.2f}% of theo<br>%{customdata[0]} players<br>"
                                         "Comps $%{customdata[1]:,.0f} / Theo $%{customdata[2]:,.0f}"
                                         "<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dot", line_color=RED,
                  annotation_text=f"overall {overall:.2f}%", annotation_font_color=RED)
    fig.update_layout(xaxis_title="Loyalty tier", yaxis_title="Comp cost as % of theo",
                      yaxis_ticksuffix="%")
    save(fig, "comp_by_tier.html")

# ------------------------------------------------------------------ model
def forecast_vs_actual():
    f = pd.read_csv(FORECAST_FILE, parse_dates=[F["date"]])
    mae_b = (f[F["actual"]] - f[F["baseline"]]).abs().mean()
    mae_f = (f[F["actual"]] - f[F["predicted"]]).abs().mean()
    fig = go.Figure([
        go.Scatter(x=f[F["date"]], y=f[F["actual"]], name="Actual", line=dict(color=CREAM, width=2)),
        go.Scatter(x=f[F["date"]], y=f[F["predicted"]], name=f"Forecast (MAE ${mae_f:,.0f})",
                   line=dict(color=GOLD, width=3)),
        go.Scatter(x=f[F["date"]], y=f[F["baseline"]], name=f"Baseline (MAE ${mae_b:,.0f})",
                   line=dict(color=GREY, dash="dot")),
    ])
    fig.update_layout(yaxis_title="Daily value ($)", yaxis_tickprefix="$")
    save(fig, "forecast_vs_actual.html")
    print(f"    -> Baseline MAE ${mae_b:,.0f} | Forecast MAE ${mae_f:,.0f} | "
          f"improvement {(1 - mae_f / mae_b):.1%}  (put these in the Model cards)")

# ------------------------------------------------------------------ budget
def budget_allocation():
    items = dict(SPEND)
    items["Unallocated / reserve"] = BUDGET - sum(SPEND.values())
    colors = [GOLD, RED, CREAM, GREY][: len(items) - 1] + ["rgba(138,138,138,0.35)"]
    fig = go.Figure([go.Bar(y=["July budget"], x=[v], name=f"{k}: ${v:,.0f}", orientation="h",
                            marker_color=c) for (k, v), c in zip(items.items(), colors)])
    fig.update_layout(barmode="stack", xaxis_tickprefix="$", xaxis_range=[0, BUDGET], height=260)
    save(fig, "budget_allocation.html")

if __name__ == "__main__":
    for fn in (rank_shift, demand_decomposition, downtime_pareto, player_concentration,
               comp_by_tier, forecast_vs_actual, budget_allocation):
        safe(fn)
