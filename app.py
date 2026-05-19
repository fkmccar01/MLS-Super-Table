#!/usr/bin/env python3
"""
app.py — MLS Super Table: Streamlit frontend.

Displays MLS standings with actual points (PTS), Vegas points (vPTS),
and expected points (xPTS) side by side.
"""

import streamlit as st
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLS Super Table",
    page_icon="⚽",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────
def load_standings():
    df = pd.read_csv("data/processed/standings.csv")
    return df

df = load_standings()

# ── Header ────────────────────────────────────────────────────────────
st.title("⚽ MLS Super Table")
st.markdown(
    "Comparing actual points (PTS), Vegas implied points (vPTS), "
    "and expected points from xG (xPTS) across MLS seasons."
)

# ── Season selector ───────────────────────────────────────────────────
seasons = sorted(df["Season"].unique(), reverse=True)
selected_season = st.selectbox("Select Season", seasons, index=0)

season_df = df[df["Season"] == selected_season].copy()
season_df = season_df.sort_values(
    ["PTS", "W", "GD"], ascending=[False, False, False]
).reset_index(drop=True)

# ── Display columns ──────────────────────────────────────────────────
display_cols = [
    "team", "GP", "W", "D", "L", "GF", "GA", "GD",
    "PTS", "PPG",
    "vPTS", "vPPG", "vPTS_diff",
    "xGF", "xGA", "xGD", "xPTS", "xPPG", "xPTS_diff",
]

# Only show columns that exist
display_cols = [c for c in display_cols if c in season_df.columns]
table = season_df[display_cols].copy()

# Rename for display
table = table.rename(columns={
    "team": "Team",
    "vPTS_diff": "PTS−vPTS",
    "xPTS_diff": "PTS−xPTS",
})

# ── Summary metrics ───────────────────────────────────────────────────
num_teams = len(table)
total_matches = season_df["GP"].sum() // 2
current_label = " 🔴 LIVE" if selected_season == seasons[0] else ""

st.markdown(
    f"### {selected_season} Season{current_label}\n"
    f"**{num_teams}** teams · **{total_matches}** matches"
)

# ── Styled table ──────────────────────────────────────────────────────
st.dataframe(
    table,
    use_container_width=True,
    height=35 * num_teams + 50,
    hide_index=True,
    column_config={
        "Team": st.column_config.TextColumn("Team", width="medium"),
        "PTS": st.column_config.NumberColumn("PTS", format="%d"),
        "vPTS": st.column_config.NumberColumn("vPTS", format="%.1f"),
        "xPTS": st.column_config.NumberColumn("xPTS", format="%.1f"),
        "PPG": st.column_config.NumberColumn("PPG", format="%.2f"),
        "vPPG": st.column_config.NumberColumn("vPPG", format="%.2f"),
        "xPPG": st.column_config.NumberColumn("xPPG", format="%.2f"),
        "PTS−vPTS": st.column_config.NumberColumn("PTS−vPTS", format="%.1f"),
        "PTS−xPTS": st.column_config.NumberColumn("PTS−xPTS", format="%.1f"),
        "xGF": st.column_config.NumberColumn("xGF", format="%.1f"),
        "xGA": st.column_config.NumberColumn("xGA", format="%.1f"),
        "xGD": st.column_config.NumberColumn("xGD", format="%.1f"),
    },
)

# ── Legend ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ Column definitions"):
    st.markdown("""
| Column | Definition |
|--------|-----------|
| **PTS** | Actual MLS points (3W + 1D) |
| **PPG** | Points per game |
| **vPTS** | Vegas Points — expected points derived from closing betting odds (football-data.co.uk) |
| **vPPG** | Vegas points per game |
| **PTS−vPTS** | Overperformance vs. Vegas expectations (positive = outperformed) |
| **xGF / xGA** | Expected goals for / against (American Soccer Analysis) |
| **xGD** | Expected goal difference |
| **xPTS** | Expected points based on xG model (American Soccer Analysis) |
| **xPPG** | Expected points per game |
| **PTS−xPTS** | Overperformance vs. xG expectations (positive = outperformed) |
""")

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
try:
    with open("data/processed/last_updated.txt") as f:
        last_updated = f.read().strip()
    st.caption(
        f"Data: football-data.co.uk (odds) · American Soccer Analysis (xG) · "
        f"Updated automatically via GitHub Actions · Last updated: {last_updated}"
    )
except FileNotFoundError:
    st.caption(
        "Data: football-data.co.uk (odds) · American Soccer Analysis (xG) · "
        "Updated automatically via GitHub Actions."
    )
