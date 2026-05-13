#!/usr/bin/env python3
"""
app.py — MLS Super Table Streamlit app.
Displays MLS standings by season with PTS, vPTS, and xPTS.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="MLS Super Table", page_icon="⚽", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/standings.csv")
    return df

df = load_data()

# ── Header ────────────────────────────────────────────────────────────
st.title("⚽ MLS Super Table")
st.markdown("Standings by **Actual Points**, **Vegas Points (vPTS)**, and **Expected Points (xPTS)**")

# ── Season selector ──────────────────────────────────────────────────
seasons = sorted(df["Season"].unique(), reverse=True)

# Label the current/latest season as "Live"
season_labels = {}
for s in seasons:
    if s == max(seasons):
        season_labels[f"{s} (Live)"] = s
    else:
        season_labels[str(s)] = s

selected_label = st.selectbox("Select Season", list(season_labels.keys()))
selected_season = season_labels[selected_label]

# ── Filter to selected season ────────────────────────────────────────
table = df[df["Season"] == selected_season].copy()
table = table.sort_values(["PTS", "GD", "GF"], ascending=[False, False, False]).reset_index(drop=True)
table.index += 1  # 1-based rank
table.index.name = "Rank"

# ── Choose display columns ───────────────────────────────────────────
base_cols = ["team", "GP", "W", "D", "L", "GF", "GA", "GD", "PTS", "PPG",
             "vPTS", "vPPG", "vPTS_diff"]

# Only show xG columns if data exists
has_xg = table["xPTS"].notna().any()
if has_xg:
    display_cols = base_cols + ["xGF", "xGA", "xGD", "xPTS", "xPPG"]
else:
    display_cols = base_cols

display = table[display_cols].copy()

# Rename for cleaner headers
display = display.rename(columns={
    "team": "Team",
    "vPTS_diff": "PTS ± vPTS",
})

# ── Sort toggle ──────────────────────────────────────────────────────
sort_options = ["PTS", "vPTS"]
if has_xg:
    sort_options.append("xPTS")

sort_by = st.radio("Sort by", sort_options, horizontal=True)

if sort_by == "vPTS":
    display = display.sort_values("vPTS", ascending=False).reset_index(drop=True)
    display.index += 1
elif sort_by == "xPTS" and has_xg:
    display = display.sort_values("xPTS", ascending=False).reset_index(drop=True)
    display.index += 1

# ── Display table ────────────────────────────────────────────────────
st.dataframe(
    display,
    use_container_width=True,
    height=35 * len(display) + 38,  # auto-size height
)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**vPTS** = Vegas Points from betting odds ([football-data.co.uk](https://www.football-data.co.uk/usa.php)) · "
    "**xPTS** = Expected Points from xG data ([FBRef](https://fbref.com/))"
)
