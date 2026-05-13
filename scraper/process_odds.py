#!/usr/bin/env python3
"""
process_odds.py — MLS Super Table: Vegas Points (vPTS) from betting odds.

Reads  : data/raw/USA.csv  (football-data.co.uk)
Writes : data/processed/matches_odds.csv
         data/processed/standings_odds.csv

Processes seasons 2022-2026.  Uses AvgC (average closing) odds columns.
vPTS = expected points derived from normalised implied probabilities.
"""

import os
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────
RAW_PATH        = "data/raw/USA.csv"
MATCHES_OUT     = "data/processed/matches_odds.csv"
STANDINGS_OUT   = "data/processed/standings_odds.csv"
SEASONS         = [2022, 2023, 2024, 2025, 2026]
ODDS_COLS       = ("AvgCH", "AvgCD", "AvgCA")   # home / draw / away


# ── Helpers ───────────────────────────────────────────────────────────
def implied_probs(oh, od, oa):
    """Convert decimal odds → normalised implied probabilities."""
    raw_h, raw_d, raw_a = 1 / oh, 1 / od, 1 / oa
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def expected_points(p_win, p_draw):
    """E[pts] = 3·P(win) + 1·P(draw)."""
    return round(3 * p_win + 1 * p_draw, 4)


# ── Main ──────────────────────────────────────────────────────────────
def run():
    os.makedirs("data/processed", exist_ok=True)

    # Load raw data
    df = pd.read_csv(RAW_PATH)
    # Drop the unnamed index column if present
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]

    print(f"✅ Loaded {len(df)} total rows from USA.csv")
    print(f"   Seasons available: {sorted(df['Season'].unique())}")
    print(f"   Columns: {list(df.columns)}\n")

    # Filter to target seasons
    df = df[df["Season"].isin(SEASONS)].copy()
    print(f"   Filtered to seasons {SEASONS}: {len(df)} rows\n")

    # ── Parse completed matches with valid odds ───────────────────────
    oh_col, od_col, oa_col = ODDS_COLS
         
    # Force odds columns to numeric (some rows have blanks/text)
    for col in [oh_col, od_col, oa_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows that have a result and valid odds
    mask = (
        df["Res"].isin(["H", "D", "A"])
        & df[oh_col].notna()
        & df[od_col].notna()
        & df[oa_col].notna()
        & (df[oh_col] > 1)
        & (df[od_col] > 1)
        & (df[oa_col] > 1)
    )
    matches = df.loc[mask].copy()
    print(f"   Parsed {len(matches)} completed matches with valid odds")
    print(f"   Odds source: {oh_col} / {od_col} / {oa_col}\n")

    # ── Compute vPTS per match ────────────────────────────────────────
    records = []
    for _, row in matches.iterrows():
        season = int(row["Season"])
        home   = row["Home"]
        away   = row["Away"]
        hg     = int(row["HG"])
        ag     = int(row["AG"])
        res    = row["Res"]

        p_h, p_d, p_a = implied_probs(row[oh_col], row[od_col], row[oa_col])
        home_vpts = expected_points(p_h, p_d)
        away_vpts = expected_points(p_a, p_d)

        # Actual points
        if res == "H":
            home_pts, away_pts = 3, 0
        elif res == "D":
            home_pts, away_pts = 1, 1
        else:
            home_pts, away_pts = 0, 3

        records.append({
            "Season": season,
            "Date":   row["Date"],
            "Home":   home,
            "Away":   away,
            "HG":     hg,
            "AG":     ag,
            "Res":    res,
            "Home_vPTS": home_vpts,
            "Away_vPTS": away_vpts,
            "Home_PTS":  home_pts,
            "Away_PTS":  away_pts,
        })

    matches_df = pd.DataFrame(records)
    matches_df.to_csv(MATCHES_OUT, index=False)
    print(f"💾 Saved {len(matches_df)} matches → {MATCHES_OUT}\n")

    # ── Build standings per season ────────────────────────────────────
    all_standings = []

    for season in SEASONS:
        sm = matches_df[matches_df["Season"] == season]
        if sm.empty:
            print(f"⚠️  No matches for {season}, skipping.")
            continue

        # Home rows
        home = sm.rename(columns={
            "Home": "team", "Away": "opponent",
            "HG": "GF", "AG": "GA",
            "Home_PTS": "PTS", "Home_vPTS": "vPTS",
        })[["Season", "team", "GF", "GA", "PTS", "vPTS", "Res"]]
        home = home.copy()
        home["W"] = (home["Res"] == "H").astype(int)
        home["D"] = (home["Res"] == "D").astype(int)
        home["L"] = (home["Res"] == "A").astype(int)

        # Away rows
        away = sm.rename(columns={
            "Away": "team", "Home": "opponent",
            "AG": "GF", "HG": "GA",
            "Away_PTS": "PTS", "Away_vPTS": "vPTS",
        })[["Season", "team", "GF", "GA", "PTS", "vPTS", "Res"]]
        away = away.copy()
        away["W"] = (away["Res"] == "A").astype(int)
        away["D"] = (away["Res"] == "D").astype(int)
        away["L"] = (away["Res"] == "H").astype(int)

        combined = pd.concat([home, away], ignore_index=True)

        standings = (
            combined.groupby(["Season", "team"])
            .agg(
                GP    = ("PTS",  "count"),
                W     = ("W",    "sum"),
                D     = ("D",    "sum"),
                L     = ("L",    "sum"),
                GF    = ("GF",   "sum"),
                GA    = ("GA",   "sum"),
                PTS   = ("PTS",  "sum"),
                vPTS  = ("vPTS", "sum"),
            )
            .reset_index()
        )

        standings["GD"]        = standings["GF"] - standings["GA"]
        standings["vPTS"]      = standings["vPTS"].round(1)
        standings["vPTS_diff"] = (standings["PTS"] - standings["vPTS"]).round(1)
        standings["PPG"]       = (standings["PTS"] / standings["GP"]).round(2)
        standings["vPPG"]      = (standings["vPTS"] / standings["GP"]).round(2)

        # Sort by PTS desc, then W, then GD
        standings = standings.sort_values(
            ["PTS", "W", "GD"], ascending=[False, False, False]
        ).reset_index(drop=True)

        # Reorder columns
        standings = standings[
            ["Season", "team", "GP", "W", "D", "L", "GF", "GA", "GD",
             "PTS", "vPTS", "vPTS_diff", "PPG", "vPPG"]
        ]

        all_standings.append(standings)

        # Print summary
        print(f"── {season} Season ({len(standings)} teams, {len(sm)} matches) ──")
        print(standings.to_string(index=False))
        print()

    # Combine all seasons and save
    final = pd.concat(all_standings, ignore_index=True)
    final.to_csv(STANDINGS_OUT, index=False)
    print(f"📊 Saved standings ({len(final)} rows, {final['Season'].nunique()} seasons) → {STANDINGS_OUT}")


if __name__ == "__main__":
    run()
