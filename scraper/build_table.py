#!/usr/bin/env python3
"""
build_table.py — MLS Super Table: Merge odds + xG standings.

Reads  : data/processed/standings_odds.csv
         data/processed/standings_xg.csv
Writes : data/processed/standings.csv
"""

import os
import pandas as pd
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────
ODDS_PATH      = "data/processed/standings_odds.csv"
XG_PATH        = "data/processed/standings_xg.csv"
STANDINGS_OUT  = "data/processed/standings.csv"


def run():
    os.makedirs("data/processed", exist_ok=True)

    # Load both standings files
    odds = pd.read_csv(ODDS_PATH)
    xg   = pd.read_csv(XG_PATH)

    print(f"✅ Loaded odds standings ({len(odds)} rows, {odds['Season'].nunique()} seasons)")
    print(f"✅ Loaded xG standings ({len(xg)} rows, {xg['Season'].nunique()} seasons)")

    # Merge on Season + team
    merged = odds.merge(
        xg[["Season", "team", "xGF", "xGA", "xGD", "xPTS", "xPPG"]],
        on=["Season", "team"],
        how="left",
    )

    print(f"✅ Merged odds + xG data")

    # Add xPTS diff column
    merged["xPTS_diff"] = (merged["PTS"] - merged["xPTS"]).round(1)

    # Sort by MLS tiebreaker: PTS → W → GD
    merged = merged.sort_values(
        ["Season", "PTS", "W", "GD"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    # Save
    merged.to_csv(STANDINGS_OUT, index=False)
    print(f"\n💾 Saved merged standings → {STANDINGS_OUT}")

    # Save timestamp
    with open("data/processed/last_updated.txt", "w") as f:
        f.write(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    # Print each season
    for season in sorted(merged["Season"].unique()):
        s = merged[merged["Season"] == season]
        print(f"\n── {season} ({len(s)} teams) ──")
        print(s.to_string(index=False))


if __name__ == "__main__":
    run()
