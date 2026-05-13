#!/usr/bin/env python3
"""
build_table.py — MLS Super Table: merge odds + xG standings.

Reads  : data/processed/standings_odds.csv
         data/processed/standings_xg.csv  (optional, skipped if missing)
Writes : data/processed/standings.csv
"""

import os
import pandas as pd

ODDS_PATH = "data/processed/standings_odds.csv"
XG_PATH   = "data/processed/standings_xg.csv"
OUT_PATH  = "data/processed/standings.csv"


def run():
    # ── Load odds standings (required) ────────────────────────────────
    if not os.path.exists(ODDS_PATH):
        print(f"❌ {ODDS_PATH} not found. Run process_odds.py first.")
        return

    odds = pd.read_csv(ODDS_PATH)
    print(f"✅ Loaded odds standings ({len(odds)} rows, {odds['Season'].nunique()} seasons)")

    # ── Load xG standings (optional) ──────────────────────────────────
    if os.path.exists(XG_PATH):
        xg = pd.read_csv(XG_PATH)
        print(f"✅ Loaded xG standings ({len(xg)} rows, {xg['Season'].nunique()} seasons)")

        # Merge on Season + team
        merged = odds.merge(
            xg[["Season", "team", "xGF", "xGA", "xGD", "xPTS", "xPPG"]],
            on=["Season", "team"],
            how="left",
        )
        print(f"✅ Merged odds + xG data")
    else:
        print(f"⚠️  {XG_PATH} not found — building table with odds data only.")
        merged = odds.copy()
        # Add placeholder xG columns so Streamlit doesn't break
        merged["xGF"]  = None
        merged["xGA"]  = None
        merged["xGD"]  = None
        merged["xPTS"] = None
        merged["xPPG"] = None

    # ── Sort and save ─────────────────────────────────────────────────
    merged = merged.sort_values(
        ["Season", "PTS", "GD", "GF"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    merged.to_csv(OUT_PATH, index=False)
    print(f"\n💾 Saved merged standings → {OUT_PATH}")

    # Print preview per season
    for season in sorted(merged["Season"].unique()):
        s = merged[merged["Season"] == season]
        print(f"\n── {season} ({len(s)} teams) ──")
        print(s.to_string(index=False))


if __name__ == "__main__":
    run()
