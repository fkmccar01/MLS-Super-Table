#!/usr/bin/env python3
"""
process_xg.py — MLS Super Table: Expected Points (xPTS) from xG data.

Reads  : American Soccer Analysis API (itscalledsoccer)
Writes : data/processed/standings_xg.csv

Processes seasons 2022-2026.
Uses ASA's xpoints directly.
Regular season only — tries stage_name filter, falls back to GP cap of 34.
"""

import os
import pandas as pd
from itscalledsoccer.client import AmericanSoccerAnalysis

# ── Config ────────────────────────────────────────────────────────────
STANDINGS_OUT = "data/processed/standings_xg.csv"
SEASONS       = [2022, 2023, 2024, 2025, 2026]
MAX_GP        = 34  # MLS regular season = 34 matches per team

# ── Team name mapping: ASA → football-data.co.uk ─────────────────────
ASA_TO_FD = {
    "Atlanta United FC":          "Atlanta Utd",
    "Austin FC":                  "Austin FC",
    "CF Montréal":                "CF Montreal",
    "Charlotte FC":               "Charlotte",
    "Chicago Fire FC":            "Chicago Fire",
    "Colorado Rapids":            "Colorado Rapids",
    "Columbus Crew":              "Columbus Crew",
    "D.C. United":                "DC United",
    "FC Cincinnati":              "FC Cincinnati",
    "FC Dallas":                  "FC Dallas",
    "Houston Dynamo FC":          "Houston Dynamo",
    "Inter Miami CF":             "Inter Miami",
    "LA Galaxy":                  "Los Angeles Galaxy",
    "Los Angeles FC":             "Los Angeles FC",
    "Minnesota United FC":        "Minnesota United",
    "Nashville SC":               "Nashville SC",
    "New England Revolution":     "New England Revolution",
    "New York City FC":           "New York City",
    "New York Red Bulls":         "New York Red Bulls",
    "Orlando City SC":            "Orlando City",
    "Philadelphia Union":         "Philadelphia Union",
    "Portland Timbers FC":        "Portland Timbers",
    "Portland Timbers":           "Portland Timbers",
    "Real Salt Lake":             "Real Salt Lake",
    "San Jose Earthquakes":       "San Jose Earthquakes",
    "Seattle Sounders FC":        "Seattle Sounders",
    "Sporting Kansas City":       "Sporting Kansas City",
    "St. Louis City SC":          "St. Louis City",
    "Toronto FC":                 "Toronto FC",
    "Vancouver Whitecaps FC":     "Vancouver Whitecaps",
    "San Diego FC":               "San Diego FC",
}


# ── Helpers ───────────────────────────────────────────────────────────
def map_team_name(asa_name):
    """Map ASA team name to football-data.co.uk name."""
    return ASA_TO_FD.get(asa_name, asa_name)


# ── Main ──────────────────────────────────────────────────────────────
def run():
    os.makedirs("data/processed", exist_ok=True)

    asa = AmericanSoccerAnalysis()

    # Build team_id → team_name lookup
    print("🌐 Loading ASA team directory...")
    try:
        teams = asa.get_teams(leagues="mls")
        team_lookup = dict(zip(teams["team_id"], teams["team_name"]))
        print(f"   Loaded {len(team_lookup)} teams")
    except Exception as e:
        print(f"⚠️  Could not load team directory: {e}")
        team_lookup = {}

    all_standings = []

    for season in SEASONS:
        print(f"\n🌐 Fetching xG data for {season} season...")

        # Try with stage_name filter first (regular season only)
        team_xg = None
        try:
            team_xg = asa.get_team_xgoals(
                leagues="mls",
                season_name=str(season),
                split_by_seasons=True,
                stage_name="Regular Season",
            )
            if team_xg is not None and not team_xg.empty:
                print(f"   ✅ Got regular season data via stage_name filter")
        except Exception as e:
            print(f"   ⚠️  stage_name filter failed: {e}")
            team_xg = None

        # Fallback: get all data (includes playoffs)
        if team_xg is None or team_xg.empty:
            try:
                team_xg = asa.get_team_xgoals(
                    leagues="mls",
                    season_name=str(season),
                    split_by_seasons=True,
                )
                if team_xg is not None and not team_xg.empty:
                    print(f"   ⚠️  Using unfiltered data (may include playoffs)")
            except Exception as e:
                print(f"   ❌ Failed to fetch {season}: {e}")
                continue

        if team_xg is None or team_xg.empty:
            print(f"   ❌ No data found for {season}, skipping.")
            continue

        # Debug: print columns on first season
        if season == SEASONS[0]:
            print(f"   ASA columns: {list(team_xg.columns)}")

        print(f"   Found {len(team_xg)} teams with xG data")

        # Build standings from team aggregates
        records = []
        for _, row in team_xg.iterrows():
            # Resolve team name
            raw_id = str(row["team_id"])
            team_name_asa = team_lookup.get(raw_id, raw_id)
            team_name = map_team_name(team_name_asa)

            gp   = int(row["count_games"])
            xgf  = float(row["xgoals_for"])
            xga  = float(row["xgoals_against"])
            xpts = float(row["xpoints"])

            if gp == 0:
                continue

            # Cap at 34 GP — scale xG stats proportionally if over
            if gp > MAX_GP:
                scale = MAX_GP / gp
                xgf  = xgf * scale
                xga  = xga * scale
                xpts = xpts * scale
                gp   = MAX_GP

            records.append({
                "Season": season,
                "team":   team_name,
                "GP":     gp,
                "xGF":    round(xgf, 1),
                "xGA":    round(xga, 1),
                "xGD":    round(xgf - xga, 1),
                "xPTS":   round(xpts, 1),
                "xPPG":   round(xpts / gp, 2),
            })

        standings = pd.DataFrame(records)
        standings = standings.sort_values(
            "xPTS", ascending=False
        ).reset_index(drop=True)
        all_standings.append(standings)

        print(f"📊 {season}: {len(standings)} teams")
        print(standings.to_string(index=False))

    # Save
    if all_standings:
        final = pd.concat(all_standings, ignore_index=True)
        final.to_csv(STANDINGS_OUT, index=False)
        print(f"\n📊 Saved standings ({len(final)} rows) → {STANDINGS_OUT}")
    else:
        print("❌ No standings generated.")


if __name__ == "__main__":
    run()
