#!/usr/bin/env python3
"""
process_xg.py — MLS Super Table: Expected Points (xPTS) from xG data.

Reads  : American Soccer Analysis API (itscalledsoccer)
Writes : data/processed/standings_xg.csv

Processes seasons 2022-2026.
xPTS = expected points derived from team-level xG using Poisson model.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import poisson
from itscalledsoccer.client import AmericanSoccerAnalysis

# ── Config ────────────────────────────────────────────────────────────
STANDINGS_OUT = "data/processed/standings_xg.csv"
SEASONS       = [2022, 2023, 2024, 2025, 2026]
MAX_GOALS     = 10  # max goals to consider in Poisson grid

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
def poisson_xpts_per_game(avg_xgf, avg_xga):
    """
    Given average xGF and xGA per game, use Poisson to compute
    expected points per game.
    """
    goals = np.arange(0, MAX_GOALS + 1)

    for_probs = poisson.pmf(goals, avg_xgf)
    against_probs = poisson.pmf(goals, avg_xga)

    prob_matrix = np.outer(for_probs, against_probs)

    p_win  = np.sum(np.tril(prob_matrix, -1))
    p_draw = np.sum(np.diag(prob_matrix))

    return round(3 * p_win + 1 * p_draw, 4)


def map_team_name(asa_name):
    """Map ASA team name to football-data.co.uk name."""
    return ASA_TO_FD.get(asa_name, asa_name)


# ── Main ──────────────────────────────────────────────────────────────
def run():
    os.makedirs("data/processed", exist_ok=True)

    asa = AmericanSoccerAnalysis()

    all_standings = []

    for season in SEASONS:
        print(f"\n🌐 Fetching xG data for {season} season...")

        try:
            team_xg = asa.get_team_xgoals(
                leagues="mls",
                season_name=str(season),
                split_by_seasons=True,
            )
        except Exception as e:
            print(f"⚠️  Failed to fetch {season}: {e}")
            continue

        if team_xg.empty:
            print(f"⚠️  No data found for {season}, skipping.")
            continue

        # Debug: print columns on first season so we know what ASA returns
        if season == SEASONS[0]:
            print(f"   ASA columns: {list(team_xg.columns)}")

        print(f"   Found {len(team_xg)} teams with xG data")

        # Build standings from team aggregates
        records = []
        for _, row in team_xg.iterrows():
            # Handle different possible column names across versions
            team_name = map_team_name(str(
                row.get("team_name", row.get("team_id", "Unknown"))
            ))

            gp  = int(row.get("count", row.get("games_played", 0)))
            xgf = float(row.get("xgoals_for", row.get("team_xgoals", 0)))
            xga = float(row.get("xgoals_against", row.get("opponent_xgoals", 0)))

            if gp == 0:
                continue

            # Poisson xPTS from per-game averages
            avg_xgf = max(xgf / gp, 0.05)
            avg_xga = max(xga / gp, 0.05)
            xpts_per_game = poisson_xpts_per_game(avg_xgf, avg_xga)
            xpts = round(xpts_per_game * gp, 1)

            records.append({
                "Season": season,
                "team":   team_name,
                "GP":     gp,
                "xGF":    round(xgf, 1),
                "xGA":    round(xga, 1),
                "xGD":    round(xgf - xga, 1),
                "xPTS":   xpts,
                "xPPG":   round(xpts / gp, 2),
            })

        standings = pd.DataFrame(records)
        standings = standings.sort_values("xPTS", ascending=False).reset_index(drop=True)
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
