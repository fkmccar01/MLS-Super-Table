#!/usr/bin/env python3
"""
process_xg.py — MLS Super Table: Expected Points (xPTS) from xG data.

Reads  : American Soccer Analysis API (itscalledsoccer)
Writes : data/processed/matches_xg.csv
         data/processed/standings_xg.csv

Processes seasons 2022-2026.
xPTS = expected points derived from match-level xG using Poisson model.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import poisson
from itscalledsoccer.client import AmericanSoccerAnalysis

# ── Config ────────────────────────────────────────────────────────────
MATCHES_OUT   = "data/processed/matches_xg.csv"
STANDINGS_OUT = "data/processed/standings_xg.csv"
SEASONS       = [2022, 2023, 2024, 2025, 2026]
MAX_GOALS     = 10  # max goals to consider in Poisson grid

# ── Team name mapping: ASA → football-data.co.uk ─────────────────────
# Ensures team names match between odds and xG standings.
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
def poisson_xpts(xg_home, xg_away):
    """
    Given match xG for home and away, use a Poisson model to compute
    expected points for each team.
    Returns (home_xpts, away_xpts).
    """
    # Build Poisson probability grids
    home_goals = np.arange(0, MAX_GOALS + 1)
    away_goals = np.arange(0, MAX_GOALS + 1)

    home_probs = poisson.pmf(home_goals, xg_home)
    away_probs = poisson.pmf(away_goals, xg_away)

    # Joint probability matrix
    prob_matrix = np.outer(home_probs, away_probs)

    # P(home win) = sum of probs where home > away
    p_home_win = np.sum(np.tril(prob_matrix, -1))
    # P(draw) = sum of diagonal
    p_draw = np.sum(np.diag(prob_matrix))
    # P(away win) = sum of probs where away > home
    p_away_win = np.sum(np.triu(prob_matrix, 1))

    home_xpts = round(3 * p_home_win + 1 * p_draw, 4)
    away_xpts = round(3 * p_away_win + 1 * p_draw, 4)

    return home_xpts, away_xpts


def map_team_name(asa_name):
    """Map ASA team name to football-data.co.uk name."""
    return ASA_TO_FD.get(asa_name, asa_name)


# ── Main ──────────────────────────────────────────────────────────────
def run():
    os.makedirs("data/processed", exist_ok=True)

    asa = AmericanSoccerAnalysis()

    all_matches = []
    all_standings = []

    for season in SEASONS:
        print(f"\n🌐 Fetching xG data for {season} season...")

        try:
            games = asa.get_games_xgoals(
                leagues="mls",
                season_name=str(season),
            )
        except Exception as e:
            print(f"⚠️  Failed to fetch {season}: {e}")
            continue

        if games.empty:
            print(f"⚠️  No games found for {season}, skipping.")
            continue

        # Filter to completed games with valid xG
        games = games.dropna(subset=["home_xgoals", "away_xgoals"])
        games = games[games["home_xgoals"] > 0]  # drop 0-0 xG edge cases
        print(f"   Found {len(games)} completed matches with xG data")

        # ── Match-level xPTS ──────────────────────────────────────────
        records = []
        for _, g in games.iterrows():
            home = map_team_name(g.get("home_team_name", g.get("home_team_id", "")))
            away = map_team_name(g.get("away_team_name", g.get("away_team_id", "")))
            xg_h = float(g["home_xgoals"])
            xg_a = float(g["away_xgoals"])

            # Clamp xG to avoid Poisson issues with 0
            xg_h = max(xg_h, 0.05)
            xg_a = max(xg_a, 0.05)

            home_xpts, away_xpts = poisson_xpts(xg_h, xg_a)

            records.append({
                "Season":     season,
                "Date":       g.get("date_time_utc", ""),
                "Home":       home,
                "Away":       away,
                "Home_xG":    round(xg_h, 2),
                "Away_xG":    round(xg_a, 2),
                "Home_xPTS":  home_xpts,
                "Away_xPTS":  away_xpts,
            })

        matches_df = pd.DataFrame(records)
        all_matches.append(matches_df)

        # ── Build standings ───────────────────────────────────────────
        home_rows = matches_df[["Season", "Home", "Home_xG", "Away_xG", "Home_xPTS"]].copy()
        home_rows.columns = ["Season", "team", "xGF", "xGA", "xPTS"]

        away_rows = matches_df[["Season", "Away", "Away_xG", "Home_xG", "Away_xPTS"]].copy()
        away_rows.columns = ["Season", "team", "xGF", "xGA", "xPTS"]

        combined = pd.concat([home_rows, away_rows], ignore_index=True)

        standings = (
            combined.groupby(["Season", "team"])
            .agg(
                GP   = ("xPTS", "count"),
                xGF  = ("xGF",  "sum"),
                xGA  = ("xGA",  "sum"),
                xPTS = ("xPTS", "sum"),
            )
            .reset_index()
        )

        standings["xGF"]  = standings["xGF"].round(1)
        standings["xGA"]  = standings["xGA"].round(1)
        standings["xGD"]  = (standings["xGF"] - standings["xGA"]).round(1)
        standings["xPTS"] = standings["xPTS"].round(1)
        standings["xPPG"] = (standings["xPTS"] / standings["GP"]).round(2)

        standings = standings.sort_values("xPTS", ascending=False).reset_index(drop=True)

        standings = standings[
            ["Season", "team", "GP", "xGF", "xGA", "xGD", "xPTS", "xPPG"]
        ]

        all_standings.append(standings)

        print(f"📊 {season}: {len(standings)} teams")
        print(standings.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────
    if all_matches:
        final_matches = pd.concat(all_matches, ignore_index=True)
        final_matches.to_csv(MATCHES_OUT, index=False)
        print(f"\n💾 Saved {len(final_matches)} matches → {MATCHES_OUT}")

    if all_standings:
        final_standings = pd.concat(all_standings, ignore_index=True)
        final_standings.to_csv(STANDINGS_OUT, index=False)
        print(f"📊 Saved standings ({len(final_standings)} rows) → {STANDINGS_OUT}")
    else:
        print("❌ No standings generated.")


if __name__ == "__main__":
    run()
