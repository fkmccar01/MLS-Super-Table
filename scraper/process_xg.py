"""
Step 2b — Scrape FBRef MLS match-level xG → xPTS (Expected Points).
Source: https://fbref.com/en/comps/22/schedule/Major-League-Soccer-Scores-and-Fixtures

⚠️  FBRef rate-limits aggressively. Respect a 3-second delay between requests.
    This script only needs ONE page request per run.
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time
import re

# ─── CONFIG ───────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# FBRef MLS Scores & Fixtures page
FBREF_URL = "https://fbref.com/en/comps/22/schedule/Major-League-Soccer-Scores-and-Fixtures"

# For Poisson model: max goals to sum over
MAX_GOALS = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─── TEAM NAME MAPPING ───────────────────────────────────────────
# FBRef names → canonical MLS names
TEAM_MAP = {
    "Atlanta United":       "Atlanta United FC",
    "Atlanta Utd":          "Atlanta United FC",
    "Austin FC":            "Austin FC",
    "Austin":               "Austin FC",
    "Charlotte FC":         "Charlotte FC",
    "Charlotte":            "Charlotte FC",
    "Chicago Fire":         "Chicago Fire FC",
    "Chicago Fire FC":      "Chicago Fire FC",
    "FC Cincinnati":        "FC Cincinnati",
    "Cincinnati":           "FC Cincinnati",
    "Colorado Rapids":      "Colorado Rapids",
    "Columbus Crew":        "Columbus Crew",
    "D.C. United":          "D.C. United",
    "DC United":            "D.C. United",
    "FC Dallas":            "FC Dallas",
    "Houston Dynamo FC":    "Houston Dynamo FC",
    "Houston Dynamo":       "Houston Dynamo FC",
    "Inter Miami CF":       "Inter Miami CF",
    "Inter Miami":          "Inter Miami CF",
    "LA Galaxy":            "LA Galaxy",
    "Los Angeles Galaxy":   "LA Galaxy",
    "Los Angeles FC":       "LAFC",
    "LAFC":                 "LAFC",
    "Minnesota United FC":  "Minnesota United FC",
    "Minnesota United":     "Minnesota United FC",
    "CF Montréal":          "CF Montréal",
    "CF Montreal":          "CF Montréal",
    "Nashville SC":         "Nashville SC",
    "Nashville":            "Nashville SC",
    "New England Revolution": "New England Revolution",
    "New England Rev":      "New England Revolution",
    "New York City FC":     "New York City FC",
    "New York City":        "New York City FC",
    "New York Red Bulls":   "New York Red Bulls",
    "NY Red Bulls":         "New York Red Bulls",
    "Orlando City SC":      "Orlando City SC",
    "Orlando City":         "Orlando City SC",
    "Philadelphia Union":   "Philadelphia Union",
    "Philadelphia":         "Philadelphia Union",
    "Portland Timbers":     "Portland Timbers",
    "Portland":             "Portland Timbers",
    "Real Salt Lake":       "Real Salt Lake",
    "Salt Lake":            "Real Salt Lake",
    "San Diego FC":         "San Diego FC",
    "San Diego":            "San Diego FC",
    "San Jose Earthquakes": "San Jose Earthquakes",
    "San Jose":             "San Jose Earthquakes",
    "Seattle Sounders FC":  "Seattle Sounders FC",
    "Seattle Sounders":     "Seattle Sounders FC",
    "Sporting Kansas City": "Sporting Kansas City",
    "Sporting KC":          "Sporting Kansas City",
    "St. Louis City SC":    "St. Louis City SC",
    "St. Louis City":       "St. Louis City SC",
    "St Louis City":        "St. Louis City SC",
    "Toronto FC":           "Toronto FC",
    "Toronto":              "Toronto FC",
    "Vancouver Whitecaps FC": "Vancouver Whitecaps FC",
    "Vancouver Whitecaps":  "Vancouver Whitecaps FC",
    "Vancouver":            "Vancouver Whitecaps FC",
}


# ─── SCRAPE FBREF ─────────────────────────────────────────────────
def scrape_fbref_schedule():
    """
    Scrape FBRef MLS Scores & Fixtures page.
    Returns a DataFrame with: date, home, away, home_goals, away_goals, home_xg, away_xg.
    """
    print(f"🌐 Fetching {FBREF_URL}")
    response = requests.get(FBREF_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    print(f"   Status: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Find the scores/fixtures table
    # FBRef uses id like "sched_YYYY_22_1" or "sched_all" or similar
    table = None
    for t in soup.find_all("table"):
        tid = t.get("id", "")
        if "sched" in tid:
            table = t
            break

    if table is None:
        raise ValueError("Could not find schedule table on FBRef page")

    print(f"   Found table: {table.get('id', 'unknown')}")

    # Parse rows
    rows = []
    tbody = table.find("tbody")
    for tr in tbody.find_all("tr"):
        # Skip spacer/header rows
        if tr.get("class") and "spacer" in " ".join(tr.get("class", [])):
            continue
        if tr.find("th", {"scope": "col"}):
            continue

        cells = {}
        for td in tr.find_all(["th", "td"]):
            data_stat = td.get("data-stat", "")
            cells[data_stat] = td.get_text(strip=True)

        # Only process rows that have a score (completed matches)
        if not cells.get("score"):
            continue

        # Parse score "2–1" or "2-1"
        score_text = cells.get("score", "")
        score_match = re.match(r"(\d+)\D+(\d+)", score_text)
        if not score_match:
            continue

        row = {
            "date": cells.get("date", ""),
            "home_team_raw": cells.get("home_team", "").strip(),
            "away_team_raw": cells.get("away_team", "").strip(),
            "home_goals": int(score_match.group(1)),
            "away_goals": int(score_match.group(2)),
            "home_xg": cells.get("home_xg", ""),
            "away_xg": cells.get("away_xg", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"   Scraped {len(df)} completed matches")

    # Clean up types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
    df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")

    # Map team names
    df["home_team"] = df["home_team_raw"].map(TEAM_MAP).fillna(df["home_team_raw"])
    df["away_team"] = df["away_team_raw"].map(TEAM_MAP).fillna(df["away_team_raw"])

    # Result
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] == df["away_goals"], "D", "A")
    )

    # Actual points
    df["home_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["away_pts"] = df["result"].map({"H": 0, "D": 1, "A": 3})

    return df


# ─── POISSON xPTS ─────────────────────────────────────────────────
def poisson_xpts(xg_home, xg_away, max_goals=MAX_GOALS):
    """
    Given match xG values, compute win/draw/loss probabilities
    using independent Poisson distributions, then return xPTS.

    Returns: (home_xpts, away_xpts, prob_h, prob_d, prob_a)
    """
    if pd.isna(xg_home) or pd.isna(xg_away):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Ensure minimum xG to avoid degenerate distributions
    xg_home = max(xg_home, 0.01)
    xg_away = max(xg_away, 0.01)

    # Probability mass for each goal count 0..max_goals
    home_probs = poisson.pmf(range(max_goals + 1), xg_home)
    away_probs = poisson.pmf(range(max_goals + 1), xg_away)

    # Joint probability matrix
    prob_matrix = np.outer(home_probs, away_probs)

    # P(home win) = sum where home > away
    prob_h = np.tril(prob_matrix, -1).sum()
    # P(draw) = sum of diagonal
    prob_d = np.trace(prob_matrix)
    # P(away win) = sum where away > home
    prob_a = np.triu(prob_matrix, 1).sum()

    # Expected points
    home_xpts = round(prob_h * 3 + prob_d * 1, 3)
    away_xpts = round(prob_a * 3 + prob_d * 1, 3)

    return home_xpts, away_xpts, round(prob_h, 4), round(prob_d, 4), round(prob_a, 4)


def add_expected_points(df):
    """Apply Poisson xPTS model to each match."""
    results = df.apply(
        lambda row: poisson_xpts(row["home_xg"], row["away_xg"]),
        axis=1,
        result_type="expand"
    )
    results.columns = ["home_xpts", "away_xpts", "xg_prob_h", "xg_prob_d", "xg_prob_a"]
    df = pd.concat([df, results], axis=1)
    return df


# ─── STANDINGS ────────────────────────────────────────────────────
def build_standings(df):
    """Aggregate match-level xG data into standings with xPTS."""
    # Drop rows without xG data
    df_xg = df.dropna(subset=["home_xpts", "away_xpts"]).copy()

    if len(df_xg) == 0:
        print("⚠️  No matches with xG data found")
        return pd.DataFrame()

    home = df_xg.groupby("home_team").agg(
        home_gp=("result", "count"),
        home_w=("result", lambda x: (x == "H").sum()),
        home_d=("result", lambda x: (x == "D").sum()),
        home_l=("result", lambda x: (x == "A").sum()),
        home_gf=("home_goals", "sum"),
        home_ga=("away_goals", "sum"),
        home_xgf=("home_xg", "sum"),
        home_xga=("away_xg", "sum"),
        home_pts=("home_pts", "sum"),
        home_xpts=("home_xpts", "sum"),
    ).rename_axis("team")

    away = df_xg.groupby("away_team").agg(
        away_gp=("result", "count"),
        away_w=("result", lambda x: (x == "A").sum()),
        away_d=("result", lambda x: (x == "D").sum()),
        away_l=("result", lambda x: (x == "H").sum()),
        away_gf=("away_goals", "sum"),
        away_ga=("home_goals", "sum"),
        away_xgf=("away_xg", "sum"),
        away_xga=("home_xg", "sum"),
        away_pts=("away_pts", "sum"),
        away_xpts=("away_xpts", "sum"),
    ).rename_axis("team")

    standings = home.join(away, how="outer").fillna(0)
    int_cols = ["home_gp", "home_w", "home_d", "home_l",
                "away_gp", "away_w", "away_d", "away_l"]
    standings[int_cols] = standings[int_cols].astype(int)

    standings["GP"]   = standings["home_gp"] + standings["away_gp"]
    standings["W"]    = standings["home_w"]  + standings["away_w"]
    standings["D"]    = standings["home_d"]  + standings["away_d"]
    standings["L"]    = standings["home_l"]  + standings["away_l"]
    standings["GF"]   = (standings["home_gf"] + standings["away_gf"]).astype(int)
    standings["GA"]   = (standings["home_ga"] + standings["away_ga"]).astype(int)
    standings["GD"]   = standings["GF"] - standings["GA"]
    standings["xGF"]  = (standings["home_xgf"] + standings["away_xgf"]).round(1)
    standings["xGA"]  = (standings["home_xga"] + standings["away_xga"]).round(1)
    standings["xGD"]  = (standings["xGF"] - standings["xGA"]).round(1)
    standings["PTS"]  = (standings["home_pts"] + standings["away_pts"]).astype(int)
    standings["xPTS"]      = (standings["home_xpts"] + standings["away_xpts"]).round(1)
    standings["xPTS_diff"] = (standings["PTS"] - standings["xPTS"]).round(1)
    standings["PPG"]       = (standings["PTS"] / standings["GP"]).round(2)
    standings["xPPG"]      = (standings["xPTS"] / standings["GP"]).round(2)

    standings = standings.sort_values(
        ["PTS", "GD", "GF"], ascending=[False, False, False]
    )

    output_cols = [
        "GP", "W", "D", "L", "GF", "GA", "GD",
        "xGF", "xGA", "xGD",
        "PTS", "xPTS", "xPTS_diff", "PPG", "xPPG",
    ]
    return standings[output_cols]


# ─── MAIN ─────────────────────────────────────────────────────────
def run():
    """Full pipeline: scrape FBRef → xPTS → standings → save."""
    # 1. Scrape
    matches = scrape_fbref_schedule()

    # 2. Add xPTS via Poisson model
    matches = add_expected_points(matches)

    # 3. Save match-level
    match_out = PROCESSED_DIR / "matches_xg.csv"
    matches.to_csv(match_out, index=False)
    print(f"\n💾 Saved {len(matches)} matches → {match_out}")

    # 4. Build & save standings
    standings = build_standings(matches)
    standings_out = PROCESSED_DIR / "standings_xg.csv"
    standings.to_csv(standings_out)
    print(f"📊 Saved standings ({len(standings)} teams) → {standings_out}")
    print(f"\n{standings.to_string()}")

    return matches, standings


if __name__ == "__main__":
    run()
