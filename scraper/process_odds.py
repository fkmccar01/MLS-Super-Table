"""
Step 2a — Process football-data.co.uk MLS CSV → vPTS (Vegas Points).
Source: https://www.football-data.co.uk/new/USA.csv
Column key: https://www.football-data.co.uk/notes.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_SEASON = 2025  # change this each year

# Odds priority — closing odds ("C" suffix)
ODDS_PRIORITY = [
    ("AvgCH",  "AvgCD",  "AvgCA"),    # Market average closing
    ("PSCH",   "PSCD",   "PSCA"),      # Pinnacle closing
    ("B365CH", "B365CD", "B365CA"),    # Bet365 closing
    ("MaxCH",  "MaxCD",  "MaxCA"),     # Max closing
]

# ─── TEAM NAME MAPPING ───────────────────────────────────────────
TEAM_MAP = {
    "Atlanta United":       "Atlanta United FC",
    "Atlanta Utd":          "Atlanta United FC",
    "Austin":               "Austin FC",
    "Austin FC":            "Austin FC",
    "Charlotte":            "Charlotte FC",
    "Charlotte FC":         "Charlotte FC",
    "Chicago Fire":         "Chicago Fire FC",
    "Cincinnati":           "FC Cincinnati",
    "FC Cincinnati":        "FC Cincinnati",
    "Colorado Rapids":      "Colorado Rapids",
    "Columbus Crew":        "Columbus Crew",
    "DC United":            "D.C. United",
    "D.C. United":          "D.C. United",
    "FC Dallas":            "FC Dallas",
    "Houston Dynamo":       "Houston Dynamo FC",
    "Inter Miami":          "Inter Miami CF",
    "LA Galaxy":            "LA Galaxy",
    "Los Angeles Galaxy":   "LA Galaxy",
    "Los Angeles FC":       "LAFC",
    "LAFC":                 "LAFC",
    "Minnesota United":     "Minnesota United FC",
    "CF Montreal":          "CF Montréal",
    "CF Montréal":          "CF Montréal",
    "Montreal Impact":      "CF Montréal",
    "Nashville SC":         "Nashville SC",
    "Nashville":            "Nashville SC",
    "New England Revolution": "New England Revolution",
    "New England Rev":      "New England Revolution",
    "New York City":        "New York City FC",
    "New York City FC":     "New York City FC",
    "New York Red Bulls":   "New York Red Bulls",
    "NY Red Bulls":         "New York Red Bulls",
    "Orlando City":         "Orlando City SC",
    "Philadelphia Union":   "Philadelphia Union",
    "Portland Timbers":     "Portland Timbers",
    "Real Salt Lake":       "Real Salt Lake",
    "San Jose Earthquakes": "San Jose Earthquakes",
    "Seattle Sounders":     "Seattle Sounders FC",
    "Sporting Kansas City": "Sporting Kansas City",
    "St. Louis City":       "St. Louis City SC",
    "St Louis City":        "St. Louis City SC",
    "Toronto FC":           "Toronto FC",
    "Vancouver Whitecaps":  "Vancouver Whitecaps FC",
    "San Diego":            "San Diego FC",
    "San Diego FC":         "San Diego FC",
    # Legacy teams
    "Chivas USA":           "Chivas USA",
}


# ─── LOAD ─────────────────────────────────────────────────────────
def load_csv(filepath=None):
    """Load a football-data.co.uk CSV. Auto-detects latest in data/raw/."""
    if filepath is None:
        files = sorted(RAW_DIR.glob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"No CSV files in {RAW_DIR}/")
        filepath = files[0]
    else:
        filepath = Path(filepath)

    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} total rows from {filepath.name}")
    print(f"   Seasons: {sorted(df['Season'].unique())}")
    print(f"   Columns: {list(df.columns)}")
    return df


# ─── PARSE ────────────────────────────────────────────────────────
def parse_matches(df, season=None):
    """
    Parse football-data.co.uk CSV.
    Columns: Home, Away, HG, AG, Res, Date, Time, Season, etc.
    """
    if season is not None:
        df = df[df["Season"] == season].copy()
        print(f"   Filtered to {season} season: {len(df)} matches")

    clean = pd.DataFrame()

    # Date
    clean["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if "Time" in df.columns:
        clean["time"] = df["Time"].values
    clean["season"] = df["Season"].values

    # Teams
    clean["home_team_raw"] = df["Home"].str.strip().values
    clean["away_team_raw"] = df["Away"].str.strip().values
    clean["home_team"] = clean["home_team_raw"].map(TEAM_MAP).fillna(clean["home_team_raw"])
    clean["away_team"] = clean["away_team_raw"].map(TEAM_MAP).fillna(clean["away_team_raw"])

    # Goals
    clean["home_goals"] = pd.to_numeric(df["HG"], errors="coerce")
    clean["away_goals"] = pd.to_numeric(df["AG"], errors="coerce")

    # Result
    clean["result"] = df["Res"].str.strip().values

    # Actual points
    clean["home_pts"] = clean["result"].map({"H": 3, "D": 1, "A": 0})
    clean["away_pts"] = clean["result"].map({"H": 0, "D": 1, "A": 3})

    # Odds
    odds_h, odds_d, odds_a = _select_odds_columns(df)
    clean["odds_h"] = pd.to_numeric(df[odds_h], errors="coerce")
    clean["odds_d"] = pd.to_numeric(df[odds_d], errors="coerce")
    clean["odds_a"] = pd.to_numeric(df[odds_a], errors="coerce")
    clean["odds_source"] = odds_h

    # Drop incomplete rows (upcoming fixtures)
    clean = clean.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)
    print(f"   Parsed {len(clean)} completed matches")
    print(f"   Odds source: {odds_h} / {odds_d} / {odds_a}")
    return clean


def _select_odds_columns(df):
    """Pick the best available odds columns from the CSV."""
    for h, d, a in ODDS_PRIORITY:
        if h in df.columns and d in df.columns and a in df.columns:
            if df[h].notna().sum() > 0:
                return h, d, a
    raise ValueError(f"No usable odds columns. Available: {list(df.columns)}")


# ─── vPTS (Vegas Points) ─────────────────────────────────────────
def add_vegas_points(df):
    """
    Decimal odds → normalized implied probabilities → vPTS.
    Removes bookmaker overround (vig) before calculating.
    """
    raw_h = 1 / df["odds_h"]
    raw_d = 1 / df["odds_d"]
    raw_a = 1 / df["odds_a"]
    overround = raw_h + raw_d + raw_a

    # Normalized probabilities
    df["prob_h"] = (raw_h / overround).round(4)
    df["prob_d"] = (raw_d / overround).round(4)
    df["prob_a"] = (raw_a / overround).round(4)

    # Vegas Points per match
    df["home_vpts"] = (df["prob_h"] * 3 + df["prob_d"] * 1).round(3)
    df["away_vpts"] = (df["prob_a"] * 3 + df["prob_d"] * 1).round(3)

    df["market_margin"] = ((overround - 1) * 100).round(1)
    return df


# ─── STANDINGS ────────────────────────────────────────────────────
def build_standings(df):
    """Aggregate match-level data into a season standings table with vPTS."""
    home = df.groupby("home_team").agg(
        home_gp=("result", "count"),
        home_w=("result", lambda x: (x == "H").sum()),
        home_d=("result", lambda x: (x == "D").sum()),
        home_l=("result", lambda x: (x == "A").sum()),
        home_gf=("home_goals", "sum"),
        home_ga=("away_goals", "sum"),
        home_pts=("home_pts", "sum"),
        home_vpts=("home_vpts", "sum"),
    ).rename_axis("team")

    away = df.groupby("away_team").agg(
        away_gp=("result", "count"),
        away_w=("result", lambda x: (x == "A").sum()),
        away_d=("result", lambda x: (x == "D").sum()),
        away_l=("result", lambda x: (x == "H").sum()),
        away_gf=("away_goals", "sum"),
        away_ga=("home_goals", "sum"),
        away_pts=("away_pts", "sum"),
        away_vpts=("away_vpts", "sum"),
    ).rename_axis("team")

    standings = home.join(away, how="outer").fillna(0)
    int_cols = ["home_gp", "home_w", "home_d", "home_l",
                "away_gp", "away_w", "away_d", "away_l"]
    standings[int_cols] = standings[int_cols].astype(int)

    standings["GP"]  = standings["home_gp"] + standings["away_gp"]
    standings["W"]   = standings["home_w"]  + standings["away_w"]
    standings["D"]   = standings["home_d"]  + standings["away_d"]
    standings["L"]   = standings["home_l"]  + standings["away_l"]
    standings["GF"]  = (standings["home_gf"] + standings["away_gf"]).astype(int)
    standings["GA"]  = (standings["home_ga"] + standings["away_ga"]).astype(int)
    standings["GD"]  = standings["GF"] - standings["GA"]
    standings["PTS"] = (standings["home_pts"] + standings["away_pts"]).astype(int)
    standings["vPTS"]      = (standings["home_vpts"] + standings["away_vpts"]).round(1)
    standings["vPTS_diff"] = (standings["PTS"] - standings["vPTS"]).round(1)
    standings["PPG"]       = (standings["PTS"] / standings["GP"]).round(2)
    standings["vPPG"]      = (standings["vPTS"] / standings["GP"]).round(2)

    standings = standings.sort_values(
        ["PTS", "GD", "GF"], ascending=[False, False, False]
    )

    output_cols = [
        "GP", "W", "D", "L", "GF", "GA", "GD",
        "PTS", "vPTS", "vPTS_diff", "PPG", "vPPG",
    ]
    return standings[output_cols]


# ─── MAIN ─────────────────────────────────────────────────────────
def run(filepath=None, season=CURRENT_SEASON):
    """Full pipeline: load → parse → vPTS → standings → save."""
    raw = load_csv(filepath)
    matches = parse_matches(raw, season=season)
    matches = add_vegas_points(matches)

    match_out = PROCESSED_DIR / "matches_odds.csv"
    matches.to_csv(match_out, index=False)
    print(f"\n💾 Saved {len(matches)} matches → {match_out}")

    standings = build_standings(matches)
    standings_out = PROCESSED_DIR / "standings_odds.csv"
    standings.to_csv(standings_out)
    print(f"📊 Saved standings ({len(standings)} teams) → {standings_out}")
    print(f"\n{standings.to_string()}")

    return matches, standings


if __name__ == "__main__":
    run()
