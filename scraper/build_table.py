"""
Step 2c — Merge odds-based vPTS and xG-based xPTS into one unified standings table.
Reads:  data/processed/matches_odds.csv  (from process_odds.py)
        data/processed/matches_xg.csv    (from process_xg.py)
Writes: data/processed/standings.csv     (merged table for Streamlit)
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def _build_from_matches(df, pts_col, team_suffix=""):
    """Generic home/away aggregation from a matches DataFrame."""
    home = df.groupby("home_team").agg(
        home_gp=("result", "count"),
        home_w=("result", lambda x: (x == "H").sum()),
        home_d=("result", lambda x: (x == "D").sum()),
        home_l=("result", lambda x: (x == "A").sum()),
        home_gf=("home_goals", "sum"),
        home_ga=("away_goals", "sum"),
        home_pts=("home_pts", "sum"),
        home_model_pts=(f"home_{pts_col}", "sum"),
    ).rename_axis("team")

    away = df.groupby("away_team").agg(
        away_gp=("result", "count"),
        away_w=("result", lambda x: (x == "A").sum()),
        away_d=("result", lambda x: (x == "D").sum()),
        away_l=("result", lambda x: (x == "H").sum()),
        away_gf=("away_goals", "sum"),
        away_ga=("home_goals", "sum"),
        away_pts=("away_pts", "sum"),
        away_model_pts=(f"away_{pts_col}", "sum"),
    ).rename_axis("team")

    s = home.join(away, how="outer").fillna(0)
    int_cols = ["home_gp", "home_w", "home_d", "home_l",
                "away_gp", "away_w", "away_d", "away_l"]
    s[int_cols] = s[int_cols].astype(int)

    out = pd.DataFrame(index=s.index)
    out["GP"]  = s["home_gp"] + s["away_gp"]
    out["W"]   = s["home_w"]  + s["away_w"]
    out["D"]   = s["home_d"]  + s["away_d"]
    out["L"]   = s["home_l"]  + s["away_l"]
    out["GF"]  = (s["home_gf"] + s["away_gf"]).astype(int)
    out["GA"]  = (s["home_ga"] + s["away_ga"]).astype(int)
    out["GD"]  = out["GF"] - out["GA"]
    out["PTS"] = (s["home_pts"] + s["away_pts"]).astype(int)
    out[pts_col.upper()] = (s["home_model_pts"] + s["away_model_pts"]).round(1)

    return out


def load_odds_standings():
    """Build standings from odds matches → vPTS."""
    path = PROCESSED_DIR / "matches_odds.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    standings = _build_from_matches(df, "vpts")
    standings["vPTS_diff"] = (standings["PTS"] - standings["VPTS"]).round(1)
    standings.rename(columns={"VPTS": "vPTS"}, inplace=True)
    standings["vPPG"] = (standings["vPTS"] / standings["GP"]).round(2)
    print(f"✅ Loaded odds standings ({len(standings)} teams)")
    return standings


def load_xg_standings():
    """Build standings from xG matches → xPTS."""
    path = PROCESSED_DIR / "matches_xg.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    # Also aggregate xGF / xGA
    home_xg = df.groupby("home_team").agg(
        home_xgf=("home_xg", "sum"),
        home_xga=("away_xg", "sum"),
    ).rename_axis("team")
    away_xg = df.groupby("away_team").agg(
        away_xgf=("away_xg", "sum"),
        away_xga=("home_xg", "sum"),
    ).rename_axis("team")

    standings = _build_from_matches(df, "xpts")
    standings.rename(columns={"XPTS": "xPTS"}, inplace=True)
    standings["xPTS_diff"] = (standings["PTS"] - standings["xPTS"]).round(1)
    standings["xPPG"] = (standings["xPTS"] / standings["GP"]).round(2)

    # Add xG totals
    xg = home_xg.join(away_xg, how="outer").fillna(0)
    standings["xGF"] = (xg["home_xgf"] + xg["away_xgf"]).round(1)
    standings["xGA"] = (xg["home_xga"] + xg["away_xga"]).round(1)
    standings["xGD"] = (standings["xGF"] - standings["xGA"]).round(1)

    print(f"✅ Loaded xG standings ({len(standings)} teams)")
    return standings


def merge_standings(odds_st, xg_st):
    """Merge odds and xG standings into one table."""
    if odds_st is not None and xg_st is not None:
        # Use odds standings as base (has PTS, W, D, L, etc.)
        base = odds_st[["GP", "W", "D", "L", "GF", "GA", "GD",
                         "PTS", "vPTS", "vPTS_diff", "vPPG"]].copy()
        # Join xG columns
        xg_cols = xg_st[["xGF", "xGA", "xGD", "xPTS", "xPTS_diff", "xPPG"]].copy()
        merged = base.join(xg_cols, how="left")
        merged["PPG"] = (merged["PTS"] / merged["GP"]).round(2)
    elif odds_st is not None:
        merged = odds_st.copy()
        merged["PPG"] = (merged["PTS"] / merged["GP"]).round(2)
    elif xg_st is not None:
        merged = xg_st.copy()
        merged["PPG"] = (merged["PTS"] / merged["GP"]).round(2)
    else:
        raise FileNotFoundError(
            "No processed data found. Run process_odds.py and/or process_xg.py first."
        )

    # Sort
    merged = merged.sort_values(
        ["PTS", "GD", "GF"], ascending=[False, False, False]
    )

    return merged


def run():
    """Load both sources, merge, save."""
    odds_st = load_odds_standings()
    xg_st = load_xg_standings()

    standings = merge_standings(odds_st, xg_st)

    out_path = PROCESSED_DIR / "standings.csv"
    standings.to_csv(out_path)
    print(f"\n💾 Saved merged standings → {out_path}")
    print(f"\n{standings.to_string()}")

    return standings


if __name__ == "__main__":
    run()
