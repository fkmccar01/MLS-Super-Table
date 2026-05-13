"""
MLS Standings Dashboard — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MLS Standings",
    page_icon="⚽",
    layout="wide",
)

PROCESSED_DIR = Path("data/processed")


# ─── LOAD DATA ────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache for 5 minutes
def load_standings():
    """Load standings with fallback priority: merged → odds → xg."""
    paths = [
        PROCESSED_DIR / "standings.csv",
        PROCESSED_DIR / "standings_odds.csv",
        PROCESSED_DIR / "standings_xg.csv",
    ]
    for path in paths:
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            return df, path.name
    return None, None


# ─── STYLING ──────────────────────────────────────────────────────
def color_diff(val):
    """Green for positive, red for negative, gray for zero."""
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: #2ecc71; font-weight: bold"
    elif val < 0:
        return "color: #e74c3c; font-weight: bold"
    return "color: #95a5a6"


def style_table(df):
    """Apply conditional formatting to the standings table."""
    styler = df.style

    # Color the diff columns
    diff_cols = [c for c in df.columns if "diff" in c.lower()]
    if diff_cols:
        styler = styler.applymap(color_diff, subset=diff_cols)

    # Format decimal columns
    float_cols = df.select_dtypes(include="float").columns.tolist()
    format_dict = {col: "{:.1f}" for col in float_cols}
    for col in float_cols:
        if "PPG" in col or "PPG" in col.upper():
            format_dict[col] = "{:.2f}"
    styler = styler.format(format_dict, na_rep="—")

    return styler


# ─── MAIN APP ─────────────────────────────────────────────────────
def main():
    # Header
    st.title("⚽ MLS Standings Dashboard")
    st.markdown(f"**2026 Season**")

    # Load data
    standings, source_file = load_standings()

    if standings is None:
        st.error(
            "No standings data found. Run the processing scripts first:\n\n"
            "```bash\n"
            "python scraper/process_odds.py\n"
            "python scraper/process_xg.py\n"
            "python scraper/build_table.py\n"
            "```"
        )
        return

    # ─── SIDEBAR ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Options")

        # Sort options — only show columns that exist
        sortable = ["PTS", "GD", "GF", "PPG"]
        if "vPTS" in standings.columns:
            sortable += ["vPTS", "vPTS_diff", "vPPG"]
        if "xPTS" in standings.columns:
            sortable += ["xPTS", "xPTS_diff", "xGD", "xPPG"]

        sort_by = st.selectbox("Sort by", sortable, index=0)
        ascending = st.toggle("Ascending", value=False)

        st.divider()

        # Data source info
        st.caption(f"📄 Source: `{source_file}`")
        if source_file and Path(PROCESSED_DIR / source_file).exists():
            mod_time = datetime.fromtimestamp(
                (PROCESSED_DIR / source_file).stat().st_mtime
            )
            st.caption(f"🕐 Updated: {mod_time.strftime('%b %d, %Y %I:%M %p')}")

    # ─── SORT & DISPLAY ──────────────────────────────────────────
    sorted_standings = standings.sort_values(sort_by, ascending=ascending)

    # Add rank column
    sorted_standings.insert(0, "#", range(1, len(sorted_standings) + 1))

    # Display styled table
    styled = style_table(sorted_standings)
    st.dataframe(
        styled,
        use_container_width=True,
        height=1100,
    )

    # ─── LEGEND ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        if "vPTS" in standings.columns:
            st.metric(
                "Biggest vPTS Overperformer",
                standings["vPTS_diff"].idxmax(),
                f"+{standings['vPTS_diff'].max():.1f}",
            )
    with col2:
        if "xPTS" in standings.columns:
            st.metric(
                "Biggest xPTS Overperformer",
                standings["xPTS_diff"].idxmax(),
                f"+{standings['xPTS_diff'].max():.1f}",
            )
    with col3:
        if "xPTS" in standings.columns:
            st.metric(
                "Biggest xPTS Underperformer",
                standings["xPTS_diff"].idxmin(),
                f"{standings['xPTS_diff'].min():.1f}",
                delta_color="inverse",
            )
        elif "vPTS" in standings.columns:
            st.metric(
                "Biggest vPTS Underperformer",
                standings["vPTS_diff"].idxmin(),
                f"{standings['vPTS_diff'].min():.1f}",
                delta_color="inverse",
            )

    # ─── HOW IT WORKS ─────────────────────────────────────────────
    with st.expander("ℹ️ How it works — vPTS vs xPTS"):
        st.markdown("""
### vPTS (Vegas Points)
- **Source:** Betting odds from [football-data.co.uk](https://www.football-data.co.uk/usa.php)
- **Method:** Closing odds → remove bookmaker margin → implied win/draw/loss probabilities → expected points
- **What it measures:** What the betting market *expected* to happen before each match
- **Interpretation:** PTS > vPTS = overperforming market expectations (luck or intangibles)

### xPTS (Expected Points)
- **Source:** Match-level xG data from [FBRef](https://fbref.com/en/comps/22/schedule/Major-League-Soccer-Scores-and-Fixtures)
- **Method:** xG/xGA per match → Poisson model → win/draw/loss probabilities → expected points
- **What it measures:** What *should have happened* based on actual chances created on the pitch
- **Interpretation:** PTS > xPTS = finishing/converting above expected rate

### Key Differences
| | vPTS | xPTS |
|---|---|---|
| **Timing** | Pre-match | Post-match |
| **Based on** | Market wisdom | On-pitch performance |
| **Tells you** | Were they favored? | Did they create/concede chances? |

A team with high PTS, low xPTS, and high vPTS was *expected* to be good but is winning ugly.
A team with low PTS but high xPTS is unlucky — regression candidate.
        """)

    # ─── FOOTER ───────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Data: [football-data.co.uk](https://www.football-data.co.uk/usa.php) (odds) · "
        "[FBRef](https://fbref.com/en/comps/22/schedule/Major-League-Soccer-Scores-and-Fixtures) (xG) · "
        "Built with [Streamlit](https://streamlit.io)"
    )


if __name__ == "__main__":
    main()
