"""
Loader for tennis-data.co.uk historical odds.

Files at http://www.tennis-data.co.uk/alldata.php
  - One Excel per year: 2024.xlsx, 2023.xlsx, etc.
  - ATP files: just YYYY.xlsx
  - Player names: "Djokovic N." format (last name + first initial)
  - Pinnacle odds: PSW (winner closing odds), PSL (loser closing odds)

Place downloaded files in ./odds_data/ before running.
Sackmann names use full names, so we need a fuzzy match.
"""

from pathlib import Path
import urllib.request
import pandas as pd
import re

ODDS_DIR = Path(__file__).parent / "odds_data"
ODDS_DIR.mkdir(exist_ok=True)

# tennis-data.co.uk URL pattern: http://www.tennis-data.co.uk/{year}/{year}.xlsx
# (older files were .xls but post-2013 are .xlsx)
ODDS_URL_ATP = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"
ODDS_URL_WTA = "http://www.tennis-data.co.uk/{year}w/{year}.xlsx"


def download_odds_year(year: int, tour: str = "atp") -> Path:
    """Download tennis-data.co.uk Excel for one year if not already cached."""
    fname = ODDS_DIR / f"{tour}_{year}.xlsx"
    if fname.exists():
        return fname
    url = (ODDS_URL_ATP if tour == "atp" else ODDS_URL_WTA).format(year=year)
    print(f"  Downloading {url} ...")
    # tennis-data.co.uk doesn't require auth; it does sometimes block default
    # urllib User-Agent though, so set a browser-ish one.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(fname, "wb") as f:
                f.write(resp.read())
    except Exception as e:
        if fname.exists():
            fname.unlink()  # remove partial file
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    return fname


def load_odds_year(year: int, tour: str = "atp") -> pd.DataFrame:
    """Load one year's tennis-data.co.uk Excel file (auto-downloads on first use)."""
    fname = download_odds_year(year, tour)
    df = pd.read_excel(fname)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_odds_years(years, tour: str = "atp") -> pd.DataFrame:
    dfs = []
    for y in years:
        try:
            d = load_odds_year(y, tour)
            d["year"] = y
            dfs.append(d)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    if not dfs:
        raise RuntimeError("No odds files loaded.")
    combined = pd.concat(dfs, ignore_index=True)
    return clean_odds(combined)


def clean_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the columns we care about, drop nulls."""
    keep = ["Date", "Tournament", "Surface", "Round", "Best of",
            "Winner", "Loser", "WRank", "LRank",
            "PSW", "PSL", "B365W", "B365L", "AvgW", "AvgL", "MaxW", "MaxL"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Winner", "Loser"])
    # Pinnacle is our preferred odds source; fall back to average if missing
    if "PSW" in out.columns:
        out["closing_odds_w"] = out["PSW"].fillna(out.get("AvgW"))
        out["closing_odds_l"] = out["PSL"].fillna(out.get("AvgL"))
    elif "AvgW" in out.columns:
        out["closing_odds_w"] = out["AvgW"]
        out["closing_odds_l"] = out["AvgL"]
    out = out.dropna(subset=["closing_odds_w", "closing_odds_l"])
    return out


# ---------- Player name matching ----------
# Sackmann: "Novak Djokovic"
# Tennis-data: "Djokovic N."
# We need to convert one to the other.

def sackmann_to_td_name(full_name: str) -> str:
    """
    'Novak Djokovic' -> 'Djokovic N.'
    Handles compound surnames imperfectly; flagged matches need manual review.
    """
    if not isinstance(full_name, str):
        return ""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    first = parts[0]
    last = " ".join(parts[1:])
    return f"{last} {first[0]}."


def td_to_sackmann_pattern(td_name: str) -> str:
    """
    'Djokovic N.' -> regex to match 'Novak Djokovic' (case-insensitive)
    Returns a regex pattern.
    """
    if not isinstance(td_name, str):
        return ""
    # Strip the period and split
    parts = td_name.replace(".", "").strip().split()
    if len(parts) < 2:
        return re.escape(td_name)
    last_initial = parts[-1]
    surname_parts = parts[:-1]
    surname = " ".join(surname_parts)
    # Pattern: starts with first name beginning with the initial, then surname
    return rf"^{re.escape(last_initial)}\w+\s+{re.escape(surname)}$"


def normalize_for_matching(name: str) -> str:
    """Lower, strip diacritics, collapse spaces, handle hyphens."""
    if not isinstance(name, str):
        return ""
    import unicodedata
    n = unicodedata.normalize("NFKD", name)
    n = "".join(ch for ch in n if not unicodedata.combining(ch))
    n = n.lower().strip()
    n = re.sub(r"\s+", " ", n)
    n = n.replace("-", " ")
    return n


def build_sackmann_to_td_index(sackmann_long: pd.DataFrame) -> dict:
    """
    Build an index: sackmann_player_id -> set of normalized TD-style names.
    For each unique sackmann player_id, generate the expected TD name.
    """
    idx = {}
    seen = sackmann_long[["player_id", "player_name"]].drop_duplicates()
    for _, r in seen.iterrows():
        td = sackmann_to_td_name(r["player_name"])
        idx[r["player_id"]] = normalize_for_matching(td)
    return idx


def merge_predictions_with_odds(predictions: pd.DataFrame,
                                odds: pd.DataFrame,
                                player_id_to_name: dict) -> pd.DataFrame:
    """
    Merge predictions (with player_a_id, player_b_id, date) with odds 
    (with Winner, Loser names).

    Strategy:
      1. For each prediction, look up the player_a name and player_b name.
      2. Convert to TD-style and normalize.
      3. Find an odds row on the same date (+/- 3 days for date drift)
         with matching player names (in either order).
      4. Determine which is the winner; assign odds accordingly.

    Returns predictions DataFrame with new columns:
      odds_a, odds_b, implied_prob_a, implied_prob_b, fair_prob_a, fair_prob_b
    """
    odds = odds.copy()
    odds["winner_norm"] = odds["Winner"].apply(normalize_for_matching)
    odds["loser_norm"] = odds["Loser"].apply(normalize_for_matching)

    # Index odds by date for fast lookup
    odds["date"] = pd.to_datetime(odds["Date"]).dt.normalize()

    out = []
    for _, r in predictions.iterrows():
        a_name = player_id_to_name.get(r["player_a_id"])
        b_name = player_id_to_name.get(r["player_b_id"])
        if not a_name or not b_name:
            continue
        a_td = normalize_for_matching(sackmann_to_td_name(a_name))
        b_td = normalize_for_matching(sackmann_to_td_name(b_name))

        # Find matching odds row
        match_date = pd.to_datetime(r["date"]).normalize()
        # Date can be off by a few days due to scheduling differences
        candidates = odds[
            (odds["date"] >= match_date - pd.Timedelta(days=3))
            & (odds["date"] <= match_date + pd.Timedelta(days=3))
            & (
                ((odds["winner_norm"] == a_td) & (odds["loser_norm"] == b_td))
                | ((odds["winner_norm"] == b_td) & (odds["loser_norm"] == a_td))
            )
        ]
        if len(candidates) == 0:
            continue
        # Take the closest by date
        candidates = candidates.copy()
        candidates["date_diff"] = (candidates["date"] - match_date).abs()
        match = candidates.sort_values("date_diff").iloc[0]

        # Determine which is A
        if match["winner_norm"] == a_td:
            odds_a = match["closing_odds_w"]
            odds_b = match["closing_odds_l"]
            actual_winner = "a"
        else:
            odds_a = match["closing_odds_l"]
            odds_b = match["closing_odds_w"]
            actual_winner = "b"

        # Sanity check: actual outcome should match
        if actual_winner == "a" and r["actual_a_won"] != 1:
            continue  # Mismatch — skip
        if actual_winner == "b" and r["actual_a_won"] != 0:
            continue

        # Implied probs (with vig)
        impl_a = 1.0 / odds_a
        impl_b = 1.0 / odds_b
        # Fair probs (vig removed by multiplicative method)
        total = impl_a + impl_b
        fair_a = impl_a / total
        fair_b = impl_b / total

        rec = dict(r)
        rec.update({
            "odds_a": float(odds_a),
            "odds_b": float(odds_b),
            "implied_prob_a": float(impl_a),
            "implied_prob_b": float(impl_b),
            "fair_prob_a": float(fair_a),
            "fair_prob_b": float(fair_b),
            "vig": float(total - 1),  # bookmaker margin
        })
        out.append(rec)

    return pd.DataFrame(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        years = [int(y) for y in sys.argv[1:]]
    else:
        years = [2024]
    print(f"Loading odds for years: {years}")
    print(f"  (auto-downloading missing files to {ODDS_DIR}/)")
    odds = load_odds_years(years)
    print(f"\n  Loaded {len(odds)} matches with odds.")
    print(f"  Columns: {list(odds.columns)}")
    print(f"\n  Sample:")
    print(odds.head(3))
    print(f"\n  Pinnacle odds present in: {odds['closing_odds_w'].notna().sum()} / {len(odds)} matches")
    print(f"  Mean vig: {(1/odds['closing_odds_w'] + 1/odds['closing_odds_l'] - 1).mean():.4f}")
