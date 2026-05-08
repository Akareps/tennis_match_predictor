"""
Data loader for Jeff Sackmann's tennis_atp / tennis_wta match data.
Downloads CSVs from GitHub on first use, caches locally.

Each year's file has match-level stats including service stats per player:
  w_svpt, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced  (and l_* for loser)

We compute per-match:
  spw = (1stWon + 2ndWon) / svpt    [serve-points won by that player]
  rpw = 1 - opponent's spw           [return-points won]
"""

import os
import urllib.request
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ATP_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
WTA_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"


def download_year(year: int, tour: str = "atp") -> Path:
    """Download one year's CSV if not already cached."""
    fname = DATA_DIR / f"{tour}_matches_{year}.csv"
    if fname.exists():
        return fname
    url = (ATP_URL if tour == "atp" else WTA_URL).format(year=year)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, fname)
    return fname


def load_years(years, tour: str = "atp") -> pd.DataFrame:
    """Load and concatenate multiple years of match data."""
    dfs = []
    for y in years:
        path = download_year(y, tour)
        df = pd.read_csv(path, low_memory=False)
        df["year"] = y
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one-row-per-match (winner/loser cols) into two rows per match,
    one per player perspective. Easier for skill estimation.

    Output cols:
      tourney_date, surface, tourney_level, best_of, year,
      player_id, player_name, opp_id, opp_name,
      won (bool),
      svpt, sv_won, rpt, rpt_won
    """
    keep = ["tourney_date", "surface", "tourney_level", "best_of", "year",
            "winner_id", "winner_name", "loser_id", "loser_name",
            "w_svpt", "w_1stWon", "w_2ndWon", "w_SvGms",
            "l_svpt", "l_1stWon", "l_2ndWon", "l_SvGms"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.dropna(subset=["w_svpt", "l_svpt", "surface"])
    df = df[df["w_svpt"] > 0]
    df = df[df["l_svpt"] > 0]
    df["surface"] = df["surface"].str.strip()

    # Winner-perspective rows
    w = pd.DataFrame({
        "date": pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce"),
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "best_of": df["best_of"],
        "player_id": df["winner_id"],
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"],
        "opp_name": df["loser_name"],
        "won": True,
        "svpt": df["w_svpt"],
        "sv_won": df["w_1stWon"] + df["w_2ndWon"],
        "rpt": df["l_svpt"],
        "rpt_won": df["l_svpt"] - (df["l_1stWon"] + df["l_2ndWon"]),
    })
    # Loser-perspective rows
    l = pd.DataFrame({
        "date": pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce"),
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "best_of": df["best_of"],
        "player_id": df["loser_id"],
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"],
        "opp_name": df["winner_name"],
        "won": False,
        "svpt": df["l_svpt"],
        "sv_won": df["l_1stWon"] + df["l_2ndWon"],
        "rpt": df["w_svpt"],
        "rpt_won": df["w_svpt"] - (df["w_1stWon"] + df["w_2ndWon"]),
    })
    long = pd.concat([w, l], ignore_index=True)
    long = long.dropna(subset=["date"])
    long["spw"] = long["sv_won"] / long["svpt"]   # serve-points won %
    long["rpw"] = long["rpt_won"] / long["rpt"]   # return-points won %
    return long.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    df = load_years([2022, 2023, 2024], tour="atp")
    print(f"Loaded {len(df)} ATP matches")
    long = to_long_format(df)
    print(f"Long format: {len(long)} player-match rows")
    print(long.head())
    print("\nSurface breakdown:")
    print(long["surface"].value_counts())
    print("\nMean spw / rpw across all matches (sanity):")
    print(f"  spw: {long['spw'].mean():.4f}")
    print(f"  rpw: {long['rpw'].mean():.4f}  (should sum to ~1.0 with spw)")
