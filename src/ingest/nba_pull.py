"""
NBA ingest: pulls Miami HEAT (or any team) game list + Play-by-Play and
caches raw parquet files under data/raw/pbp/<SEASON>/<GAME_ID>.parquet

Usage examples:
  python -m src.ingest.nba_pull --team MIA --seasons 2022-23 2023-24 --limit 5
  python -m src.ingest.nba_pull --team MIA --seasons 2023-24 --overwrite

Notes:
- Respects a simple rate limit (NBA_API_RATE_LIMIT env). Default 0.6s between calls.
- Safe to rerun; skips files unless --overwrite.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV2
from nba_api.stats.static import teams as static_teams

RATE_LIMIT = float(os.getenv("NBA_API_RATE_LIMIT", "0.6"))
RAW_DIR = Path("data/raw")
PBP_DIR = RAW_DIR / "pbp"


def _sleep():
    time.sleep(RATE_LIMIT)


def get_team_id(team_abbr: str) -> int:
    abbr = team_abbr.strip().upper()
    for t in static_teams.get_teams():
        if t["abbreviation"].upper() == abbr:
            return int(t["id"])
    raise ValueError(f"Unknown team abbreviation: {team_abbr}")


def find_team_games(team_id: int, seasons: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for season in seasons:
        lgf = LeagueGameFinder(team_id_nullable=team_id, season_nullable=season)
        df = lgf.get_data_frames()[0].copy()
        df["SEASON"] = season
        frames.append(df)
        _sleep()
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Keep only regular season + playoffs; sort newest first
    if not out.empty:
        out = out.sort_values(["GAME_DATE"], ascending=False)
    return out


def save_pbp_for_game(game_id: str, season: str, overwrite: bool = False) -> Path:
    season_dir = PBP_DIR / season
    season_dir.mkdir(parents=True, exist_ok=True)
    out_path = season_dir / f"{game_id}.parquet"
    if out_path.exists() and not overwrite:
        return out_path

    pbp = PlayByPlayV2(game_id=game_id)
    df = pbp.get_data_frames()[0]
    # Basic hygiene
    df["GAME_ID"] = game_id
    df["SEASON"] = season

    df.to_parquet(out_path, index=False)
    _sleep()
    return out_path


def pull_team_pbp(team_abbr: str, seasons: Iterable[str], limit: int | None, overwrite: bool) -> None:
    team_id = get_team_id(team_abbr)
    games = find_team_games(team_id, seasons)
    if games.empty:
        print("No games found.")
        return

    # Use GAME_ID column (string like '0022300001')
    game_ids = games["GAME_ID"].astype(str).tolist()
    if limit:
        game_ids = game_ids[: int(limit)]

    print(f"📦 Will fetch PBP for {len(game_ids)} games → {PBP_DIR}")
    for i, gid in enumerate(game_ids, 1):
        season = str(games.loc[games["GAME_ID"].astype(str) == gid, "SEASON"].iloc[0])
        path = save_pbp_for_game(gid, season, overwrite=overwrite)
        print(f"[{i}/{len(game_ids)}] {gid} → {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull team games and Play-by-Play")
    p.add_argument("--team", default="MIA", help="Team abbreviation (e.g., MIA)")
    p.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="Seasons like 2022-23 2023-24",
    )
    p.add_argument("--limit", type=int, default=5, help="Limit number of games for quick runs")
    p.add_argument("--overwrite", action="store_true", help="Overwrite cached parquet files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pull_team_pbp(args.team, args.seasons, args.limit, args.overwrite)


if __name__ == "__main__":
    main()