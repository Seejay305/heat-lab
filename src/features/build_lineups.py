# src/features/build_lineups.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import glob

RAW_DIR = Path("data/raw/pbp/2023-24")
OUT = Path("data/derived/lineups_events.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)


def _norm_game_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)


def load_all_pbp() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files under {RAW_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["GAME_ID"] = _norm_game_id(df["GAME_ID"])
    return df


def build_lineups(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple lineup tracker: walk events, apply subs, maintain 5 per side.
    Assumes PBP has TEAM_ID, PLAYER1_ID/PLAYER2_ID for subs.
    """
    out_rows = []
    for gid, gdf in pbp.groupby("GAME_ID"):
        gdf = gdf.sort_values(["PERIOD", "EVENTNUM"]).reset_index(drop=True)

        home_abbr = gdf["HOMEDESCRIPTION"].dropna().head(1).tolist()
        away_abbr = gdf["VISITORDESCRIPTION"].dropna().head(1).tolist()
        home_abbr = home_abbr[0] if home_abbr else "HOME"
        away_abbr = away_abbr[0] if away_abbr else "AWAY"

        on_home, on_away = set(), set()

        for _, r in gdf.iterrows():
            evnum = int(r["EVENTNUM"])
            period = int(r["PERIOD"])

            # handle substitutions
            if str(r.get("EVENTMSGTYPE")) == "8":  # substitution
                out_id = r.get("PLAYER1_ID")
                in_id = r.get("PLAYER2_ID")
                if r.get("PLAYER1_TEAM_ID") == r.get("HOME_TEAM_ID"):
                    if out_id:
                        on_home.discard(out_id)
                    if in_id:
                        on_home.add(in_id)
                else:
                    if out_id:
                        on_away.discard(out_id)
                    if in_id:
                        on_away.add(in_id)

            # record snapshot
            out_rows.append(
                {
                    "GAME_ID": gid,
                    "PERIOD": period,
                    "EVENTNUM": evnum,
                    "ON_COURT_HOME": ";".join(str(x) for x in sorted(on_home)),
                    "ON_COURT_AWAY": ";".join(str(x) for x in sorted(on_away)),
                    "HOME_TEAM_ABBR": home_abbr,
                    "AWAY_TEAM_ABBR": away_abbr,
                    "OFF_TEAM_ABBR": r.get("TEAM_ABBREVIATION", None),
                    "DEF_TEAM_ABBR": (
                        away_abbr
                        if r.get("TEAM_ABBREVIATION") == home_abbr
                        else home_abbr
                    ),
                }
            )

    return pd.DataFrame(out_rows)


def main():
    pbp = load_all_pbp()
    lu = build_lineups(pbp)
    lu.to_parquet(OUT, index=False)
    print(f"✅ Wrote {OUT} with {len(lu)} rows, {lu['GAME_ID'].nunique()} games")


if __name__ == "__main__":
    main()
