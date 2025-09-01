# Minimal roster extractor so the app can show players by team
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd

RAW = Path("data/raw/pbp/2023-24")
OUT = Path("data/derived/lineups_events.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

PLAYER_COLS = [
    ("PLAYER1_NAME", "PLAYER1_TEAM_ABBREVIATION"),
    ("PLAYER2_NAME", "PLAYER2_TEAM_ABBREVIATION"),
    ("PLAYER3_NAME", "PLAYER3_TEAM_ABBREVIATION"),
]


def _norm_gid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)


def main():
    files = sorted(glob.glob(str(RAW / "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet under {RAW}")

    rows = []
    for fp in files:
        df = pd.read_parquet(fp)
        if "GAME_ID" not in df.columns:
            continue
        gid = str(df["GAME_ID"].iloc[0])
        gid = "".join([ch for ch in gid if ch.isdigit()]).zfill(10)

        for name_col, team_col in PLAYER_COLS:
            if name_col in df.columns and team_col in df.columns:
                tmp = df[[name_col, team_col]].dropna()
                tmp = tmp.rename(columns={name_col: "PLAYER", team_col: "TEAM"})
                tmp["PLAYER"] = tmp["PLAYER"].astype(str).str.strip()
                tmp["TEAM"] = tmp["TEAM"].astype(str).str.strip()
                tmp = tmp[(tmp["PLAYER"] != "") & (tmp["TEAM"] != "")]
                if not tmp.empty:
                    tmp["GAME_ID"] = gid
                    rows.append(tmp[["GAME_ID", "TEAM", "PLAYER"]])

    if not rows:
        raise SystemExit("Couldn’t find PLAYER*/TEAM* columns in PBP.")

    out = (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates()
        .sort_values(["GAME_ID", "TEAM", "PLAYER"])
        .reset_index(drop=True)
    )

    out.to_parquet(OUT, index=False)
    print(f"✅ Wrote {OUT} ({len(out)} rows across {out['GAME_ID'].nunique()} games)")
    print("Sample:", out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
