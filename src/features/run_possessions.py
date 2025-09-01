from __future__ import annotations
from pathlib import Path
import pandas as pd

from .possession_builder import (
    build_possessions,
    normalize_event_type,  # kept for parity; used in _attach_possession_seq
)

RAW_DIR = Path("data/raw/pbp/2023-24")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_PATH = OUT_DIR / "event_possess_map.parquet"  # authoritative event→possession map


def _attach_possession_seq(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Assign POSS_SEQ to each event using the SAME boundary logic as the possession builder.
    If your build_possessions() already annotates pbp with POSS_SEQ, you can skip this and
    just return pbp.

    IMPORTANT: Keep these START/END sets in sync with the builder!
    """
    df = pbp.copy()

    # sort & normalize
    df["GAME_ID"] = (
        df["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
    )
    df["EVENTNUM"] = pd.to_numeric(df["EVENTNUM"], errors="coerce")
    df = df.sort_values(["GAME_ID", "PERIOD", "EVENTNUM"]).reset_index(drop=True)

    # unify text for a robust EVTYPE
    for c in ["HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]:
        if c not in df.columns:
            df[c] = ""
    df["TEXT"] = (
        df[["HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )
    df["EVTYPE"] = df["TEXT"].map(normalize_event_type)

    # MUST mirror possession_builder's rules:
    START_EVENTS = {"jump ball", "start of period", "rebound_off"}
    END_EVENTS = {"made shot", "turnover", "end of period", "rebound_def", "ft_end"}

    prev_end = df["EVTYPE"].shift(1).isin(END_EVENTS).fillna(False)
    new_pos = (df["EVTYPE"].isin(START_EVENTS) | prev_end).astype(int)
    df["POSS_SEQ"] = new_pos.groupby(df["GAME_ID"]).cumsum()

    return df


def main() -> None:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet in {RAW_DIR}")

    all_poss = []
    # Load the existing map if present (so we can append)
    if MAP_PATH.exists():
        map_accum = pd.read_parquet(MAP_PATH)
    else:
        map_accum = pd.DataFrame(columns=["GAME_ID", "PERIOD", "EVENTNUM", "POSS_SEQ"])

    for fp in files:
        pbp = pd.read_parquet(fp).copy()
        # normalize GAME_ID to a 10-char string
        pbp["GAME_ID"] = (
            pbp["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
        )
        pbp["EVENTNUM"] = pd.to_numeric(pbp["EVENTNUM"], errors="coerce")

        # Build possessions table (your existing function)
        poss = build_possessions(pbp)
        all_poss.append(poss)

        # Ensure PBP has POSS_SEQ per event. If build_possessions() already wrote it into pbp,
        # you can skip the call below. Otherwise, attach it here with the SAME boundary logic:
        if "POSS_SEQ" not in pbp.columns or pbp["POSS_SEQ"].isna().any():
            pbp = _attach_possession_seq(pbp)

        # Create event→possession mapping for THIS game and append to global map
        event_map = pbp.loc[:, ["GAME_ID", "PERIOD", "EVENTNUM", "POSS_SEQ"]].dropna()
        event_map["EVENTNUM"] = event_map["EVENTNUM"].astype(int)

        map_accum = pd.concat([map_accum, event_map], ignore_index=True)
        map_accum = map_accum.drop_duplicates(
            ["GAME_ID", "PERIOD", "EVENTNUM"], keep="last"
        )

        gid = poss["GAME_ID"].iloc[0]
        print(f"✅ Event map rows for game {gid}: {len(event_map)}")

        # Save per-game possessions parquet (optional but handy)
        (OUT_DIR / f"{gid}_possessions.parquet").parent.mkdir(
            parents=True, exist_ok=True
        )
        poss.to_parquet(OUT_DIR / f"{gid}_possessions.parquet", index=False)

    # Write the authoritative event→possession map
    map_accum.to_parquet(MAP_PATH, index=False)
    print(f"✅ Updated {MAP_PATH} with {len(map_accum)} total rows.")

    # Concatenate all games and write the aggregate CSV (used by the app)
    poss_all = pd.concat(all_poss, ignore_index=True)
    poss_all.to_csv(OUT_DIR / "possessions.csv", index=False)

    print(f"Games processed: {len(files)}")
    print("GAME_IDs:", sorted(poss_all["GAME_ID"].unique().tolist()))
    print("Total possessions:", len(poss_all))
    if "PERIOD" in poss_all.columns:
        print(poss_all["PERIOD"].value_counts().sort_index())


if __name__ == "__main__":
    main()
