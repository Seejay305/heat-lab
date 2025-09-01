# src/features/build_event_features.py
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from src.features.possession_builder import normalize_event_type  # reuse classifier
from collections import defaultdict

RAW_DIR = Path("data/raw/pbp/2023-24")
LINEUP_PATH = Path("data/derived/lineups_events.parquet")
ENRICHED_PATH = Path("data/processed/possessions_enriched.csv")
OUT = Path("data/derived/event_features.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

def coalesce_exact_duplicate_names(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    name_to_idx = defaultdict(list)
    for i, c in enumerate(cols):
        name_to_idx[c].append(i)

    # build replacement columns with non-null-first logic
    replacements = {}
    drop_indices = []
    for name, idxs in name_to_idx.items():
        if len(idxs) > 1:
            base = df.iloc[:, idxs[0]]
            for j in idxs[1:]:
                base = base.combine_first(df.iloc[:, j])
            # stash the combined series under a temporary unique name
            tmp_name = name + "__TMP__"
            replacements[tmp_name] = base
            # mark originals for drop
            drop_indices.extend(idxs)

    if not replacements:
        return df  # nothing to do

    # drop all originals for duplicated names
    df = df.drop(columns=[cols[i] for i in drop_indices])

    # add combined columns, then rename back to the original names
    for tmp_name, s in replacements.items():
        df[tmp_name] = s
    df = df.rename(columns={k: k.replace("__TMP__", "") for k in replacements.keys()})

    # final sanity
    assert df.columns.is_unique, "Columns still not unique after coalescing."
    return df

def _clock_to_seconds(t: str) -> int:
    try:
        m, s = str(t).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return -1

def _unify_text(df: pd.DataFrame) -> pd.Series:
    cols = ["HOMEDESCRIPTION","VISITORDESCRIPTION","NEUTRALDESCRIPTION"]
    return df[cols].fillna("").agg(" ".join, axis=1).str.strip()

def _attach_possession_seq(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    df["EVENTNUM"] = pd.to_numeric(df["EVENTNUM"], errors="coerce")
    df = df.sort_values(["GAME_ID","PERIOD","EVENTNUM"]).reset_index(drop=True)
    df["TEXT"] = _unify_text(df)
    df["EVTYPE"] = df["TEXT"].map(normalize_event_type)

    START_EVENTS = {"jump ball", "start of period"}  # offensive rebounds DO NOT start new poss
    END_EVENTS   = {"made shot", "turnover", "end of period", "rebound_def"}

    prev_end = df["EVTYPE"].shift(1).isin(END_EVENTS).fillna(False)
    new_pos = (df["EVTYPE"].isin(START_EVENTS) | prev_end).astype(int)
    df["POSS_SEQ"] = new_pos.groupby(df["GAME_ID"]).cumsum()
    return df


def build_features():
    # ---- load pbp
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("No raw PBP parquet in data/raw/pbp/2023-24")
    pbp = pd.read_parquet(files[0]).copy()
    pbp = pbp.sort_values(["GAME_ID","PERIOD","EVENTNUM"]).reset_index(drop=True)

    # ---- attach POSS_SEQ to pbp events first
    pbp = _attach_possession_seq(pbp)

    # ---- clocks / bins
    pbp["CLOCK_SEC"] = pbp["PCTIMESTRING"].map(_clock_to_seconds)
    pbp["CLOCK_BIN"] = (pbp["CLOCK_SEC"] // 180)  # 3-min bins

    # ---- load enriched possessions
    if not ENRICHED_PATH.exists():
        raise SystemExit("Missing possessions_enriched.csv. Run enrich first.")
    poss = pd.read_csv(ENRICHED_PATH)
    pbp["GAME_ID"] = pbp["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
    poss["GAME_ID"] = poss["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)

    # ---- single merge with all needed columns
    keep = ["GAME_ID", "POSS_SEQ", "MARGIN_PRE", "MARGIN_POST", "POINTS", "RESULT_CLASS", "PERIOD"]
    keep = [c for c in keep if c in poss.columns]
    pbp = pbp.merge(poss[keep], on=["GAME_ID", "POSS_SEQ"], how="left", validate="m:1")
    pbp = pbp.rename(columns={"POINTS": "TARGET_POINTS_AHEAD"})

    # normalize GAME_IDs to 10-digit strings on BOTH sides
    pbp["GAME_ID"] = pbp["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
    poss["GAME_ID"] = poss["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)

    # ---- cast keys to str and assert keys
    for df in (pbp, poss):
        df["GAME_ID"] = df["GAME_ID"].astype(str)
    assert "POSS_SEQ" in pbp.columns, "POSS_SEQ missing on PBP"
    assert "POSS_SEQ" in poss.columns, "POSS_SEQ missing on possessions_enriched.csv"

    # ---- merge margin + target points ON two keys
    pbp = pbp.merge(
        poss[["GAME_ID","POSS_SEQ","MARGIN_PRE","MARGIN_POST","POINTS"]],
        on=["GAME_ID","POSS_SEQ"], how="left", validate="m:1"
    ).rename(columns={"POINTS": "TARGET_POINTS_AHEAD"})

    # ---- optional lineups join (robust to missing PERIOD/POSS_SEQ) ----
    if LINEUP_PATH.exists():
        lineup = pd.read_parquet(LINEUP_PATH).copy()
        # normalize types
        lineup["GAME_ID"] = (
            lineup["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
        )
        lineup["EVENTNUM"] = pd.to_numeric(lineup.get("EVENTNUM"), errors="coerce")

        # pick merge keys that exist on BOTH sides, in this priority order
        candidates = [
            ["GAME_ID", "POSS_SEQ", "EVENTNUM", "PERIOD"],
            ["GAME_ID", "POSS_SEQ", "EVENTNUM"],
            ["GAME_ID", "EVENTNUM", "PERIOD"],
            ["GAME_ID", "EVENTNUM"],
        ]
        for keys in candidates:
            if all(k in pbp.columns for k in keys) and all(k in lineup.columns for k in keys):
                merge_keys = keys
                break
        else:
            merge_keys = None

        if merge_keys:
            # minimal columns to bring over
            bring = ["ON_COURT_MIA", "ON_COURT_BOS"]
            cols = [c for c in (merge_keys + bring) if c in lineup.columns]
            # de-dupe lineups on the chosen key set
            lineup_small = lineup[cols].drop_duplicates(subset=merge_keys)
            pbp = pbp.merge(lineup_small, on=merge_keys, how="left", validate="m:1")
        else:
            print("⚠️  Skipping lineup merge: no compatible key set found.")

    # ---- n-grams within a possession
    # EVTYPE already set in _attach_possession_seq
    pbp["BIGRAM"] = pbp.groupby(["GAME_ID","POSS_SEQ"])["EVTYPE"].shift(1).fillna("START") + " → " + pbp["EVTYPE"]
    pbp["TRIGRAM"] = (
        pbp.groupby(["GAME_ID","POSS_SEQ"])["EVTYPE"].shift(2).fillna("START") + " → " +
        pbp.groupby(["GAME_ID","POSS_SEQ"])["EVTYPE"].shift(1).fillna("MID")   + " → " +
        pbp["EVTYPE"]
    )

    # --- 1) Normalize GAME_ID and join possessions for margin (if not already) ---
    pbp["GAME_ID"] = pbp["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)

    # If you haven’t already merged margins, pull from enriched possessions
    try:
        poss = pd.read_csv("data/processed/possessions_enriched.csv")
        poss["GAME_ID"] = poss["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
        keep = ["GAME_ID", "POSS_SEQ", "MARGIN_PRE", "MARGIN_POST", "POINTS", "RESULT_CLASS", "PERIOD"]
        keep = [c for c in keep if c in poss.columns]
        pbp = pbp.merge(poss[keep], on=["GAME_ID", "POSS_SEQ"], how="left", validate="m:1")
        pbp = pbp.rename(columns={"POINTS": "TARGET_POINTS_AHEAD"})

    except Exception:
        pass

    # --- 2) Infer offense side for each event (who generated the text) ---
    def who_offense(row):
        # Use whichever description is present; fallback to neutral as "UNK"
        h = str(row.get("HOMEDESCRIPTION", "") or "")
        v = str(row.get("VISITORDESCRIPTION", "") or "")
        if h and not v:
            return "HOME"
        if v and not h:
            return "AWAY"
        return "UNK"

    if "HOMEDESCRIPTION" in pbp.columns and "VISITORDESCRIPTION" in pbp.columns:
        pbp["OFF_SIDE"] = pbp.apply(who_offense, axis=1)
    else:
        pbp["OFF_SIDE"] = "UNK"

    # (If you know MIA is home in this game, you can map to team codes later.
    # For now, OFF_SIDE is a categorical feature HOME/AWAY/UNK.)

    # --- 3) Team fouls per period -> BONUS flags (coarse, good enough for EPV-lite) ---
    # Count defensive/common shooting/personal fouls as team fouls.
    def is_team_foul(text: str) -> bool:
        t = (text or "").lower()
        # count personal/shooting fouls; exclude 'offensive' / 'loose ball' if you want to be strict
        if "p.foul" in t or "s.foul" in t or "shooting foul" in t or "personal foul" in t:
            # rough exclusion for player-control (offensive) fouls:
            if "off" in t or "charge" in t or "player control" in t:
                return False
            return True
        return False

    for side, col in [("HOME", "HOMEDESCRIPTION"), ("AWAY", "VISITORDESCRIPTION")]:
        tag = f"{side}_TF"  # team fouls
        if col in pbp.columns:
            pbp[tag] = pbp[col].fillna("").apply(is_team_foul).astype(int)
        else:
            pbp[tag] = 0

    # cumulative team fouls per period
    pbp = pbp.sort_values(["GAME_ID", "PERIOD", "EVENTNUM"]).reset_index(drop=True)
    pbp["HOME_TF_CUM"] = pbp.groupby(["GAME_ID", "PERIOD"])["HOME_TF"].cumsum()
    pbp["AWAY_TF_CUM"] = pbp.groupby(["GAME_ID", "PERIOD"])["AWAY_TF"].cumsum()

    # NBA bonus is nuanced by period/time, but a coarse ≥5 threshold works well for EPV-lite
    pbp["HOME_BONUS"] = (pbp["HOME_TF_CUM"] >= 5).astype(int)
    pbp["AWAY_BONUS"] = (pbp["AWAY_TF_CUM"] >= 5).astype(int)

    # OFFENSE in bonus flag
    def off_bonus(row):
        if row["OFF_SIDE"] == "HOME":
            return row["HOME_BONUS"]
        if row["OFF_SIDE"] == "AWAY":
            return row["AWAY_BONUS"]
        return 0

    pbp["OFF_BONUS"] = pbp.apply(off_bonus, axis=1).astype(int)

    # --- 4) Score margin bins (behavior differs by state) ---
    if "MARGIN_PRE" in pbp.columns:
        bins = [-50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50]
        labels = list(range(len(bins) - 1))
        pbp["MARGIN_BIN"] = pd.cut(pbp["MARGIN_PRE"].fillna(0), bins=bins, labels=labels, include_lowest=True)
        pbp["MARGIN_BIN"] = pbp["MARGIN_BIN"].astype("Int64").fillna(-1).astype(int)
    else:
        pbp["MARGIN_BIN"] = -1

    # --- 5) Recent-context counts within the possession (last 3 events) ---
    pbp["EVENT_IDX"] = pbp.groupby(["GAME_ID", "POSS_SEQ"]).cumcount()

    def rolling_counts(g):
        g = g.sort_values("EVENT_IDX")
        is_make = (g["EVTYPE"] == "made shot").astype(int)
        is_miss = (g["EVTYPE"] == "missed shot").astype(int)
        is_to = (g["EVTYPE"] == "turnover").astype(int)
        g["RCNT3_MAKE"] = is_make.rolling(3, min_periods=1).sum().shift(1).fillna(0)
        g["RCNT3_MISS"] = is_miss.rolling(3, min_periods=1).sum().shift(1).fillna(0)
        g["RCNT3_TO"] = is_to.rolling(3, min_periods=1).sum().shift(1).fillna(0)
        return g

    pbp = pbp.groupby(["GAME_ID", "POSS_SEQ"], group_keys=False).apply(rolling_counts)

    # --- keep writing your existing columns PLUS the new ones ---
    # Example: if you build a final df 'out_df' to write, add these to it:
    extra_cols = [
        "OFF_SIDE", "HOME_BONUS", "AWAY_BONUS", "OFF_BONUS",
        "MARGIN_BIN", "EVENT_IDX", "RCNT3_MAKE", "RCNT3_MISS", "RCNT3_TO"
    ]
    for c in extra_cols:
        if c not in pbp.columns:
            pbp[c] = 0

    # ensure types
    for c in ["HOME_BONUS", "AWAY_BONUS", "OFF_BONUS", "MARGIN_BIN", "EVENT_IDX",
              "RCNT3_MAKE", "RCNT3_MISS", "RCNT3_TO"]:
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0).astype(int)

    # and then proceed to save:
    # pbp.to_parquet("data/derived/event_features.parquet", index=False)

    # ---- debug prints (tiny but useful)
    print("pbp rows:", len(pbp))
    print("cols:", sorted(pbp.columns.tolist())[:12], "…")
    print("GAME_ID dtype:", pbp["GAME_ID"].dtype)
    print("Nulls snapshot:",
          pbp[["GAME_ID","POSS_SEQ","MARGIN_PRE","MARGIN_POST","TARGET_POINTS_AHEAD"]].isna().mean().round(3).to_dict())

    # ---- write
    # ---- coalesce possible _x/_y after merges (safety) ----
    def _coalesce_xy(df, base):
        x, y = f"{base}_x", f"{base}_y"
        if x in df.columns and y in df.columns:
            df[base] = df[x].combine_first(df[y])
            df.drop(columns=[x, y], inplace=True, errors="ignore")

    for base in ["PERIOD", "MARGIN_PRE", "MARGIN_POST", "RESULT_CLASS"]:
        _coalesce_xy(pbp, base)

    # If we somehow duplicated TARGET_POINTS_AHEAD, keep the first and drop the rest
    if (list(pbp.columns).count("TARGET_POINTS_AHEAD") > 1):
        seen = 0
        keep_cols = []
        for c in pbp.columns:
            if c == "TARGET_POINTS_AHEAD":
                seen += 1
                if seen > 1:
                    continue
            keep_cols.append(c)
        pbp = pbp.loc[:, keep_cols]
        pbp = coalesce_exact_duplicate_names(pbp)

    # ---- select a thin, unique schema for event_features ----
    KEEP = [
        # keys / indexing
        "GAME_ID", "POSS_SEQ", "EVENTNUM", "PERIOD", "PCTIMESTRING",
        # event typing & context
        "EVTYPE", "BIGRAM", "TRIGRAM",
        # clocks / bins
        "CLOCK_SEC", "CLOCK_BIN", "EVENT_IDX",
        # margin & labels
        "MARGIN_PRE", "MARGIN_POST", "MARGIN_BIN", "RESULT_CLASS", "TARGET_POINTS_AHEAD",
        # offense / bonus
        "OFF_SIDE", "HOME_BONUS", "AWAY_BONUS", "OFF_BONUS",
        # recent context
        "RCNT3_MAKE", "RCNT3_MISS", "RCNT3_TO",
        # optional lineups
        "ON_COURT_MIA", "ON_COURT_BOS",
        # (optional) helpful text for UI hover
        "HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION", "TEXT",
    ]
    KEEP = [c for c in KEEP if c in pbp.columns]
    pbp = pbp.loc[:, KEEP]

    pbp.to_parquet(OUT, index=False)
    print(f"✅ Wrote {OUT} with {len(pbp)} rows.")

if __name__ == "__main__":
    build_features()
