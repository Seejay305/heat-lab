# src/features/build_event_features.py
"""
Build per-event features for EPV training/inference.

Inputs
------
- data/raw/pbp/<SEASON>/*.parquet         (raw NBA pbp, one parquet per game)
- data/processed/event_possess_map.parquet (event -> possession map from run_possessions)
- data/processed/possessions_enriched.csv  (possession labels from enrich_possessions)
- data/derived/lineups_events.parquet      (optional; on-court strings per event)

Output
------
- data/derived/event_features.parquet
"""

from __future__ import annotations

from pathlib import Path
import glob
import re
import pandas as pd

# ---------- Paths ----------
RAW_DIR = Path("data/raw/pbp/2023-24")
MAP_PATH = Path("data/processed/event_possess_map.parquet")
ENRICHED_PATH = Path("data/processed/possessions_enriched.csv")
LINEUPS_PATH = Path("data/derived/lineups_events.parquet")  # optional
OUT_PATH = Path("data/derived/event_features.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------- Small helpers ----------


def _norm_game_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)


def _coerce_int(df: pd.DataFrame, col: str, default=None) -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce")


def _split_names(s: str) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


def _off_string(r: pd.Series) -> str:
    """Safe OFF_SIDE string ('HOME'/'AWAY' or '') without NA truthiness issues."""
    v = r.get("OFF_SIDE", "")
    if isinstance(v, str):
        return v.upper()
    return ""


# =========================
# SECTION A — LOAD & KEYS
# =========================


def load_all_pbp(raw_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw_dir / "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found under {raw_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    need = {"GAME_ID", "PERIOD", "EVENTNUM"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise SystemExit(f"PBP missing required columns: {miss}")

    df["GAME_ID"] = _norm_game_id(df["GAME_ID"])
    _coerce_int(df, "PERIOD")
    _coerce_int(df, "EVENTNUM")
    return df


def attach_event_possess_map(pbp: pd.DataFrame) -> pd.DataFrame:
    if not MAP_PATH.exists():
        raise SystemExit(
            f"Missing {MAP_PATH}. Run: python -m src.features.run_possessions"
        )

    emap = pd.read_parquet(MAP_PATH).copy()
    for c in ["GAME_ID", "PERIOD", "EVENTNUM", "POSS_SEQ"]:
        if c not in emap.columns:
            raise SystemExit(f"{MAP_PATH} missing required column: {c}")
    emap["GAME_ID"] = _norm_game_id(emap["GAME_ID"])
    for c in ["PERIOD", "EVENTNUM", "POSS_SEQ"]:
        _coerce_int(emap, c)

    out = pbp.merge(
        emap[["GAME_ID", "PERIOD", "EVENTNUM", "POSS_SEQ"]],
        on=["GAME_ID", "PERIOD", "EVENTNUM"],
        how="inner",
        validate="1:1",
    )
    return out


# ==================================
# SECTION B — ENRICHMENT MERGE-INs
# ==================================


def attach_possession_labels(pbp: pd.DataFrame) -> pd.DataFrame:
    if not ENRICHED_PATH.exists():
        raise SystemExit(
            f"Missing {ENRICHED_PATH}. Run: python -m src.features.enrich_possessions"
        )

    enr = pd.read_csv(ENRICHED_PATH)
    enr["GAME_ID"] = _norm_game_id(enr["GAME_ID"])
    for c in ("POSS_SEQ", "POINTS", "MARGIN_PRE", "MARGIN_POST", "PERIOD"):
        if c in enr.columns:
            _coerce_int(enr, c)

    keep = [
        c
        for c in [
            "GAME_ID",
            "POSS_SEQ",
            "POINTS",
            "MARGIN_PRE",
            "MARGIN_POST",
            "RESULT_CLASS",
            "TEAM",
            "PERIOD",
        ]
        if c in enr.columns
    ]
    pbp = pbp.merge(
        enr[keep].drop_duplicates(subset=["GAME_ID", "POSS_SEQ"]),
        on=["GAME_ID", "POSS_SEQ"],
        how="left",
        validate="m:1",
    )
    return pbp


def attach_lineups_if_present(pbp: pd.DataFrame) -> pd.DataFrame:
    if not LINEUPS_PATH.exists():
        print(
            "⚠️ lineups_events.parquet not found — continuing without on-court metadata."
        )
        return pbp

    lu = pd.read_parquet(LINEUPS_PATH).copy()
    lu["GAME_ID"] = _norm_game_id(lu["GAME_ID"])
    for c in ("PERIOD", "EVENTNUM", "POSS_SEQ"):
        if c in lu.columns:
            _coerce_int(lu, c)

    # prefer event-level join using (GAME_ID, PERIOD, EVENTNUM)
    keep_lu = [
        c
        for c in [
            "GAME_ID",
            "PERIOD",
            "EVENTNUM",
            "HOME_TEAM_ABBR",
            "AWAY_TEAM_ABBR",
            "ON_COURT_HOME",
            "ON_COURT_AWAY",
            "OFF_TEAM_ABBR",
            "DEF_TEAM_ABBR",
        ]
        if c in lu.columns
    ]

    lu_small = lu[keep_lu].drop_duplicates(subset=["GAME_ID", "PERIOD", "EVENTNUM"])
    overlap = (set(lu_small.columns) & set(pbp.columns)) - {
        "GAME_ID",
        "PERIOD",
        "EVENTNUM",
    }
    if overlap:
        lu_small = lu_small.drop(columns=list(overlap))

    if "PERIOD" not in pbp.columns:
        # normalize suffixes from previous merges, if any
        for src in ("PERIOD_x", "PERIOD_y"):
            if src in pbp.columns and "PERIOD" not in pbp.columns:
                pbp = pbp.rename(columns={src: "PERIOD"})
        if "PERIOD" not in pbp.columns:
            # recover via emap, last resort
            emap_period = pd.read_parquet(MAP_PATH)[
                ["GAME_ID", "EVENTNUM", "PERIOD"]
            ].copy()
            emap_period["GAME_ID"] = _norm_game_id(emap_period["GAME_ID"])
            _coerce_int(emap_period, "EVENTNUM")
            _coerce_int(emap_period, "PERIOD")
            pbp = pbp.merge(
                emap_period, on=["GAME_ID", "EVENTNUM"], how="left", validate="m:1"
            )

    _coerce_int(pbp, "PERIOD")
    _coerce_int(pbp, "EVENTNUM")

    pbp = pbp.merge(
        lu_small, on=["GAME_ID", "PERIOD", "EVENTNUM"], how="left", validate="1:1"
    )
    return pbp


# =====================================
# SECTION C — TEAM ABBR CLEAN + OFFSIDE
# =====================================

_ABBR_RE = r"^[A-Z]{2,4}$"


def clean_team_abbrs(
    df: pd.DataFrame,
    cols=("HOME_TEAM_ABBR", "AWAY_TEAM_ABBR", "OFF_TEAM_ABBR", "DEF_TEAM_ABBR"),
) -> pd.DataFrame:
    """
    1) Blank out anything that doesn't look like a 2–4 letter abbreviation.
    2) Within each GAME_ID, backfill HOME/AWAY via mode to stabilize.
    """
    # step 1 — blank bad values
    for col in cols:
        if col in df.columns:
            bad = ~df[col].astype("string").str.fullmatch(_ABBR_RE, na=True)
            n_bad = int(bad.sum())
            if n_bad:
                df.loc[bad, col] = pd.NA
                print(
                    f"[CLEAN] Blank/cleaned {n_bad} values in {col} (non-abbreviation text)."
                )

    # step 2 — backfill HOME/AWAY using per-game mode (only if needed)
    for col in ("HOME_TEAM_ABBR", "AWAY_TEAM_ABBR"):
        if col not in df.columns:
            continue
        # per-game mode
        mode_map = (
            df[["GAME_ID", col]]
            .dropna()
            .groupby("GAME_ID")[col]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
        )
        # use 'where' to avoid .fillna downcasting warning
        df[col] = df[col].where(df[col].notna(), df["GAME_ID"].map(mode_map))

    return df


def ensure_off_side(df: pd.DataFrame) -> pd.DataFrame:
    """Derive OFF_SIDE if missing, using OFF_TEAM_ABBR vs HOME/AWAY abbreviations."""
    if "OFF_SIDE" in df.columns:
        return df

    for col in ("HOME_TEAM_ABBR", "AWAY_TEAM_ABBR", "OFF_TEAM_ABBR"):
        if col not in df.columns:
            df[col] = pd.NA

    def _side(r):
        off = r.get("OFF_TEAM_ABBR")
        home, away = r.get("HOME_TEAM_ABBR"), r.get("AWAY_TEAM_ABBR")
        if pd.isna(off) or (pd.isna(home) and pd.isna(away)):
            return pd.NA
        if off == home:
            return "HOME"
        if off == away:
            return "AWAY"
        return pd.NA

    df["OFF_SIDE"] = df.apply(_side, axis=1)
    return df


# ============================================
# SECTION D — LINEUP COUNTS / ON–OFF FEATURES
# ============================================


def add_lineup_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """Create ON_COURT_* counts and generic OFF/DEF 5-on-the-floor counts."""
    for side in ("HOME", "AWAY"):
        col = f"ON_COURT_{side}"
        if col not in pbp.columns:
            pbp[col] = pd.NA
        names = pbp[col].fillna("").astype(str)
        pbp[f"ON_COURT_{side}_COUNT"] = names.apply(lambda s: len(_split_names(s)))

    # generic: offense/defense counts derived from OFF_SIDE
    def _off_count(r):
        side = _off_string(r)
        if side == "HOME":
            return r.get("ON_COURT_HOME_COUNT", 0)
        if side == "AWAY":
            return r.get("ON_COURT_AWAY_COUNT", 0)
        return 0

    def _def_count(r):
        side = _off_string(r)
        if side == "HOME":
            return r.get("ON_COURT_AWAY_COUNT", 0)
        if side == "AWAY":
            return r.get("ON_COURT_HOME_COUNT", 0)
        return 0

    pbp["ON_COURT_OFF_COUNT"] = pbp.apply(_off_count, axis=1)
    pbp["ON_COURT_DEF_COUNT"] = pbp.apply(_def_count, axis=1)
    return pbp


# ======================================
# SECTION E — BONUS / TEAM FOUL CONTEXT
# ======================================

FOUL_EVTS = {"foul_personal", "foul_shooting", "foul_loose"}


def add_bonus_flags(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple team-fouls tracker per (GAME_ID, PERIOD).
    - Count defensive fouls (personal/shooting/loose) against the defensive side.
    - Set HOME_BONUS / AWAY_BONUS once team-fouls >= 5 in a period.
    - OFF_BONUS mirrors the current offense's bonus flag.
    """
    for col in ("HOME_BONUS", "AWAY_BONUS", "OFF_BONUS"):
        pbp[col] = 0

    # Ensure ingredients
    if "EVTYPE" not in pbp.columns:
        pbp["EVTYPE"] = pd.NA
    if "OFF_SIDE" not in pbp.columns:
        pbp = ensure_off_side(pbp)

    pbp = pbp.sort_values(["GAME_ID", "PERIOD", "EVENTNUM"]).reset_index(drop=True)

    # track fouls per game+period
    def _flag_bonus(group: pd.DataFrame) -> pd.DataFrame:
        home_fouls = 0
        away_fouls = 0
        hb = []
        ab = []

        for _, r in group.iterrows():
            ev = str(r.get("EVTYPE", "")).lower()
            off = _off_string(r)
            # defensive side is the opposite of offense
            if ev in FOUL_EVTS:
                if off == "HOME":
                    away_fouls += 1
                elif off == "AWAY":
                    home_fouls += 1
            hb.append(1 if home_fouls >= 5 else 0)
            ab.append(1 if away_fouls >= 5 else 0)

        group["HOME_BONUS"] = hb
        group["AWAY_BONUS"] = ab
        return group

    pbp = pbp.groupby(["GAME_ID", "PERIOD"], group_keys=False).apply(_flag_bonus)

    # OFF_BONUS = offense's current bonus state
    def _off_bonus(r):
        off = _off_string(r)
        if off == "HOME":
            return r.get("HOME_BONUS", 0)
        if off == "AWAY":
            return r.get("AWAY_BONUS", 0)
        return 0

    pbp["OFF_BONUS"] = pbp.apply(_off_bonus, axis=1)
    return pbp


# =========================================
# SECTION F — RECENT MEMORY (last 3 events)
# =========================================


def add_recent_memory_flags(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    RCNT3_MAKE / RCNT3_MISS / RCNT3_TO within the same possession window
    based on EVTYPE/BIGRAM/TRIGRAM text signals.
    """
    for c in ("BIGRAM", "TRIGRAM"):
        if c not in pbp.columns:
            pbp[c] = pd.NA

    # simple textual signals
    def _is_make(row) -> bool:
        txt = " ".join(
            str(row.get(c, "")) for c in ("EVTYPE", "BIGRAM", "TRIGRAM")
        ).lower()
        return bool(re.search(r"\bmade shot\b|\bmake\b|\bgood\b", txt))

    def _is_miss(row) -> bool:
        txt = " ".join(
            str(row.get(c, "")) for c in ("EVTYPE", "BIGRAM", "TRIGRAM")
        ).lower()
        return bool(re.search(r"\bmissed shot\b|\bmiss\b", txt))

    def _is_turnover(row) -> bool:
        txt = " ".join(
            str(row.get(c, "")) for c in ("EVTYPE", "BIGRAM", "TRIGRAM")
        ).lower()
        return bool(re.search(r"\bturnover\b|\bsteal\b|\bviolation\b", txt))

    pbp["EV_IS_MAKE"] = pbp.apply(_is_make, axis=1).astype(int)
    pbp["EV_IS_MISS"] = pbp.apply(_is_miss, axis=1).astype(int)
    pbp["EV_IS_TO"] = pbp.apply(_is_turnover, axis=1).astype(int)

    pbp = pbp.sort_values(["GAME_ID", "POSS_SEQ", "EVENTNUM"]).reset_index(drop=True)
    pbp["EVENT_IDX"] = pbp.groupby(["GAME_ID", "POSS_SEQ"]).cumcount()

    def _roll3(group: pd.DataFrame) -> pd.DataFrame:
        # lookback on the previous 3 events (exclude current)
        group["RCNT3_MAKE"] = (
            group["EV_IS_MAKE"]
            .shift(1)
            .rolling(3, min_periods=1)
            .sum()
            .fillna(0)
            .astype(int)
        )
        group["RCNT3_MISS"] = (
            group["EV_IS_MISS"]
            .shift(1)
            .rolling(3, min_periods=1)
            .sum()
            .fillna(0)
            .astype(int)
        )
        group["RCNT3_TO"] = (
            group["EV_IS_TO"]
            .shift(1)
            .rolling(3, min_periods=1)
            .sum()
            .fillna(0)
            .astype(int)
        )
        return group

    pbp = (
        pbp.groupby(["GAME_ID", "POSS_SEQ"], group_keys=False)[
            ["EV_IS_MAKE", "EV_IS_MISS", "EV_IS_TO", "EVENTNUM", "EVENT_IDX"]
        ]
        .apply(_roll3)
        .join(
            pbp.drop(columns=["RCNT3_MAKE", "RCNT3_MISS", "RCNT3_TO"], errors="ignore"),
            how="right",
        )
    )

    # clean temp flags
    pbp.drop(
        columns=["EV_IS_MAKE", "EV_IS_MISS", "EV_IS_TO"], inplace=True, errors="ignore"
    )
    return pbp


# ======================================
# SECTION G — TARGETS, TYPES, DIAGNOSTIC
# ======================================


def finalize_targets_and_types(df: pd.DataFrame) -> pd.DataFrame:
    # stable ordering for trainer
    df = df.sort_values(["GAME_ID", "POSS_SEQ", "EVENTNUM"]).reset_index(drop=True)
    df["EVENT_IDX"] = df.groupby(["GAME_ID", "POSS_SEQ"]).cumcount()

    # main target (next points within possession already computed upstream)
    df["TARGET_POINTS_AHEAD"] = (
        pd.to_numeric(df.get("POINTS"), errors="coerce").fillna(0).astype(int)
    )

    # numeric casts / fill
    num_cols = [
        "PERIOD",
        "CLOCK_SEC",
        "CLOCK_BIN",
        "EVENT_IDX",
        "MARGIN_PRE",
        "MARGIN_POST",
        "ON_COURT_HOME_COUNT",
        "ON_COURT_AWAY_COUNT",
        "ON_COURT_OFF_COUNT",
        "ON_COURT_DEF_COUNT",
        "HOME_BONUS",
        "AWAY_BONUS",
        "OFF_BONUS",
        "RCNT3_MAKE",
        "RCNT3_MISS",
        "RCNT3_TO",
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # text/categorical existence
    for c in ["EVTYPE", "BIGRAM", "TRIGRAM", "MARGIN_BIN", "OFF_SIDE"]:
        if c not in df.columns:
            df[c] = pd.NA

    return df


def diagnostics_and_write(df: pd.DataFrame) -> None:
    vc = df["TARGET_POINTS_AHEAD"].value_counts().sort_index().to_frame("count")
    nonzero = int((df["TARGET_POINTS_AHEAD"] > 0).sum())
    print("TARGET_POINTS_AHEAD value counts:\n", vc)
    print(f"Nonzero-label rows: {nonzero}")

    per_game_counts = df["GAME_ID"].value_counts().sort_index().to_dict()
    per_game_targets = (
        df.groupby("GAME_ID")["TARGET_POINTS_AHEAD"]
        .apply(lambda s: s.notna().sum())
        .to_dict()
    )
    print("EF rows / games:", len(df), df["GAME_ID"].nunique())
    print("EF per game:", per_game_counts)
    print("Non-null TARGET_POINTS_AHEAD per game:", per_game_targets)

    df.to_parquet(OUT_PATH, index=False)
    print(
        f"✅ Wrote {OUT_PATH} with {len(df)} rows and {df['GAME_ID'].nunique()} game(s)."
    )
    print("Games:", sorted(df["GAME_ID"].unique().tolist())[:10], "...")


# ============
# MAIN PIPE
# ============


def main():
    # A) load + map
    pbp = load_all_pbp(RAW_DIR)
    pbp = attach_event_possess_map(pbp)

    # B) labels + lineups
    pbp = attach_possession_labels(pbp)
    pbp = attach_lineups_if_present(pbp)

    # C) team abbr cleanup + OFF_SIDE derivation
    pbp = clean_team_abbrs(
        pbp, cols=("HOME_TEAM_ABBR", "AWAY_TEAM_ABBR", "OFF_TEAM_ABBR", "DEF_TEAM_ABBR")
    )
    # optional post-check (warn only on remaining bad)
    for c in ["HOME_TEAM_ABBR", "AWAY_TEAM_ABBR"]:
        if c in pbp.columns:
            bad = ~pbp[c].astype("string").str.fullmatch(_ABBR_RE, na=True)
            if int(bad.sum()):
                print(
                    f"[WARN] {c} still has {int(bad.sum())} unexpected values; leaving as-is."
                )

    pbp = ensure_off_side(pbp)

    # D) lineup counts (on floor)
    pbp = add_lineup_counts(pbp)

    # E) bonus flags (per period team fouls)
    pbp = add_bonus_flags(pbp)

    # F) recent memory flags (within possession)
    pbp = add_recent_memory_flags(pbp)

    # G) finalize types/targets + write
    pbp = finalize_targets_and_types(pbp)
    diagnostics_and_write(pbp)


if __name__ == "__main__":
    main()
