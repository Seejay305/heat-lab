"""
enrich_possessions.py

Build possession-level labels from raw play-by-play + the (GAME_ID, PERIOD, EVENTNUM)→POSS_SEQ map.

Outputs
-------
data/processed/possessions_enriched.csv
  Columns (minimum):
    GAME_ID, POSS_SEQ, PERIOD, MARGIN_PRE, MARGIN_POST, POINTS, RESULT_CLASS, TEAM

Assumptions
-----------
- Raw PBP parquet per game exists under data/raw/pbp/<SEASON>/<GAME_ID>.parquet
- Event→possession map exists at data/processed/event_possess_map.parquet
- PBP parquet has standard NBA stats columns including: GAME_ID, EVENTNUM, PERIOD,
  SCORE, SCOREMARGIN, HOMEDESCRIPTION, VISITORDESCRIPTION, NEUTRALDESCRIPTION,
  PLAYER1_TEAM_ABBREVIATION/PLAYER2_TEAM_ABBREVIATION (optional).

Usage
-----
python -m src.features.enrich_possessions
"""

from __future__ import annotations

from pathlib import Path
import glob
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS & CONSTANTS
# =============================================================================
RAW_DIR = Path("data/raw/pbp")  # contains <SEASON>/<GAME_ID>.parquet
MAP_PATH = Path("data/processed/event_possess_map.parquet")
OUT_CSV = Path("data/processed/possessions_enriched.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================
def _norm_game_id(s: pd.Series) -> pd.Series:
    """Zero-pad and strip non-digits to match 10-char GAME_ID strings."""
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)


def _find_game_parquet(game_id: str) -> Optional[str]:
    """
    Locate the parquet file for a given GAME_ID across any season dir.
    Searches: data/raw/pbp/**/<GAME_ID>.parquet
    """
    pattern = str(RAW_DIR / "**" / f"{game_id}.parquet")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def _parse_score_pair(score: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse 'XX-YY' to (away, home). NBA API SCORE is 'VISITORS-HOME'.
    Return (None, None) if unparsable.
    """
    if not isinstance(score, str) or "-" not in score:
        return (None, None)
    try:
        a, h = score.split("-")
        return (int(a), int(h))
    except Exception:
        return (None, None)


def _score_total(s: pd.Series) -> pd.Series:
    """Convert SCORE to total points (away+home)."""
    ah = s.fillna("").astype(str).str.extract(r"(\d+)-(\d+)")
    out = pd.to_numeric(ah[0], errors="coerce") + pd.to_numeric(ah[1], errors="coerce")
    return out


def _margin_number(scoremargin: pd.Series) -> pd.Series:
    """
    SCOREMARGIN column can be like '3', 'TIE', '', '-5'.
    Convert to signed integer from the HOME perspective:
      positive => home leads, negative => home trails.
    """
    s = scoremargin.astype(str).str.upper()
    s = s.replace({"TIE": "0", "": np.nan, "NONE": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _guess_team_from_row(row: pd.Series) -> Optional[str]:
    """
    Try to infer the team abbreviation responsible for the event:
    - Use PLAYER*_TEAM_ABBREVIATION if present.
    - Else scrape last ALLCAPS token from descriptions (HOME/VISITOR/NEUTRAL).
    """
    for c in (
        "PLAYER1_TEAM_ABBREVIATION",
        "PLAYER2_TEAM_ABBREVIATION",
        "PLAYER3_TEAM_ABBREVIATION",
    ):
        if c in row and isinstance(row[c], str) and len(row[c]) in (2, 3, 4):
            return row[c].strip()

    for c in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"):
        if c in row and isinstance(row[c], str) and row[c]:
            m = re.findall(r"\b[A-Z]{2,4}\b", row[c])
            if m:
                return m[-1]
    return None


# =============================================================================
# MAIN
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 0) Load (GAME_ID, PERIOD, EVENTNUM) → POSS_SEQ map
    # -------------------------------------------------------------------------
    if not MAP_PATH.exists():
        raise SystemExit(
            f"Missing {MAP_PATH}. Run: python -m src.features.run_possessions"
        )

    emap = pd.read_parquet(MAP_PATH).copy()
    for c in ["GAME_ID", "EVENTNUM", "POSS_SEQ"]:
        if c not in emap.columns:
            raise SystemExit(f"{MAP_PATH} missing required column: {c}")

    emap["GAME_ID"] = _norm_game_id(emap["GAME_ID"])
    for c in ["EVENTNUM", "POSS_SEQ", "PERIOD"]:
        if c in emap.columns:
            emap[c] = pd.to_numeric(emap[c], errors="coerce")

    # -------------------------------------------------------------------------
    # 1) Load raw PBP for each game referenced in the map
    # -------------------------------------------------------------------------
    game_ids = sorted(emap["GAME_ID"].unique().tolist())
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for gid in game_ids:
        f = _find_game_parquet(gid)
        if not f:
            missing.append(gid)
            continue

        df = pd.read_parquet(f).copy()
        df["GAME_ID"] = _norm_game_id(df["GAME_ID"])

        # Required keys
        for need in ["EVENTNUM", "PERIOD"]:
            if need not in df.columns:
                raise SystemExit(f"{f} missing required column: {need}")
        df["EVENTNUM"] = pd.to_numeric(df["EVENTNUM"], errors="coerce")
        df["PERIOD"] = pd.to_numeric(df["PERIOD"], errors="coerce")

        # Ensure score/margin columns exist
        if "SCORE" not in df.columns:
            df["SCORE"] = np.nan
        if "SCOREMARGIN" not in df.columns:
            df["SCOREMARGIN"] = np.nan

        # Precompute helpers
        df["TOTAL_SCORE"] = _score_total(df["SCORE"])
        df["MARGIN_NUM"] = _margin_number(df["SCOREMARGIN"])

        # Keep only needed columns (others are optional and may be missing)
        keep = [
            "GAME_ID",
            "EVENTNUM",
            "PERIOD",
            "SCORE",
            "SCOREMARGIN",
            "TOTAL_SCORE",
            "MARGIN_NUM",
            "HOMEDESCRIPTION",
            "VISITORDESCRIPTION",
            "NEUTRALDESCRIPTION",
            "PLAYER1_TEAM_ABBREVIATION",
            "PLAYER2_TEAM_ABBREVIATION",
            "PLAYER3_TEAM_ABBREVIATION",
        ]
        have = [c for c in keep if c in df.columns]
        frames.append(df[have].copy())

    if missing:
        print(
            f"⚠️ Missing raw PBP parquets for {len(missing)} game(s): {missing[:5]}{'...' if len(missing)>5 else ''}"
        )
    if not frames:
        raise SystemExit("No raw PBP loaded. Aborting.")

    pbp = pd.concat(frames, ignore_index=True)

    # -------------------------------------------------------------------------
    # 2) Attach POSS_SEQ to each PBP row (inner join on GAME_ID & EVENTNUM)
    # -------------------------------------------------------------------------
    pbp = pbp.merge(
        emap[["GAME_ID", "EVENTNUM", "POSS_SEQ"]].drop_duplicates(),
        on=["GAME_ID", "EVENTNUM"],
        how="left",
        validate="1:1",
    )
    pbp = pbp[pbp["POSS_SEQ"].notna()].copy()
    pbp["POSS_SEQ"] = pd.to_numeric(pbp["POSS_SEQ"], errors="coerce")

    # Sort for within-game diffs
    pbp = pbp.sort_values(["GAME_ID", "POSS_SEQ", "EVENTNUM"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 3) EVENT-LEVEL SCORING
    #    3a. Primary: use SCORE deltas when available
    #    3b. Fallback: infer scoring from text when SCORE is missing/unchanged
    # -------------------------------------------------------------------------
    # 3a) SCORE-based delta
    pbp["TOTAL_SCORE_PREV"] = pbp.groupby("GAME_ID")["TOTAL_SCORE"].shift(1)
    points_from_score = (pbp["TOTAL_SCORE"] - pbp["TOTAL_SCORE_PREV"]).astype("float64")

    # 3b) Text-based fallback
    text_home = pbp["HOMEDESCRIPTION"].fillna("").astype(str)
    text_away = pbp["VISITORDESCRIPTION"].fillna("").astype(str)
    text_neut = pbp["NEUTRALDESCRIPTION"].fillna("").astype(str)
    text_all = (text_home + " " + text_away + " " + text_neut).str.lower()

    # Free throws made → +1 (exclude explicit misses)
    ft_made = text_all.str.contains(
        r"\bfree throw\b", regex=True
    ) & ~text_all.str.contains(r"\bmiss(?:es|ed)?\b", regex=True)

    # 3PT made → +3 (exclude misses)
    made_3 = (
        text_all.str.contains(r"\b(?:3\s?pt|3-?pointer|three)\b", regex=True)
        & text_all.str.contains(r"\b(?:make|makes|made|hits|hit|good)\b", regex=True)
        & ~text_all.str.contains(r"\bmiss(?:es|ed)?\b", regex=True)
    )

    # Generic made shot → +2 (not FT, not 3PT)
    made_generic = (
        text_all.str.contains(r"\b(?:make|makes|made|hits|hit|good)\b", regex=True)
        & ~ft_made
        & ~made_3
    )

    fallback_points = (
        (ft_made.astype(int) * 1)
        + (made_3.astype(int) * 3)
        + (made_generic.astype(int) * 2)
    ).astype("float64")

    # Prefer SCORE-based; fallback only when SCORE delta is NaN or (0 with text points)
    use_fallback = points_from_score.isna() | (
        (points_from_score == 0) & (fallback_points > 0)
    )
    pbp["POINTS_DELTA"] = points_from_score.where(~use_fallback, fallback_points)
    pbp["POINTS_DELTA"] = pbp["POINTS_DELTA"].clip(lower=0, upper=4).fillna(0)

    # -------------------------------------------------------------------------
    # 4) POSSESSION-LEVEL AGGREGATES
    #    - POINTS: sum of event deltas within possession
    #    - PERIOD: first event's period in the possession
    #    - MARGIN_PRE/MARGIN_POST: first/last MARGIN_NUM
    #    - TEAM: guessed from row data
    # -------------------------------------------------------------------------
    first = pbp.groupby(["GAME_ID", "POSS_SEQ"]).head(1).copy()
    last = pbp.groupby(["GAME_ID", "POSS_SEQ"]).tail(1).copy()

    poss = pbp.groupby(["GAME_ID", "POSS_SEQ"], as_index=False).agg(
        POINTS=("POINTS_DELTA", "sum")
    )
    poss["POINTS"] = (
        pd.to_numeric(poss["POINTS"], errors="coerce").fillna(0).astype(int)
    )

    keep_first = first[
        [
            "GAME_ID",
            "POSS_SEQ",
            "PERIOD",
            "MARGIN_NUM",
            "HOMEDESCRIPTION",
            "VISITORDESCRIPTION",
            "NEUTRALDESCRIPTION",
            "PLAYER1_TEAM_ABBREVIATION",
            "PLAYER2_TEAM_ABBREVIATION",
            "PLAYER3_TEAM_ABBREVIATION",
        ]
    ].rename(columns={"MARGIN_NUM": "MARGIN_PRE"})

    keep_last = last[["GAME_ID", "POSS_SEQ", "MARGIN_NUM"]].rename(
        columns={"MARGIN_NUM": "MARGIN_POST"}
    )

    poss = poss.merge(
        keep_first, on=["GAME_ID", "POSS_SEQ"], how="left", validate="1:1"
    )
    poss = poss.merge(keep_last, on=["GAME_ID", "POSS_SEQ"], how="left", validate="1:1")

    # TEAM guess (offense)
    poss["TEAM"] = poss.apply(_guess_team_from_row, axis=1)

    # -------------------------------------------------------------------------
    # 5) RESULT CLASS
    #    'score' if POINTS>0; else 'turnover' if text signals TO; else 'empty'
    # -------------------------------------------------------------------------
    def _turnover_flag(df: pd.DataFrame) -> pd.Series:
        text = (
            df["HOMEDESCRIPTION"].fillna("").astype(str)
            + " "
            + df["VISITORDESCRIPTION"].fillna("").astype(str)
            + " "
            + df["NEUTRALDESCRIPTION"].fillna("").astype(str)
        ).str.lower()
        pat = r"\bturnover\b|\bstolen\b|\bsteal\b|\boffensive foul\b|\b3 sec turnover\b|\bviolation\b"
        return text.str.contains(pat, regex=True)

    pbp["IS_TOV"] = _turnover_flag(pbp)
    tov = (
        pbp.groupby(["GAME_ID", "POSS_SEQ"], as_index=False)["IS_TOV"]
        .any()
        .rename(columns={"IS_TOV": "HAS_TURNOVER"})
    )
    poss = poss.merge(tov, on=["GAME_ID", "POSS_SEQ"], how="left", validate="1:1")
    poss["HAS_TURNOVER"] = poss["HAS_TURNOVER"].fillna(False)

    def _result_class(row: pd.Series) -> str:
        if (row.get("POINTS", 0) or 0) > 0:
            return "score"
        if bool(row.get("HAS_TURNOVER", False)):
            return "turnover"
        return "empty"

    poss["RESULT_CLASS"] = poss.apply(_result_class, axis=1)

    # -------------------------------------------------------------------------
    # 6) CLEANUP, ORDERING & OUTPUT
    # -------------------------------------------------------------------------
    poss["GAME_ID"] = _norm_game_id(poss["GAME_ID"])
    poss["PERIOD"] = (
        pd.to_numeric(poss["PERIOD"], errors="coerce").fillna(0).astype(int)
    )
    poss["MARGIN_PRE"] = pd.to_numeric(poss["MARGIN_PRE"], errors="coerce")
    poss["MARGIN_POST"] = pd.to_numeric(poss["MARGIN_POST"], errors="coerce")

    keep_cols = [
        "GAME_ID",
        "POSS_SEQ",
        "PERIOD",
        "MARGIN_PRE",
        "MARGIN_POST",
        "POINTS",
        "RESULT_CLASS",
        "TEAM",
    ]
    extra_cols = [c for c in poss.columns if c not in keep_cols]
    out = poss[keep_cols + extra_cols].copy()

    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV} ({len(out)} rows)")

    # -------------------------------------------------------------------------
    # 7) QA SUMMARIES
    # -------------------------------------------------------------------------
    if "RESULT_CLASS" in out.columns:
        print(
            "\nRESULT_CLASS counts:\n",
            out["RESULT_CLASS"].value_counts().to_frame("count"),
        )
    if "TEAM" in out.columns:
        print(
            "\nTEAM counts:\n", out["TEAM"].value_counts(dropna=False).to_frame("count")
        )

    # PPP (points per possession)
    ppp_overall = out["POINTS"].mean()
    print(f"\nPPP (overall): {ppp_overall:.3f}")
    if "TEAM" in out.columns:
        ppp_by_team = (
            out.groupby("TEAM", dropna=False)["POINTS"]
            .mean()
            .sort_values(ascending=False)
        )
        print("\nPPP by TEAM:\n", ppp_by_team)


if __name__ == "__main__":
    main()
