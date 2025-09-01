from __future__ import annotations
import pandas as pd


# Possession boundaries
START_EVENTS = {"jump ball", "start of period", "rebound_off"}
# End a possession on these (incl. defensive rebound)
END_EVENTS = {"made shot", "turnover", "end of period", "rebound_def", "ft_end"}

# Raw PBP columns we require
REQUIRED = {
    "GAME_ID", "EVENTNUM", "PERIOD", "PCTIMESTRING",
    "HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION",
}

def _clock_to_seconds(t: str) -> int:
    """'MM:SS' → total seconds; -1 on parse failure."""
    try:
        m, s = str(t).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return -1

def normalize_event_type(text: str) -> str:
    t = str(text or "").lower()

    SHOT_KWS = (
        "jump shot", "layup", "dunk", "hook", "tip", "fadeaway", "turnaround",
        "bank shot", "pull-up", "floater", "alley-oop", "putback", "three", "3pt", "3-pt"
    )

    # 1) Timeouts & subs (diagnostic only; do NOT end possessions)
    if "timeout" in t:
        return "timeout"
    if " sub:" in t or t.startswith("sub:"):
        return "substitution"

    # 2) Goaltending (defensive adds points = treat as made shot; offensive is turnover)
    if "goaltending" in t or "goal tending" in t:
        if "offensive" in t:
            return "turnover"  # offensive GT is a turnover
        return "made shot"  # defensive GT counts the bucket

    # 3) Violations (most keep possession; offensive 3-sec is a turnover; defensive 3-sec = tech FT)
    if "violation" in t:
        if "defensive 3" in t or "defensive three" in t or "def 3" in t:
            return "ft_special"  # technical FT; same team retains ball
        if "offensive 3" in t or "offensive three" in t or "off 3" in t:
            return "turnover"  # offensive 3-sec → turnover
        if "kicked ball" in t or "lane violation" in t:
            return "violation"  # minor reset; same possession
        return "violation"

    # Period boundaries (handle variants)
    if ("start" in t and ("period" in t or "q" in t)) or "start of period" in t:
        return "start of period"
    if ("end" in t and ("period" in t or "q" in t)) or "end of period" in t:
        return "end of period"

    if "turnover" in t:  return "turnover"
    if "free throw" in t: return "free throw"

    # Shots (field goals, not free throws)
    if "free throw" not in t and any(kw in t for kw in SHOT_KWS):
        return "missed shot" if "miss" in t else "made shot"

    # Free throws
    if "free throw" in t:
        if "technical" in t or "clear path" in t or "flagrant" in t:
            return "ft_special"  # does NOT end possession
        if any(x in t for x in ("1 of 1", "2 of 2", "3 of 3")):
            return "ft_end"  # can end possession
        return "free throw"

    # Fouls (diagnostic labels; do not end possessions)
    if "foul" in t:
        if "l.b.foul" in t or "loose ball" in t:
            return "foul_loose"
        if "s.foul" in t or "shooting" in t:
            return "foul_shooting"
        # 'p.foul' and everything else
        return "foul_personal"

    # Rebounds
    if "rebound" in t:
        # explicit words
        if "offensive rebound" in t: return "rebound_off"
        if "defensive rebound" in t: return "rebound_def"
        # pattern Off:x Def:y
        if "off:" in t:
            # crude parse: look for a nonzero after 'off:'
            # works on strings like '(Off:1 Def:2)'
            try:
                after = t.split("off:")[1].strip()
                num = ""
                for ch in after:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num and int(num) > 0:
                    return "rebound_off"
                return "rebound_def"
            except Exception:
                return "rebound_def"
        # generic/team rebound -> assume defensive
        return "rebound_def"

    return "other"

def build_possessions(pbp: pd.DataFrame) -> pd.DataFrame:
    # Guard against passing processed data back into the builder
    missing = REQUIRED - set(pbp.columns)
    if missing:
        raise ValueError(
            f"build_possessions: input missing columns {sorted(missing)} "
            f"(did you pass processed possessions instead of raw PBP?)"
        )

    df = pbp.copy()
    df["EVENTNUM"] = pd.to_numeric(df["EVENTNUM"], errors="coerce")
    df = df.sort_values(["GAME_ID", "PERIOD", "EVENTNUM"]).reset_index(drop=True)

    # Unified text for parsing event type
    text_cols = ["HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]
    df["TEXT"] = df[text_cols].fillna("").agg(" ".join, axis=1).str.strip()
    df["EVTYPE"] = df["TEXT"].map(normalize_event_type)
    print("EVTYPE counts:\n", df["EVTYPE"].value_counts().head(20))

    # Stitch possessions: start at START_EVENTS or right after an END_EVENTS
    prev_end = df["EVTYPE"].shift(1).isin(END_EVENTS).fillna(False)
    new_pos = (df["EVTYPE"].isin(START_EVENTS) | prev_end).astype(int)
    df["POSS_SEQ"] = new_pos.groupby(df["GAME_ID"]).cumsum()

    # Aggregate to one row per possession
    gcols = ["GAME_ID", "POSS_SEQ"]
    poss = (
        df.groupby(gcols)
        .agg(
            PERIOD=("PERIOD", "first"),
            START_EVENTNUM=("EVENTNUM", "first"),
            END_EVENTNUM=("EVENTNUM", "last"),
            START_TIME=("PCTIMESTRING", "first"),
            END_TIME=("PCTIMESTRING", "last"),
            START_TEXT=("TEXT", "first"),
            END_TEXT=("TEXT", "last"),
            EVENTS=("EVENTNUM", "count"),
        )
        .reset_index()
    )

    # Result label (bring only EVTYPE to avoid PERIOD_x/_y)
    last_rows = df.groupby(gcols).tail(1)[["GAME_ID", "POSS_SEQ", "EVTYPE"]]
    poss = poss.merge(last_rows, on=["GAME_ID", "POSS_SEQ"], how="left")
    poss.rename(columns={"EVTYPE": "RESULT"}, inplace=True)

    # Safety: if any legacy PERIOD duplicates crept in
    if "PERIOD_x" in poss.columns:
        poss.rename(columns={"PERIOD_x": "PERIOD"}, inplace=True)
    if "PERIOD_y" in poss.columns:
        poss.drop(columns=["PERIOD_y"], inplace=True, errors="ignore")

    # Duration in seconds (clock counts down)
    st_sec = poss["START_TIME"].map(_clock_to_seconds)
    en_sec = poss["END_TIME"].map(_clock_to_seconds)
    poss["DURATION_SEC"] = (st_sec - en_sec).clip(lower=0)

    return poss
