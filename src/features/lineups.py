# src/features/lineups.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

from src.features.roster import PLAYER_TO_TEAM
from src.features.possession_builder import normalize_event_type  # reuse classifier

RAW_DIR = Path("data/raw/pbp/2023-24")
OUT = Path("data/derived/lineups_events.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

SUB_PAT = re.compile(r"\bsub:\s*(?P<inn>.+?)\s+for\s+(?P<out>.+?)\b", re.I)

def _unify_text(df: pd.DataFrame) -> pd.Series:
    cols = ["HOMEDESCRIPTION","VISITORDESCRIPTION","NEUTRALDESCRIPTION"]
    return df[cols].fillna("").agg(" ".join, axis=1).str.strip()

def _detect_team_for_name(name: str) -> str | None:
    # best-effort: exact key or stripped "Jr." variant
    if name in PLAYER_TO_TEAM:
        return PLAYER_TO_TEAM[name]
    n = name.replace(" Jr.", "").replace(" Jr", "")
    return PLAYER_TO_TEAM.get(n)

def _parse_sub(text: str) -> tuple[str,str] | None:
    m = SUB_PAT.search(text or "")
    if not m:
        return None
    inn = m.group("inn").strip().rstrip(".")
    out = m.group("out").strip().rstrip(".")
    return inn, out

def _attach_possession_seq(pbp: pd.DataFrame) -> pd.DataFrame:
    # mirror your builder’s boundaries (no change to logic)
    df = pbp.copy()
    df["EVENTNUM"] = pd.to_numeric(df["EVENTNUM"], errors="coerce")
    df = df.sort_values(["GAME_ID","PERIOD","EVENTNUM"]).reset_index(drop=True)
    df["TEXT"] = _unify_text(df)
    df["EVTYPE"] = df["TEXT"].map(normalize_event_type)

    START_EVENTS = {"jump ball", "start of period", "rebound_off"}
    END_EVENTS   = {"made shot", "turnover", "end of period", "rebound_def", "ft_end"}

    prev_end = df["EVTYPE"].shift(1).isin(END_EVENTS).fillna(False)
    new_pos = (df["EVTYPE"].isin(START_EVENTS) | prev_end).astype(int)
    df["POSS_SEQ"] = new_pos.groupby(df["GAME_ID"]).cumsum()
    return df

def build_lineups_for_game(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return per-event on-court lineups (5 per team) inferred from SUB: events.
       Strategy: collect first 5 unique players per team seen before any sub;
       then apply SUB: X FOR Y updates in order.
    """
    df = _attach_possession_seq(pbp)
    df["TEXT"] = _unify_text(df)

    # Collect early-appearance players per team to seed lineups
    seen_by_team: dict[str, list[str]] = {"MIA": [], "BOS": []}
    def _collect(name: str):
        team = _detect_team_for_name(name)
        if team and name not in seen_by_team[team]:
            seen_by_team[team].append(name)

    # mine names from early events
    name_pat = re.compile(r"\b([A-Z][a-z]+(?:\sJr\.)?)\b")  # simple LastName / LastName Jr.
    for txt in df["TEXT"].head(80).tolist():  # enough to capture starters
        for cand in name_pat.findall(txt or ""):
            if cand in PLAYER_TO_TEAM or cand.replace(" Jr.", "") in PLAYER_TO_TEAM:
                _collect(cand)

    # seed lineups to first 5 per team (if fewer found, we’ll grow as we go)
    on_court: dict[str, list[str]] = {
        "MIA": seen_by_team["MIA"][:5],
        "BOS": seen_by_team["BOS"][:5],
    }

    # helper to ensure max-5 & no duplicates
    def _ensure_five(team: str, name: str | None = None):
        five = on_court[team]
        # trim extras
        if len(five) > 5:
            on_court[team] = five[:5]
        # grow if <5 and we have a valid name
        if name and name not in on_court[team] and len(on_court[team]) < 5:
            on_court[team].append(name)

    # iterate through events; update on_court via SUB: inn FOR out
    mia_list, bos_list = [], []
    for txt in df["TEXT"]:
        sub = _parse_sub(txt)
        if sub:
            inn, out = sub
            t_in  = _detect_team_for_name(inn)
            t_out = _detect_team_for_name(out)
            # pick the team we can determine (prefer t_out for robustness)
            team = t_out or t_in
            if team in ("MIA","BOS"):
                # add missing names into pool to avoid KeyErrors
                if inn and inn not in on_court[team]:
                    on_court[team].append(inn)
                if out and out in on_court[team]:
                    on_court[team].remove(out)
                # keep roster to 5
                on_court[team] = [p for p in on_court[team] if p]  # clean Nones
                if len(on_court[team]) > 5:
                    on_court[team] = on_court[team][-5:]

        # After applying subs at this event, record snapshots
        # also opportunistically grow to five when we see new names via actions
        for cand in name_pat.findall(txt or ""):
            if cand in PLAYER_TO_TEAM or cand.replace(" Jr.", "") in PLAYER_TO_TEAM:
                tm = _detect_team_for_name(cand)
                if tm in ("MIA","BOS"):
                    _ensure_five(tm, cand)

        # pad with blanks so length is 5 (stable serialization)
        mia = (on_court["MIA"] + [""]*5)[:5]
        bos = (on_court["BOS"] + [""]*5)[:5]
        mia_list.append(";".join(mia))
        bos_list.append(";".join(bos))

    out = df[["GAME_ID","PERIOD","EVENTNUM","POSS_SEQ"]].copy()
    out["ON_COURT_MIA"] = mia_list
    out["ON_COURT_BOS"] = bos_list
    return out

def main():
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("No raw PBP parquet found in data/raw/pbp/2023-24")
    # For now: process the first game (matches your current workflow)
    pbp = pd.read_parquet(files[0])
    line = build_lineups_for_game(pbp)
    line.to_parquet(OUT, index=False)
    print(f"✅ Wrote {OUT} with {len(line)} event rows.")

if __name__ == "__main__":
    main()
