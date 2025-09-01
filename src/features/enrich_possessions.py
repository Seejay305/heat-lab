from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from .roster import PLAYER_TO_TEAM  # NEW

def _name_variants(name: str) -> list[str]:
    n = name
    out = {n}
    if " Jr." in n or " Jr" in n:
        out.add(n.replace(" Jr.", " Jr"))
        out.add(n.replace(" Jr", " Jr."))
        out.add(n.replace(" Jr.", ""))
        out.add(n.replace(" Jr", ""))
    return list(out)

PLAYER_PATTERNS = []
for name, team in PLAYER_TO_TEAM.items():
    for variant in _name_variants(name):
        # word-boundary, case-insensitive
        PLAYER_PATTERNS.append((name, re.compile(rf"\b{re.escape(variant)}\b", re.I), team))

TEAM_WORD_PATTERNS = [
    ("MIA", re.compile(r"\bheat\b", re.I)),
    ("BOS", re.compile(r"\bceltics\b", re.I)),
]

IN_csv  = Path("data/processed/possessions.csv")
OUT_csv = Path("data/processed/possessions_enriched.csv")

THREE_PAT = re.compile(r"\b(3pt|3-pt|three)\b", re.I)
MISS_PAT  = re.compile(r"\bmiss(es|ed)?\b", re.I)
MADE_PAT  = re.compile(r"\bmade|makes\b", re.I)

# Compile player patterns once (word boundaries; case-insensitive)
PLAYER_PATTERNS = [(name, re.compile(rf"\b{re.escape(name)}\b", re.I), team)
                   for name, team in PLAYER_TO_TEAM.items()]

TEAM_WORD_PATTERNS = [
    ("MIA", re.compile(r"\bheat\b", re.I)),
    ("BOS", re.compile(r"\bceltics\b", re.I)),
]

def infer_team_from_text(start_text: str, end_text: str) -> str | None:
    s = str(start_text or "")
    e = str(end_text or "")

    # 1) Player hit wins
    for _name, pat, team in PLAYER_PATTERNS:
        if pat.search(s) or pat.search(e):
            return team

    # 2) Explicit team words
    for code, pat in TEAM_WORD_PATTERNS:
        if pat.search(s) or pat.search(e):
            return code

    return None

def points_from_end(result: str, end_text: str) -> int:
    r = (result or "").lower()
    t = (end_text or "")

    # Field goals
    if r == "made shot":
        return 3 if THREE_PAT.search(t) else 2
    if r == "missed shot":
        return 0

    # Free throws (final ordinary FT)
    if r == "ft_end":
        return 0 if MISS_PAT.search(t) else 1

    # Turnovers / defensive rebound / end period → no points
    if r in {"turnover", "rebound_def", "end of period"}:
        return 0

    # Offensive rebound shouldn't be an end row, but be safe
    if r == "rebound_off":
        return 0

    return 0

def classify_result(result: str, pts: int) -> str:
    r = (result or "").lower()
    if r == "turnover":
        return "turnover"
    if pts > 0:
        return "score"
    return "empty"

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # TEAM from start/end text using roster
    out["TEAM"] = [infer_team_from_text(s, e) for s, e in zip(out.get("START_TEXT",""), out.get("END_TEXT",""))]

    # POINTS from ending event
    out["POINTS"] = [points_from_end(res, txt) for res, txt in zip(out.get("RESULT",""), out.get("END_TEXT",""))]

    # RESULT_CLASS
    out["RESULT_CLASS"] = [classify_result(res, pts) for res, pts in zip(out.get("RESULT",""), out["POINTS"])]

    def _flip(team: str | None) -> str | None:
        if team == "MIA": return "BOS"
        if team == "BOS": return "MIA"
        return None

    # Second pass: fill TEAM using previous possession’s end-team + result semantics
    out = out.sort_values(["GAME_ID", "POSS_SEQ"]).reset_index(drop=True)

    # First, infer the end-team (who performed END_TEXT)
    end_team = [infer_team_from_text("", et) for et in out.get("END_TEXT", "")]
    out["_END_TEAM"] = end_team

    for i in range(1, len(out)):
        if pd.isna(out.at[i, "TEAM"]) or not out.at[i, "TEAM"]:
            prev_end_team = out.at[i - 1, "_END_TEAM"]
            prev_res = str(out.at[i - 1, "RESULT"]).lower()

            if prev_end_team in {"MIA", "BOS"}:
                if prev_res == "rebound_def":
                    # defense got the ball ⇒ next possession belongs to them
                    out.at[i, "TEAM"] = prev_end_team
                elif prev_res in {"made shot", "ft_end", "turnover"}:
                    # change of possession ⇒ next possession belongs to the other team
                    out.at[i, "TEAM"] = _flip(prev_end_team)

    # Clean temp
    out.drop(columns=["_END_TEAM"], inplace=True, errors="ignore")

    # EVENT_COUNT (alias)
    out["EVENT_COUNT"] = out["EVENTS"].astype(int)

    # EFFICIENCY (PPP)
    out["EFFICIENCY"] = out["POINTS"].astype(float)

    # ---- Margin tracking (HEAT perspective) ----
    out = out.sort_values(["GAME_ID", "POSS_SEQ"]).reset_index(drop=True)

    def margin_delta_row(team, pts):
        if pd.isna(team) or int(pts) == 0:
            return 0
        return int(pts) if str(team) == "MIA" else -int(pts)

    out["MARGIN_DELTA"] = [margin_delta_row(t, p) for t, p in zip(out.get("TEAM", ""), out["POINTS"])]

    # running margin BEFORE and AFTER each possession
    out["MARGIN_PRE"] = out.groupby("GAME_ID")["MARGIN_DELTA"].cumsum().shift(fill_value=0)
    out["MARGIN_POST"] = out["MARGIN_PRE"] + out["MARGIN_DELTA"]

    # Column order
    cols = [
        "GAME_ID","POSS_SEQ","PERIOD","TEAM",
        "START_TIME","END_TIME","DURATION_SEC",
        "EVENT_COUNT","RESULT","RESULT_CLASS","POINTS","EFFICIENCY",
        "START_TEXT","END_TEXT"
    ]
    cols = [c for c in cols if c in out.columns] + [c for c in out.columns if c not in cols]
    return out[cols]


def main() -> None:
    if not IN_csv.exists():
        raise SystemExit(f"Missing {IN_csv}. Run: python -m src.features.run_possessions")

    df = pd.read_csv(IN_csv)
    enr = enrich(df)
    enr.to_csv(OUT_csv, index=False)

    print(f"✅ Wrote {OUT_csv} ({len(enr)} rows)")
    print("RESULT_CLASS counts:\n", enr["RESULT_CLASS"].value_counts())
    if "TEAM" in enr.columns:
        print("\nTEAM counts:\n", enr["TEAM"].value_counts(dropna=False))
        print("\nPPP by TEAM:\n", enr.groupby("TEAM")["EFFICIENCY"].mean())
    print("\nPPP (overall):", round(enr["EFFICIENCY"].mean(), 3))



if __name__ == "__main__":
    main()