from __future__ import annotations
from pathlib import Path
import pandas as pd

INP = Path("data/processed/possessions_enriched.csv")
OUT = Path("data/qa/findings.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def add(findings, poss_seq, kind, detail):
    findings.append({"POSS_SEQ": int(poss_seq), "TYPE": kind, "DETAIL": detail})

def main():
    if not INP.exists():
        raise SystemExit("Missing possessions_enriched.csv. Run: python -m src.features.enrich_possessions")

    df = pd.read_csv(INP)

    findings = []

    # 1) Core NA checks
    must_have = ["GAME_ID","POSS_SEQ","PERIOD","EVENT_COUNT","POINTS","RESULT_CLASS","DURATION_SEC","TEAM","MARGIN_DELTA","MARGIN_PRE","MARGIN_POST"]
    for c in must_have:
        if c not in df.columns:
            add(findings, -1, "MISSING_COLUMN", f"{c} not in file")

    # If critical columns exist, run per-row checks
    if all(c in df.columns for c in ["POSS_SEQ","EVENT_COUNT","POINTS","DURATION_SEC","RESULT","RESULT_CLASS"]):
        for _, r in df.iterrows():
            ps = r["POSS_SEQ"]

            # 2) At least one event
            if int(r["EVENT_COUNT"]) < 1:
                add(findings, ps, "EVENT_COUNT_LT1", f"EVENT_COUNT={r['EVENT_COUNT']}")

            # 3) Duration non-negative and not absurd
            if float(r["DURATION_SEC"]) < 0:
                add(findings, ps, "NEG_DURATION", f"DURATION_SEC={r['DURATION_SEC']}")
            if float(r["DURATION_SEC"]) > 400:  # > 6:40 is suspicious for a single possession
                add(findings, ps, "LONG_DURATION", f"DURATION_SEC={r['DURATION_SEC']}")

            # 4) Points bounds (allow 0–4, flag others)
            if int(r["POINTS"]) < 0 or int(r["POINTS"]) > 4:
                add(findings, ps, "POINTS_OUT_OF_RANGE", f"POINTS={r['POINTS']}")

            # 5) Result consistency
            if r["RESULT_CLASS"] == "score" and int(r["POINTS"]) == 0:
                add(findings, ps, "RESULT_MISMATCH", f"RESULT_CLASS=score but POINTS=0")
            if r["RESULT_CLASS"] == "turnover" and ("turnover" not in str(r.get("RESULT","")).lower()):
                # Soft warning: turnover class without end label turnover
                add(findings, ps, "TURNOVER_LABEL_MISMATCH", f"RESULT={r.get('RESULT','')}")

    # 6) Monotonicity: start <= end EVENTNUM if present
    if all(c in df.columns for c in ["START_EVENTNUM","END_EVENTNUM"]):
        bad = df[df["START_EVENTNUM"] > df["END_EVENTNUM"]]
        for _, r in bad.iterrows():
            add(findings, r["POSS_SEQ"], "EVENTNUM_ORDER", f"START_EVENTNUM={r['START_EVENTNUM']} > END_EVENTNUM={r['END_EVENTNUM']}")

    # Write findings
    out_df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["POSS_SEQ","TYPE","DETAIL"])
    out_df.to_csv(OUT, index=False)
    print(f"✅ Wrote {OUT} with {len(out_df)} finding(s).")

if __name__ == "__main__":
    main()
