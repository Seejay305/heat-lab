from __future__ import annotations
from pathlib import Path
import pandas as pd
from .possession_builder import build_possessions, normalize_event_type  # keep import of normalize for debug print

RAW = Path("data/raw/pbp/2023-24")   # or wherever your raw PBP lives
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        print("No raw PBP parquet files found. Put some in data/raw/pbp/2023-24/")
        return

    df = pd.read_parquet(files[0])

    # Debug: show EVTYPE coverage so we know splitting is healthy
    text_cols = ["HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]
    df["TEXT"] = df[text_cols].fillna("").agg(" ".join, axis=1).str.strip()
    df["EVTYPE"] = df["TEXT"].map(normalize_event_type)
    print("EVTYPE counts:\n", df["EVTYPE"].value_counts().head(20))


    mask = df["EVTYPE"] == "other"
    print("\nOTHER samples:")
    print(df.loc[mask, ["EVENTNUM", "PCTIMESTRING", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]]
          .head(15).to_string(index=False))

    poss = build_possessions(df)

    # Save BOTH parquet (canonical) and CSV (Streamlit-friendly)
    game_id = str(df["GAME_ID"].iloc[0])
    parquet_path = OUT_DIR / f"{game_id}_possessions.parquet"
    csv_path = OUT_DIR / "possessions.csv"

    poss.to_parquet(parquet_path, index=False)
    poss.to_csv(csv_path, index=False)

    print(f"Total possessions: {len(poss)}")
    print(poss["PERIOD"].value_counts().sort_index())
    print("Top long (by EVENTS):")
    print(poss.nlargest(5, "EVENTS")[['POSS_SEQ','EVENTS','START_TEXT','END_TEXT']])
    print(f"Saved parquet → {parquet_path}")
    print(f"Saved CSV     → {csv_path}")

if __name__ == "__main__":
    main()

 # python -m src.features.run_possessions to run
