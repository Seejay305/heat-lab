from __future__ import annotations
from pathlib import Path
import random
import pandas as pd

RAW_DIR = Path("data/raw")
PBP_DIR = RAW_DIR / "pbp"
DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)

REQUIRED_COLS = [
    "GAME_ID","EVENTNUM","PERIOD","PCTIMESTRING",
    "HOMEDESCRIPTION","VISITORDESCRIPTION","NEUTRALDESCRIPTION"
]

def list_pbp_files() -> dict[str, list[Path]]:
    seasons: dict[str, list[Path]] = {}
    if not PBP_DIR.exists():
        return seasons
    for season_dir in sorted(PBP_DIR.iterdir()):
        if season_dir.is_dir():
            files = sorted(season_dir.glob("*.parquet"))
            if files:
                seasons[season_dir.name] = files
    return seasons

def read_pbp(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def check_event_order(df: pd.DataFrame) -> bool:
    s = pd.to_numeric(df["EVENTNUM"], errors="coerce")
    return (s.diff().fillna(1) >= 0).all()

def missing_required_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLS if c not in df.columns]

def null_report(df: pd.DataFrame) -> dict[str, float]:
    cols = ["PERIOD","PCTIMESTRING","HOMEDESCRIPTION","VISITORDESCRIPTION","NEUTRALDESCRIPTION"]
    return {c: float(df[c].isna().mean()) for c in cols if c in df.columns}

def dup_count(df: pd.DataFrame) -> int:
    if not {"GAME_ID","EVENTNUM"}.issubset(df.columns):
        return -1
    return int(df.duplicated(subset=["GAME_ID","EVENTNUM"]).sum())

def write_report(text: str) -> Path:
    out = DOCS / "dqr.md"
    out.write_text(text, encoding="utf-8")
    return out

def main() -> None:
    seasons = list_pbp_files()
    lines: list[str] = []
    lines.append("# Data Quality Report (v1)\n")

    if not seasons:
        lines.append("No pbp files found. Run the ingest script first.\n")
        write_report("\n".join(lines)); print("No pbp files found."); return

    total = 0
    for s, files in seasons.items():
        lines.append(f"## Season {s}")
        lines.append(f"- Files: {len(files)}")
        total += len(files)

        # sample a few files for deeper checks
        sample = random.sample(files, k=min(3, len(files)))
        for p in sample:
            try:
                df = read_pbp(p)
                miss = missing_required_cols(df)
                mono = check_event_order(df) if not miss else False
                dups = dup_count(df)
                nulls = null_report(df)
                lines.append(f"  - {p.name}:")
                lines.append(f"    - missing columns: {miss if miss else 'none'}")
                lines.append(f"    - EVENTNUM monotonic: {mono}")
                lines.append(f"    - duplicate (GAME_ID, EVENTNUM): {dups}")
                if nulls:
                    pretty = ', '.join(f'{k}={v:.3f}' for k,v in nulls.items())
                    lines.append(f"    - null rates: {pretty}")
            except Exception as e:
                lines.append(f"  - {p.name}: read error: {e}")
        lines.append("")
    lines.append(f"**Total PBP parquet files:** {total}\n")
    out = write_report("\n".join(lines))
    print(f"✅ Wrote {out}")

if __name__ == "__main__":
    main()
