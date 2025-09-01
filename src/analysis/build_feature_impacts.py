from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PRED_ALL = Path("data/derived/event_epv_all.parquet")   # preferred (all events)
PRED_VAL = Path("data/derived/event_epv.parquet")       # fallback (val-only)
FEATS    = Path("data/derived/event_features.parquet")

OUT = Path("data/derived/feature_impacts.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

def _norm_game_id(df: pd.DataFrame, col: str = "GAME_ID") -> pd.DataFrame:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
                  .str.replace(r"\D", "", regex=True)
                  .str.zfill(10)
        )
    return df

def _load_predictions() -> pd.DataFrame:
    if PRED_ALL.exists():
        df = pd.read_parquet(PRED_ALL)
    elif PRED_VAL.exists():
        df = pd.read_parquet(PRED_VAL)
    else:
        raise SystemExit("No predictions found. Train EPV first.")
    return _norm_game_id(df)

def _load_features() -> pd.DataFrame:
    if not FEATS.exists():
        raise SystemExit("Missing event_features.parquet. Build features first.")
    df = pd.read_parquet(FEATS)
    return _norm_game_id(df)

def _ensure_bins(df: pd.DataFrame) -> pd.DataFrame:
    # EVENT_IDX_BIN: early / mid / late in possession
    if "EVENT_IDX" in df.columns and "EVENT_IDX_BIN" not in df.columns:
        cut = pd.cut(df["EVENT_IDX"],
                     bins=[-1, 2, 5, 9999],
                     labels=["early(0-2)", "mid(3-5)", "late(6+)"])
        df["EVENT_IDX_BIN"] = cut.astype("object").fillna("UNK")
    if "CLOCK_BIN" in df.columns:
        df["CLOCK_BIN"] = df["CLOCK_BIN"].astype("Int64").fillna(-1).astype(int)
    if "MARGIN_BIN" in df.columns:
        df["MARGIN_BIN"] = df["MARGIN_BIN"].astype("Int64").fillna(-1).astype(int)
    # BONUS flags as int
    for c in ["HOME_BONUS","AWAY_BONUS","OFF_BONUS"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def _bootstrap_ci_delta(epv_all: np.ndarray,
                        epv_group: np.ndarray,
                        B: int = 500,
                        alpha: float = 0.05) -> tuple[float,float]:
    """
    Return (lo, hi) of bootstrap CI for delta = mean(group) - mean(all).
    Resample both the group and the overall with replacement.
    """
    n_all   = len(epv_all)
    n_group = len(epv_group)
    if n_group == 0:
        return (np.nan, np.nan)

    deltas = []
    for _ in range(B):
        m_all   = epv_all[RNG.integers(0, n_all, n_all)].mean()
        m_group = epv_group[RNG.integers(0, n_group, n_group)].mean()
        deltas.append(m_group - m_all)
    q_lo = np.quantile(deltas, alpha/2)
    q_hi = np.quantile(deltas, 1 - alpha/2)
    return float(q_lo), float(q_hi)

def compute_impacts(df: pd.DataFrame,
                    feature: str,
                    min_n: int = 8,
                    B: int = 500) -> pd.DataFrame:
    """
    For a single feature, compute ΔEPV per value with bootstrap CIs.
    """
    if feature not in df.columns:
        return pd.DataFrame(columns=["feature","value","delta","delta_lo","delta_hi","n"])

    eps = df.dropna(subset=["EPV"]).copy()
    eps["val"] = eps[feature].astype("object")  # categorical safety
    base_mean = eps["EPV"].mean()
    out_rows = []

    # group stats
    gstats = (eps.groupby("val", dropna=False)["EPV"]
                .agg(mu="mean", n="size")
                .reset_index())

    epv_all = eps["EPV"].to_numpy()

    for _, row in gstats.iterrows():
        v = row["val"]
        n = int(row["n"])
        if n < min_n:
            continue
        mu = float(row["mu"])
        delta = mu - base_mean

        # bootstrap CI
        epv_group = eps.loc[eps["val"] == v, "EPV"].to_numpy()
        lo, hi = _bootstrap_ci_delta(epv_all, epv_group, B=B)

        out_rows.append({
            "feature": feature,
            "value":   str(v),
            "delta":   float(delta),
            "delta_lo": lo,
            "delta_hi": hi,
            "n":       n
        })
    return pd.DataFrame(out_rows)

def main():
    pred = _load_predictions()   # needs EPV + keys
    feats = _load_features()     # engineered categorical/numeric features

    # Merge: keys present in both
    keys = [k for k in ["GAME_ID","POSS_SEQ","EVENTNUM"] if k in pred.columns and k in feats.columns]
    if not keys:
        raise SystemExit("No shared keys between predictions and features.")
    df = pred.merge(feats, on=keys, how="left")

    # safety
    if "EPV" not in df.columns:
        raise SystemExit("Predictions missing EPV column.")
    df = _ensure_bins(df)

    # Choose a robust set; only keep those present
    candidates = [
        "EVTYPE",
        "CLOCK_BIN",
        "EVENT_IDX_BIN",
        "MARGIN_BIN",
        "OFF_BONUS",
        "HOME_BONUS",
        "AWAY_BONUS",
        "OFF_SIDE",
        "PERIOD",
    ]
    features = [f for f in candidates if f in df.columns]

    all_impacts = []
    for f in features:
        imp = compute_impacts(df, f, min_n=8, B=500)
        if not imp.empty:
            all_impacts.append(imp)

    if not all_impacts:
        print("No impacts computed (check that features exist and have support).")
        pd.DataFrame(columns=["feature","value","delta","delta_lo","delta_hi","n"]).to_csv(OUT, index=False)
        return

    out = pd.concat(all_impacts, ignore_index=True)
    out = out.sort_values(["feature","value"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    # quick console preview
    print("✅ Wrote", OUT, f"with {len(out)} rows.")
    print(out.head(12))

if __name__ == "__main__":
    main()
