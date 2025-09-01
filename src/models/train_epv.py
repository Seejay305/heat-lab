from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import lightgbm as lgb
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import OrdinalEncoder

INP = Path("data/derived/event_features.parquet")
O_PRED = Path("data/derived/event_epv.parquet")
O_PRED_ALL = Path("data/derived/event_epv_all.parquet")
O_CAL = Path("data/derived/calibration.csv")
O_IMP = Path("data/derived/feature_importance.csv")
O_CV = Path("data/derived/cv_metrics.csv")
O_MDL = Path("models/epv_lgb.txt")
for p in [O_PRED, O_PRED_ALL, O_CAL, O_IMP, O_CV, O_MDL]:
    p.parent.mkdir(parents=True, exist_ok=True)


def clip_points(y: pd.Series) -> np.ndarray:
    # Clamp to {0,1,2}. If you want 3-pt explicitly, change to 0..3 AND num_class=4 below.
    return np.clip(pd.to_numeric(y, errors="coerce").fillna(0).astype(int), 0, 2)


def make_design(df: pd.DataFrame):
    """Return X (numpy), y, enc, num_cols, cat_cols, meta."""
    df = df.copy()
    # Keep only labeled events
    df["TARGET_POINTS_AHEAD"] = pd.to_numeric(
        df["TARGET_POINTS_AHEAD"], errors="coerce"
    )
    df = df[df["TARGET_POINTS_AHEAD"].notna()].copy()

    by_game = df["GAME_ID"].value_counts().sort_index().to_dict()
    print("Labeled events by GAME_ID:", by_game, "| games:", len(by_game))

    df = df.sort_values(["GAME_ID", "POSS_SEQ", "EVENTNUM"]).reset_index(drop=True)

    # Ensure basics
    df["EVENT_IDX"] = df.groupby(["GAME_ID", "POSS_SEQ"]).cumcount()
    df["PERIOD"] = pd.to_numeric(df["PERIOD"], errors="coerce").fillna(0).astype(int)
    df["CLOCK_SEC"] = (
        pd.to_numeric(df.get("CLOCK_SEC", 0), errors="coerce").fillna(0).astype(int)
    )
    df["CLOCK_BIN"] = (
        pd.to_numeric(df.get("CLOCK_BIN", -1), errors="coerce").fillna(-1).astype(int)
    )
    df["MARGIN_PRE"] = (
        pd.to_numeric(df.get("MARGIN_PRE", 0), errors="coerce").fillna(0).astype(int)
    )
    df["MARGIN_POST"] = (
        pd.to_numeric(df.get("MARGIN_POST", 0), errors="coerce").fillna(0).astype(int)
    )

    # Lineup / flags
    for c in [
        "ON_COURT_MIA_COUNT",
        "ON_COURT_BOS_COUNT",
        "HERRO_ON",
        "ADEBAYO_ON",
        "HOME_BONUS",
        "AWAY_BONUS",
        "OFF_BONUS",
        "RCNT3_MAKE",
        "RCNT3_MISS",
        "RCNT3_TO",
    ]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # numeric feature set (team-agnostic + backward compat)
    num_cols = [
        "PERIOD",
        "CLOCK_SEC",
        "CLOCK_BIN",
        "EVENT_IDX",
        "MARGIN_PRE",
        "MARGIN_POST",
        # team-agnostic lineup counts
        "ON_COURT_OFF_COUNT",
        "ON_COURT_DEF_COUNT",
        # keep these if present (no harm if filled with 0)
        "ON_COURT_MIA_COUNT",
        "ON_COURT_BOS_COUNT",
        "HERRO_ON",
        "ADEBAYO_ON",
        # fouls / bonus context
        "HOME_BONUS",
        "AWAY_BONUS",
        "OFF_BONUS",
        # short memory
        "RCNT3_MAKE",
        "RCNT3_MISS",
        "RCNT3_TO",
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # --- categorical features ---
    cat_cols = ["EVTYPE", "BIGRAM", "TRIGRAM", "OFF_SIDE", "MARGIN_BIN"]

    for c in cat_cols:
        if c not in df.columns:
            df[c] = "UNK"

        # work on a copy to avoid chained-assign warnings
        col = df[c].copy()

        # if it's categorical, make sure "UNK" exists, then fillna
        if isinstance(col.dtype, CategoricalDtype):
            if "UNK" not in col.cat.categories:
                col = col.cat.add_categories(["UNK"])
            col = col.fillna("UNK").astype(str)
        else:
            # coerce to pandas string dtype first (handles mixed/object cleanly)
            col = col.astype("string").fillna("UNK").astype(str)

        df[c] = col

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = enc.fit_transform(df[cat_cols])

    X = np.hstack([df[num_cols].to_numpy(), X_cat])
    y = clip_points(df["TARGET_POINTS_AHEAD"])

    meta = df[
        ["GAME_ID", "POSS_SEQ", "EVENTNUM", "PERIOD", "CLOCK_SEC", "EVENT_IDX"]
    ].copy()
    return X, y, enc, num_cols, cat_cols, meta


def calibration_table(p_true, y_true, bins=10):
    q = pd.qcut(p_true, q=bins, duplicates="drop")
    tab = pd.DataFrame({"bin": q, "p": p_true, "y": y_true})
    agg = (
        tab.groupby("bin", observed=False)
        .agg(
            pred=("p", "mean"),
            emp=("y", "mean"),
            n=("y", "size"),
        )
        .reset_index()
    )
    return agg


def rolling_time_splits(n_rows: int, n_folds: int = 3, val_size: int = 101):
    """Grows train, fixed-size validation tail per fold."""
    splits = []
    while len(splits) < n_folds:
        n_tr = n_rows - (len(splits) + 1) * val_size
        if n_tr <= 0:
            break
        tr_idx = np.arange(n_tr)
        va_idx = np.arange(n_tr, min(n_tr + val_size, n_rows))
        if len(va_idx) == 0:
            break
        splits.append((tr_idx, va_idx))
    return splits


def main():
    if not INP.exists():
        raise SystemExit(f"Missing {INP}. Build event_features first.")

    df = pd.read_parquet(INP)
    X_all, y_all, enc, num_cols, cat_cols, meta_all = make_design(df)

    print("EF rows / games:", len(df), df["GAME_ID"].nunique())
    print("EF per game:", df["GAME_ID"].value_counts().sort_index().to_dict())
    print(
        "Non-null TARGET_POINTS_AHEAD per game:",
        df.groupby("GAME_ID")["TARGET_POINTS_AHEAD"]
        .apply(lambda s: s.notna().sum())
        .to_dict(),
    )

    # ----- rolling CV
    splits = rolling_time_splits(len(X_all), n_folds=3, val_size=101)
    rows_cv = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]

        # Safety: skip if val has only one class
        if len(np.unique(y_va)) < 2 or len(np.unique(y_tr)) < 2:
            print(f"[fold {fold}] Skipped (insufficient class variety).")
            continue

        print(f"X_tr shape: {X_tr.shape} X_va shape: {X_va.shape}")
        print("y_tr dist:", dict(pd.Series(y_tr).value_counts()))
        print("y_va dist:", dict(pd.Series(y_va).value_counts()))
        print(
            "X_tr finite?",
            np.isfinite(X_tr).all(),
            "X_va finite?",
            np.isfinite(X_va).all(),
        )

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        params = dict(
            objective="multiclass",
            num_class=3,  # matches clip_points 0..2
            metric="multi_logloss",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=40,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            verbose=-1,
            seed=42,
        )

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=100),
            ],
        )

        P = model.predict(X_va, num_iteration=model.best_iteration)
        p_any = 1.0 - P[:, 0]
        y_any = (y_va > 0).astype(int)

        logloss_bin = log_loss(y_any, np.clip(p_any, 1e-6, 1 - 1e-6))
        brier_bin = brier_score_loss(y_any, np.clip(p_any, 1e-6, 1 - 1e-6))
        epv = (P * np.arange(3)).sum(axis=1)

        rows_cv.append(
            {
                "fold": fold,
                "n_train": len(X_tr),
                "n_val": len(X_va),
                "val_logloss_any": round(logloss_bin, 4),
                "val_brier_any": round(brier_bin, 4),
                "val_mean_epv": round(float(epv.mean()), 4),
                "best_iter": int(model.best_iteration),
            }
        )

    if rows_cv:
        cv = pd.DataFrame(rows_cv)
        cv.to_csv(O_CV, index=False)
        print(f"✅ Wrote rolling-CV metrics → {O_CV}")
        print(cv.to_string(index=False))
    else:
        print("No CV rows written (all folds skipped?).")

    # ----- final model on last split
    if not splits:
        raise SystemExit("Not enough rows for a rolling split. Add more games.")
    tr_idx, va_idx = splits[-1]
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    params = dict(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=40,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        verbose=-1,
        seed=42,
    )

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100),
        ],
    )

    # Final val diagnostics
    P_va = model.predict(X_va, num_iteration=model.best_iteration)
    p_any = 1.0 - P_va[:, 0]  # assumes class 0 = "no score"
    y_any = (y_va > 0).astype(int)

    # Guard against single-class y_any on a fold
    try:
        logloss_bin = log_loss(
            y_any,
            np.clip(p_any, 1e-6, 1 - 1e-6),
            labels=[0, 1],  # <- key fix
        )
    except ValueError:
        logloss_bin = float("nan")

    brier_bin = brier_score_loss(y_any, np.clip(p_any, 1e-6, 1 - 1e-6))
    epv_va = (P_va * np.arange(P_va.shape[1])).sum(axis=1)

    print(
        f"[VAL] logloss(any-score)={logloss_bin:.4f}  "
        f"brier={brier_bin:.4f}  mean_EPV={epv_va.mean():.3f}"
    )

    # Calibration
    calibration_table(p_any, y_any, bins=10).to_csv(O_CAL, index=False)
    print(f"✅ Wrote calibration → {O_CAL}")

    # Per-event preds (val only)
    pred_val = meta_all.iloc[va_idx].copy()
    pred_val[["p0", "p1", "p2"]] = P_va
    pred_val["EPV"] = epv_va
    pred_val.to_parquet(O_PRED, index=False)
    print(f"✅ Wrote event EPV → {O_PRED} ({len(pred_val)} rows)")

    # ALL events (optional)
    P_all = model.predict(X_all, num_iteration=model.best_iteration)
    epv_all = (P_all * np.arange(3)).sum(axis=1)
    pred_all = meta_all.copy()
    pred_all[["p0", "p1", "p2"]] = P_all
    pred_all["EPV"] = epv_all
    pred_all.to_parquet(O_PRED_ALL, index=False)
    print(f"✅ Wrote ALL-event EPV → {O_PRED_ALL} ({len(pred_all)} rows)")

    # Feature importance
    imp = pd.DataFrame(
        {
            "feature": list(num_cols) + list(cat_cols),
            "gain": list(model.feature_importance(importance_type="gain")),
            "split": list(model.feature_importance(importance_type="split")),
        }
    )
    imp.to_csv(O_IMP, index=False)
    print(f"✅ Wrote feature importance → {O_IMP}")

    # Save model
    model.save_model(str(O_MDL))
    print(f"✅ Saved model → {O_MDL}  (best_iter={model.best_iteration})")


if __name__ == "__main__":
    main()
