# src/models/train_epv.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb

INP   = Path("data/derived/event_features.parquet")
O_PRED = Path("data/derived/event_epv.parquet")
O_PRED_ALL = Path("data/derived/event_epv_all.parquet")
O_CAL  = Path("data/derived/calibration.csv")
O_MDL  = Path("models/epv_lgb.txt")
O_IMP  = Path("data/derived/feature_importance.csv")
O_CV   = Path("data/derived/cv_metrics.csv")

for p in [O_PRED.parent, O_MDL.parent]:
    p.mkdir(parents=True, exist_ok=True)

# -------------------------
# helpers
# -------------------------
def clip_points(y):
    return np.clip(y.astype(int), 0, 3)

def _make_basic_features(df: pd.DataFrame):
    """Builds X (numpy), y (numpy), cat encoder, numeric/cat col names, and sorted df index."""
    needed = ["GAME_ID","POSS_SEQ","EVENTNUM","PERIOD","CLOCK_SEC","CLOCK_BIN",
              "EVTYPE","MARGIN_PRE","MARGIN_POST","TARGET_POINTS_AHEAD"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise SystemExit(f"event_features missing {miss}")

    df = df.copy()
    df = df.sort_values(["GAME_ID","POSS_SEQ","EVENTNUM"]).reset_index(drop=True)

    # Stable ints
    df["PERIOD"]       = pd.to_numeric(df["PERIOD"], errors="coerce").fillna(0).astype(int)
    df["CLOCK_SEC"]    = pd.to_numeric(df["CLOCK_SEC"], errors="coerce").fillna(0).astype(int)
    df["CLOCK_BIN"]    = pd.to_numeric(df["CLOCK_BIN"], errors="coerce").fillna(-1).astype(int)
    df["MARGIN_PRE"]   = pd.to_numeric(df["MARGIN_PRE"], errors="coerce").fillna(0).astype(int)
    df["MARGIN_POST"]  = pd.to_numeric(df["MARGIN_POST"], errors="coerce").fillna(0).astype(int)
    df["EVENT_IDX"]    = df.groupby(["GAME_ID","POSS_SEQ"]).cumcount()

    # Optional/nice numeric features (present in your latest build)
    num_cols = [
        "PERIOD","CLOCK_SEC","CLOCK_BIN","EVENT_IDX",
        "MARGIN_PRE","MARGIN_POST",
        "HOME_BONUS","AWAY_BONUS","OFF_BONUS",
        "RCNT3_MAKE","RCNT3_MISS","RCNT3_TO",
    ]
    for c in num_cols:
        if c not in df.columns: df[c] = 0
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Categorical features (robust to missing)
    cat_cols = ["EVTYPE","BIGRAM","TRIGRAM","OFF_SIDE","MARGIN_BIN"]
    for c in cat_cols:
        if c not in df.columns: df[c] = "UNK"

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = enc.fit_transform(df[cat_cols])

    X_num = df[num_cols].to_numpy()
    X = np.hstack([X_num, X_cat])
    y = clip_points(df["TARGET_POINTS_AHEAD"].fillna(0))

    return df, X, y, enc, num_cols, cat_cols

def _any_score_metrics(P, y):
    """Binary any-score sanity metrics from multiclass probs."""
    p_any = 1.0 - P[:,0]
    y_any = (y > 0).astype(int)
    ll = log_loss(y_any, np.clip(p_any, 1e-6, 1-1e-6))
    br = brier_score_loss(y_any, np.clip(p_any, 1e-6, 1-1e-6))
    epv = (P @ np.arange(P.shape[1])).mean()
    return ll, br, epv

def _train_one(X_tr, y_tr, X_va, y_va, params, num_round=2000):
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        valid_sets=[dtr, dva],
        valid_names=["train","val"],
        num_boost_round=num_round,
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)],
    )
    P_val = model.predict(X_va, num_iteration=model.best_iteration)
    return model, P_val

def _rolling_folds(n_rows: int, n_splits: int = 5):
    """Indices for time-aware rolling CV: earlier rows train, later rows validate."""
    # require at least 2*splits chunks; fallback if tiny
    if n_rows < n_splits + 1:
        n_splits = max(2, min(3, n_rows // 5 or 2))
    fold_edges = np.linspace(0, n_rows, n_splits + 1, dtype=int)
    folds = []
    for k in range(1, len(fold_edges)):
        end = fold_edges[k]
        start = fold_edges[k-1]
        if start == 0:     # avoid empty train
            continue
        tr_idx = np.arange(0, start)
        va_idx = np.arange(start, end)
        if len(va_idx) == 0:  # skip empty val
            continue
        folds.append((tr_idx, va_idx))
    return folds

# -------------------------
# main
# -------------------------
def main():
    if not INP.exists():
        raise SystemExit(f"Missing {INP}. Build event_features first.")

    df0 = pd.read_parquet(INP)
    df, X, y, enc, num_cols, cat_cols = _make_basic_features(df0)

    # LightGBM params
    params = dict(
        objective="multiclass",
        num_class=4,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=40,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        seed=42,
        verbose=-1,
    )

    # ========= Rolling CV (time-aware) =========
    rows = len(df)
    folds = _rolling_folds(rows, n_splits=5)
    cv_records = []

    for i, (tr_idx, va_idx) in enumerate(folds, start=1):
        model_i, P_va = _train_one(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], params)
        ll, br, epv = _any_score_metrics(P_va, y[va_idx])
        cv_records.append({
            "fold": i,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "val_logloss_any": ll,
            "val_brier_any": br,
            "val_mean_epv": epv,
            "best_iter": model_i.best_iteration,
        })

    if cv_records:
        cv_df = pd.DataFrame(cv_records)
        cv_df.to_csv(O_CV, index=False)
        print(f"✅ Wrote rolling-CV metrics → {O_CV}")
        print(cv_df.round(4).to_string(index=False))

    # ========= Original single train/val (keep behavior) =========
    # chronological split: 80/20
    cut = int(0.8 * rows)
    tr_idx = np.arange(0, cut)
    va_idx = np.arange(cut, rows)

    model, P_val = _train_one(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], params)

    ll, br, epv = _any_score_metrics(P_val, y[va_idx])
    print(f"[VAL] logloss(any-score)={ll:.4f}  brier={br:.4f}  mean_EPV={epv:.3f}")

    # calibration (any-score)
    p_any = 1.0 - P_val[:,0]
    y_any = (y[va_idx] > 0).astype(int)
    q = pd.qcut(p_any, q=10, duplicates="drop")
    tab = pd.DataFrame({"bin": q, "p": p_any, "y": y_any})
    cal = tab.groupby("bin", observed=False).agg(pred=("p","mean"), emp=("y","mean"), n=("y","size")).reset_index()
    cal.to_csv(O_CAL, index=False)
    print(f"✅ Wrote calibration → {O_CAL}")

    # per-event predictions (val split) for app
    pred = df.loc[va_idx, ["GAME_ID","POSS_SEQ","EVENTNUM","PERIOD","CLOCK_SEC","EVENT_IDX"]].copy()
    pred[["p0","p1","p2","p3"]] = P_val
    pred["EPV"] = (P_val * np.arange(4)).sum(axis=1)
    pred.to_parquet(O_PRED, index=False)
    print(f"✅ Wrote event EPV → {O_PRED} ({len(pred)} rows)")

    # ALL-event predictions (optional, slower but useful for analyses)
    P_all = model.predict(X, num_iteration=model.best_iteration)
    pred_all = df[["GAME_ID","POSS_SEQ","EVENTNUM","PERIOD","CLOCK_SEC","EVENT_IDX"]].copy()
    pred_all[["p0","p1","p2","p3"]] = P_all
    pred_all["EPV"] = (P_all * np.arange(4)).sum(axis=1)
    pred_all.to_parquet(O_PRED_ALL, index=False)
    print(f"✅ Wrote ALL-event EPV → {O_PRED_ALL} ({len(pred_all)} rows)")

    # feature importance
    imp = pd.DataFrame({
        "feature": num_cols + cat_cols,
        "gain":    model.feature_importance(importance_type="gain").tolist()
    })
    imp.to_csv(O_IMP, index=False)
    print(f"✅ Wrote feature importance → {O_IMP}")

    # save model
    model.save_model(str(O_MDL))
    print(f"✅ Saved model → {O_MDL}  (best_iter={model.best_iteration})")

if __name__ == "__main__":
    main()
