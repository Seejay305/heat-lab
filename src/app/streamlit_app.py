# src/app/streamlit_app.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt
import glob

# -----------------------------
# Config & paths
# -----------------------------
st.set_page_config(page_title="Heat-Lab — Possession Analysis", layout="wide")

POSSESSIONS_CSV      = Path("data/processed/possessions.csv")
POSSESSIONS_ENRICHED = Path("data/processed/possessions_enriched.csv")
EVENT_EPV            = Path("data/derived/event_epv.parquet")
EVENT_EPV_ALL        = Path("data/derived/event_epv_all.parquet")  # optional
EVENT_FEATURES       = Path("data/derived/event_features.parquet") # optional
LINEUPS_EVENTS       = Path("data/derived/lineups_events.parquet") # optional
RAW_PBP_DIR          = Path("data/raw/pbp/2023-24")

CALIB_PATH           = Path("data/derived/calibration.csv")
FEAT_IMP_PATH        = Path("data/derived/feature_importance.csv")
FEAT_EFF_PATH        = Path("data/derived/feature_impacts.csv")    # optional


# -----------------------------
# Helpers (cached loaders)
# -----------------------------
@st.cache_data
def _norm_game_id(df: pd.DataFrame, col: str = "GAME_ID") -> pd.DataFrame:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
                  .str.replace(r"\D", "", regex=True)
                  .str.zfill(10)
        )
    return df

@st.cache_data
def load_possessions() -> pd.DataFrame:
    if POSSESSIONS_CSV.exists():
        df = pd.read_csv(POSSESSIONS_CSV)
    else:
        st.warning("Missing data/processed/possessions.csv — run the features pipeline.")
        return pd.DataFrame()
    return _norm_game_id(df)

@st.cache_data
def load_enriched() -> pd.DataFrame:
    if POSSESSIONS_ENRICHED.exists():
        df = pd.read_csv(POSSESSIONS_ENRICHED)
        return _norm_game_id(df)
    return pd.DataFrame()

@st.cache_data
def load_predictions() -> pd.DataFrame:
    if EVENT_EPV.exists():
        df = pd.read_parquet(EVENT_EPV)
        return _norm_game_id(df)
    return pd.DataFrame()

@st.cache_data
def load_event_features() -> pd.DataFrame:
    if EVENT_FEATURES.exists():
        df = pd.read_parquet(EVENT_FEATURES)
        return _norm_game_id(df)
    return pd.DataFrame()

@st.cache_data
def load_raw_pbp_for(game_id: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_PBP_DIR / "*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = _norm_game_id(df)
    df = df[df["GAME_ID"] == game_id].copy()
    for c in ["HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]:
        if c not in df.columns:
            df[c] = ""
    df["TEXT"] = df[["HOMEDESCRIPTION","VISITORDESCRIPTION","NEUTRALDESCRIPTION"]].fillna("").agg(" ".join, axis=1).str.strip()
    keep = [c for c in ["GAME_ID","PERIOD","EVENTNUM","PCTIMESTRING","TEXT"] if c in df.columns]
    return df[keep]

def compute_epv_delta(df_pred: pd.DataFrame) -> pd.DataFrame:
    if df_pred.empty:
        return df_pred
    df = df_pred.sort_values(["GAME_ID", "POSS_SEQ", "EVENT_IDX"])
    df["EPV_DELTA"] = df.groupby(["GAME_ID", "POSS_SEQ"])["EPV"].diff().fillna(0.0)
    return df

def _split5(x: str) -> list[str]:
    if isinstance(x, str) and x.strip():
        parts = [p.strip() for p in x.split(";") if p.strip()]
        return (parts + [""] * 5)[:5]
    return [""] * 5

def build_roster(df_any: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("MIA", "BOS"):
        col = f"ON_COURT_{side}"
        if col in df_any.columns:
            names = df_any[col].dropna().astype(str).apply(_split5)
            flat = pd.Series([n for row in names for n in row if n])
            if not flat.empty:
                rows.append(pd.DataFrame({"PLAYER": flat.unique(), "TEAM": side}))
    if rows:
        return (pd.concat(rows, ignore_index=True)
                  .drop_duplicates()
                  .sort_values(["TEAM","PLAYER"])
                  .reset_index(drop=True))
    if {"PLAYER","TEAM"} <= set(df_any.columns):
        return (df_any[["PLAYER","TEAM"]]
                  .drop_duplicates()
                  .sort_values(["TEAM","PLAYER"])
                  .reset_index(drop=True))
    return pd.DataFrame(columns=["PLAYER","TEAM"])


# -----------------------------
# Load data
# -----------------------------
poss = load_possessions()
enr  = load_enriched()
pred = compute_epv_delta(load_predictions())
ef   = load_event_features()

# Join labels into predictions (if present)
if not pred.empty and not enr.empty:
    keep = [c for c in ["GAME_ID","POSS_SEQ","PERIOD","RESULT_CLASS","POINTS"] if c in enr.columns]
    if keep:
        pred = pred.merge(enr[keep], on=["GAME_ID","POSS_SEQ"], how="left")

# Roster source
if LINEUPS_EVENTS.exists():
    try:
        roster_source = pd.read_parquet(LINEUPS_EVENTS)
        roster_source = _norm_game_id(roster_source)
    except Exception:
        roster_source = pred if not pred.empty else poss
else:
    roster_source = pred if not pred.empty else poss


# -----------------------------
# UI — Filters
# -----------------------------
st.title("Heat-Lab: Possession & EPV-Lite Explorer")

teams = ["All"]
if "TEAM" in poss.columns:
    teams = ["All"] + sorted([t for t in poss["TEAM"].dropna().unique().tolist() if t])

result_classes = ["All"]
if "RESULT_CLASS" in poss.columns:
    result_classes = ["All"] + sorted([c for c in poss["RESULT_CLASS"].dropna().unique().tolist() if c])

periods = ["All"]
if "PERIOD" in poss.columns:
    periods = ["All"] + sorted([int(x) for x in poss["PERIOD"].dropna().unique().tolist()])

colA, colB, colC = st.columns(3)
team_choice   = colA.selectbox("Team", teams, index=0, key="flt_team")
rc_choice     = colB.selectbox("Result class", result_classes, index=0, key="flt_rc")
period_choice = colC.selectbox("Period", periods, index=0, key="flt_period")

# Apply filters to a preview slice (non-blocking)
poss_view = poss.copy()
if team_choice != "All" and "TEAM" in poss_view.columns:
    poss_view = poss_view[poss_view["TEAM"] == team_choice]
if rc_choice != "All" and "RESULT_CLASS" in poss_view.columns:
    poss_view = poss_view[poss_view["RESULT_CLASS"] == rc_choice]
if period_choice != "All" and "PERIOD" in poss_view.columns:
    poss_view = poss_view[poss_view["PERIOD"] == int(period_choice)]

# -----------------------------
# Roster (dynamic)
# -----------------------------
st.subheader("Roster (dynamic from data)")
roster = build_roster(roster_source if roster_source is not None else poss_view)
if roster.empty:
    st.info("No roster columns found yet (ON_COURT_* or PLAYER/TEAM). This will populate once lineups are merged.")
else:
    st.dataframe(roster, use_container_width=True, hide_index=True)

# -----------------------------
# Co-presence picker (A while B on court)
# -----------------------------
st.subheader("Co-presence filter (prototype)")
players = ["None"] + (roster["PLAYER"].dropna().unique().tolist() if not roster.empty else [])
c1, c2 = st.columns(2)
_ = c1.selectbox("Focus player (A)", options=players, index=0, key="co_A")
_ = c2.selectbox("With player (B)", options=players, index=0, key="co_B")
st.caption("Hook this to event-level ON_COURT_* in a later iteration to truly filter possessions by co-presence.")

# -----------------------------
# EPV Scrubber
# -----------------------------
st.subheader("EPV Scrubber")

if pred.empty:
    st.info("No EPV predictions found. Train the model first: `python -m src.models.train_epv`.")
else:
    # Game selector
    game_ids = sorted(pred["GAME_ID"].dropna().unique().tolist())
    sel_game = st.selectbox("Game", options=game_ids, index=0, key="epv_game")

    # Slice predictions for this game
    pred_g = pred[pred["GAME_ID"] == sel_game].copy()

    # Load raw PBP for hover
    pbp_g = load_raw_pbp_for(sel_game)

    # Possession selector with robust labeling
    need_cols = ["POSS_SEQ","PERIOD","RESULT_CLASS","POINTS"]
    have_cols = [c for c in need_cols if c in pred_g.columns]
    opts = pred_g[have_cols].drop_duplicates()
    sort_cols = [c for c in ["PERIOD","POSS_SEQ"] if c in opts.columns]
    if sort_cols:
        opts = opts.sort_values(sort_cols)

    def _get(r, col, default=None):
        return r[col] if (col in r.index and pd.notna(r[col])) else default

    def _fmt(r):
        possq = int(_get(r,"POSS_SEQ",-1) or -1)
        q = f"Q{int(_get(r,'PERIOD',0))}" if _get(r,"PERIOD") is not None else "Q?"
        rc = str(_get(r,"RESULT_CLASS","—"))
        pts = int(_get(r,"POINTS",0) or 0)
        return f"POSS {possq} | {q} | {rc} ({pts} pts)"

    if opts.empty:
        st.warning("No possession metadata found; selecting by POSS_SEQ only.")
        sel_poss = st.selectbox("Possession", options=sorted(pred_g["POSS_SEQ"].unique().tolist()), index=0, key="epv_poss_raw")
    else:
        labels = opts.apply(_fmt, axis=1).tolist()
        mapping = dict(zip(labels, opts["POSS_SEQ"].tolist()))
        sel_label = st.selectbox("Possession", options=labels, index=0, key="epv_poss_lbl")
        sel_poss = mapping[sel_label]

    # Build trace & enrich for hover
    trace = pred_g[pred_g["POSS_SEQ"] == sel_poss].sort_values("EVENT_IDX").copy()
    if not pbp_g.empty and "EVENTNUM" in trace.columns:
        trace = trace.merge(pbp_g[["EVENTNUM","PCTIMESTRING","TEXT"]], on="EVENTNUM", how="left")

    # Bring EVTYPE from event_features if available
    if not ef.empty:
        ef_small = ef.loc[(ef["GAME_ID"] == sel_game) & (ef["POSS_SEQ"] == sel_poss),
                          ["GAME_ID","POSS_SEQ","EVENTNUM","EVTYPE"]]
        trace = trace.merge(ef_small, on=["GAME_ID","POSS_SEQ","EVENTNUM"], how="left")

    # Plot EPV curve
    if not trace.empty and "EPV" in trace.columns:
        trace["EVENT_IDX"] = trace["EVENT_IDX"].astype(int)
        tooltip_cols = [c for c in ["EVENT_IDX","EPV","EPV_DELTA","EVTYPE","PCTIMESTRING","TEXT"] if c in trace.columns]
        chart = (
            alt.Chart(trace)
            .mark_line(point=True)
            .encode(
                x=alt.X("EVENT_IDX:Q", title="Event index in possession"),
                y=alt.Y("EPV:Q", title="Expected Points"),
                color=alt.Color("EVTYPE:N", title="Event", legend=None) if "EVTYPE" in trace.columns else alt.value(
                    None),
                tooltip=tooltip_cols
            )
            .properties(height=260, width="container")
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No EPV curve for this selection.")

    # Top EPV movers within this possession
    if "EPV_DELTA" in trace.columns:
        t2 = trace.copy()
        t2["ABS_DELTA"] = t2["EPV_DELTA"].abs()
        cols = ["EVENT_IDX","EVENTNUM","PCTIMESTRING","EVTYPE","EPV","EPV_DELTA","TEXT"]
        cols = [c for c in cols if c in t2.columns]
        st.caption("Top |ΔEPV| within this possession")
        st.dataframe(t2.sort_values("ABS_DELTA", ascending=False).head(8)[cols],
                     use_container_width=True, hide_index=True)

    # Game-level swings
    st.subheader("Game-level EPV swings")
    if not pred_g.empty and "EPV_DELTA" in pred_g.columns:
        g2 = pred_g.copy()
        g2["ABS_DELTA"] = g2["EPV_DELTA"].abs()
        if not pbp_g.empty and "EVENTNUM" in g2.columns:
            g2 = g2.merge(pbp_g[["EVENTNUM","PCTIMESTRING","TEXT"]], on="EVENTNUM", how="left")
        # try to attach EVTYPE if missing
        if "EVTYPE" not in g2.columns and not ef.empty:
            g2 = g2.merge(ef_small[["EVENTNUM","EVTYPE"]], on="EVENTNUM", how="left")
        cols = [c for c in ["POSS_SEQ","EVENT_IDX","EVENTNUM","PCTIMESTRING","EVTYPE","EPV_DELTA","TEXT"] if c in g2.columns]
        st.dataframe(g2.sort_values("ABS_DELTA", ascending=False).head(15)[cols],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No EPV deltas available for game-level swings.")

# -----------------------------
# Calibration
# -----------------------------
st.subheader("Calibration: P(any points)")

if CALIB_PATH.exists():
    cal = pd.read_csv(CALIB_PATH)
    # expected: pred, emp, n
    if {"pred","emp"} <= set(cal.columns):
        line = pd.DataFrame({"x":[0,1],"y":[0,1]})
        chart_cal = alt.layer(
            alt.Chart(cal).mark_line(point=True).encode(
                x=alt.X("pred:Q", title="Predicted P(score remainder of possession)"),
                y=alt.Y("emp:Q",  title="Empirical frequency"),
                tooltip=[c for c in ["pred","emp","n"] if c in cal.columns]
            ),
            alt.Chart(line).mark_line(strokeDash=[4,4]).encode(x="x:Q", y="y:Q")
        ).properties(height=260, width="container")
        st.altair_chart(chart_cal, use_container_width=True)
    else:
        st.info("calibration.csv is missing required columns.")
else:
    st.info("No calibration.csv yet — run the trainer to generate it.")

# -----------------------------
# QA: End-of-possession EPV vs actual points
# -----------------------------
st.subheader("QA: End-of-possession EPV vs actual points")
if not pred.empty and not enr.empty and "POINTS" in enr.columns:
    last_epv = (
        pred.sort_values(["GAME_ID","POSS_SEQ","EVENT_IDX"])
            .groupby(["GAME_ID","POSS_SEQ"])
            .tail(1)[["GAME_ID","POSS_SEQ","EPV"]]
            .rename(columns={"EPV":"EPV_END"})
    )
    comp = last_epv.merge(
        enr[["GAME_ID","POSS_SEQ","POINTS","RESULT_CLASS"]].drop_duplicates(),
        on=["GAME_ID","POSS_SEQ"], how="left"
    ).dropna(subset=["POINTS"])
    if comp.empty:
        st.info("No overlap between predictions and enriched possessions.")
    else:
        comp["RESIDUAL"] = comp["EPV_END"] - comp["POINTS"]
        chart_sc = (
            alt.Chart(comp)
            .mark_circle()
            .encode(
                x=alt.X("EPV_END:Q", title="End EPV"),
                y=alt.Y("POINTS:Q",  title="Actual points"),
                tooltip=["GAME_ID","POSS_SEQ","EPV_END","POINTS","RESULT_CLASS"]
            ).properties(height=260, width="container")
        )
        st.altair_chart(chart_sc, use_container_width=True)
        mae  = comp["RESIDUAL"].abs().mean()
        bias = comp["RESIDUAL"].mean()
        st.write(f"**MAE (|EPV−points|):** {mae:.3f}  |  **Bias (EPV−points):** {bias:+.3f}")

        worst = comp.reindex(comp["RESIDUAL"].abs().sort_values(ascending=False).index).head(10)
        st.caption("Largest absolute mismatches")
        st.dataframe(worst[["GAME_ID","POSS_SEQ","EPV_END","POINTS","RESULT_CLASS","RESIDUAL"]],
                     use_container_width=True, hide_index=True)
else:
    st.info("Need predictions and possessions_enriched with POINTS for this QA.")

# -----------------------------
# Feature Importance (robust)
# -----------------------------
st.subheader("Feature importance (LightGBM gain)")
if not FEAT_IMP_PATH.exists():
    st.info("Train the model to produce feature_importance.csv.")
else:
    imp = pd.read_csv(FEAT_IMP_PATH)
    # Normalize expected columns
    if "feature" not in imp.columns:
        st.warning("feature_importance.csv is missing a 'feature' column.")
    if "gain" not in imp.columns:
        if "importance" in imp.columns:
            imp["gain"] = imp["importance"]
        else:
            imp["gain"] = 0

    imp["gain"] = pd.to_numeric(imp["gain"], errors="coerce")
    if "split" in imp.columns:
        imp["split"] = pd.to_numeric(imp["split"], errors="coerce")

    imp = imp.dropna(subset=["gain"]).sort_values("gain", ascending=False)
    if imp.empty:
        st.info("No valid rows to plot in feature_importance.csv.")
    else:
        st.dataframe(
            imp[["feature","gain"] + (["split"] if "split" in imp.columns else [])],
            use_container_width=True, hide_index=True
        )
        tooltip_cols = ["feature","gain"] + (["split"] if "split" in imp.columns else [])
        chart_imp = (
            alt.Chart(imp.head(20))
            .mark_bar()
            .encode(
                x=alt.X("gain:Q", title="Gain"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=tooltip_cols
            )
            .properties(height=min(24*len(imp.head(20)) + 40, 520))
        )
        st.altair_chart(chart_imp, use_container_width=True)

# -----------------------------
# Feature Impacts / Partial Effects (robust)
# -----------------------------
st.subheader("Feature impacts (ΔEPV vs baseline)")
if not FEAT_EFF_PATH.exists():
    st.info("No feature_impacts.csv found. Generate it with your analysis script.")
else:
    eff = pd.read_csv(FEAT_EFF_PATH)
    # Expect at least: feature, value, delta (or delta_epv)
    if "delta" not in eff.columns:
        if "delta_epv" in eff.columns:
            eff["delta"] = eff["delta_epv"]
        else:
            st.warning("feature_impacts.csv is missing a 'delta' (or 'delta_epv') column.")
            eff["delta"] = 0

    # Coerce types
    for c in ["delta"]:
        eff[c] = pd.to_numeric(eff[c], errors="coerce")
    if "value" not in eff.columns:
        eff["value"] = eff.get("bin", eff.get("category", "UNK"))

    eff = eff.dropna(subset=["delta"])
    if eff.empty:
        st.info("No valid rows in feature_impacts.csv to plot.")
    else:
        # Choose one top feature (by mean |delta|) and plot its curve; allow user selection
        ranks = (eff.groupby("feature")["delta"]
                   .apply(lambda s: s.abs().mean())
                   .sort_values(ascending=False))
        top_feats = ranks.index.tolist()
        sel_feat = st.selectbox("Select feature", options=top_feats, index=0, key="impact_feat")

        eff_f = eff[eff["feature"] == sel_feat].copy()
        # Attempt to sort 'value' smartly
        try:
            eff_f["_order"] = pd.to_numeric(eff_f["value"], errors="coerce")
        except Exception:
            eff_f["_order"] = None
        eff_f = eff_f.sort_values(["_order","value"])

        chart_eff = (
            alt.Chart(eff_f)
            .mark_line(point=True)
            .encode(
                x=alt.X("value:N", title=f"{sel_feat}"),
                y=alt.Y("delta:Q", title="Δ EPV"),
                tooltip=["feature","value","delta"]
            )
            .properties(height=260, width="container")
        )
        st.altair_chart(chart_eff, use_container_width=True)
        st.caption("Interpretation: positive Δ means higher expected points when this feature value holds, holding others roughly constant.")

    st.subheader("Rolling CV (time-aware)")
    cv_path = Path("data/derived/cv_metrics.csv")
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        st.dataframe(cv.round(4), use_container_width=True, hide_index=True)
        st.write(
            "**Avg logloss(any):**",
            f"{cv['val_logloss_any'].mean():.4f}",
            " | **Avg Brier:**",
            f"{cv['val_brier_any'].mean():.4f}"
        )
    else:
        st.info("Run training to generate rolling CV metrics.")




