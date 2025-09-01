# src/app/streamlit_app.py
from __future__ import annotations

from pathlib import Path
import glob
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------
# Config & paths
# -----------------------------
st.set_page_config(page_title="Heat-Lab — Possession Analysis", layout="wide")

POSSESSIONS_CSV = Path("data/processed/possessions.csv")
POSSESSIONS_ENRICH = Path("data/processed/possessions_enriched.csv")
EVENT_EPV = Path("data/derived/event_epv.parquet")
EVENT_EPV_ALL = Path("data/derived/event_epv_all.parquet")
EVENT_FEATURES = Path("data/derived/event_features.parquet")
LINEUPS_EVENTS = Path("data/derived/lineups_events.parquet")
RAW_PBP_DIR = Path("data/raw/pbp/2023-24")

CALIB_PATH = Path("data/derived/calibration.csv")
FEAT_IMP_PATH = Path("data/derived/feature_importance.csv")
FEAT_EFF_PATH = Path("data/derived/feature_impacts.csv")  # optional


# -----------------------------
# Helpers (cached loaders)
# -----------------------------
@st.cache_data
def _norm_game_id(df: pd.DataFrame, col: str = "GAME_ID") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
    return df


@st.cache_data
def _harmonize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "GAME_ID" in df.columns:
        df["GAME_ID"] = (
            df["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
        )
    if "POSS_SEQ" in df.columns:
        df["POSS_SEQ"] = pd.to_numeric(df["POSS_SEQ"], errors="coerce").astype("Int64")
    if "PERIOD" in df.columns:
        df["PERIOD"] = pd.to_numeric(df["PERIOD"], errors="coerce").astype("Int64")
    return df


@st.cache_data
def load_possessions() -> pd.DataFrame:
    if POSSESSIONS_CSV.exists():
        return _harmonize_keys(pd.read_csv(POSSESSIONS_CSV))
    st.warning("Missing data/processed/possessions.csv — run the features pipeline.")
    return pd.DataFrame()


@st.cache_data
def load_enriched() -> pd.DataFrame:
    if POSSESSIONS_ENRICH.exists():
        return _harmonize_keys(pd.read_csv(POSSESSIONS_ENRICH))
    return pd.DataFrame()


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if EVENT_EPV.exists():
        return _harmonize_keys(pd.read_parquet(EVENT_EPV))
    return pd.DataFrame()


@st.cache_data
def load_predictions_all() -> pd.DataFrame:
    if EVENT_EPV_ALL.exists():
        return _harmonize_keys(pd.read_parquet(EVENT_EPV_ALL))
    return pd.DataFrame()


@st.cache_data
def load_event_features() -> pd.DataFrame:
    if EVENT_FEATURES.exists():
        return _harmonize_keys(pd.read_parquet(EVENT_FEATURES))
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
    df["TEXT"] = (
        df[["HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.strip()
    )
    keep = [
        c
        for c in ["GAME_ID", "PERIOD", "EVENTNUM", "PCTIMESTRING", "TEXT"]
        if c in df.columns
    ]
    return df[keep]


def compute_epv_delta(df_pred: pd.DataFrame) -> pd.DataFrame:
    if df_pred.empty:
        return df_pred
    df = df_pred.sort_values(["GAME_ID", "POSS_SEQ", "EVENT_IDX"])
    df["EPV_DELTA"] = df.groupby(["GAME_ID", "POSS_SEQ"])["EPV"].diff().fillna(0.0)
    return df


def _is_flat_epv(series, min_std=0.02, min_unique=3):
    """Heuristic: EPV is 'flat' if very low variance OR almost constant."""
    if series is None or len(series) == 0:
        return True
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return True
    return (s.std() < min_std) or (s.nunique() < min_unique)


def _split5(x: str) -> list[str]:
    if isinstance(x, str) and x.strip():
        parts = [p.strip() for p in x.split(";") if p.strip()]
        return (parts + [""] * 5)[:5]
    return [""] * 5


def build_roster(df_any: pd.DataFrame) -> pd.DataFrame:
    # accept any ON_COURT_* column (HOME/AWAY or team abbr if present)
    on_cols = [c for c in df_any.columns if c.startswith("ON_COURT_")]
    rows = []
    for col in on_cols:
        side = col.replace("ON_COURT_", "")
        names = df_any[col].dropna().astype(str).apply(_split5)
        flat = pd.Series([n for row in names for n in row if n])
        if not flat.empty:
            rows.append(pd.DataFrame({"PLAYER": flat.unique(), "TEAM": side}))
    if rows:
        return (
            pd.concat(rows, ignore_index=True)
            .drop_duplicates()
            .sort_values(["TEAM", "PLAYER"])
            .reset_index(drop=True)
        )
    # fallback if a PLAYER/TEAM pair exists
    if {"PLAYER", "TEAM"} <= set(df_any.columns):
        return (
            df_any[["PLAYER", "TEAM"]]
            .drop_duplicates()
            .sort_values(["TEAM", "PLAYER"])
            .reset_index(drop=True)
        )
    return pd.DataFrame(columns=["PLAYER", "TEAM"])


# -----------------------------
# Load data
# -----------------------------
poss = load_possessions()
enr = load_enriched()
pred = compute_epv_delta(load_predictions())  # val fold only
pred_all = compute_epv_delta(load_predictions_all())  # ALL events
ef = load_event_features()

st.caption(
    f"EF rows/games: {len(ef)} / {ef['GAME_ID'].nunique() if 'GAME_ID' in ef.columns else 0} | "
    f"Pred ALL rows/games: {len(pred_all)} / {pred_all['GAME_ID'].nunique() if 'GAME_ID' in pred_all.columns else 0}"
)

# Join labels (PERIOD / RESULT_CLASS / POINTS) into ALL-event predictions for UI
if not pred_all.empty and not enr.empty:
    keep = [
        c
        for c in ["GAME_ID", "POSS_SEQ", "PERIOD", "RESULT_CLASS", "POINTS"]
        if c in enr.columns
    ]
    if keep:
        pred_all = pred_all.merge(
            enr[keep].drop_duplicates(subset=["GAME_ID", "POSS_SEQ"]),
            on=["GAME_ID", "POSS_SEQ"],
            how="left",
            validate="m:1",
        )

# Pick a unified game source (prefer enriched -> poss -> pred_all -> pred -> ef)
_game_source = None
for cand in [enr, poss, pred_all, pred, ef]:
    if cand is not None and not cand.empty and "GAME_ID" in cand.columns:
        _game_source = cand
        break

st.title("Heat-Lab: Possession & EPV-Lite Explorer")

if _game_source is None or _game_source.empty:
    st.info("No data found to populate games. Train or rebuild features first.")
    st.stop()

game_ids = (
    _game_source["GAME_ID"]
    .astype(str)
    .str.replace(r"\D", "", regex=True)
    .str.zfill(10)
    .dropna()
    .unique()
    .tolist()
)
game_ids = sorted(game_ids)
sel_game = st.selectbox("Game", options=game_ids, index=0, key="game_picker")

# Choose table with TEAM/RESULT_CLASS for filters
flt_src = (
    enr if (not enr.empty and {"TEAM", "RESULT_CLASS"} <= set(enr.columns)) else poss
)

# -----------------------------
# UI — Filters (preview table)
# -----------------------------
# --- Build Team options robustly (prefer lineups' home/away) ---
teams_pool = set()

# From lineups (best source for many opponents)
try:
    if LINEUPS_EVENTS.exists():
        _lu = pd.read_parquet(LINEUPS_EVENTS)
        for c in ["HOME_TEAM_ABBR", "AWAY_TEAM_ABBR"]:
            if c in _lu.columns:
                teams_pool |= set(_lu[c].dropna().astype(str).unique().tolist())
except Exception:
    pass

# Fallback: from enriched possessions if available
if not enr.empty and "TEAM" in enr.columns:
    teams_pool |= set(enr["TEAM"].dropna().astype(str).unique().tolist())

# Finalize list
teams = ["All"] + sorted(t for t in teams_pool if t and t.lower() != "none")

result_classes = ["All"] + (
    sorted(
        [
            c
            for c in flt_src.get("RESULT_CLASS", pd.Series([]))
            .dropna()
            .unique()
            .tolist()
        ]
    )
    if "RESULT_CLASS" in flt_src.columns
    else []
)
periods = ["All"] + (
    sorted(
        [
            int(x)
            for x in flt_src.get("PERIOD", pd.Series([])).dropna().unique().tolist()
        ]
    )
    if "PERIOD" in flt_src.columns
    else []
)

colA, colB, colC = st.columns(3)
team_choice = colA.selectbox("Team", teams, index=0, key="flt_team")
rc_choice = colB.selectbox("Result class", result_classes, index=0, key="flt_rc")
period_choice = colC.selectbox("Period", periods, index=0, key="flt_period")

poss_view = flt_src.copy()
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
if LINEUPS_EVENTS.exists():
    try:
        roster_source = _norm_game_id(pd.read_parquet(LINEUPS_EVENTS))
    except Exception:
        roster_source = pred if not pred.empty else poss
else:
    roster_source = pred if not pred.empty else poss

roster = build_roster(roster_source if roster_source is not None else poss_view)
if roster.empty:
    st.info("No roster columns found yet (ON_COURT_* or PLAYER/TEAM).")
else:
    st.dataframe(roster, use_container_width=True, hide_index=True)

# -----------------------------
# Co-presence picker (prototype)
# -----------------------------
st.subheader("Co-presence filter (prototype)")
players = ["None"] + (
    roster["PLAYER"].dropna().unique().tolist() if not roster.empty else []
)
c1, c2 = st.columns(2)
_ = c1.selectbox("Focus player (A)", options=players, index=0, key=f"co_A_{sel_game}")
_ = c2.selectbox("With player (B)", options=players, index=0, key=f"co_B_{sel_game}")
st.caption("Hook this to event-level ON_COURT_* later to filter by co-presence.")

# -----------------------------
# EPV Scrubber
# -----------------------------
st.subheader("EPV Scrubber")

pred_source = pred_all if not pred_all.empty else pred
pred_g = pred_source[pred_source["GAME_ID"] == sel_game].copy()
if pred_g.empty:
    st.info("No predictions available for this game yet. Train or rebuild features.")
    st.stop()
# Quick game-level flatness pulse
if not pred_g.empty and "EPV" in pred_g.columns:
    flat_by_poss = (
        pred_g.sort_values(["POSS_SEQ", "EVENT_IDX"])
        .groupby("POSS_SEQ")["EPV"]
        .apply(_is_flat_epv)
    )
    share_flat = flat_by_poss.mean() if len(flat_by_poss) else 0.0
    if share_flat >= 0.5:
        st.warning(
            f"Note: {share_flat:.0%} of possessions in this game have flat EPV curves. Add more features/games for richer traces."
        )


# Possession selector with robust labeling (unique keys per game)
def _fmt(r: pd.Series) -> str:
    possq = int(r.get("POSS_SEQ", -1) or -1)
    per = r.get("PERIOD", pd.NA)
    qtxt = f"Q{int(per)}" if pd.notna(per) else "Q?"
    rc = r.get("RESULT_CLASS", "—")
    pts = r.get("POINTS", pd.NA)
    pts_txt = f"{int(pts)} pts" if pd.notna(pts) else "0 pts"
    return f"POSS {possq} | {qtxt} | {rc} ({pts_txt})"


need_cols = ["POSS_SEQ", "PERIOD", "RESULT_CLASS", "POINTS"]
have_cols = [c for c in need_cols if c in pred_g.columns]
opts = pred_g[have_cols].drop_duplicates()
pos_key_lbl = f"epv_poss_lbl_{sel_game}"
pos_key_raw = f"epv_poss_raw_{sel_game}"

if "POSS_SEQ" not in have_cols or opts.empty:
    st.warning("No possession metadata found; selecting by POSS_SEQ only.")
    sel_poss = st.selectbox(
        "Possession",
        options=sorted(pred_g["POSS_SEQ"].dropna().unique().tolist()),
        index=0,
        key=pos_key_raw,
    )
else:
    sort_cols = [c for c in ["PERIOD", "POSS_SEQ"] if c in opts.columns]
    if sort_cols:
        opts = opts.sort_values(sort_cols)
    labels = opts.apply(_fmt, axis=1).tolist()
    mapping = dict(zip(labels, opts["POSS_SEQ"].tolist()))
    sel_label = st.selectbox("Possession", options=labels, index=0, key=pos_key_lbl)
    sel_poss = mapping[sel_label]

# Build trace & enrich for hover
pbp_g = load_raw_pbp_for(sel_game)
trace = pred_g[pred_g["POSS_SEQ"] == sel_poss].sort_values("EVENT_IDX").copy()
if not pbp_g.empty and "EVENTNUM" in trace.columns:
    trace = trace.merge(
        pbp_g[["EVENTNUM", "PCTIMESTRING", "TEXT"]], on="EVENTNUM", how="left"
    )

# Bring EVTYPE from event_features if available
ef_small = pd.DataFrame()
if not ef.empty:
    base_cols = ["GAME_ID", "POSS_SEQ", "EVENTNUM"]
    cols = base_cols + (["EVTYPE"] if "EVTYPE" in ef.columns else [])
    ef_small = ef.loc[
        (ef["GAME_ID"] == sel_game) & (ef["POSS_SEQ"] == sel_poss),
        [c for c in cols if c in ef.columns],
    ].copy()
if not ef_small.empty:
    trace = trace.merge(
        ef_small[["GAME_ID", "POSS_SEQ", "EVENTNUM"]],
        on=["GAME_ID", "POSS_SEQ", "EVENTNUM"],
        how="left",
    )
    if "EVTYPE" in ef_small.columns and "EVTYPE" not in trace.columns:
        trace = trace.merge(
            ef_small[["EVENTNUM", "EVTYPE"]].drop_duplicates(),
            on="EVENTNUM",
            how="left",
        )

# ---- EPV or probability fallback ----
if trace.empty:
    st.info("No EPV data for this possession.")
else:
    trace["EVENT_IDX"] = pd.to_numeric(trace["EVENT_IDX"], errors="coerce").astype(
        "Int64"
    )
    has_epv = "EPV" in trace.columns and trace["EPV"].notna().any()
    epv_is_flat = False
    if has_epv:
        epv_is_flat = _is_flat_epv(trace["EPV"])

    # Prefer EPV when present and non-flat
    if has_epv and not epv_is_flat:
        tooltip_cols = [
            c
            for c in ["EVENT_IDX", "EPV", "EPV_DELTA", "EVTYPE", "PCTIMESTRING", "TEXT"]
            if c in trace.columns
        ]
        chart = (
            alt.Chart(trace)
            .mark_line(point=True)
            .encode(
                x=alt.X("EVENT_IDX:Q", title="Event index in possession"),
                y=alt.Y("EPV:Q", title="Expected Points"),
                color=(
                    alt.Color("EVTYPE:N", title="Event", legend=None)
                    if "EVTYPE" in trace.columns
                    else alt.value(None)
                ),
                tooltip=tooltip_cols,
            )
            .properties(height=260, width="container")
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        # Fallback to probabilities if EPV is flat or missing
        if has_epv and epv_is_flat:
            st.warning(
                "EPV curve is nearly flat — showing class probabilities instead."
            )
        else:
            st.info("No EPV values available — showing class probabilities instead.")

        need_probs = {"p0", "p1", "p2"} <= set(trace.columns)
        if not need_probs:
            st.error(
                "This possession has no p0/p1/p2 columns. Re-run training to regenerate event_epv_all.parquet."
            )
        else:
            tprob = trace[["EVENT_IDX", "p0", "p1", "p2"]].copy()
            tprob = tprob.melt(
                id_vars=["EVENT_IDX"],
                value_vars=["p0", "p1", "p2"],
                var_name="class",
                value_name="p",
            )
            prob_chart = (
                alt.Chart(tprob)
                .mark_line(point=True)
                .encode(
                    x=alt.X("EVENT_IDX:Q", title="Event index in possession"),
                    y=alt.Y("p:Q", title="Probability"),
                    color=alt.Color("class:N", title="Class"),
                    tooltip=["EVENT_IDX", "class", "p"],
                )
                .properties(height=260, width="container")
            )
            st.altair_chart(prob_chart, use_container_width=True)

            # Also show P(any points) = 1 - p0
            if "p0" in trace.columns:
                tprob2 = trace[["EVENT_IDX", "p0"]].copy()
                tprob2["p_any"] = 1.0 - tprob2["p0"]
                chart_any = (
                    alt.Chart(tprob2)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("EVENT_IDX:Q", title="Event index in possession"),
                        y=alt.Y("p_any:Q", title="P(any points)"),
                        tooltip=["EVENT_IDX", "p_any"],
                    )
                    .properties(height=200, width="container")
                )
                st.altair_chart(chart_any, use_container_width=True)

# Top EPV movers within this possession
if "EPV_DELTA" in trace.columns:
    t2 = trace.copy()
    t2["ABS_DELTA"] = t2["EPV_DELTA"].abs()
    cols = [
        c
        for c in [
            "EVENT_IDX",
            "EVENTNUM",
            "PCTIMESTRING",
            "EVTYPE",
            "EPV",
            "EPV_DELTA",
            "TEXT",
        ]
        if c in t2.columns
    ]
    st.caption("Top |ΔEPV| within this possession")
    st.dataframe(
        t2.sort_values("ABS_DELTA", ascending=False).head(8)[cols],
        use_container_width=True,
        hide_index=True,
    )

# Game-level swings
st.subheader("Game-level EPV swings")
if not pred_g.empty and "EPV_DELTA" in pred_g.columns:
    g2 = pred_g.copy()
    g2["ABS_DELTA"] = g2["EPV_DELTA"].abs()
    if not pbp_g.empty and "EVENTNUM" in g2.columns:
        g2 = g2.merge(
            pbp_g[["EVENTNUM", "PCTIMESTRING", "TEXT"]], on="EVENTNUM", how="left"
        )
    if (
        "EVTYPE" not in g2.columns
        and not ef_small.empty
        and "EVTYPE" in ef_small.columns
    ):
        g2 = g2.merge(
            ef_small[["EVENTNUM", "EVTYPE"]].drop_duplicates(),
            on="EVENTNUM",
            how="left",
        )
    cols = [
        c
        for c in [
            "POSS_SEQ",
            "EVENT_IDX",
            "EVENTNUM",
            "PCTIMESTRING",
            "EVTYPE",
            "EPV_DELTA",
            "TEXT",
        ]
        if c in g2.columns
    ]
    st.dataframe(
        g2.sort_values("ABS_DELTA", ascending=False).head(15)[cols],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No EPV deltas available for game-level swings.")

# -----------------------------
# Calibration
# -----------------------------
st.subheader("Calibration: P(any points)")
if CALIB_PATH.exists():
    cal = pd.read_csv(CALIB_PATH)
    if {"pred", "emp"} <= set(cal.columns):
        line = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        chart_cal = alt.layer(
            alt.Chart(cal)
            .mark_line(point=True)
            .encode(
                x=alt.X("pred:Q", title="Predicted P(score remainder of possession)"),
                y=alt.Y("emp:Q", title="Empirical frequency"),
                tooltip=[c for c in ["pred", "emp", "n"] if c in cal.columns],
            ),
            alt.Chart(line).mark_line(strokeDash=[4, 4]).encode(x="x:Q", y="y:Q"),
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
        pred.sort_values(["GAME_ID", "POSS_SEQ", "EVENT_IDX"])
        .groupby(["GAME_ID", "POSS_SEQ"])
        .tail(1)[["GAME_ID", "POSS_SEQ", "EPV"]]
        .rename(columns={"EPV": "EPV_END"})
    )
    comp = last_epv.merge(
        enr[["GAME_ID", "POSS_SEQ", "POINTS", "RESULT_CLASS"]].drop_duplicates(),
        on=["GAME_ID", "POSS_SEQ"],
        how="left",
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
                y=alt.Y("POINTS:Q", title="Actual points"),
                tooltip=["GAME_ID", "POSS_SEQ", "EPV_END", "POINTS", "RESULT_CLASS"],
            )
            .properties(height=260, width="container")
        )
        st.altair_chart(chart_sc, use_container_width=True)
        mae = comp["RESIDUAL"].abs().mean()
        bias = comp["RESIDUAL"].mean()
        st.write(
            f"**MAE (|EPV−points|):** {mae:.3f}  |  **Bias (EPV−points):** {bias:+.3f}"
        )
        worst = comp.reindex(
            comp["RESIDUAL"].abs().sort_values(ascending=False).index
        ).head(10)
        st.caption("Largest absolute mismatches")
        st.dataframe(
            worst[
                ["GAME_ID", "POSS_SEQ", "EPV_END", "POINTS", "RESULT_CLASS", "RESIDUAL"]
            ],
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Need predictions and possessions_enriched with POINTS for this QA.")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature importance (LightGBM gain)")
if not FEAT_IMP_PATH.exists():
    st.info("Train the model to produce feature_importance.csv.")
else:
    imp = pd.read_csv(FEAT_IMP_PATH)
    if "feature" not in imp.columns:
        st.warning("feature_importance.csv is missing a 'feature' column.")
    if "gain" not in imp.columns:
        imp["gain"] = imp.get("importance", 0)
    imp["gain"] = pd.to_numeric(imp["gain"], errors="coerce")
    if "split" in imp.columns:
        imp["split"] = pd.to_numeric(imp["split"], errors="coerce")
    imp = imp.dropna(subset=["gain"]).sort_values("gain", ascending=False)
    if imp.empty:
        st.info("No valid rows to plot in feature_importance.csv.")
    else:
        st.dataframe(
            imp[["feature", "gain"] + (["split"] if "split" in imp.columns else [])],
            use_container_width=True,
            hide_index=True,
        )
        tooltip_cols = ["feature", "gain"] + (
            ["split"] if "split" in imp.columns else []
        )
        chart_imp = (
            alt.Chart(imp.head(20))
            .mark_bar()
            .encode(
                x=alt.X("gain:Q", title="Gain"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=tooltip_cols,
            )
            .properties(height=min(24 * len(imp.head(20)) + 40, 520))
        )
        st.altair_chart(chart_imp, use_container_width=True)

# -----------------------------
# Feature Impacts / Partial Effects (optional)
# -----------------------------
st.subheader("Feature impacts (ΔEPV vs baseline)")
if not FEAT_EFF_PATH.exists():
    st.info("No feature_impacts.csv found. Generate it with your analysis script.")
else:
    eff = pd.read_csv(FEAT_EFF_PATH)
    if "delta" not in eff.columns:
        eff["delta"] = eff.get("delta_epv", 0)
    eff["delta"] = pd.to_numeric(eff["delta"], errors="coerce")
    if "value" not in eff.columns:
        eff["value"] = eff.get("bin", eff.get("category", "UNK"))
    eff = eff.dropna(subset=["delta"])
    if eff.empty:
        st.info("No valid rows in feature_impacts.csv to plot.")
    else:
        ranks = (
            eff.groupby("feature")["delta"]
            .apply(lambda s: s.abs().mean())
            .sort_values(ascending=False)
        )
        top_feats = ranks.index.tolist()
        sel_feat = st.selectbox(
            "Select feature", options=top_feats, index=0, key="impact_feat"
        )
        eff_f = eff[eff["feature"] == sel_feat].copy()
        try:
            eff_f["_order"] = pd.to_numeric(eff_f["value"], errors="coerce")
        except Exception:
            eff_f["_order"] = None
        eff_f = eff_f.sort_values(["_order", "value"])
        chart_eff = (
            alt.Chart(eff_f)
            .mark_line(point=True)
            .encode(
                x=alt.X("value:N", title=f"{sel_feat}"),
                y=alt.Y("delta:Q", title="Δ EPV"),
                tooltip=["feature", "value", "delta"],
            )
            .properties(height=260, width="container")
        )
        st.altair_chart(chart_eff, use_container_width=True)

# -----------------------------
# Rolling CV table
# -----------------------------
st.subheader("Rolling CV (time-aware)")
cv_path = Path("data/derived/cv_metrics.csv")
if cv_path.exists():
    cv = pd.read_csv(cv_path)
    st.dataframe(cv.round(4), use_container_width=True, hide_index=True)
    if {"val_logloss_any", "val_brier_any"} <= set(cv.columns):
        st.write(
            "**Avg logloss(any):**",
            f"{cv['val_logloss_any'].mean():.4f}",
            " | **Avg Brier:**",
            f"{cv['val_brier_any'].mean():.4f}",
        )
else:
    st.info("Run training to generate rolling CV metrics.")
