# file: app.py
"""
Streamlit dashboard for anomaly-based fraud detection.
- Loads scored CSVs produced by modeling notebook (train_scored.csv, test_scored.csv) and metrics/sweep JSON.
- Interactive contamination slider to re-threshold model scores live.
- Filtering by time, location, device, transaction type, amount, provider, user type.
- KPIs, charts, and anomaly table with downloads.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="FraudWatch — Anomaly Dashboard", page_icon="🛰️", layout="wide")

DEFAULT_TRAIN = Path("train_scoreds.csv")
DEFAULT_TEST = Path("test_scoreds.csv")
DEFAULT_METRICS = Path("metrics.json")
DEFAULT_SWEEP = Path("sweep.json")

MODELS = {
    "Isolation Forest": "iso",
    "One-Class SVM": "svm",
    "Local Outlier Factor": "lof",
    "Ensemble (majority)": "ensemble",
}

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure datetime typed if present
    for col in ["datetime", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}

def retitle_flag_col(model_key: str) -> str:
    return "ensemble_flag" if model_key == "ensemble" else f"{model_key}_flag"

def retitle_score_col(model_key: str) -> str:
    return f"{model_key}_score"

def apply_contamination_threshold(scores: np.ndarray, contamination: float) -> np.ndarray:
    # Quantile threshold; higher scores = more anomalous (our scores are normalized that way)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.array([], dtype=int)
    q = np.quantile(scores, 1 - contamination)
    return (scores >= q).astype(int)

def recompute_flags(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    out = df.copy()
    for key in ["iso", "svm", "lof"]:
        sc = retitle_score_col(key)
        if sc in out.columns:
            out[retitle_flag_col(key)] = apply_contamination_threshold(out[sc].values, contamination)
    # Ensemble vote
    cols = [c for c in ["iso_flag", "svm_flag", "lof_flag"] if c in out.columns]
    if cols:
        votes = out[cols].sum(axis=1)
        out["ensemble_flag"] = (votes >= 2).astype(int)
    return out

def kpi_card(label: str, value, help_txt: str = ""):
    st.metric(label, value)  # minimalist KPI
    if help_txt:
        st.caption(help_txt)

# ----------------------------
# Sidebar — data sources & controls
# ----------------------------
st.sidebar.title("⚙️ Controls")

st.sidebar.subheader("Data sources")
train_file = st.sidebar.file_uploader("Train scored CSV", type=["csv"], key="train")
test_file = st.sidebar.file_uploader("Test scored CSV", type=["csv"], key="test")
metrics_file = st.sidebar.file_uploader("Metrics JSON (optional)", type=["json"], key="metrics")
sweep_file = st.sidebar.file_uploader("Sweep JSON (optional)", type=["json"], key="sweep")

if train_file is not None:
    train_df = pd.read_csv(train_file)
    for col in ["datetime", "date"]:
        if col in train_df.columns:
            train_df[col] = pd.to_datetime(train_df[col], errors="coerce")
elif DEFAULT_TRAIN.exists():
    train_df = load_csv(DEFAULT_TRAIN)
else:
    train_df = pd.DataFrame()

if test_file is not None:
    test_df = pd.read_csv(test_file)
    for col in ["datetime", "date"]:
        if col in test_df.columns:
            test_df[col] = pd.to_datetime(test_df[col], errors="coerce")
elif DEFAULT_TEST.exists():
    test_df = load_csv(DEFAULT_TEST)
else:
    test_df = pd.DataFrame()

metrics = load_json(metrics_file) if metrics_file else (load_json(DEFAULT_METRICS) if DEFAULT_METRICS.exists() else {})
sweep = load_json(sweep_file) if sweep_file else (load_json(DEFAULT_SWEEP) if DEFAULT_SWEEP.exists() else {})

split_choice = st.sidebar.radio("Dataset split", ["Test", "Train", "Both"], index=0, horizontal=True)
model_choice = st.sidebar.selectbox("Model", list(MODELS.keys()), index=3)
model_key = MODELS[model_choice]

contam = st.sidebar.slider("Contamination (flag rate)", 0.005, 0.10, 0.02, 0.005, help="Re-threshold scores at this rate.")

# Merge splits if needed
if split_choice == "Both":
    df = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)
elif split_choice == "Train":
    df = train_df.assign(split="train")
else:
    df = test_df.assign(split="test")

if df.empty:
    st.warning("No data found. Upload CSVs in the sidebar or place default files next to app.py.")
    st.stop()

# Recompute flags with chosen contamination
if retitle_score_col("iso") in df.columns:
    df = recompute_flags(df, contamination=contam)

# ----------------------------
# Sidebar — filters (built dynamically)
# ----------------------------
with st.sidebar.expander("Filters", expanded=True):
    # Time
    if "datetime" in df.columns and df["datetime"].notna().any():
        dt_min = pd.to_datetime(df["datetime"].min())
        dt_max = pd.to_datetime(df["datetime"].max())
        date_range = st.date_input("Date range", (dt_min.date(), dt_max.date()))
    else:
        date_range = None

    # Categorical filters
    def multi(label: str, col: str):
        if col in df.columns:
            options = sorted([x for x in df[col].dropna().unique().tolist()])
            return st.multiselect(label, options, default=options)
        return None

    loc_sel = multi("Location", "location")
    type_sel = multi("Transaction type", "transaction_type")
    dev_sel = multi("Device type", "device_type")
    net_sel = multi("Network provider", "network_provider")
    user_sel = multi("User type", "user_type")

    # Amount
    if "amount" in df.columns:
        amt_min, amt_max = float(df["amount"].min()), float(df["amount"].max())
        amt_sel = st.slider("Amount range", min_value=0.0, max_value=max(amt_max, 1.0), value=(amt_min, amt_max), step=max((amt_max-amt_min)/100, 0.01))
    else:
        amt_sel = None

# Apply filters
mask = pd.Series(True, index=df.index)
if date_range and "datetime" in df.columns:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    mask &= df["datetime"].between(start, end)
if loc_sel is not None:
    mask &= df["location"].isin(loc_sel)
if type_sel is not None:
    mask &= df["transaction_type"].isin(type_sel)
if dev_sel is not None:
    mask &= df["device_type"].isin(dev_sel)
if net_sel is not None:
    mask &= df["network_provider"].isin(net_sel)
if user_sel is not None:
    mask &= df["user_type"].isin(user_sel)
if amt_sel is not None and "amount" in df.columns:
    mask &= (df["amount"].between(amt_sel[0], amt_sel[1]))

fdf = df.loc[mask].copy()

# Current model columns
flag_col = retitle_flag_col(model_key)
score_col = retitle_score_col(model_key) if model_key != "ensemble" else None

# ----------------------------
# Header + KPIs
# ----------------------------
st.title("🛰️ FraudWatch — Anomaly Dashboard")
left, mid, right, right2 = st.columns(4)

total_tx = int(fdf.shape[0])
flag_rate = float(fdf.get(flag_col, pd.Series(0)).mean()) if flag_col in fdf else 0.0
num_flags = int(flag_rate * total_tx)

with left:
    kpi_card("Transactions", f"{total_tx:,}")
with mid:
    kpi_card("Flagged (current model)", f"{num_flags:,}")
with right:
    kpi_card("Flag rate", f"{flag_rate:.2%}", "Driven by contamination slider")
with right2:
    kpi_card("Model", model_choice)

# Metrics (if available)
if metrics and model_key != "ensemble":
    mname = {
        "iso": "isolation_forest",
        "svm": "oneclass_svm",
        "lof": "lof",
    }.get(model_key)
    if mname in metrics:
        st.subheader("Evaluation (labeled data)")
        m = metrics[mname]
        cols = st.columns(5)
        cols[0].metric("Precision", f"{m['precision']:.3f}")
        cols[1].metric("Recall", f"{m['recall']:.3f}")
        cols[2].metric("F1", f"{m['f1']:.3f}")
        cols[3].metric("TP", f"{m['tp']}")
        cols[4].metric("FN", f"{m['fn']}")

# ----------------------------
# Tabs: Trends | Segments | Table
# ----------------------------
trend_tab, seg_tab, table_tab = st.tabs(["📈 Trends", "🧭 Segments", "📋 Table"])

with trend_tab:
    # Time series of flags
    if "datetime" in fdf.columns and flag_col in fdf.columns:
        ts = (
            fdf.set_index("datetime").resample("D")[flag_col].agg(["count", "sum"]).reset_index()
        )
        ts.rename(columns={"count": "transactions", "sum": "flags"}, inplace=True)
        ts["flag_rate"] = ts["flags"] / ts["transactions"].replace(0, np.nan)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=ts["datetime"], y=ts["transactions"], name="Transactions", marker_color="#7ab3ff", opacity=0.5))
        fig.add_trace(go.Scatter(x=ts["datetime"], y=ts["flags"], name="Flags", mode="lines+markers", marker_color="#e45756"))
        fig.add_trace(go.Scatter(x=ts["datetime"], y=ts["flag_rate"]*ts["transactions"].max(), name="Flag rate (scaled)", mode="lines", line=dict(dash="dash", color="#2ca02c")))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Amount distribution by flag
    if "amount" in fdf.columns and flag_col in fdf.columns:
        fig2 = px.histogram(
            fdf, x="amount", color=flag_col, nbins=50, barmode="overlay",
            color_discrete_map={0: "#4CAF50", 1: "#e45756"}, opacity=0.6
        )
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

with seg_tab:
    # Location bar
    if "location" in fdf.columns and flag_col in fdf.columns:
        loc = fdf.groupby("location", dropna=False)[flag_col].agg(["count", "sum"]).reset_index()
        loc.rename(columns={"count": "transactions", "sum": "flags"}, inplace=True)
        fig3 = px.bar(loc.sort_values("flags", ascending=False), x="location", y="flags", title="Anomalies by location", color="flags", color_continuous_scale="Reds")
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    # Hourly heatmap by day of week
    if {"hour", "day_of_week", flag_col}.issubset(fdf.columns):
        heat = fdf.pivot_table(index="day_of_week", columns="hour", values=flag_col, aggfunc="mean").fillna(0)
        fig4 = px.imshow(heat, aspect="auto", color_continuous_scale="Reds", origin="lower",
                         labels=dict(color="Flag rate"))
        fig4.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig4, use_container_width=True)

with table_tab:
    st.subheader("Flagged transactions")
    # Compute score column for sorting (for ensemble, average of component scores)
    if model_key == "ensemble":
        parts = [c for c in ["iso_score", "svm_score", "lof_score"] if c in fdf.columns]
        if parts:
            fdf["_ensemble_score"] = fdf[parts].mean(axis=1)
            score_for_sort = "_ensemble_score"
        else:
            score_for_sort = None
    else:
        score_for_sort = score_col if score_col in fdf.columns else None

    show_cols = [c for c in [
        "datetime", "transaction_id", "user_id", "transaction_type", "amount", "location",
        "device_type", "network_provider", "user_type", "is_foreign_number",
        "is_sim_recently_swapped", "has_multiple_accounts", flag_col, score_for_sort
    ] if c in fdf.columns]

    flagged = fdf[fdf[flag_col] == 1] if flag_col in fdf.columns else fdf
    flagged = flagged.sort_values(score_for_sort, ascending=False) if score_for_sort else flagged

    st.dataframe(flagged[show_cols].head(1000), use_container_width=True, height=420)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download filtered data (CSV)", data=fdf.to_csv(index=False).encode("utf-8"), file_name="filtered_transactions.csv", mime="text/csv")
    with c2:
        st.download_button("⬇️ Download flagged only (CSV)", data=flagged.to_csv(index=False).encode("utf-8"), file_name="flagged_transactions.csv", mime="text/csv")

# ----------------------------
# Optional: contamination sweep plot
# ----------------------------
if sweep and isinstance(sweep, dict) and model_key != "ensemble":
    st.subheader("Threshold sweep — flagged rate / metrics")
    mkey = {
        "iso": "isolation_forest",
        "svm": "oneclass_svm",
        "lof": "lof",
    }.get(model_key)
    if mkey in sweep:
        df_sweep = pd.DataFrame.from_dict(sweep[mkey], orient="index").reset_index().rename(columns={"index": "contamination"})
        # Two-axis plot depending on available fields
        if {"precision", "recall", "f1"}.issubset(df_sweep.columns):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sweep["contamination"], y=df_sweep["precision"], name="Precision"))
            fig.add_trace(go.Scatter(x=df_sweep["contamination"], y=df_sweep["recall"], name="Recall"))
            fig.add_trace(go.Scatter(x=df_sweep["contamination"], y=df_sweep["f1"], name="F1"))
            if "cost" in df_sweep.columns:
                fig.add_trace(go.Scatter(x=df_sweep["contamination"], y=df_sweep["cost"], name="Cost", yaxis="y2", line=dict(dash="dash")))
                fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Cost"))
        else:
            fig = px.line(df_sweep, x="contamination", y="flag_rate", title="Flag rate vs contamination")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.caption("Tip: Use the contamination slider to explore sensitivity and control false positives.")
