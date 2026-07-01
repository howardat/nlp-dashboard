"""Sentiment trajectory features and their correlation with conversion."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FEATURE_COLS = ["mean", "final", "min", "max_drop", "slope", "neg_run_len", "start_to_end", "n_client_turns"]


def logistic_auc(features: pd.DataFrame, convs: pd.DataFrame) -> float:
    """5-fold CV ROC-AUC of a logistic model over all trajectory features. NaN if not enough data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    df = features.merge(convs[["conversation_id", "converted"]], on="conversation_id")
    feats = [c for c in FEATURE_COLS if c in df.columns]
    y = df["converted"].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 10:
        return float("nan")
    x = StandardScaler().fit_transform(df[feats].to_numpy(dtype=float))
    scores = cross_val_score(LogisticRegression(max_iter=1000), x, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def _max_drop(vals: np.ndarray) -> float:
    """Largest peak-to-trough decline along the series (>=0)."""
    peak = vals[0]
    drop = 0.0
    for v in vals:
        peak = max(peak, v)
        drop = max(drop, peak - v)
    return float(drop)


def _neg_run(vals: np.ndarray) -> int:
    best = cur = 0
    for v in vals:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def trajectory_features(signals: pd.DataFrame) -> pd.DataFrame:
    cl = signals[(signals["role"] == "client") & signals["sentiment"].notna()]
    out = []
    for conv, g in cl.sort_values("turn_index").groupby("conversation_id"):
        v = g["sentiment"].to_numpy(dtype=float)
        if len(v) == 0:
            continue
        slope = float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) > 1 else 0.0
        out.append({
            "conversation_id": conv,
            "mean": float(v.mean()),
            "final": float(v[-1]),
            "min": float(v.min()),
            "max_drop": _max_drop(v),
            "slope": slope,
            "neg_run_len": _neg_run(v),
            "start_to_end": float(v[-1] - v[0]),
            "n_client_turns": int(len(v)),
        })
    return pd.DataFrame(out, columns=["conversation_id"] + FEATURE_COLS)


def correlations(features: pd.DataFrame, convs: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(convs[["conversation_id", "converted"]], on="conversation_id")
    y = df["converted"].astype(int).to_numpy()
    rows = []
    feats = [c for c in FEATURE_COLS if c in df.columns]
    for f in feats:
        x = df[f].to_numpy(dtype=float)
        if np.std(x) == 0 or len(np.unique(y)) < 2:
            r, p = 0.0, 1.0
        else:
            r, p = stats.pointbiserialr(y, x)
        rows.append({"feature": f, "r": float(r), "p_value": float(p)})
    res = pd.DataFrame(rows)
    # Benjamini-Hochberg adjusted p-values
    m = len(res)
    res = res.sort_values("p_value").reset_index(drop=True)
    res["p_adj"] = (res["p_value"] * m / (res.index + 1)).clip(upper=1.0)
    res["p_adj"] = res["p_adj"][::-1].cummin()[::-1]
    res["direction"] = np.where(res["r"] >= 0, "higher → converts", "higher → loses")
    return res.sort_values("r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def avg_trajectory(signals: pd.DataFrame, convs: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Mean sentiment per normalized-progress bin, split by converted.
    Columns: bin, converted, mean, n."""
    cl = signals[(signals["role"] == "client") & signals["sentiment"].notna()].merge(
        convs[["conversation_id", "converted"]], on="conversation_id"
    )
    rows = []
    for conv, g in cl.sort_values("turn_index").groupby("conversation_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        conv_flag = bool(g["converted"].iloc[0])
        for i in range(n):
            b = int(i / n * bins) if n > 1 else 0
            rows.append({"bin": min(b, bins - 1), "converted": conv_flag, "sentiment": g["sentiment"].iloc[i]})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    return (d.groupby(["bin", "converted"])["sentiment"]
            .agg(["mean", "count"]).reset_index().rename(columns={"count": "n"}))
