"""Shared evaluation code for every model: GroupKFold CV by trip, pair-level and observation-level metrics."""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold

from config import LABEL_DIST_MAX, RESULTS, SEED, STEST_DIR
from preprocessing.network import RoadGraph
from metrics import classifier_only_pred, top1_accuracy
from preprocessing.features import FEATURES
from preprocessing.io import load_trip


def excluded_train_trips():
    """Training trips whose reference route does not match their GPS (see config.LABEL_DIST_MAX)."""
    q = pd.read_csv(RESULTS / "candidate_quality.csv", dtype={"trip_id": str})
    q["trip_id"] = q["trip_id"].str.zfill(8)
    bad = q[(q["split"] == "train") & (q["label_dist_median"] > LABEL_DIST_MAX)]
    return bad[["trip_id", "n_obs", "miss_rate", "label_dist_median"]]


def load_tables():
    tr = pd.read_parquet(RESULTS / "features_train.parquet")
    te = pd.read_parquet(RESULTS / "features_stest.parquet")
    otr = pd.read_parquet(RESULTS / "obs_train.parquet")
    ote = pd.read_parquet(RESULTS / "obs_stest.parquet")
    bad = excluded_train_trips()
    bad.to_csv(RESULTS / "excluded_trips.csv", index=False)
    ids = set(bad["trip_id"])
    if ids:
        print(f"excluding {len(ids)} mismatched training trips: {sorted(ids)}")
    tr = tr[~tr["trip_id"].isin(ids)]
    otr = otr[~otr["trip_id"].isin(ids)]
    return tr.reset_index(drop=True), te.reset_index(drop=True), otr, ote


def obs_top1(pairs, obs, prob):
    """Observation-level Top-1 pooled over all trips in `pairs`/`obs`."""
    prob = np.asarray(prob)
    hits, total = 0, 0
    for tid, pos in pairs.groupby("trip_id").indices.items():   # positional row numbers
        o = obs[obs["trip_id"] == tid].sort_values("obs_idx")
        pred = classifier_only_pred(pairs.iloc[pos], prob[pos], len(o))
        hits += (pred == o["true_arc"].values).sum()
        total += len(o)
    return hits / total


def pair_metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                macro_f1=f1_score(y, pred, average="macro", zero_division=0),
                log_loss=log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7)))


def cross_validate(model, pairs, obs, n_splits=5, seed=SEED):
    """GroupKFold by trip. Returns dict of mean/std over folds and the per-fold table."""
    X, y, groups = pairs[FEATURES].values, pairs["label"].values, pairs["trip_id"].values
    rows = []
    for k, (tri, vai) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = clone(model).fit(X[tri], y[tri])
        prob = m.predict_proba(X[vai])[:, 1]
        vp = pairs.iloc[vai].reset_index(drop=True)
        vo = obs[obs["trip_id"].isin(vp["trip_id"].unique())]
        r = pair_metrics(y[vai], prob)
        r["top1"] = obs_top1(vp, vo, prob)
        r["fold"] = k
        rows.append(r)
    df = pd.DataFrame(rows)
    return df


_graph_cache = {}


def continuity_of(pred_by_trip):
    """pred_by_trip: {trip_id: array of predicted arc per observation} for student-test trips."""
    vals = []
    for tid, pred in pred_by_trip.items():
        if tid not in _graph_cache:
            _graph_cache[tid] = RoadGraph(load_trip(STEST_DIR / tid))
        vals.append(_graph_cache[tid].continuity(pred))
    return float(np.nanmean(vals))


def evaluate_on_stest(model, pairs, obs, with_continuity=True):
    """Fit nothing here: model must already be fitted on the training split."""
    t0 = time.perf_counter()
    prob = model.predict_proba(pairs[FEATURES].values)[:, 1]
    inf_time = time.perf_counter() - t0
    out = pair_metrics(pairs["label"].values, prob)
    out["cm"] = confusion_matrix(pairs["label"].values, (prob >= 0.5).astype(int)).tolist()
    out["inference_ms_per_1000_pairs"] = 1000 * inf_time / len(pairs) * 1000

    preds = {}
    hits = total = 0
    for tid, pos in pairs.groupby("trip_id").indices.items():
        o = obs[obs["trip_id"] == tid].sort_values("obs_idx")
        p = classifier_only_pred(pairs.iloc[pos], prob[pos], len(o))
        preds[tid] = p
        hits += (p == o["true_arc"].values).sum()
        total += len(o)
    out["top1"] = hits / total
    if with_continuity:
        out["route_continuity"] = continuity_of(preds)
    return out, prob
