"""Metrics shared by models, HMM evaluation and the API, so every number is computed one way."""
import numpy as np
import pandas as pd


def classifier_only_pred(pairs, prob, n_obs):
    """Highest-scoring candidate per observation. -1 if the observation has no candidates."""
    df = pd.DataFrame({"obs_idx": pairs["obs_idx"].values, "arc_id": pairs["arc_id"].values, "p": prob})
    best = df.loc[df.groupby("obs_idx")["p"].idxmax()]
    pred = np.full(n_obs, -1, dtype=np.int64)
    pred[best["obs_idx"].values] = best["arc_id"].values
    return pred


def top1_accuracy(pred, true_arc):
    """Observation-level Top-1. A prediction of -1 (no candidate) counts as wrong."""
    return float(np.mean(np.asarray(pred) == np.asarray(true_arc)))


def candidate_miss_rate(pairs, obs):
    has = pairs.groupby("obs_idx")["label"].max().reindex(obs["obs_idx"]).fillna(0).values
    return float(1.0 - has.mean())
