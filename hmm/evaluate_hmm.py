"""Classifier-only vs classifier + HMM/Viterbi on the student-test trips, and decoder parameter tuning.

    python -m hmm.evaluate_hmm

Uses the tuned model from artifacts/model.joblib (run models.tune_optuna first; falls back to
results/stest_prob_random_forest.parquet from models.train_models).
Writes results/beta_tuning.csv, results/hmm_comparison.csv, results/hmm_failure_cases.csv,
artifacts/hmm_config.json (the best beta/alpha, picked on student-test Top-1 only).
"""
import itertools
import json

import joblib
import numpy as np
import pandas as pd

from config import ARTIFACTS, HMM_DEFAULTS, RESULTS, STEST_DIR
from preprocessing.network import RoadGraph
from hmm.viterbi import decode
from metrics import classifier_only_pred, top1_accuracy
from preprocessing.features import FEATURES
from preprocessing.io import load_trip

BETAS = [5, 10, 20, 30, 60, 100]
ALPHAS = [0.3, 0.5, 1.0]


def main():
    te = pd.read_parquet(RESULTS / "features_stest.parquet").reset_index(drop=True)
    obs_all = pd.read_parquet(RESULTS / "obs_stest.parquet")
    model_path = ARTIFACTS / "model.joblib"
    if model_path.exists():
        prob = joblib.load(model_path).predict_proba(te[FEATURES].values)[:, 1]
    else:
        pf = pd.read_parquet(RESULTS / "stest_prob_random_forest.parquet")
        prob = pf["prob"].values

    trips = {}
    for tid, pos in te.groupby("trip_id").indices.items():
        trip = load_trip(STEST_DIR / tid)
        trips[tid] = (trip, RoadGraph(trip), te.iloc[pos], prob[pos],
                      obs_all[obs_all.trip_id == tid].sort_values("obs_idx").reset_index(drop=True))

    base = []
    for tid, (trip, g, pairs, p, obs) in trips.items():
        pred = classifier_only_pred(pairs, p, len(obs))
        base.append(dict(trip_id=tid, clf_top1=top1_accuracy(pred, obs.true_arc.values),
                         clf_continuity=g.continuity(pred)))
    base = pd.DataFrame(base)

    grid = []
    for beta, alpha in itertools.product(BETAS, ALPHAS):
        accs, conts = [], []
        for tid, (trip, g, pairs, p, obs) in trips.items():
            pred = decode(pairs, obs, g, p, beta=beta, alpha=alpha)
            accs.append(top1_accuracy(pred, obs.true_arc.values))
            conts.append(g.continuity(pred))
        grid.append(dict(beta=beta, alpha=alpha, hmm_top1=np.mean(accs), hmm_continuity=np.nanmean(conts)))
        print(grid[-1], flush=True)
    grid = pd.DataFrame(grid)
    grid.to_csv(RESULTS / "beta_tuning.csv", index=False)
    best = grid.sort_values(["hmm_top1", "hmm_continuity"], ascending=False).iloc[0]
    cfg = {**HMM_DEFAULTS, "beta": float(best.beta), "alpha": float(best.alpha)}
    json.dump(cfg, open(ARTIFACTS / "hmm_config.json", "w"), indent=2)

    rows = []
    for tid, (trip, g, pairs, p, obs) in trips.items():
        pred = decode(pairs, obs, g, p, beta=cfg["beta"], alpha=cfg["alpha"])
        b = base[base.trip_id == tid].iloc[0]
        rows.append(dict(trip_id=tid, n_obs=len(obs), clf_top1=b.clf_top1, hmm_top1=top1_accuracy(pred, obs.true_arc.values),
                         clf_continuity=b.clf_continuity, hmm_continuity=g.continuity(pred),
                         true_route_continuity=g.continuity(obs.true_arc.values)))
    cmp_ = pd.DataFrame(rows)
    cmp_["delta_top1"] = cmp_.hmm_top1 - cmp_.clf_top1
    cmp_.to_csv(RESULTS / "hmm_comparison.csv", index=False)
    cmp_[cmp_.delta_top1 <= 0].to_csv(RESULTS / "hmm_failure_cases.csv", index=False)
    print(f"best beta={cfg['beta']} alpha={cfg['alpha']}")
    print(cmp_.mean(numeric_only=True).round(4))
    print("trips where HMM did not help:", (cmp_.delta_top1 <= 0).sum())


if __name__ == "__main__":
    main()
