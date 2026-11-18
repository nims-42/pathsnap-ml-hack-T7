"""Bayesian hyperparameter optimisation (Optuna, TPE sampler) with group-aware cross-validation.

    python -m models.tune_optuna --model random_forest --trials 15

Search      : GroupKFold(5) by trip on the training trips (the same exclusion rule as the comparison)
Objective   : minimise mean cross-validated log loss (probability quality matters, the HMM uses it)
Outputs     : results/optuna_trials.csv, results/optuna_summary.json, results/tuned_vs_untuned.csv,
              artifacts/model.joblib (tuned model, compressed), artifacts/model_info.json,
              artifacts/feature_list.json
The student-test trips are used only for the final tuned-vs-untuned comparison, never inside the search.
"""
import argparse
import json
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from config import ARTIFACTS, RESULTS, SEED
from models.harness import cross_validate, evaluate_on_stest, load_tables
from models.train_models import LGBM_DEFAULT, RF_DEFAULT
from preprocessing.features import FEATURES

SPACES = {
    "lightgbm": {
        "num_leaves": "int 15..127", "learning_rate": "log-float 0.01..0.2", "n_estimators": "int 100..600",
        "min_child_samples": "int 10..100", "subsample": "float 0.6..1.0 (subsample_freq=1)",
        "colsample_bytree": "float 0.6..1.0", "reg_lambda": "float 0..5"},
    # ranges are kept moderate on purpose: they bound the saved model size (GitHub rejects files > 100 MB)
    "random_forest": {
        "n_estimators": "int 100..300", "max_depth": "int 8..40", "min_samples_leaf": "int 2..30",
        "max_features": "float 0.3..1.0", "max_samples": "float 0.5..1.0"},
}
UNTUNED = {"lightgbm": LGBM_DEFAULT, "random_forest": RF_DEFAULT}


def suggest(name, trial):
    if name == "lightgbm":
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            random_state=SEED, n_jobs=-1, verbose=-1)
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 8, 40),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 30),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
        max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
        random_state=SEED, n_jobs=-1)


def make_model(name, params):
    return LGBMClassifier(**params) if name == "lightgbm" else RandomForestClassifier(**params)


def main(name, n_trials):
    tr, te, otr, ote = load_tables()
    X, y, groups = tr[FEATURES].values, tr["label"].values, tr["trip_id"].values
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))

    def objective(trial):
        params = suggest(name, trial)
        losses = []
        for tri, vai in folds:
            m = make_model(name, params).fit(X[tri], y[tri])
            losses.append(log_loss(y[vai], np.clip(m.predict_proba(X[vai])[:, 1], 1e-7, 1 - 1e-7)))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_time = time.perf_counter() - t0
    study.trials_dataframe().to_csv(RESULTS / "optuna_trials.csv", index=False)
    best_full = suggest(name, optuna.trial.FixedTrial(study.best_params))
    json.dump(dict(model=name, space=SPACES[name], objective="mean GroupKFold(5) log loss on the training trips",
                   n_trials=n_trials, best_value=study.best_value, best_params=study.best_params,
                   search_seconds=search_time), open(RESULTS / "optuna_summary.json", "w"), indent=2)
    print("best CV log loss", round(study.best_value, 5), study.best_params, flush=True)

    rows = []
    for label, params in [("untuned", UNTUNED[name]), ("tuned", best_full)]:
        model = make_model(name, params)
        cv = cross_validate(model, tr, otr)
        t1 = time.perf_counter()
        model.fit(X, y)
        fit_s = time.perf_counter() - t1
        res, prob = evaluate_on_stest(model, te, ote)
        row = dict(model=f"{name} ({label})", cv_macro_f1=cv.macro_f1.mean(), cv_log_loss=cv.log_loss.mean(),
                   test_top1=res["top1"], test_macro_f1=res["macro_f1"], test_log_loss=res["log_loss"],
                   route_continuity=res["route_continuity"],
                   inference_ms_per_1000_pairs=res["inference_ms_per_1000_pairs"],
                   fit_seconds=fit_s, search_seconds=search_time if label == "tuned" else 0.0)
        if label == "tuned":
            path = ARTIFACTS / "model.joblib"
            joblib.dump(model, path, compress=3)
            row["model_size_mb"] = path.stat().st_size / 1e6
            json.dump(dict(model=name, params={k: v for k, v in params.items()}), open(ARTIFACTS / "model_info.json", "w"))
            json.dump(FEATURES, open(ARTIFACTS / "feature_list.json", "w"))
            pd.DataFrame({"trip_id": te.trip_id, "obs_idx": te.obs_idx, "arc_id": te.arc_id, "prob": prob}) \
                .to_parquet(RESULTS / "stest_prob_tuned.parquet")
            if row["model_size_mb"] > 90:
                print(f"WARNING: model.joblib is {row['model_size_mb']:.0f} MB, close to GitHub's 100 MB limit")
        rows.append(row)
        print(label, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "tuned_vs_untuned.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lightgbm", "random_forest"], default="random_forest")
    ap.add_argument("--trials", type=int, default=15)
    a = ap.parse_args()
    main(a.model, a.trials)
