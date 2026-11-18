"""Train and compare five model families.

    python -m models.train_models                 # all five
    python -m models.train_models --only lightgbm # re-run one

For each model: GroupKFold(5) by trip on the 80 training trips, then a fit on all 80 and an
evaluation on the 10 student-test trips. Results are appended to results/model_comparison.csv
(this file is the single source for the report table). Fitted models go to artifacts/candidates/.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS, RESULTS, SEED
from models.harness import cross_validate, evaluate_on_stest, load_tables
from preprocessing.features import FEATURES

LGBM_DEFAULT = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED,
                    n_jobs=-1, verbose=-1)
RF_DEFAULT = dict(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)


def model_zoo():
    return {
        "logreg": ("Linear / probabilistic",
                   make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
                   "log loss (cross-entropy)"),
        "knn": ("Instance-based",
                make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance",
                                                                     algorithm="kd_tree", n_jobs=-1)),
                "N/A (non-parametric, no loss minimised)"),
        "random_forest": ("Ensemble (bagging)",
                          RandomForestClassifier(**RF_DEFAULT),
                          "Gini impurity"),
        "lightgbm": ("Boosting", LGBMClassifier(**LGBM_DEFAULT), "log loss (cross-entropy)"),
        "mlp": ("Neural network",
                make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                              learning_rate_init=1e-3, early_stopping=True,
                                                              max_iter=200, random_state=SEED)),
                "log loss (cross-entropy), Adam"),
    }


def main(only=None):
    tr, te, otr, ote = load_tables()
    (ARTIFACTS / "candidates").mkdir(exist_ok=True)
    out_path = RESULTS / "model_comparison.csv"
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() and only else []
    for name, (family, model, objective) in model_zoo().items():
        if only and name != only:
            continue
        rows = [r for r in rows if r["model"] != name]
        print(f"== {name}: cross-validation", flush=True)
        cv = cross_validate(model, tr, otr)
        cv.to_csv(RESULTS / f"cv_folds_{name}.csv", index=False)

        print(f"== {name}: fit on all training trips", flush=True)
        t0 = time.perf_counter()
        model.fit(tr[FEATURES].values, tr["label"].values)
        train_time = time.perf_counter() - t0
        res, prob = evaluate_on_stest(model, te, ote)
        joblib.dump(model, ARTIFACTS / "candidates" / f"{name}.joblib")
        pd.DataFrame({"trip_id": te.trip_id, "obs_idx": te.obs_idx, "arc_id": te.arc_id, "prob": prob}) \
            .to_parquet(RESULTS / f"stest_prob_{name}.parquet")
        json.dump(res["cm"], open(RESULTS / f"confusion_{name}.json", "w"))
        if name == "mlp":
            mlp = model[-1]
            pd.DataFrame({"epoch": range(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                          "val_score": mlp.validation_scores_[:len(mlp.loss_curve_)]}) \
                .to_csv(RESULTS / "mlp_loss_curve.csv", index=False)

        rows.append(dict(
            model=name, family=family, objective=objective,
            config=json.dumps({k: v for k, v in model.get_params().items()
                               if isinstance(v, (int, float, str, bool, tuple, type(None)))
                               and k not in ("memory", "verbose", "n_jobs")}, default=str)[:600],
            seed=SEED,
            cv_macro_f1=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(),
            cv_top1=cv.top1.mean(), cv_top1_std=cv.top1.std(), cv_log_loss=cv.log_loss.mean(),
            test_top1=res["top1"], test_accuracy=res["accuracy"], test_precision=res["precision"],
            test_recall=res["recall"], test_macro_f1=res["macro_f1"], test_log_loss=res["log_loss"],
            route_continuity=res["route_continuity"], train_time_s=train_time,
            inference_ms_per_1000_pairs=res["inference_ms_per_1000_pairs"]))
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"   top1 {res['top1']:.4f}  macroF1 {res['macro_f1']:.4f}  logloss {res['log_loss']:.4f}  "
              f"cont {res['route_continuity']:.4f}  cv F1 {cv.macro_f1.mean():.4f}", flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    main(ap.parse_args().only)
