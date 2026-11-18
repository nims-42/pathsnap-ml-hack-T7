"""Build the training tables for every trip.

    python -m preprocessing.build_features

Writes
  results/features_train.parquet, results/features_stest.parquet   (pair rows)
  results/obs_train.parquet,      results/obs_stest.parquet        (one row per observation)
  results/candidate_quality.csv   per-trip candidate miss rate and label-to-GPS distance
"""
import time

import pandas as pd

from config import RESULTS, STEST_DIR, TRAIN_DIR
from preprocessing.features import load_and_build
from preprocessing.io import list_trips


def run(split_name, folder):
    pairs_all, obs_all, quality = [], [], []
    trips = list_trips(folder)
    if not trips:
        raise SystemExit(f"no trips found in {folder}. Set PATHSNAP_DATA or copy the data into raw_data/")
    for k, d in enumerate(trips):
        t0 = time.time()
        trip, pairs, obs = load_and_build(d, with_labels=True)
        pairs["split"] = split_name
        obs["trip_id"], obs["split"] = trip["trip_id"], split_name
        has_true = pairs.groupby("obs_idx")["label"].max().reindex(obs["obs_idx"]).fillna(0).values
        quality.append(dict(
            split=split_name, trip_id=trip["trip_id"], n_obs=len(obs), n_pairs=len(pairs),
            miss_rate=1 - has_true.mean(), mean_cands=len(pairs) / max(len(obs), 1),
            label_dist_median=obs["true_dist"].median(), label_dist_p95=obs["true_dist"].quantile(0.95),
            seconds=round(time.time() - t0, 2)))
        pairs_all.append(pairs)
        obs_all.append(obs)
        print(f"[{split_name}] {k + 1}/{len(trips)} {trip['trip_id']}: {len(obs)} obs, "
              f"miss {quality[-1]['miss_rate']:.3f}, label dist median {quality[-1]['label_dist_median']:.1f} m")
    pd.concat(pairs_all, ignore_index=True).to_parquet(RESULTS / f"features_{split_name}.parquet")
    pd.concat(obs_all, ignore_index=True).to_parquet(RESULTS / f"obs_{split_name}.parquet")
    return quality


if __name__ == "__main__":
    q = run("train", TRAIN_DIR) + run("stest", STEST_DIR)
    qdf = pd.DataFrame(q)
    qdf.to_csv(RESULTS / "candidate_quality.csv", index=False)
    print(qdf.groupby("split")[["miss_rate", "mean_cands", "label_dist_median", "label_dist_p95"]].mean())
