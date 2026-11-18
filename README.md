# PathSnap (phase 1: data preprocessing, feature engineering, model selection)

Noisy GPS points are matched to road-network arcs. Phase 1 covers everything up to model selection:
raw data inspection, candidate generation, label alignment, features, and a five-model comparison.
Tuning, the HMM and the API/UI come in the next phase.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Data: put `train/` and `student_test/` in `raw_data/`, or set `PATHSNAP_DATA` to the folder that contains them.
If this repo sits next to `Student folder/Student folder/` it is found automatically.
Run everything from the repository root.

## Run order

```
python -m preprocessing.eda               # inventory, plots, Folium map  -> results/
python -m preprocessing.build_features    # features_*.parquet, candidate_quality.csv
python -m models.train_models             # five models -> results/model_comparison.csv
python -m models.plots                    # comparison figures
```

`train_models` accepts `--only <name>` (logreg, knn, random_forest, lightgbm, mlp) and keeps the other
rows already in the comparison CSV. Run those one at a time, never two copies at once.

## What each folder does

preprocessing/   loading (`io`), projection to metres (`geo`), candidate arcs (`candidates`), label alignment
                 (`labels`), features (`features`), road graph and route continuity (`network`), Folium map
                 (`maps`), exploration (`eda`), table builder (`build_features`)
models/          shared CV/metric code (`harness`), the five models (`train_models`), figures (`plots`)
results/         every CSV, figure and map that goes in the report
config.py        all constants in one place (projection, step, radius, seed)

## Design choices to be able to explain

Coordinates are projected to UTM zone 40N so all distances are in metres. Every 5th GPS point is kept.
Candidates are arcs within 50 m of a point (KD-tree over points sampled every 10 m along each arc, then exact
distance), at most 8 per point. The route file lists arcs, not one label per GPS point, so labels come from a
monotonic dynamic-programming alignment of the points to the route. Each row is a (point, candidate arc) pair
with eight features: distance, distance rank, distance minus the nearest candidate, heading difference, speed,
arc length, projection fraction along the arc, and candidate count. Cross-validation is GroupKFold by trip, so
a trip never appears in both training and validation. Models are chosen on student-test Top-1, macro F1 and log
loss, not on training accuracy.


