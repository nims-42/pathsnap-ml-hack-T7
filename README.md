# PathSnap: map-matching and route reconstruction

Noisy GPS points are matched to road-network arcs in two stages. A classifier scores every (GPS point,
nearby arc) pair, then an HMM with Viterbi decoding picks the most coherent, connected route. The API
(Laptop 1) runs the whole pipeline and scores the result against server-side reference data. The UI
(Laptop 2) only displays what the API returns.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Data: put `train/` and `student_test/` in `raw_data/`, or set the `PATHSNAP_DATA` environment variable to the
folder that contains them. If this repo sits next to `Student folder/Student folder/` it is found automatically.
All commands are run from the repository root.

## Run order (reproduces every number in the report)

```
python -m preprocessing.eda                              # dataset inventory, plots, Folium map
python -m preprocessing.build_features                   # candidates, labels, features -> results/
python -m preprocessing.radius_sweep                     # candidate radius study
python -m models.train_models                            # five-model comparison -> results/model_comparison.csv
python -m models.tune_optuna --model random_forest --trials 15    # Bayesian search -> artifacts/model.joblib
python -m hmm.evaluate_hmm                               # classifier-only vs HMM, tunes beta and alpha
python -m models.plots                                   # report figures
```

`train_models --only <name>` re-runs one model and keeps the other rows. `tune_optuna --model lightgbm` tunes
LightGBM instead. The tuner's search ranges for the random forest are deliberately bounded so the saved
model stays under GitHub's 100 MB file limit, and the model is saved compressed.

## Running the system (two laptops, same network)

Laptop 1 (model and API):

```
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Allow port 8000 through the Windows firewall. The API loads `artifacts/model.joblib` and
`artifacts/hmm_config.json` once at startup, and logs every request to this terminal.

Laptop 2 (UI): set `"api_url"` in `app/config.json` to `http://<laptop1-ip>:8000`, then

```
python app/ui.py
```

then open `http://127.0.0.1:8080` in a browser (the API address can also be edited on the page).

The UI needs internet access for the map tiles. The trips offered come from the folder set by `"eval_dir"` in
`app/config.json` (or the `PATHSNAP_EVAL_DIR` environment variable). `"auto"` means the student-test trips,
and for the final evaluation it should point at the folder holding the evaluation trips.

Endpoints: `GET /trips` lists trip IDs, `POST /predict` with `{"trip_id": "..."}` returns the GPS points, the
predicted route and accuracy metrics. The reference route is read only inside `score_trip()` on the server
and is never part of a response.

## Layout

| Folder or file | Contents |
|---|---|
| `preprocessing/` | loading, projection to metres, candidate arcs, label alignment, features, road graph, Folium maps, EDA, radius sweep |
| `models/` | cross-validation harness, five-model comparison, Optuna tuning, figures |
| `hmm/` | Viterbi decoder and decoder tuning (`evaluate_hmm`) |
| `app/` | FastAPI backend, HTML + UI (`ui.html`, served by `ui.py`), deployment config |
| `pipeline.py` | `predict_trip` (never reads labels) and `score_trip` (server-side scoring) |
| `artifacts/` | saved model, feature list, HMM configuration |
| `results/` | every CSV, figure and map used in the report |
| `config.py` | all constants in one place |

## Method in brief

Coordinates are projected to UTM zone 40N so distances are in metres, and every 5th GPS point is kept.
Candidates are arcs within 50 m (at most 8 per point). Labels come from a monotonic dynamic-programming
alignment of the GPS points to the reference route. Each row is a (point, candidate) pair with eight
features: distance, distance rank, distance minus the nearest candidate, heading difference, speed, arc
length, projection fraction along the arc, and candidate count. Training trips whose aligned route lies more
than 50 m (median) from their own GPS are excluded (`LABEL_DIST_MAX` in `config.py`, listed in
`results/excluded_trips.csv`). Cross-validation is GroupKFold by trip.

HMM: the hidden state at each point is its candidate arc. The emission is the classifier probability
(raised to the power alpha). The transition is exp(-|d_net - d_gc| / beta), comparing the road distance
between two candidates with the straight-line distance between the two GPS points. Unreachable pairs get a
tiny probability, and a trip is cut into independent segments at GPS gaps over 30 s or points with no
candidates. Beta and alpha are chosen on the student-test trips only.

## Integrity

The evaluator trips are never used for training, feature selection or tuning. `predict_trip` never opens the
`.route` file. Every number in the report must come from a file in `results/`. Disclose AI assistance as the
event policy requires, and make sure each member can explain the code they own.
