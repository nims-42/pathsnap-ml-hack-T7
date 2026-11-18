"""Candidate radius sweep: how often is the true arc missing from the candidate set at each radius?

    python -m preprocessing.radius_sweep

Uses the 80 training trips only. Writes results/radius_sweep.csv.
"""
import pandas as pd

from config import MAX_CANDS, RESULTS, TRAIN_DIR
from preprocessing import candidates as cand
from preprocessing.features import observation_table
from preprocessing.io import list_trips, load_trip
from preprocessing.labels import align_route

RADII = [30, 50, 75, 100]

rows = []
for d in list_trips(TRAIN_DIR):
    trip = load_trip(d, load_route=True)
    obs = observation_table(trip)
    x, y, arcs = obs["x"].values, obs["y"].values, trip["arcs"]
    true_arc, _ = align_route(x, y, trip["route"], arcs)
    index = cand.build_arc_index(arcs)
    for r in RADII:
        c = cand.generate(x, y, arcs, index=index, radius=r)
        found = c[c["arc_id"].values == true_arc[c["obs_idx"].values]]["obs_idx"].nunique()
        rows.append(dict(trip_id=trip["trip_id"], radius_m=r, n_obs=len(obs), true_arc_found=found,
                         n_pairs=len(c), obs_with_no_candidates=len(obs) - c["obs_idx"].nunique()))
    print("done", trip["trip_id"], flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "radius_sweep_per_trip.csv", index=False)
s = df.groupby("radius_m")[["n_obs", "true_arc_found", "n_pairs", "obs_with_no_candidates"]].sum()
s["miss_rate"] = 1 - s["true_arc_found"] / s["n_obs"]
s["mean_candidates_per_obs"] = s["n_pairs"] / s["n_obs"]
s["pct_obs_no_candidates"] = 100 * s["obs_with_no_candidates"] / s["n_obs"]
s = s[["miss_rate", "mean_candidates_per_obs", "pct_obs_no_candidates"]].round(4)
s.to_csv(RESULTS / "radius_sweep.csv")
print(f"\n(candidates capped at the nearest {MAX_CANDS} per observation)\n")
print(s)