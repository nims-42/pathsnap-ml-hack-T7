"""Turn one trip into (observation, candidate arc) rows with features.

One row = "is arc A the arc the vehicle was on at observation i?"  Target: label (0/1).
The route is touched ONLY when with_labels=True (training / scoring), never for prediction.
"""
import numpy as np
import pandas as pd

from config import STEP
from preprocessing import candidates as cand
from preprocessing.geo import angle_diff, bearing
from preprocessing.io import load_trip
from preprocessing.labels import align_route

FEATURES = ["dist", "dist_rank", "dist_minus_min", "heading_diff",
            "speed", "arc_len", "proj_frac", "n_cands"]


def observation_table(trip, step=STEP):
    tr = trip["track"]
    idx = np.arange(0, len(tr), step)
    obs = tr.iloc[idx][["lon", "lat", "t", "x", "y"]].reset_index(drop=True)
    obs.insert(0, "obs_idx", np.arange(len(obs)))
    obs["orig_idx"] = idx
    return obs


def _gps_motion(obs):
    """Central-difference heading (deg) and speed (m/s) per observation."""
    x, y, t = obs["x"].values, obs["y"].values, obs["t"].values
    n = len(obs)
    lo = np.maximum(np.arange(n) - 1, 0)
    hi = np.minimum(np.arange(n) + 1, n - 1)
    dx, dy, dt = x[hi] - x[lo], y[hi] - y[lo], t[hi] - t[lo]
    dt = np.where(dt > 0, dt, 1.0)
    return bearing(dx, dy), np.hypot(dx, dy) / dt


def build_pairs(trip, step=STEP, with_labels=False, radius=None):
    """Returns (pairs, obs). obs gets a true_arc column when with_labels=True."""
    obs = observation_table(trip, step)
    arcs = trip["arcs"]
    kw = {} if radius is None else {"radius": radius}
    c = cand.generate(obs["x"].values, obs["y"].values, arcs, **kw)

    gps_head, gps_speed = _gps_motion(obs)
    arc_head = bearing(arcs["bx"].values - arcs["ax"].values, arcs["by"].values - arcs["ay"].values)

    p = c.copy()
    p["dist_rank"] = p.groupby("obs_idx").cumcount()
    p["dist_minus_min"] = p["dist"] - p.groupby("obs_idx")["dist"].transform("min")
    p["n_cands"] = p.groupby("obs_idx")["arc_id"].transform("size")
    p["heading_diff"] = angle_diff(gps_head[p["obs_idx"].values], arc_head[p["arc_id"].values])
    p["speed"] = gps_speed[p["obs_idx"].values]
    p["arc_len"] = arcs["length"].values[p["arc_id"].values]

    p.insert(0, "trip_id", trip["trip_id"])
    p["t"] = obs["t"].values[p["obs_idx"].values]
    p["x"] = obs["x"].values[p["obs_idx"].values]
    p["y"] = obs["y"].values[p["obs_idx"].values]

    if with_labels:
        if trip["route"] is None:
            raise ValueError(f"trip {trip['trip_id']} has no route to label from")
        true_arc, true_dist = align_route(obs["x"].values, obs["y"].values, trip["route"], arcs)
        obs["true_arc"] = true_arc
        obs["true_dist"] = true_dist
        p["label"] = (p["arc_id"].values == true_arc[p["obs_idx"].values]).astype(int)
    return p.reset_index(drop=True), obs


def load_and_build(trip_dir, with_labels=True):
    trip = load_trip(trip_dir, load_route=with_labels)
    return trip, *build_pairs(trip, with_labels=with_labels)
