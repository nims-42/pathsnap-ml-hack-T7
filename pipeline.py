"""End-to-end prediction for one trip folder. The API only ever calls predict_trip and score_trip.

predict_trip never opens the .route file. score_trip is the only place that does, and it runs
server-side after the prediction is finished, so ground truth cannot reach the model or the UI.
"""
import json
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from config import ARTIFACTS, HMM_DEFAULTS
from preprocessing.network import RoadGraph
from hmm.viterbi import decode
from metrics import classifier_only_pred, top1_accuracy
from preprocessing.features import FEATURES, build_pairs, observation_table
from preprocessing.io import load_trip
from preprocessing.labels import align_route


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(ARTIFACTS / "model.joblib")


@lru_cache(maxsize=1)
def get_hmm_config():
    p = ARTIFACTS / "hmm_config.json"
    return {**HMM_DEFAULTS, **(json.load(open(p)) if p.exists() else {})}


def _arc_polyline(trip, arc):
    n, a = trip["nodes"], trip["arcs"]
    u, v = int(a["u"].iloc[arc]), int(a["v"].iloc[arc])
    return [[float(n["lat"].iloc[u]), float(n["lon"].iloc[u])], [float(n["lat"].iloc[v]), float(n["lon"].iloc[v])]]


def predict_trip(trip_dir, use_hmm=True):
    t0 = time.perf_counter()
    trip = load_trip(trip_dir, load_route=False)             # labels are NOT loaded here
    pairs, obs = build_pairs(trip, with_labels=False)
    model = get_model()
    prob = model.predict_proba(pairs[FEATURES].values)[:, 1] if len(pairs) else np.array([])
    clf_pred = classifier_only_pred(pairs, prob, len(obs))
    graph = RoadGraph(trip)
    cfg = get_hmm_config()
    hmm_pred = decode(pairs, obs, graph, prob, beta=cfg["beta"], alpha=cfg["alpha"],
                      max_gap_s=cfg["max_gap_s"], limit=cfg["dijkstra_limit"]) if use_hmm else clf_pred
    route = graph.fill_route(hmm_pred, limit=cfg["dijkstra_limit"])
    return dict(
        trip_id=trip["trip_id"], n_obs=len(obs),
        gps=obs[["lat", "lon"]].values.tolist(),
        pred_arcs=hmm_pred.tolist(), clf_arcs=clf_pred.tolist(), route_arcs=route,
        pred_polylines=[_arc_polyline(trip, a) for a in route],
        seconds=time.perf_counter() - t0, _trip=trip, _graph=graph)


def score_trip(trip_dir, result):
    """Server-side scoring against the reference route. Returns metrics only."""
    trip = load_trip(trip_dir, load_route=True)
    if trip["route"] is None:
        return dict(scored=False)
    obs = observation_table(trip)
    true_arc, _ = align_route(obs["x"].values, obs["y"].values, trip["route"], trip["arcs"])
    graph = result["_graph"]
    hmm_pred, clf_pred = np.array(result["pred_arcs"]), np.array(result["clf_arcs"])
    return dict(scored=True,
                top1_accuracy=top1_accuracy(hmm_pred, true_arc),
                classifier_only_top1=top1_accuracy(clf_pred, true_arc),
                route_continuity=graph.continuity(hmm_pred),
                classifier_only_continuity=graph.continuity(clf_pred),
                n_observations=int(len(obs)))
