"""Candidate generation: for each GPS point, which road segments could it belong to?

Method: sample points every SAMPLE_SPACING metres along every arc, put them in a KD-tree,
ask the tree for samples near each GPS point, then compute the EXACT point-to-segment
distance for the arcs those samples belong to. Keep arcs within CAND_RADIUS, nearest MAX_CANDS.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import CAND_RADIUS, MAX_CANDS, SAMPLE_SPACING
from preprocessing.geo import point_to_segment


def build_arc_index(arcs):
    length = arcs["length"].values
    n = np.maximum(np.ceil(length / SAMPLE_SPACING).astype(int), 1) + 1
    arc_ids = np.repeat(np.arange(len(arcs)), n)
    # position of each sample along its arc, 0..1
    offsets = np.concatenate([np.arange(k) for k in n])
    frac = offsets / np.repeat(n - 1, n)
    ax, ay = arcs["ax"].values[arc_ids], arcs["ay"].values[arc_ids]
    bx, by = arcs["bx"].values[arc_ids], arcs["by"].values[arc_ids]
    pts = np.column_stack([ax + frac * (bx - ax), ay + frac * (by - ay)])
    return cKDTree(pts), arc_ids


def generate(obs_x, obs_y, arcs, index=None, radius=CAND_RADIUS, max_cands=MAX_CANDS):
    """Returns DataFrame[obs_idx, arc_id, dist, proj_frac] (obs_idx = position in obs_x)."""
    tree, sample_arc = index if index is not None else build_arc_index(arcs)
    # a sample can be up to half a spacing away from the true closest point on the arc
    hits = tree.query_ball_point(np.column_stack([obs_x, obs_y]), r=radius + SAMPLE_SPACING)
    counts = np.fromiter((len(h) for h in hits), dtype=int, count=len(hits))
    if counts.sum() == 0:
        return pd.DataFrame(columns=["obs_idx", "arc_id", "dist", "proj_frac"])
    oi = np.repeat(np.arange(len(hits)), counts)
    ai = sample_arc[np.concatenate([np.asarray(h, dtype=int) for h in hits if len(h)])]
    df = pd.DataFrame({"obs_idx": oi, "arc_id": ai}).drop_duplicates()

    a = arcs
    d, t = point_to_segment(
        obs_x[df["obs_idx"].values], obs_y[df["obs_idx"].values],
        a["ax"].values[df["arc_id"].values], a["ay"].values[df["arc_id"].values],
        a["bx"].values[df["arc_id"].values], a["by"].values[df["arc_id"].values])
    df["dist"], df["proj_frac"] = d, t
    df = df[df["dist"] <= radius].sort_values(["obs_idx", "dist"])
    df = df.groupby("obs_idx", sort=True).head(max_cands).reset_index(drop=True)
    return df
