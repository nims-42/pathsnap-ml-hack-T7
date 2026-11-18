"""Reading a raw trip folder.

Files (all whitespace separated, no header):
  .track  lon lat t      GPS observations, t in seconds
  .nodes  lon lat        road-network nodes, row index = node id (0-based)
  .arcs   from to        directed road segments, row index = arc id (0-based)
  .route  arc_id         reference route, chronological

The route is only read when load_route=True. The prediction pipeline never asks for it,
so labels cannot leak into predictions by accident.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.geo import project


def _read(path, names, dtype=None):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="python")


def load_trip(trip_dir, load_route=False):
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"])
    nodes = _read(trip_dir / f"{tid}.nodes", ["lon", "lat"])
    arcs = _read(trip_dir / f"{tid}.arcs", ["u", "v"], dtype=np.int64)

    gx, gy = project(track["lon"].values, track["lat"].values)
    nx_, ny_ = project(nodes["lon"].values, nodes["lat"].values)
    track = track.assign(x=gx, y=gy)
    nodes = nodes.assign(x=nx_, y=ny_)

    u, v = arcs["u"].values, arcs["v"].values
    ax, ay = nodes["x"].values[u], nodes["y"].values[u]
    bx, by = nodes["x"].values[v], nodes["y"].values[v]
    arcs = arcs.assign(ax=ax, ay=ay, bx=bx, by=by, length=np.hypot(bx - ax, by - ay))

    route = None
    if load_route:
        rp = trip_dir / f"{tid}.route"
        if rp.exists():
            route = pd.read_csv(rp, sep=r"\s+", header=None, engine="python")[0].values.astype(np.int64)

    return dict(trip_id=tid, dir=trip_dir, track=track, nodes=nodes, arcs=arcs, route=route)


def list_trips(partition_dir):
    partition_dir = Path(partition_dir)
    if not partition_dir.exists():
        return []
    return sorted(p for p in partition_dir.iterdir() if p.is_dir())
