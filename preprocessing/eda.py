"""Dataset exploration for the report.

    python -m preprocessing.eda

Writes results/eda_inventory.csv, results/eda_summary.txt, results/eda_*.png and one Folium map
(results/map_<trip>.html) for the first student-test trip.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from preprocessing.maps import make_trip_map
from config import RESULTS, STEST_DIR, TRAIN_DIR
from preprocessing.io import list_trips, load_trip


def inspect(trip, partition):
    tr, nd, ar = trip["track"], trip["nodes"], trip["arcs"]
    dt = np.diff(tr["t"].values)
    n = len(nd)
    g = coo_matrix((np.ones(len(ar)), (ar["u"], ar["v"])), shape=(n, n))
    n_comp, comp = connected_components(g, directed=True, connection="weak")
    comp_sizes = np.bincount(comp)
    deg = np.bincount(ar["u"], minlength=n) + np.bincount(ar["v"], minlength=n)
    route = trip["route"]
    return dict(
        partition=partition, trip_id=trip["trip_id"], n_points=len(tr),
        duration_s=tr["t"].iloc[-1] - tr["t"].iloc[0],
        n_missing=int(tr.isna().sum().sum()), non_monotonic_t=int((dt <= 0).sum()),
        dt_not_1s=int((dt != 1).sum()), max_gap_s=dt.max(),
        track_extent_km=float(max(tr["x"].max() - tr["x"].min(), tr["y"].max() - tr["y"].min()) / 1000),
        n_nodes=n, n_arcs=len(ar), n_components=int(n_comp), largest_component=int(comp_sizes.max()),
        isolated_nodes=int((deg == 0).sum()), mean_arc_len_m=float(ar["length"].mean()),
        zero_len_arcs=int((ar["length"] == 0).sum()),
        route_arcs=None if route is None else len(route),
        route_bad_ids=None if route is None else int(((route < 0) | (route >= len(ar))).sum()),
        route_connected_frac=None if route is None or len(route) < 2 else
        float((ar["v"].values[route[:-1]] == ar["u"].values[route[1:]]).mean()))


def main():
    rows = []
    first_trip = None
    for name, folder in [("train", TRAIN_DIR), ("student_test", STEST_DIR)]:
        for d in list_trips(folder):
            trip = load_trip(d, load_route=True)
            rows.append(inspect(trip, name))
            if name == "student_test" and first_trip is None:
                first_trip = trip
            print(name, trip["trip_id"], rows[-1]["n_points"], "points")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "eda_inventory.csv", index=False)

    with open(RESULTS / "eda_summary.txt", "w") as f:
        f.write(f"trips per partition:\n{df.partition.value_counts().to_string()}\n\n")
        f.write(df.groupby("partition")[["n_points", "max_gap_s", "dt_not_1s", "n_nodes", "n_arcs",
                                         "n_components", "isolated_nodes", "route_arcs",
                                         "route_connected_frac"]].describe().T.to_string())
        f.write(f"\n\ntotal missing values: {df.n_missing.sum()}  non-monotonic timestamps: {df.non_monotonic_t.sum()}\n")
        f.write(f"trips with a gap > 5 s: {(df.max_gap_s > 5).sum()}\n")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, c in [("train", "#1f77b4"), ("student_test", "#ff7f0e")]:
        s = df[df.partition == p]
        ax[0].hist(s.n_points, bins=20, alpha=0.6, color=c, label=p)
        ax[1].hist(s.max_gap_s, bins=20, alpha=0.6, color=c, label=p)
        ax[2].scatter(s.n_arcs, s.n_points, alpha=0.7, color=c, label=p)
    ax[0].set(title="GPS points per trip", xlabel="points")
    ax[1].set(title="largest time gap per trip", xlabel="seconds")
    ax[2].set(title="network size vs trip length", xlabel="arcs", ylabel="points")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "eda_distributions.png", dpi=150)

    if first_trip is not None:
        tr = first_trip["track"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ar, nd = first_trip["arcs"], first_trip["nodes"]
        segs = np.stack([ar[["ax", "ay"]].values, ar[["bx", "by"]].values], axis=1)
        for s in segs[::3]:
            ax[0].plot(s[:, 0], s[:, 1], color="#cccccc", lw=0.4)
        ax[0].plot(tr["x"], tr["y"], ".", ms=1, color="#1f77b4")
        ax[0].set_title(f"trip {first_trip['trip_id']}: GPS (blue) over road network (m, UTM 40N)")
        ax[0].set_aspect("equal")
        ax[1].plot(np.diff(tr["t"].values), lw=0.6)
        ax[1].set(title="time step between observations", xlabel="observation", ylabel="seconds")
        plt.tight_layout()
        plt.savefig(RESULTS / "eda_trajectory.png", dpi=150)
        make_trip_map(first_trip, out_html=RESULTS / f"map_{first_trip['trip_id']}.html")
    print("done, see results/")


if __name__ == "__main__":
    main()
