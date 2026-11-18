"""Road network as a graph: shortest paths, route continuity, filling gaps in a predicted route."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from config import CONTINUITY_HOPS


class RoadGraph:
    def __init__(self, trip):
        a = trip["arcs"]
        self.n = len(trip["nodes"])
        self.u = a["u"].values
        self.v = a["v"].values
        self.length = a["length"].values
        w = np.maximum(self.length, 1e-3)  # zero weights are ambiguous in sparse csgraph
        df = pd.DataFrame({"u": self.u, "v": self.v, "w": w, "arc": np.arange(len(a))})
        df = df.sort_values("w").drop_duplicates(["u", "v"])  # parallel arcs: keep the shortest
        self.csr = csr_matrix((df["w"].values, (df["u"].values, df["v"].values)), shape=(self.n, self.n))
        self.uv2arc = {(int(x), int(y)): int(z) for x, y, z in zip(df["u"], df["v"], df["arc"])}
        order = np.argsort(self.u, kind="stable")
        self._out_start = np.searchsorted(self.u[order], np.arange(self.n + 1))
        self._out_v = self.v[order]
        self._reach_cache = {}

    def dist_from(self, nodes, limit):
        """Shortest road distance (m) from each node to every node, inf beyond `limit`."""
        return dijkstra(self.csr, directed=True, indices=nodes, limit=limit)

    def _within_edges(self, start, max_edges):
        key = (start, max_edges)
        if key in self._reach_cache:
            return self._reach_cache[key]
        seen, frontier = {start}, {start}
        for _ in range(max_edges):
            nxt = set()
            for node in frontier:
                nxt.update(self._out_v[self._out_start[node]:self._out_start[node + 1]].tolist())
            frontier = nxt - seen
            seen |= nxt
        self._reach_cache[key] = seen
        return seen

    def connected(self, a, b, hops=CONTINUITY_HOPS):
        """Same arc, or b reachable from a by traversing at most `hops` arcs."""
        if a == b:
            return True
        return int(self.u[b]) in self._within_edges(int(self.v[a]), hops - 1)

    def continuity(self, pred):
        """Share of consecutive observation pairs that are same-arc or connected within CONTINUITY_HOPS."""
        pred = np.asarray(pred)
        ok = n = 0
        for a, b in zip(pred[:-1], pred[1:]):
            if a < 0 or b < 0:
                continue
            n += 1
            ok += self.connected(int(a), int(b))
        return ok / n if n else float("nan")

    def fill_route(self, pred, limit=1500.0):
        """Ordered arc list: collapse repeats and insert the shortest road path between disconnected arcs."""
        seq = [int(a) for a in pred if a >= 0]
        out = []
        for a in seq:
            if not out or out[-1] == a:
                if not out:
                    out.append(a)
                continue
            prev = out[-1]
            if self.v[prev] != self.u[a]:
                d, pred_nodes = dijkstra(self.csr, directed=True, indices=int(self.v[prev]),
                                         limit=limit, return_predecessors=True)
                tgt = int(self.u[a])
                if np.isfinite(d[tgt]):
                    path, node = [tgt], tgt
                    while node != int(self.v[prev]) and pred_nodes[node] >= 0:
                        node = int(pred_nodes[node])
                        path.append(node)
                    path = path[::-1]
                    for x, y in zip(path[:-1], path[1:]):
                        arc = self.uv2arc.get((x, y))
                        if arc is not None and arc != out[-1]:
                            out.append(arc)
            out.append(a)
        return out
