"""HMM + Viterbi route reconstruction.

Hidden state at observation i  : which candidate arc the vehicle is on (the candidate set of that observation).
Emission  P(obs i | arc)       : classifier probability for that (obs, arc) row, normalised per observation,
                                 raised to the power alpha (alpha < 1 softens an over-confident classifier).
Transition P(arc_j | arc_k)    : exp(-|d_net - d_gc| / beta)   (Newson & Krumm 2009)
    d_gc  = straight-line distance between the two GPS points
    d_net = road distance between the two projected positions:
              same arc:       |frac_j - frac_k| * length
              different arcs: (1 - frac_k) * len_k  +  shortest_path(end of k -> start of j)  +  frac_j * len_j
    unreachable (beyond the Dijkstra limit) -> probability 1e-12.

Viterbi recurrence, in log space:
    delta_i(j) = log emission_i(j) + max_k [ delta_{i-1}(k) + log transition(k -> j) ]
with back-pointers, then backtrack from argmax delta_last.

Gaps: the sequence is cut into independent segments (and each decoded separately) wherever
  - an observation has no candidate arcs (it gets -1), or
  - the time gap to the previous observation exceeds max_gap_s.
"""
import numpy as np

from config import HMM_DEFAULTS

LOG_FLOOR = np.log(1e-12)


def distance_prob(pairs, sigma=10.0):
    """Stand-in emission before a classifier exists: Gaussian on perpendicular distance."""
    return np.exp(-(pairs["dist"].values ** 2) / (2 * sigma ** 2))


def _log_transition(graph, A, B, beta, limit):
    """A, B: dicts with arc, frac, x, y for two consecutive steps. Returns (K_A, K_B) log probs."""
    d_gc = np.hypot(B["x"] - A["x"], B["y"] - A["y"])
    la, lb = graph.length[A["arc"]], graph.length[B["arc"]]
    nodes_from = graph.v[A["arc"]]
    uniq, inv = np.unique(nodes_from, return_inverse=True)
    D = graph.dist_from(uniq, limit)[inv][:, graph.u[B["arc"]]]          # (K_A, K_B)
    d_net = (1.0 - A["frac"])[:, None] * la[:, None] + D + (B["frac"] * lb)[None, :]
    same = A["arc"][:, None] == B["arc"][None, :]
    d_same = np.abs(B["frac"][None, :] - A["frac"][:, None]) * la[:, None]
    d_net = np.where(same, d_same, d_net)
    logt = -np.abs(d_net - d_gc) / beta
    logt = np.where(np.isfinite(d_net), logt, LOG_FLOOR)
    return np.maximum(logt, LOG_FLOOR)


def _viterbi_segment(graph, steps, beta, limit):
    score = steps[0]["logp"].copy()
    back = []
    for s in range(1, len(steps)):
        trans = _log_transition(graph, steps[s - 1], steps[s], beta, limit)
        cand = score[:, None] + trans
        back.append(np.argmax(cand, axis=0))
        score = cand[back[-1], np.arange(cand.shape[1])] + steps[s]["logp"]
    j = int(np.argmax(score))
    states = [j]
    for bp in reversed(back):
        j = int(bp[j])
        states.append(j)
    states.reverse()
    return [int(steps[i]["arc"][k]) for i, k in enumerate(states)]


def decode(pairs, obs, graph, prob, beta=None, alpha=None, max_gap_s=None, limit=None):
    """pairs: candidate rows (obs_idx, arc_id, proj_frac, x, y); obs: observation table (obs_idx, t);
    prob: classifier probability per pairs row. Returns predicted arc per observation (-1 = none)."""
    cfg = {**HMM_DEFAULTS}
    for k, v in dict(beta=beta, alpha=alpha, max_gap_s=max_gap_s, dijkstra_limit=limit).items():
        if v is not None:
            cfg[k] = v

    n_obs = len(obs)
    pred = np.full(n_obs, -1, dtype=np.int64)
    order = np.argsort(pairs["obs_idx"].values, kind="stable")
    oi = pairs["obs_idx"].values[order]
    arc = pairs["arc_id"].values[order].astype(int)
    frac = pairs["proj_frac"].values[order]
    px, py = pairs["x"].values[order], pairs["y"].values[order]
    p = np.asarray(prob)[order]
    bounds = np.searchsorted(oi, np.arange(n_obs + 1))
    t = obs["t"].values

    segment, last_t = [], None

    def flush():
        if segment:
            arcs = _viterbi_segment(graph, segment, cfg["beta"], cfg["dijkstra_limit"])
            for st, a in zip(segment, arcs):
                pred[st["obs"]] = a
        segment.clear()

    for i in range(n_obs):
        lo, hi = bounds[i], bounds[i + 1]
        if hi == lo:                                   # no candidates: break the chain
            flush()
            last_t = None
            continue
        if last_t is not None and t[i] - last_t > cfg["max_gap_s"]:   # GPS gap: break the chain
            flush()
        pi = p[lo:hi]
        pi = pi / max(pi.sum(), 1e-12)
        segment.append(dict(obs=i, arc=arc[lo:hi], frac=frac[lo:hi], x=px[lo], y=py[lo],
                            logp=cfg["alpha"] * np.log(np.clip(pi, 1e-9, 1.0))))
        last_t = t[i]
    flush()
    return pred
