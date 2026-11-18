"""Turning the .route file into one label per GPS observation.

The route is an ordered list of arcs (179 arcs for ~2500 GPS points in the sample trip),
so there is no row-to-row correspondence. We align them with a monotonic dynamic programme:

    cost[i, j]  = distance from observation i to route arc j
    dp[i, j]    = cost[i, j] + min over k <= j of dp[i-1, k]

i.e. the car can stay on an arc or move forward along the route, never backwards.
Backtracking gives the route index for each observation. Complexity O(n_obs * n_route),
trivial for these trip sizes.

This replaces the greedy "look ahead 10 arcs" idea: greedy can get stuck after one noisy
point, the DP optimises the whole trip at once.
"""
import numpy as np

from preprocessing.geo import point_to_segment


def align_route(obs_x, obs_y, route, arcs):
    """Returns (label_arc_id per obs, distance from obs to its label arc)."""
    a = arcs
    rx = route
    ax, ay = a["ax"].values[rx][None, :], a["ay"].values[rx][None, :]
    bx, by = a["bx"].values[rx][None, :], a["by"].values[rx][None, :]
    cost, _ = point_to_segment(obs_x[:, None], obs_y[:, None], ax, ay, bx, by)  # (n_obs, n_route)

    n, m = cost.shape
    dp = np.empty_like(cost)
    back = np.zeros((n, m), dtype=np.int32)
    dp[0] = cost[0]
    for i in range(1, n):
        prev = dp[i - 1]
        # running argmin over k <= j
        run_min = np.minimum.accumulate(prev)
        is_new = np.concatenate([[True], prev[1:] < run_min[:-1]])
        argmin_idx = np.maximum.accumulate(np.where(is_new, np.arange(m), 0))
        dp[i] = cost[i] + run_min
        back[i] = argmin_idx
    j = int(np.argmin(dp[-1]))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    labels = route[path]
    return labels, cost[np.arange(n), path]
