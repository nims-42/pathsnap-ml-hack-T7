"""Folium maps.

make_trip_map        analysis map: GPS, road network, reference route and predicted route (report only,
                     uses the reference route).
make_prediction_map  what the UI shows: GPS + predicted route only, built from the API response.
"""
import folium
import numpy as np


def _arc_latlon(trip, arc_ids):
    n, a = trip["nodes"], trip["arcs"]
    pts = []
    for k in arc_ids:
        u, v = int(a["u"].iloc[k]), int(a["v"].iloc[k])
        pts.append([[n["lat"].iloc[u], n["lon"].iloc[u]], [n["lat"].iloc[v], n["lon"].iloc[v]]])
    return pts


def _legend(html_items):
    rows = "".join(f'<div><span style="background:{c};width:14px;height:10px;display:inline-block;'
                   f'margin-right:6px"></span>{t}</div>' for c, t in html_items)
    return folium.Element(
        '<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;'
        f'padding:8px 10px;border:1px solid #888;font-size:12px">{rows}</div>')


def make_trip_map(trip, pred_obs=None, pred_arcs=None, out_html=None, network_margin_m=300):
    """pred_obs: DataFrame with lat/lon of the observations used for prediction (optional)."""
    tr = trip["track"]
    m = folium.Map(location=[tr["lat"].mean(), tr["lon"].mean()], zoom_start=14, tiles="OpenStreetMap")
    m.fit_bounds([[tr["lat"].min(), tr["lon"].min()], [tr["lat"].max(), tr["lon"].max()]])
    legend = []

    # road network: only arcs near the track so the HTML stays small
    arcs = trip["arcs"]
    near = ((arcs[["ax", "bx"]].min(axis=1) <= tr["x"].max() + network_margin_m) &
            (arcs[["ax", "bx"]].max(axis=1) >= tr["x"].min() - network_margin_m) &
            (arcs[["ay", "by"]].min(axis=1) <= tr["y"].max() + network_margin_m) &
            (arcs[["ay", "by"]].max(axis=1) >= tr["y"].min() - network_margin_m))
    g_net = folium.FeatureGroup(name="road network", show=False)
    for line in _arc_latlon(trip, np.flatnonzero(near.values)):
        folium.PolyLine(line, color="#999999", weight=1, opacity=0.6).add_to(g_net)
    g_net.add_to(m)
    legend.append(("#999999", "road network"))

    g_gps = folium.FeatureGroup(name="GPS observations")
    for lat, lon in zip(tr["lat"].values[::5], tr["lon"].values[::5]):
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True, weight=1).add_to(g_gps)
    g_gps.add_to(m)
    legend.append(("#1f77b4", "GPS observations (every 5th)"))

    if trip.get("route") is not None:
        g_ref = folium.FeatureGroup(name="reference route")
        for line in _arc_latlon(trip, trip["route"]):
            folium.PolyLine(line, color="#2ca02c", weight=5, opacity=0.8).add_to(g_ref)
        g_ref.add_to(m)
        legend.append(("#2ca02c", "reference route"))

    if pred_arcs is not None:
        g_pred = folium.FeatureGroup(name="predicted route")
        for line in _arc_latlon(trip, [a for a in pred_arcs if a >= 0]):
            folium.PolyLine(line, color="#d62728", weight=3, opacity=0.9, dash_array="6").add_to(g_pred)
        g_pred.add_to(m)
        legend.append(("#d62728", "predicted route"))

    m.get_root().html.add_child(_legend(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    if out_html:
        m.save(str(out_html))
    return m


def make_prediction_map(gps_latlon, pred_polylines, title=None):
    """gps_latlon: [[lat, lon], ...]; pred_polylines: [[[lat, lon], ...], ...] (one per segment)."""
    gps = np.asarray(gps_latlon)
    m = folium.Map(location=gps.mean(axis=0).tolist(), zoom_start=14, tiles="OpenStreetMap")
    m.fit_bounds([gps.min(axis=0).tolist(), gps.max(axis=0).tolist()])
    g_gps = folium.FeatureGroup(name="GPS observations")
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True, weight=1).add_to(g_gps)
    g_gps.add_to(m)
    g_pred = folium.FeatureGroup(name="predicted route")
    for line in pred_polylines:
        folium.PolyLine(line, color="#d62728", weight=4, opacity=0.9).add_to(g_pred)
    g_pred.add_to(m)
    m.get_root().html.add_child(_legend([("#1f77b4", "GPS observations"), ("#d62728", "predicted route")]))
    folium.LayerControl(collapsed=False).add_to(m)
    return m
