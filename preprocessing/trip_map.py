"""Folium map of one trip with all four layers: road network, GPS, reference route, predicted route.

    python -m preprocessing.trip_map <trip_folder>

"""
import sys
from pathlib import Path

from config import RESULTS
from pipeline import predict_trip
from preprocessing.io import load_trip
from preprocessing.maps import make_trip_map


def main(trip_dir):
    trip_dir = Path(trip_dir)
    res = predict_trip(trip_dir)
    trip = load_trip(trip_dir, load_route=True)
    out = RESULTS / f"map_{trip_dir.name}_predicted.html"
    make_trip_map(trip, pred_arcs=res["route_arcs"], out_html=out)
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1])
