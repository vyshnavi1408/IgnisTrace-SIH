#joind the farmland and mining.
import json
import os
 
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
 
EARTH_RADIUS_M = 6371000
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "outputs", "features.geojson")
 

LANDCOVER_SOURCES = [
    ("osm_farmland.geojson", "farmland_distance_m"),
    ("osm_quarry.geojson", "mining_distance_m"),
]
 
 
def load_points(path):
    with open(path, "r") as f:
        geo = json.load(f)
    rows = []
    for feature in geo["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        props = feature["properties"]
        rows.append({**props, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)
 
 
def add_distance_column(firms_df, source_path, column_name):
    print(f"Loading {source_path}")
    source_df = load_points(source_path)
    print(f"  -> {len(source_df)} points")
 
    coords_rad = np.radians(source_df[["lat", "lon"]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")
 
    query_rad = np.radians(firms_df[["lat", "lon"]].to_numpy())
    dist_rad, _ = tree.query(query_rad, k=1)
    firms_df[column_name] = (dist_rad[:, 0] * EARTH_RADIUS_M).round(1)
    print(f"  -> added '{column_name}'")
 
 
def run():
    print(f"Loading FIRMS features from {FEATURES_PATH}")
    firms_df = load_points(FEATURES_PATH)
    print(f"  -> {len(firms_df)} points")
 
    for filename, column_name in LANDCOVER_SOURCES:
        source_path = os.path.join(BASE_DIR, "data", filename)
        print(f"\nComputing {column_name}...")
        add_distance_column(firms_df, source_path, column_name)
 
    
    property_cols = [c for c in firms_df.columns if c not in ("lat", "lon")]
    records = firms_df[property_cols].to_dict("records")
    lons = firms_df["lon"].to_numpy()
    lats = firms_df["lat"].to_numpy()
 
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lons[i], lats[i]]},
            "properties": records[i],
        }
        for i in range(len(firms_df))
    ]
 
    output_geo = {"type": "FeatureCollection", "features": features}
    with open(FEATURES_PATH, "w") as f:
        json.dump(output_geo, f, indent=2)
 
    print(f"\nDone. Updated {len(features)} features in {FEATURES_PATH}")
    for _, column_name in LANDCOVER_SOURCES:
        print(f"\n{column_name}:")
        print(firms_df[column_name].describe())
 
 
if __name__ == "__main__":
    run()
 