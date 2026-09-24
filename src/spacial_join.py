import pandas as pd 
import os
import math
import json
import numpy as np
from sklearn.neighbors import BallTree

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRMS_PATH = os.path.join(BASE_DIR, "data", "firms_prepared.csv")
OSM_PATH = os.path.join(BASE_DIR, "data", "osm_industrial.geojson")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "features.geojson")
R = 6371000

def load_firms(path):
    df = pd.read_csv(path)
    required = {"lat","lon","frp","brightness","acq_date","acq_time","confidence"}
    missing =required - set(df.columns)
    if missing:
        raise ValueError(f"firms data is missing columns:{missing}")
    return df

def load_facilities(path):
    with open(path,"r") as f:
        geo = json.load(f) 

    facilities=[]
    for feature in geo["features"]:
        lon, lat = feature["geometry"]["coordinates"] 
        facilities.append(
                {
                    "name": feature["properties"].get("name", "unknown"),
                    "facility_type": feature["properties"].get("facility_type", "unknown"),
                    "lat": lat,
                    "lon": lon,
                }
            )
    
    if not facilities:
        raise ValueError("osm_industrial.geojson has no facilities")
    return facilities
    
def build_facility_tree(facilities):
    coords_rad = np.radians([[fac["lat"], fac["lon"]] for fac in facilities])
    tree = BallTree(coords_rad, metric="haversine")
    return tree

def nearest_facility_batch(firms_df,facilities,tree):
    query_rad = np.radians(firms_df[["lat", "lon"]].to_numpy())
    dist_rad, idx = tree.query(query_rad, k=1)  
    
    dist_m = (dist_rad[:, 0] * R).round(1)
    nearest_idx = idx[:, 0]
    
    names = [facilities[i]["name"] for i in nearest_idx]
    types = [facilities[i]["facility_type"] for i in nearest_idx]
    return dist_m, names, types

def run():
    firms_df = load_firms(FIRMS_PATH)
    facilities = load_facilities(OSM_PATH)
    tree = build_facility_tree(facilities)
    
    distances, names, types = nearest_facility_batch(firms_df, facilities, tree)  
    firms_df["industrial_distance_m"] = distances
    firms_df["nearest_facility_name"] = names
    firms_df["nearest_facility_type"] = types

    features = []
    for _, row in firms_df.iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
                "properties": {
                    "frp": row["frp"],
                    "brightness": row["brightness"],
                    "acq_date": row["acq_date"],
                    "acq_time": row["acq_time"],
                    "confidence": row["confidence"],
                    "industrial_distance_m": row["industrial_distance_m"],
                    "nearest_facility_name": row["nearest_facility_name"],
                    "nearest_facility_type": row["nearest_facility_type"],
                },
            }
        )
 
    output_geo = {"type": "FeatureCollection", "features": features}
 
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_geo, f, indent=2)

    print(f" Wrote {len(features)} features to {OUTPUT_PATH}")
    print(firms_df[["lat", "lon", "industrial_distance_m", "nearest_facility_type"]])

if __name__ == "__main__":
        run()