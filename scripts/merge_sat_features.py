import json
import os
 
import pandas as pd
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "outputs", "features.geojson")
SATELLITE_PATH = os.path.join(BASE_DIR, "outputs", "satellite_features.csv")
 
JOIN_KEYS = ["lat", "lon", "acq_date"]
 
 
def load_points(path):
    with open(path, "r") as f:
        geo = json.load(f)
    rows = []
    for feature in geo["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        props = feature["properties"]
        rows.append({**props, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)
 
 
def run():
    print(f"Loading {FEATURES_PATH}")
    firms_df = load_points(FEATURES_PATH)
    print(f"  -> {len(firms_df)} points")
 
    print(f"Loading {SATELLITE_PATH}")
    sat_df = pd.read_csv(SATELLITE_PATH)
    print(f"  -> {len(sat_df)} points with satellite data")
 
    sat_subset = sat_df[JOIN_KEYS + ["ndvi", "nbr", "dnbr"]].drop_duplicates(subset=JOIN_KEYS)
 
    merged = firms_df.merge(sat_subset, on=JOIN_KEYS, how="left")
    matched = merged["ndvi"].notna().sum()
    print(f"  -> matched ndvi/nbr for {matched} / {len(merged)} points ({matched/len(merged):.1%})")
 
    property_cols = [c for c in merged.columns if c not in ("lat", "lon")]
    records = merged[property_cols].to_dict("records")
    lons = merged["lon"].to_numpy()
    lats = merged["lat"].to_numpy()
 
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lons[i], lats[i]]},
            "properties": records[i],
        }
        for i in range(len(merged))
    ]
 
    output_geo = {"type": "FeatureCollection", "features": features}
    with open(FEATURES_PATH, "w") as f:
        json.dump(output_geo, f, indent=2)
 
    print(f"Done. Updated {len(features)} features in {FEATURES_PATH}")
 
 
if __name__ == "__main__":
    run()
 