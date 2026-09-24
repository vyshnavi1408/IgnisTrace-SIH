
import json
import os
 
import ee
import pandas as pd
 
GCP_PROJECT_ID = "ignistrace-sih"  
SAMPLE_SIZE = 5000          
PRE_FIRE_START_DAYS = -60   
PRE_FIRE_END_DAYS = -15
POST_FIRE_START_DAYS = 10   
POST_FIRE_END_DAYS = 45
CLOUD_FILTER_PCT = 40       
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "outputs", "features_labeled.geojson")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "satellite_features.csv")
 
 
def load_points(path):
    with open(path, "r") as f:
        geo = json.load(f)
    rows = []
    for feature in geo["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        props = feature["properties"]
        rows.append({**props, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)
 
 
def mask_s2_clouds(image):
   
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000) 
 
 
def build_composite(start, end, region):
    
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER_PCT))
        .map(mask_s2_clouds)
    )
    return collection.median()
 
 
def get_pre_post_bands(anchor_date_str, region):
    
    anchor = ee.Date(anchor_date_str)
 
    pre_start = anchor.advance(PRE_FIRE_START_DAYS, "day")
    pre_end = anchor.advance(PRE_FIRE_END_DAYS, "day")
    post_start = anchor.advance(POST_FIRE_START_DAYS, "day")
    post_end = anchor.advance(POST_FIRE_END_DAYS, "day")
 
    pre_composite = build_composite(pre_start, pre_end, region)
    post_composite = build_composite(post_start, post_end, region)
 
    ndvi = post_composite.normalizedDifference(["B8", "B4"]).rename("ndvi")
    nbr_post = post_composite.normalizedDifference(["B8", "B12"]).rename("nbr")
    nbr_pre = pre_composite.normalizedDifference(["B8", "B12"]).rename("nbr_pre")
 
    return ndvi.addBands(nbr_post).addBands(nbr_pre)
 
 
def run():
    print(f"Initializing Earth Engine (project={GCP_PROJECT_ID})...")
    ee.Initialize(project=GCP_PROJECT_ID)
 
    print(f"Loading points from {INPUT_PATH}")
    df = load_points(INPUT_PATH)
    print(f"  -> {len(df)} total points")
 
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"  -> sampled down to {len(df)} points (SAMPLE_SIZE={SAMPLE_SIZE})")
 
    df["row_id"] = df.index.astype(str)
    df["year_month"] = pd.to_datetime(df["acq_date"]).dt.to_period("M").astype(str)
 
    all_results = []
    groups = df.groupby("year_month")
    print(f"\nProcessing {len(groups)} monthly groups...")
 
    for i, (year_month, group) in enumerate(groups):
        print(f"  [{i + 1}/{len(groups)}] {year_month}: {len(group)} points")
 
        ee_points = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([row["lon"], row["lat"]]), {"row_id": row["row_id"]})
            for _, row in group.iterrows()
        ])
 
        region = ee_points.geometry().bounds()
        center_date = f"{year_month}-15"  
        try:
            image = get_pre_post_bands(center_date, region)
            sampled = image.reduceRegions(
                collection=ee_points, reducer=ee.Reducer.first(), scale=10
            )
            results = sampled.getInfo()["features"]
            for feat in results:
                props = feat["properties"]
                all_results.append({
                    "row_id": props.get("row_id"),
                    "ndvi": props.get("ndvi"),
                    "nbr": props.get("nbr"),
                    "nbr_pre": props.get("nbr_pre"),
                })
        except Exception as e:
            print(f"    ERROR on {year_month}: {e} -- skipping this group")
            continue
 
    result_df = pd.DataFrame(all_results)
    merged = df.merge(result_df, on="row_id", how="left")
 
    
    merged["dnbr"] = merged["nbr_pre"] - merged["nbr"]
 
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
 
    retrieved = merged["ndvi"].notna().sum() if "ndvi" in merged.columns else 0
    print(f"\nDone. Wrote {len(merged)} rows to {OUTPUT_PATH}")
    print(f"NDVI/NBR/dNBR successfully retrieved for {retrieved} / {len(merged)} points")
    print("(missing values usually mean no sufficiently cloud-free Sentinel-2")
    print(" scene existed in that point's date window -- expected during monsoon)")
    print(merged[["lat", "lon", "acq_date", "ndvi", "nbr", "dnbr"]].head(10))
 
 
if __name__ == "__main__":
    run()
 


