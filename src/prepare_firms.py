import os
 
import pandas as pd
 

RAW_FILENAME = "VIIRS_SNPP_archive_cleaned.csv"
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", RAW_FILENAME)
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "firms_prepared.csv")
 
RENAME_MAP = {
    "latitude": "lat",
    "longitude": "lon",
}
 
EXTRA_KEEP_COLUMNS = ["brightness_diff", "confidence_score", "is_night", "bright_t31","type"]
 
REQUIRED_CORE_COLUMNS = ["lat", "lon", "frp", "brightness", "acq_date", "acq_time", "confidence"]
 
 
def run():
    print(f"Loading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"  -> {len(df)} rows, {len(df.columns)} columns")
 
    df = df.rename(columns=RENAME_MAP)
 
    missing_core = [c for c in REQUIRED_CORE_COLUMNS if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing required columns after rename: {missing_core}")
 
    keep_cols = REQUIRED_CORE_COLUMNS + [c for c in EXTRA_KEEP_COLUMNS if c in df.columns]
    df = df[keep_cols]
 

    before = len(df)
    df = df.dropna(subset=["lat", "lon", "frp"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    dropped = before - len(df)
    if dropped:
        print(f"  -> dropped {dropped} rows with missing/invalid lat, lon, or frp")
 
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
 
    print(f"Done. Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nDate range: {df['acq_date'].min()} to {df['acq_date'].max()}")
    print(f"Confidence value counts:\n{df['confidence'].value_counts()}")
 
 
if __name__ == "__main__":
    run()
 