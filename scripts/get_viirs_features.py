import ee
import json
import pandas as pd
from datetime import timedelta


PROJECT_ID = "ignistrace-sih"

ee.Initialize(project=PROJECT_ID)



INPUT_FILE = "../outputs/features.geojson"
OUTPUT_FILE = "../outputs/features.geojson"



with open(INPUT_FILE, "r") as f:
    geojson = json.load(f)

features = geojson["features"]

print(f"Loaded {len(features)} features")


def get_viirs_data(lat, lon, date_string):

    point = ee.Geometry.Point([lon, lat])

    
    start_date = ee.Date(date_string)
    end_date = start_date.advance(1, "day")

    collection = (
        ee.ImageCollection("NASA/VIIRS/002/VNP14A1")
        .filterDate(start_date, end_date)
    )

    
    max_frp_image = collection.select("MaxFRP").max()

    
    firemask_image = collection.select("FireMask").max()

    
    image = max_frp_image.addBands(firemask_image)

    result = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=1000,
        maxPixels=1e6
    )

    values = result.getInfo()

    max_frp = values.get("MaxFRP")
    firemask = values.get("FireMask")

    
    if firemask is not None and firemask >= 7:
        detected = 1
    else:
        detected = 0

    return max_frp, firemask, detected




for i, feature in enumerate(features):

    properties = feature["properties"]

    lat = properties.get("lat")
    lon = properties.get("lon")
    date = properties.get("acq_date")

    
    if lat is None or lon is None or date is None:

        properties["viirs_max_frp"] = None
        properties["viirs_firemask"] = None
        properties["viirs_detected"] = 0

        continue

    try:

        max_frp, firemask, detected = get_viirs_data(
            lat,
            lon,
            date
        )

        properties["viirs_max_frp"] = max_frp
        properties["viirs_firemask"] = firemask
        properties["viirs_detected"] = detected

    except Exception as e:

        print(f"Error at feature {i}: {e}")

        properties["viirs_max_frp"] = None
        properties["viirs_firemask"] = None
        properties["viirs_detected"] = 0

    
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(features)}")




with open(OUTPUT_FILE, "w") as f:
    json.dump(geojson, f, indent=2)

print()
print("Done!")
print(f"Saved to: {OUTPUT_FILE}")