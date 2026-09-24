from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARCHIVE_FILE = os.path.join(BASE_DIR, "data", "MODIS_archive_cleaned.csv")
FEATURES_FILE = os.path.join(BASE_DIR, "outputs", "features.geojson")
MANIFEST_FILE = os.path.join(BASE_DIR, "outputs", "feature_manifest.json")

RADIUS_M = 1000          
EARTH_RADIUS_M = 6_371_000
MIN_OBS_FOR_STD = 2       
MIN_OBS_FOR_NIGHT = 5     
REF_SAMPLE = 4000         

RULE_INPUT_COLUMNS = ("persistence_count", "historical_n_detections", "type")

REQUIRED_ARCHIVE = {"latitude", "longitude", "acq_date", "acq_datetime", "frp", "is_night"}

FLOAT_KEYS = ()




def clean(v):
    #fills nan values as geojso can only have None vlaues.
    
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _std(vals, ddof=0):
    #caluclates the standard deviation.
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < MIN_OBS_FOR_STD:
        return None
    return float(np.std(vals, ddof=ddof))


def _cv(vals):
    #calculates coefficient of variation.how variable is the thermal intensity to its typical intensity.
    
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < MIN_OBS_FOR_STD:
        return None
    m = float(vals.mean())
    if m == 0.0:
        return None
    return float(np.std(vals, ddof=0) / abs(m))


def _slope(t, y):
    #took look at the tend of frp.
    
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < MIN_OBS_FOR_STD:
        return None
    m = float(y.mean())
    if m == 0.0:
        return None
    if np.ptp(t) == 0.0:
        return None
    return float(np.polyfit(t, y / m, 1)[0])


def _bbox_area_km2(lat, lon):
    if len(lat) < 2:
        return None
    cosl = float(np.cos(np.radians(np.mean(lat))))
    dl = float(np.max(lat) - np.min(lat)) * 111.32
    dn = float(np.max(lon) - np.min(lon)) * 111.32 * cosl
    return dl * dn



def load_archive(path):
    a = pd.read_csv(path)
    missing = REQUIRED_ARCHIVE - set(a.columns)
    if missing:
        raise ValueError(f"MODIS archive is missing columns: {sorted(missing)}")

    a["acq_date"] = pd.to_datetime(a["acq_date"], errors="coerce").dt.normalize()
    a["acq_datetime"] = pd.to_datetime(a["acq_datetime"], errors="coerce")

    n0 = len(a)
    a = a.dropna(subset=["latitude", "longitude", "acq_date"]).copy()
    
    bad_dt = int(a["acq_datetime"].isna().sum())
    a = a[a["acq_datetime"].notna()].copy()
    if bad_dt:
        print(f"  dropped {bad_dt:,} archive rows with unparseable acq_datetime")
    print(f"Loaded {len(a):,} historical observations "
          f"({n0 - len(a):,} dropped), "
          f"{a['acq_date'].min().date()} .. {a['acq_date'].max().date()}")

    if "brightness_diff" not in a.columns and {"brightness", "bright_t31"} <= set(a.columns):
        a["brightness_diff"] = a["brightness"] - a["bright_t31"]

    a = a.sort_values("acq_datetime").reset_index(drop=True)
    a["_lat_r"] = np.radians(a["latitude"].to_numpy(float))
    a["_lon_r"] = np.radians(a["longitude"].to_numpy(float))
    a["_dt64"] = a["acq_datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")

    tree = BallTree(a[["_lat_r", "_lon_r"]].to_numpy(), metric="haversine")
    return a, tree



def features_from_hist(hist, archive, cur_dt64, cur_day, cur_is_night):
    """hist must already be causally masked (strictly before the event)."""
    out = {}
    n = len(hist)
    out["historical_has_prev"] = int(n > 0)
    out["historical_n_detections"] = int(n)

    if n == 0:
        for k in ("frp_mean", "frp_std", "frp_cv", "frp_max", "frp_slope_per_day",
                  "bt31_mean", "bt31_std", "bdiff_mean", "bdiff_max",
                  "confidence_mean", "cluster_night_fraction", "gap_median_days",
                  "gap_std_days", "duration_h", "rise_to_peak_h",
                  "centroid_jitter_m", "bbox_area_km2", "n_distinct_days",
                  "span_days", "n_det_z"):
            out[k] = None
        out["historical_span_days"] = None
        out["n_satellites"] = 0
        out["same_day_earlier_dets"] = 0
        return out

    lat = hist["latitude"].to_numpy(float)
    lon = hist["longitude"].to_numpy(float)
    dt = pd.to_datetime(hist["acq_datetime"])
    dt64 = hist["_dt64"].to_numpy()
    t_days = (dt64 - dt64.min()) / 8.64e10
    dates = pd.to_datetime(hist["acq_date"]).drop_duplicates()
    n_days = int(len(dates))
    span = int((dates.max() - dates.min()).days + 1) if n_days else None
    daily = hist.groupby("acq_date").size()

    out["n_distinct_days"] = n_days
    out["historical_span_days"] = span
    out["max_det_per_day"] = int(daily.max()) if len(daily) else 0
    out["n_satellites"] = int(hist["satellite"].nunique()) if "satellite" in hist else 0
    
    out["same_day_earlier_dets"] = int((dt64 // 8.64e10 == cur_day).sum())


    frp = pd.to_numeric(hist["frp"], errors="coerce").to_numpy(float)
    out["frp_mean"] = float(np.nanmean(frp)) if np.isfinite(frp).any() else None
    out["frp_max"] = float(np.nanmax(frp)) if np.isfinite(frp).any() else None
    out["frp_std"] = _std(frp)
    out["frp_cv"] = _cv(frp)
    out["frp_slope_per_day"] = _slope(t_days, frp)

    for src, dst in (("bright_t31", "bt31"), ("brightness_diff", "bdiff"),
                     ("brightness", "brightness")):
        if src in hist:
            v = pd.to_numeric(hist[src], errors="coerce").to_numpy(float)
            out[f"{dst}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else None
            out[f"{dst}_std"] = _std(v)
            if dst == "bdiff":
                out["bdiff_max"] = float(np.nanmax(v)) if np.isfinite(v).any() else None

    if "confidence" in hist:
        c = pd.to_numeric(hist["confidence"], errors="coerce").to_numpy(float)
        out["confidence_mean"] = float(np.nanmean(c)) if np.isfinite(c).any() else None

    
    night = pd.to_numeric(hist["is_night"], errors="coerce").to_numpy(float)
    nf = night[np.isfinite(night)]
    out["cluster_night_fraction"] = float(nf.mean()) if nf.size >= MIN_OBS_FOR_NIGHT else None


    if n_days >= MIN_OBS_FOR_STD:
        gaps = np.diff(np.sort(dt.to_numpy().astype("datetime64[D]").astype("int64")))
        out["gap_median_days"] = float(np.median(gaps))
        out["gap_std_days"] = float(np.std(gaps, ddof=0))
    else:
        out["gap_median_days"] = None
        out["gap_std_days"] = None

    
    if len(dt) >= 2:
        out["duration_h"] = float((dt.max() - dt.min()).total_seconds() / 3600.0)
    else:
        out["duration_h"] = 0.0

    peak_i = int(np.nanargmax(frp)) if np.isfinite(frp).any() else None
    if peak_i is not None and len(dt) >= 2:
        first_t = dt.iloc[0]
        peak_t = dt.iloc[peak_i]
        out["rise_to_peak_h"] = float((peak_t - first_t).total_seconds() / 3600.0)
    else:
        out["rise_to_peak_h"] = None

    
    if n >= 4:
        cosl = float(np.cos(np.mean(np.radians(lat))))
        out["centroid_jitter_m"] = 111_320.0 * float(
            np.sqrt(np.var(lat) + np.var(lon) * cosl ** 2)
        )
        out["bbox_area_km2"] = _bbox_area_km2(lat, lon)
    else:
        out["centroid_jitter_m"] = None
        out["bbox_area_km2"] = None

    return out



def build_reference(archive, tree, radius_m, n_sample=REF_SAMPLE, seed=0):
    
    rng = np.random.default_rng(seed)
    n = len(archive)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    rad = radius_m / EARTH_RADIUS_M
    recs = []
    for i in idx:
        row = archive.iloc[i]
        nb = tree.query_radius(np.radians([[row.latitude, row.longitude]])[0:1], r=rad)[0]
        nb = nb[archive["_dt64"].to_numpy()[nb] < row._dt64]
        recs.append((int(row.month), int(row.is_night), len(nb)))
    ref = pd.DataFrame(recs, columns=["month", "is_night", "cnt"])
    g = ref.groupby(["month", "is_night"])["cnt"].agg(["mean", "std", "size"])
    g["std"] = g["std"].replace(0.0, np.nan)
    glob = float(ref.cnt.std() or 1.0)
    print(f"  z-score reference from {len(ref):,} sampled points, "
          f"global sd={glob:.2f}; cells with <50 samples fall back to global")
    return g.to_dict("index"), (float(ref.cnt.mean()), glob)



def run(archive_path, features_path, radius_m, limit=None):
    archive, tree = load_archive(archive_path)
    dt64_all = archive["_dt64"].to_numpy()
    lat_all = archive["latitude"].to_numpy(float)
    lon_all = archive["longitude"].to_numpy(float)
    rad = radius_m / EARTH_RADIUS_M

    with open(features_path, "r", encoding="utf-8") as f:
        geo = json.load(f)
    feats = geo["features"]
    if limit:
        feats = feats[:limit]
    print(f"Loaded {len(feats):,} current features. Radius {radius_m} m.")

    print("Building (month x is_night) persistence reference...")
    ref_cells, (ref_gmean, ref_gsd) = build_reference(archive, tree, radius_m)


    @lru_cache(maxsize=1 << 16)
    def neighbours(lat, lon):
        return tree.query_radius(np.radians([[lat, lon]]), r=rad)[0]

    n_no_date = n_zero_hist = 0
    keys_seen = set()

    for k, feature in enumerate(feats):
        geom = feature.get("geometry") or {}
        if geom.get("type") != "Point" or len(geom.get("coordinates", [])) < 2:
            continue
        props = feature.setdefault("properties", {})
        lon, lat = geom["coordinates"][0], geom["coordinates"][1]

        cur_time = pd.to_datetime(props.get("acq_datetime"), errors="coerce")
        cur_date = pd.to_datetime(props.get("acq_date"), errors="coerce")
        if pd.isna(cur_time) and pd.isna(cur_date):
            n_no_date += 1
            continue
        if pd.isna(cur_time):
            cur_time = cur_date                          

        nb = neighbours(float(lat), float(lon))
        mask = dt64_all[nb] < int(np.datetime64(cur_time, "ns").astype("int64"))
        sub = archive.iloc[nb[mask]]

        out = features_from_hist(
            sub, archive, int(np.datetime64(cur_time, "ns").astype("int64")),
            int(np.datetime64(cur_time).astype("datetime64[D]").astype("int64")),
            int(props.get("is_night", 0) or 0),
        )
        if pd.isna(cur_date):
            out["historical_span_days"] = None           
        
        out["persistence_count"] = out["historical_n_detections"]
        n = out["historical_n_detections"]
        try:
            mo = int(pd.to_datetime(props.get("acq_date")).month)
            isn = int(props.get("is_night", 0) or 0)
        except Exception:
            mo, isn = None, None
        cell = ref_cells.get((mo, isn)) if mo is not None else None
        mu = cell["mean"] if cell and cell["size"] >= 50 else ref_gmean
        sd = cell["std"] if cell and cell["size"] >= 50 and np.isfinite(cell["std"]) else ref_gsd
        out["n_det_z"] = clean((n - mu) / sd) if sd else None

        if n == 0:
            n_zero_hist += 1

        for key, val in out.items():
            props[key] = clean(val)
            keys_seen.add(key)

        if (k + 1) % 2000 == 0:
            print(f"  {k + 1:,}/{len(feats):,}")

    print(f"\npoints with no usable date (skipped, not zero-filled): {n_no_date:,}")
    print(f"points with no prior detection within {radius_m} m      : {n_zero_hist:,} "
          f"({n_zero_hist / max(len(feats), 1):.1%}) -> historical_* written as null")
    if n_no_date and n_no_date == len(feats):
        raise RuntimeError(
            "EVERY point lacked acq_date/acq_datetime in properties - the feature "
            "block would have been silently all-zero. Check the property names."
        )

    # GeoJSON forbids NaN; convert any stragglers (e.g. from an upstream step) to null
    def sanitize(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(v) for v in o]
        return o

    geo["features"] = feats
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(sanitize(geo), f, allow_nan=False)
    print(f"Wrote {features_path}")

    model_safe = sorted(keys_seen - set(RULE_INPUT_COLUMNS))
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "radius_m": radius_m,
            "rule_input_columns": [c for c in RULE_INPUT_COLUMNS if c in keys_seen],
            "model_safe_columns": model_safe,
        }, f, indent=2)
    print(f"Wrote {MANIFEST_FILE} ({len(model_safe)} model-safe historical features)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE_FILE)
    ap.add_argument("--features", default=FEATURES_FILE)
    ap.add_argument("--radius", type=int, default=RADIUS_M)
    ap.add_argument("--limit", type=int, default=None, help="process first N features (smoke test)")
    a = ap.parse_args()
    run(a.archive, a.features, a.radius, a.limit)
