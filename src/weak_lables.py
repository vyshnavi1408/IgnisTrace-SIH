import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import label_industrial as LI  # noqa: E402


# ============================================================
# LABEL NAMES
# ============================================================

FINAL_CLASSES = (
    "industrial",
    "wildfire",
    "agricultural_burn",
    "gasflare",
    "mining",
)

ABSTAIN = "abstain"


# ============================================================
# LOAD / SAVE GEOJSON
# ============================================================

def load_geojson(path: str) -> pd.DataFrame:
    """
    Read a Point GeoJSON file into a pandas DataFrame.

    Geometry coordinates become:
        latitude
        longitude
    """
    with open(path, "r", encoding="utf-8") as f:
        geo = json.load(f)

    rows = []

    for ft in geo.get("features", []):
        geometry = ft.get("geometry")

        if not geometry:
            continue

        coords = geometry.get("coordinates", [])

        if len(coords) < 2:
            continue

        lon, lat = coords[:2]

        rows.append({
            **ft.get("properties", {}),
            "latitude": lat,
            "longitude": lon,
        })

    if not rows:
        raise SystemExit(f"{path} holds no usable features")

    df = pd.DataFrame(rows)

    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(
            df["acq_date"],
            errors="coerce"
        )

    return df


def _safe(v):
    """
    Convert values into JSON-safe values.
    """
    if v is None:
        return None

    if isinstance(v, float) and not np.isfinite(v):
        return None

    if isinstance(v, (np.integer,)):
        return int(v)

    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)

    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    return v


def save_geojson(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame as Point GeoJSON.
    """

    features = []

    for _, row in df.iterrows():

        props = {
            k: _safe(v)
            for k, v in row.items()
            if k not in ("latitude", "longitude")
        }

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(row["longitude"]),
                    float(row["latitude"]),
                ],
            },
            "properties": props,
        })

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            f,
        )


# ============================================================
# FACILITY DISTANCES
# ============================================================

def haversine_m(lat0, lon0, lat1, lon1):
    """
    Calculate haversine distance in metres.
    """

    lat0, lon0, lat1, lon1 = (
        np.radians(np.asarray(x, dtype=float))
        for x in (lat0, lon0, lat1, lon1)
    )

    a = (
        np.sin((lat1 - lat0) / 2) ** 2
        + np.cos(lat0)
        * np.cos(lat1)
        * np.sin((lon1 - lon0) / 2) ** 2
    )

    return (
        6371000.0
        * 2
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


def facility_distances(
    df: pd.DataFrame,
    fac_path: str,
    chunk: int = 20000,
) -> pd.DataFrame:
    """
    Calculate distance to nearest industrial facility.

    Only used if industrial_distance_m does not already exist.
    """

    fac = pd.read_csv(fac_path)

    latc = next(
        (
            c
            for c in ("latitude", "lat", "y")
            if c in fac.columns
        ),
        None,
    )

    lonc = next(
        (
            c
            for c in ("longitude", "lon", "lng", "long", "x")
            if c in fac.columns
        ),
        None,
    )

    if latc is None or lonc is None:
        raise SystemExit(
            f"{fac_path}: no lat/lon columns found "
            f"(have {list(fac.columns)})"
        )

    namec = next(
        (
            c
            for c in ("name", "facility_name", "site")
            if c in fac.columns
        ),
        None,
    )

    typec = next(
        (
            c
            for c in ("type", "facility_type", "kind", "sector")
            if c in fac.columns
        ),
        None,
    )

    fl_lat = fac[latc].to_numpy(float)
    fl_lon = fac[lonc].to_numpy(float)

    out = []

    for start in range(0, len(df), chunk):

        p = df.iloc[start:start + chunk]

        d = haversine_m(
            p["latitude"].to_numpy(float)[:, None],
            p["longitude"].to_numpy(float)[:, None],
            fl_lat[None, :],
            fl_lon[None, :],
        )

        j = np.nanargmin(d, axis=1)

        result = {
            "industrial_distance_m": d[
                np.arange(len(p)), j
            ]
        }

        if namec:
            result["nearest_facility_name"] = (
                fac[namec]
                .astype(str)
                .to_numpy()[j]
            )
        else:
            result["nearest_facility_name"] = None

        if typec:
            result["nearest_facility_type"] = (
                fac[typec]
                .astype(str)
                .to_numpy()[j]
            )
        else:
            result["nearest_facility_type"] = None

        out.append(pd.DataFrame(result))

    result = pd.concat(
        out,
        ignore_index=True,
    )

    result.index = df.index

    return result


# ============================================================
# EXISTING SENTINEL-2 AGRICULTURE LABELS
# ============================================================

def load_agri_labels(
    path: str,
) -> pd.DataFrame:
    """
    Load the Sentinel-2 agriculture labels that were
    already generated.

    Expected file:
        outputs/agri_s2_labels.csv
    """

    if not os.path.exists(path):
        raise SystemExit(
            f"Could not find agriculture label file:\n"
            f"  {path}\n\n"
            f"Run the Sentinel-2 agriculture script first."
        )

    agri = pd.read_csv(path)

    if "agri_label" not in agri.columns:
        raise SystemExit(
            f"{path} does not contain 'agri_label'.\n"
            f"Available columns:\n"
            f"{list(agri.columns)}"
        )

    print(
        f"loaded existing Sentinel-2 labels: "
        f"{len(agri):,} rows"
    )

    return agri


def merge_agri_labels(
    df: pd.DataFrame,
    agri: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach previously-generated Sentinel-2 agriculture
    labels to the FIRMS dataframe.

    The agriculture CSV produced by your current
    label_agriculture.py corresponds to the rows that
    were processed by that script.

    We first try an explicit _row/index mapping.

    If that is unavailable, we fall back to matching:
        latitude
        longitude
        acq_date
    """

    df = df.copy()

    # --------------------------------------------------------
    # Case 1: agriculture CSV has _row
    # --------------------------------------------------------

    if "_row" in agri.columns:

        print("merging agriculture labels using _row")

        agri2 = agri.copy()

        agri2["_row"] = pd.to_numeric(
            agri2["_row"],
            errors="coerce",
        )

        agri2 = agri2.dropna(
            subset=["_row"]
        )

        agri2["_row"] = agri2["_row"].astype(int)

        df["_row"] = np.arange(len(df))

        label_cols = [
            c
            for c in (
                "agri_label",
                "s2_class",
                "s2_note",
                "nbr_pre",
                "nbr_post",
                "dnbr",
                "post_scenes",
                "ba_burn",
                "ba_seen",
            )
            if c in agri2.columns
        ]

        lookup = agri2[
            ["_row"] + label_cols
        ].drop_duplicates("_row")

        df = df.merge(
            lookup,
            on="_row",
            how="left",
            suffixes=("", "_s2"),
        )

        df = df.drop(
            columns=["_row"],
            errors="ignore",
        )

        return df

    # --------------------------------------------------------
    # Case 2: match using lat/lon/date
    # --------------------------------------------------------

    required = {
        "latitude",
        "longitude",
        "acq_date",
        "agri_label",
    }

    if required.issubset(df.columns) and required.issubset(
        agri.columns
    ):

        print(
            "merging agriculture labels using "
            "latitude + longitude + acq_date"
        )

        left = df.copy()
        right = agri.copy()

        left["acq_date"] = pd.to_datetime(
            left["acq_date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        right["acq_date"] = pd.to_datetime(
            right["acq_date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        # Round coordinates slightly so tiny floating-point
        # differences don't prevent matching.
        left["_lat_key"] = left["latitude"].round(6)
        left["_lon_key"] = left["longitude"].round(6)

        right["_lat_key"] = right["latitude"].round(6)
        right["_lon_key"] = right["longitude"].round(6)

        keys = [
            "_lat_key",
            "_lon_key",
            "acq_date",
        ]

        label_cols = [
            c
            for c in (
                "agri_label",
                "s2_class",
                "s2_note",
                "nbr_pre",
                "nbr_post",
                "dnbr",
                "post_scenes",
                "ba_burn",
                "ba_seen",
            )
            if c in right.columns
        ]

        lookup = right[
            keys + label_cols
        ].drop_duplicates(keys)

        left = left.merge(
            lookup,
            on=keys,
            how="left",
            suffixes=("", "_s2"),
        )

        left = left.drop(
            columns=[
                "_lat_key",
                "_lon_key",
            ],
            errors="ignore",
        )

        return left

    raise SystemExit(
        "Could not determine how to merge the "
        "Sentinel-2 agriculture labels.\n\n"
        "The agriculture CSV needs either:\n"
        "  1. '_row'\n"
        "or:\n"
        "  2. latitude + longitude + acq_date\n\n"
        f"CSV columns are:\n{list(agri.columns)}"
    )


# ============================================================
# COMBINE INDUSTRIAL + AGRICULTURE
# ============================================================

def combine(
    industrial_label,
    agri_label,
):
    """
    Decide the final class.

    Priority:
        industrial
        wildfire
        agricultural_burn
        abstain
    """

    ind = pd.to_numeric(
        pd.Series([industrial_label]),
        errors="coerce",
    ).iloc[0]

    ag = pd.to_numeric(
        pd.Series([agri_label]),
        errors="coerce",
    ).iloc[0]

    if ind == LI.POSITIVE:
        return "industrial", "industrial_rule"

    if ag == 2:
        return "wildfire", "s2_scar"

    if ag == 1:
        return "agricultural_burn", "s2_scar"

    if ag == 0:
        return ABSTAIN, "s2_looked_no_scar"

    if ind == LI.NEGATIVE:
        return "agricultural_burn", "default_not_industrial"

    return ABSTAIN, "insufficient_evidence"


def combine_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:

    pairs = [
        combine(i, a)
        for i, a in zip(
            df.get(
                "industrial_label",
                pd.Series([np.nan] * len(df)),
            ),
            df.get(
                "agri_label",
                pd.Series([np.nan] * len(df)),
            ),
        )
    ]

    df = df.copy()

    df["final_label"] = [
        p[0] for p in pairs
    ]

    df["label_source"] = [
        p[1] for p in pairs
    ]

    return df


# ============================================================
# FEATURE LEAK FENCE
# ============================================================

NEVER_FEATURES = (
    "final_label",
    "label_source",
    "s2_note",
    "cluster_id",
    "acq_date",
    "acq_datetime",
    "latitude",
    "longitude",
    "_row",
    "_i",
)

RULE_OUTPUT_COLUMNS = (
    "industrial_label",
    "agri_label",
    "s2_class",
    "final_label",
    "label_source",
)


def resolve_features(
    df: pd.DataFrame,
):
    """
    Select numeric model features while blocking:
      - label-rule inputs
      - label outputs
      - identifiers
      - dates
    """

    fence = (
    set(LI.RULE_INPUT_COLUMNS)
    | set(RULE_OUTPUT_COLUMNS)
    | set(NEVER_FEATURES)
)
    

    cols = []
    blocked = []

    for c in df.columns:

        if c in fence:
            blocked.append(c)
            continue

        raw = df[c]

        nonnull = int(
            raw.notna().sum()
        )

        if nonnull == 0:
            continue

        num = pd.to_numeric(
            raw,
            errors="coerce",
        )

        if num.notna().sum() >= 0.9 * nonnull:
            cols.append(c)

    return cols, sorted(blocked)


def feature_report(
    df: pd.DataFrame,
):

    avail, blocked = resolve_features(df)

    print(
        f"\nModel features resolved from the file: "
        f"{len(avail)}"
    )

    print(
        "  used: "
        + ", ".join(avail)
    )

    if blocked:
        print(
            "  BLOCKED: "
            + ", ".join(blocked)
        )

    if len(avail) < 12:
        print(
            "\nWARNING: only "
            f"{len(avail)} model features were found."
        )

    return avail


# ============================================================
# LABEL REPORT
# ============================================================

def label_report(
    df: pd.DataFrame,
):

    vc = df["final_label"].value_counts()

    print(
        "\n"
        + "=" * 58
        + "\nFINAL WEAK LABELS\n"
        + "=" * 58
    )

    for k in FINAL_CLASSES + (ABSTAIN,):

        if vc.get(k, 0):

            print(
                f"  {k:<20}: "
                f"{int(vc[k]):>7,} "
                f"({vc[k] / len(df):.1%})"
            )

    print("\nlabel_source:")

    for k, v in df[
        "label_source"
    ].value_counts().items():

        print(
            f"  {k:<28}: "
            f"{int(v):>7,}"
        )

    return vc.to_dict()


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(
    features="../outputs/features.geojson",
    out="../outputs/features_labeled.geojson",
    agri_labels="../outputs/agri_s2_labels.csv",
    facilities=None,
    allow_sparse=False,
    verbose=True,
):

    # --------------------------------------------------------
    # 1. LOAD FEATURES
    # --------------------------------------------------------

    df = load_geojson(features)

    if verbose:
        print(
            f"loaded {len(df):,} rows from "
            f"{features}"
        )

    feature_report(df)

    # --------------------------------------------------------
    # 2. INDUSTRIAL DISTANCES
    # --------------------------------------------------------

    if "industrial_distance_m" not in df.columns:

        if not facilities:

            raise SystemExit(
                "\nindustrial_distance_m is missing.\n"
                "Pass --facilities <facility CSV> "
                "to calculate it."
            )

        fac = facility_distances(
            df,
            facilities,
        )

        for c in fac.columns:
            df[c] = fac[c]

        if verbose:
            print(
                "facility distances computed."
            )

    # --------------------------------------------------------
    # 3. INDUSTRIAL WEAK LABEL
    # --------------------------------------------------------

    print(
        "\nRunning industrial weak labels..."
    )

    df, ind_diag = LI.assign_labels(df)

    print(
        "industrial labeling complete."
    )

    if isinstance(ind_diag, dict):

        for k, v in ind_diag.items():

            print(
                f"  {k}: {v}"
            )

    # --------------------------------------------------------
    # 4. LOAD EXISTING SENTINEL-2 LABELS
    # --------------------------------------------------------

    print(
        "\nLoading existing Sentinel-2 "
        "agriculture labels..."
    )

    agri = load_agri_labels(
        agri_labels
    )

    # --------------------------------------------------------
    # 5. MERGE EXISTING S2 LABELS
    # --------------------------------------------------------

    df = merge_agri_labels(
        df,
        agri,
    )

    if "agri_label" not in df.columns:

        raise SystemExit(
            "agri_label was not created "
            "after the Sentinel-2 merge."
        )

    # --------------------------------------------------------
    # 6. COMBINE LABELS
    # --------------------------------------------------------

    print(
        "\nCombining industrial + "
        "Sentinel-2 labels..."
    )

    df = df.drop(
        columns=["final_label"],
        errors="ignore",
    )

    df = combine_frame(df)

    counts = label_report(df)

    # --------------------------------------------------------
    # 7. RESOLVE MODEL FEATURES
    # --------------------------------------------------------

    feature_cols, blocked = resolve_features(
        df
    )

    if len(feature_cols) < 12 and not allow_sparse:

        raise SystemExit(
            "\nToo few model features.\n"
            f"Only {len(feature_cols)} "
            "usable features were found.\n"
            "Use --allow-sparse only for testing."
        )

    # --------------------------------------------------------
    # 8. SAVE
    # --------------------------------------------------------

    print(
        f"\nSaving:\n"
        f"  {out}"
    )

    save_geojson(
        df,
        out,
    )

    # --------------------------------------------------------
    # 9. MANIFEST
    # --------------------------------------------------------

    manifest = {
        "features_in": features,
        "agri_labels_in": agri_labels,
        "rows": int(len(df)),
        "feature_columns": feature_cols,
        "feature_columns_blocked_by_fence": blocked,
        "final_counts": {
            str(k): int(v)
            for k, v in counts.items()
        },
        "sentinel2_reused": True,
        "earth_engine_called": False,
    }

    manifest_path = os.path.join(
        os.path.dirname(out) or ".",
        "label_manifest.json",
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        "\n=========================================="
    )
    print(
        "WEAK LABEL PIPELINE COMPLETE"
    )
    print(
        "=========================================="
    )
    print(
        f"rows: {len(df):,}"
    )
    print(
        f"features: {len(feature_cols)}"
    )
    print(
        f"output: {out}"
    )
    print(
        f"manifest: {manifest_path}"
    )
    print(
        "\nEarth Engine was NOT called."
    )

    return df


# ============================================================
# CLI
# ============================================================

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Build features_labeled.geojson "
            "using industrial weak labels and "
            "existing Sentinel-2 agriculture labels."
        )
    )

    ap.add_argument(
        "--features",
        default="outputs/features.geojson",
    )

    ap.add_argument(
        "--out",
        default="outputs/features_labeled.geojson",
    )

    ap.add_argument(
        "--agri-labels",
        default="outputs/agri_s2_labels.csv",
        help=(
            "Existing Sentinel-2 agriculture "
            "label CSV."
        ),
    )

    ap.add_argument(
        "--facilities",
        default=None,
        help=(
            "Facility CSV with lat/lon/name/type. "
            "Only required if industrial_distance_m "
            "is missing."
        ),
    )

    ap.add_argument(
        "--allow-sparse",
        action="store_true",
        help="Allow fewer than 12 model features.",
    )

    a = ap.parse_args()

    run(
        features=a.features,
        out=a.out,
        agri_labels=a.agri_labels,
        facilities=a.facilities,
        allow_sparse=a.allow_sparse,
    )


if __name__ == "__main__":
    main()
