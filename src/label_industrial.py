from __future__ import annotations

import numpy as np
import pandas as pd


INDUSTRIAL_DISTANCE_THRESHOLD_M = 500.0
PERSISTENCE_UNIT = "days"
PERSISTENCE_MIN_DAYS = 2
PERSISTENCE_MIN_DETECTIONS = 3

MIN_CLUSTER_DETECTIONS = 2      
NEGATIVE_RULE = "not_close"
VALID_NEGATIVE_RULES = ("not_close", "neither")
VALID_PERSISTENCE_UNITS = ("days", "detections")
INDUSTRIAL_FACILITY_TYPES = (
    "refinery", "petrochemical", "chemical", "power_plant", "thermal",
    "steel", "metal", "smelter", "cement", "lng", "terminal", "plant",
    "industry", "manufacturing", "coal_mine", "mine",
)

ABSTAIN = -1
POSITIVE = 1
NEGATIVE = 0
LABEL_NAME = "industrial"
OTHER_LABEL = "non_industrial"


RULE_INPUT_COLUMNS = (
    "industrial_distance_m",
    "persistence_count",
    "historical_n_detections",
    "n_det_z",
    "n_distinct_days",
    "span_days",
    "historical_span_days",
    "max_det_per_day",
    "same_day_earlier_dets",
    "n_satellites",
    "nearest_facility_name",
    "nearest_facility_type",
    "mining_distance_m",
    "farmland_distance_m",
    "cluster_n_detections",
)


MODEL_FEATURES = (
    "frp", "log_frp", "frp_mean", "frp_std", "frp_cv", "frp_max",
    "frp_slope_per_day", "brightness", "bright_t31", "brightness_diff",
    "bt31_mean", "bt31_std", "bdiff_mean", "bdiff_max", "brightness_mean",
    "brightness_std", "confidence", "confidence_mean", "scan", "track",
    "pixel_area_km2", "is_night", "cluster_night_fraction", "hour_sin",
    "hour_cos", "sin_month", "cos_month", "gap_median_days", "gap_std_days",
    "duration_h", "rise_to_peak_h", "centroid_jitter_m", "det_per_pixel",
    "n_distinct_pixels", "frac_within_400m", "bbox_area_km2",
    "ndvi", "nbr", "dnbr", "viirs_max_frp", "viirs_firemask", "viirs_detected",
    "lc_built_frac", "lc_crop_frac", "lc_tree_frac", "lc_bare_frac",
    "nightlights_rad",
)


def assert_no_leak(feature_cols) -> None:
    bad = sorted(set(map(str, feature_cols)) & set(RULE_INPUT_COLUMNS))
    if bad:
        raise ValueError(
            "LEAK GUARD: these columns created the labels and cannot be model "
            "inputs: " + ", ".join(bad)
            + "\nRemove them from the feature list (they belong to the label side)."
        )


def _validated(value, allowed, name):
    """A typo must not silently select a different branch -- that is how
    `else: np.zeros(...)` quietly redefined the label in the original file."""
    if value is None:
        return None
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not one of {allowed}")
    return value


def _need(df: pd.DataFrame, cols, what: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"cannot compute {what}; missing columns {missing}. Available: "
            + ", ".join(sorted(df.columns)[:40])
        )



def proximity_vote(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """1 = inside the industrial-buffer. NaN distance is counted, not assumed."""
    _need(df, ["industrial_distance_m"], "the proximity vote")
    d = pd.to_numeric(df["industrial_distance_m"], errors="coerce").to_numpy(float)
    stats = {
        "rows": len(d),
        "distance_nan": int(np.isnan(d).sum()),
        "distance_nan_pct": float(np.isnan(d).mean()),
        "median_distance_m": float(np.nanmedian(d)) if np.isfinite(d).any() else None,
    }
    if stats["distance_nan"] and stats["distance_nan"] / len(d) > 0.5:
        raise ValueError(
            f"{stats['distance_nan']} of {len(d)} industrial_distance_m are NaN - the "
            "OSM join looks broken, not 'far'. Fix the join before labelling."
        )
    with np.errstate(invalid="ignore"):
        close = (d <= INDUSTRIAL_DISTANCE_THRESHOLD_M) & ~np.isnan(d)
    return close.astype(int), stats


def recurrence_vote(df: pd.DataFrame, unit: str | None = None) -> tuple[np.ndarray, dict]:
    """1 = this location keeps lighting up. In 'days' mode Terra+Aqua on one
    date count once, which is the point."""
    unit = _validated(unit if unit is not None else PERSISTENCE_UNIT,
                      VALID_PERSISTENCE_UNITS, "PERSISTENCE_UNIT")
    stats = {"unit": unit}
    if unit == "days":
        if "n_distinct_days" in df:
            v = pd.to_numeric(df["n_distinct_days"], errors="coerce").to_numpy(float)
        elif "historical_n_distinct_days" in df:
            v = pd.to_numeric(df["historical_n_distinct_days"], errors="coerce").to_numpy(float)
        else:
            _need(df, ["cluster_id", "acq_date"], "distinct-day recurrence")
            v = df.groupby("cluster_id")["acq_date"].transform(
                lambda s: pd.to_datetime(s).dt.normalize().nunique())
            v = pd.to_numeric(v, errors="coerce").to_numpy(float)
        thr = PERSISTENCE_MIN_DAYS
        stats["threshold"] = thr
    else:
        col = "persistence_count" if "persistence_count" in df else "historical_n_detections"
        _need(df, [col], "detection-count recurrence")
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        thr = PERSISTENCE_MIN_DETECTIONS
        stats["threshold"] = thr
    out = (np.nan_to_num(v, nan=0.0) >= thr).astype(int)
    stats["positive_pct"] = float(out.mean())
    return out, stats


def scope_mask(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Rows with too few observations to judge are ABSTAIN, not negative."""
    if "cluster_n_detections" in df:
        n = pd.to_numeric(df["cluster_n_detections"], errors="coerce")
    elif "historical_n_detections" in df:
        n = pd.to_numeric(df["historical_n_detections"], errors="coerce").fillna(0) + 1
    elif "persistence_count" in df:
        n = pd.to_numeric(df["persistence_count"], errors="coerce")
    else:
        _need(df, ["cluster_id"], "cluster sizes for scoping")
        n = df.groupby("cluster_id")["cluster_id"].transform("size")
    n = pd.to_numeric(n, errors="coerce").fillna(0).to_numpy(float)
    return (n >= MIN_CLUSTER_DETECTIONS), {"in_scope_pct": float((n >= MIN_CLUSTER_DETECTIONS).mean())}



def assign_labels(df: pd.DataFrame, label_col: str = "industrial_label",
                  negative_rule: str | None = None,
                  persistence_unit: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Adds industrial_label (1/0/-1) and final_label; returns (df, diagnostics).

    Three-branch, asymmetric by design:
      POSITIVE  close AND recurring   -- needs real evidence to claim industrial
      NEGATIVE  not close             -- being far from any facility is enough to
                                         rule out "industrial", and needs no support
      ABSTAIN   close but NOT recurring  (or unjoinable distance)

    The abstain bucket is deliberately NOT dumped into negatives: "inside a refinery
    perimeter but only seen once" is exactly the transient/accidental fire the PS
    asks for, and labelling it negative would teach the model to suppress it.
    """
    negative_rule = _validated(
        negative_rule if negative_rule is not None else NEGATIVE_RULE,
        VALID_NEGATIVE_RULES, "NEGATIVE_RULE")
    out = df.copy()
    close, s_prox = proximity_vote(out)
    recur, s_rec = recurrence_vote(out, unit=persistence_unit)

    pos = (close == 1) & (recur == 1)
    
    n = _cluster_sizes(out)
    pos &= n >= MIN_CLUSTER_DETECTIONS
    neg = (close == 0) if negative_rule == "not_close" else ((close == 0) & (recur == 0))

    lab = np.full(len(out), ABSTAIN, dtype=int)
    lab[pos.to_numpy() if hasattr(pos, "to_numpy") else pos] = POSITIVE
    negm = (~pos) & np.asarray(neg, dtype=bool)
    lab[negm] = NEGATIVE

    out[label_col] = lab
    has_other = ("final_label" in out.columns
                 and out["final_label"].notna().any()
                 and bool((out["final_label"].astype(str).str.strip() != "").any()))
    if has_other:
       
        fl = out["final_label"].astype(str).where(out["final_label"].notna(), OTHER_LABEL)
        out["final_label"] = np.where(lab == POSITIVE, LABEL_NAME,
                              np.where((lab == NEGATIVE) & (fl == LABEL_NAME), OTHER_LABEL, fl))
    else:
        out["final_label"] = np.where(lab == POSITIVE, LABEL_NAME,
                              np.where(lab == NEGATIVE, OTHER_LABEL, "abstain"))

    close_a = np.asarray(close, dtype=bool)
    recur_a = np.asarray(recur, dtype=bool)
    diag = {
        "proximity": s_prox,
        "recurrence": s_rec,
        "scope": {"negative_rule": negative_rule,
                  "min_detections_for_positive": MIN_CLUSTER_DETECTIONS},
        "counts": {"positive": int((lab == POSITIVE).sum()),
                   "negative": int((lab == NEGATIVE).sum()),
                   "abstain": int((lab == ABSTAIN).sum())},
        "rates": {"positive": float((lab == POSITIVE).mean()),
                  "abstain": float((lab == ABSTAIN).mean()),
                  "votes_disagree": float(((close_a != recur_a)).mean())},
    }
    return out, diag


def _cluster_sizes(df: pd.DataFrame) -> pd.Series:
    for c in ("cluster_n_detections", "persistence_count", "historical_n_detections"):
        if c in df:
            return pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "cluster_id" in df:
        return df.groupby("cluster_id")["cluster_id"].transform("size")
    return pd.Series(1, index=df.index)


def model_features(df: pd.DataFrame, extra: tuple = ()) -> pd.DataFrame:
    """Feature frame guaranteed free of the labeling family."""
    wanted = [c for c in (MODEL_FEATURES + tuple(extra)) if c in df.columns]
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        print(f"  note: {len(missing)} intended features absent from data: {', '.join(missing)}")
    assert_no_leak(wanted)
    return df[wanted].astype(float)



def agreement_with_firms_type(df: pd.DataFrame, label_col: str = "industrial_label") -> dict:
    """Independent check: FIRMS `type` is NOT a label input here, so scoring the
    labels against it is not a self-score.

    returns AUC of the (binary) label predicting type==2, plus the type==2 rate
    inside each label bucket.
    """
    if "type" not in df.columns:
        return {"error": "no `type` column; cannot compute the independent check"}
    lab = df[label_col].to_numpy()
    t2 = (pd.to_numeric(df["type"], errors="coerce").to_numpy(float) == 2)
    have = lab != ABSTAIN
    res: dict = {}
    if int(have.sum()) > 0:
        y = t2[have].astype(int)
        l = (lab[have] == POSITIVE).astype(int)
        if len(np.unique(y)) == 2 and len(np.unique(l)) == 2:
            from sklearn.metrics import roc_auc_score
            res["auc_label_predicts_type2"] = float(roc_auc_score(y, l))
        for name, sel in (("positive", l == 1), ("negative", l == 0)):
            res[f"type2_rate_{name}"] = float(y[sel].mean()) if int(sel.sum()) else float("nan")
            res[f"n_{name}"] = int(sel.sum())
    if int((~have).sum()) > 0:
        res["type2_rate_abstain"] = float(t2[~have].mean())
        res["n_abstain"] = int((~have).sum())
    res["note"] = ("type==2 is NASA's own static-source inference. type2_rate_positive should "
                   "clearly exceed type2_rate_negative. A high rate in the ABSTAIN bucket means "
                   "the discarded region holds real sources -- that is the recall cost of scoping, "
                   "and you should report it rather than hide it.")
    return res


def print_report(diag: dict, df: pd.DataFrame, label_col: str = "industrial_label") -> None:
    c, r = diag["counts"], diag["rates"]
    print("\n" + "=" * 66 + "\nLABEL REPORT\n" + "=" * 66)
    print(f"  rows                     : {c['positive'] + c['negative'] + c['abstain']:,}")
    print(f"  {LABEL_NAME:<24s}: {c['positive']:>7,}  ({r['positive']:.1%})")
    print(f"  {OTHER_LABEL:<24s}: {c['negative']:>7,}  ({1 - r['positive'] - r['abstain']:.1%})")
    print(f"  ABSTAIN (excluded)       : {c['abstain']:>7,}  ({r['abstain']:.1%})")
    print(f"  negative rule            : {diag['scope']['negative_rule']}"
          f"   (positive needs n>={diag['scope']['min_detections_for_positive']})")
    print(f"  votes disagree           : {r['votes_disagree']:.1%}"
          "   <- close XOR recurring; the close-but-once part is your abstain bucket")
    print(f"  recurrence unit            : {diag['recurrence']['unit']} >= {diag['recurrence']['threshold']}"
          f"  (positive {diag['recurrence']['positive_pct']:.1%})")
    print(f"  NaN distances              : {diag['proximity']['distance_nan']:,}"
          f"  ({diag['proximity']['distance_nan_pct']:.1%})")
    md = diag["proximity"]["median_distance_m"]
    if md is not None:
        print(f"  median distance            : {md:,.0f} m")
    for k, v in agreement_with_firms_type(df, label_col).items():
        if k == "note":
            continue
        print(f"  {k:<26s}: {v:.3f}" if isinstance(v, float) else f"  {k:<26s}: {v}")
    print("\n  FEATURES the model may see (" + str(len([c for c in MODEL_FEATURES if c in df.columns])) + "):")
    print("    " + ", ".join(c for c in MODEL_FEATURES if c in df.columns))
    print("  FORBIDDEN (created the labels):")
    print("    " + ", ".join(c for c in RULE_INPUT_COLUMNS if c in df.columns))
    print("=" * 66)
