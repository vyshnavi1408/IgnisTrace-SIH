
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(0, HERE) 

from src import weak_lables as W        # noqa: E402  (loader, fence resolver)
from src import label_industrial as LI  # noqa: E402  (industrial half of the fence)
from src import label_agriculture as S2     # noqa: E402  (scar half of the fence)

from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import (StratifiedGroupKFold, GroupShuffleSplit,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

BASE_DIR = os.path.dirname(HERE)
DEFAULT_FEATURES = os.path.join(BASE_DIR, "outputs", "features_labeled.geojson")
DEFAULT_MANIFEST = os.path.join(BASE_DIR, "outputs", "label_manifest.json")
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "xgb_classifier.json")
DEFAULT_ENCODER = os.path.join(BASE_DIR, "models", "label_encoders.json")

ABSTAIN = "abstain"


def feature_columns(df: pd.DataFrame, manifest_path: str) -> list:
    """Manifest list first, fence-resolved fallback second; fence-checked either way."""
    cols = None
    if os.path.exists(manifest_path):
        man = json.load(open(manifest_path))
        cols = man.get("feature_columns")
        if cols:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"  note: {len(missing)} manifest columns absent from this file: "
                      f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
            cols = [c for c in cols if c in df.columns]
    if not cols:
        print("  !! no usable manifest -- resolving features from the file itself")
        cols, _ = W.resolve_features(df)
    if not cols:
        raise SystemExit("no model features at all -- run the feature builder, then weaklabels")
    # defense in depth: the manifest should already be clean; this catches a stale one
    LI.assert_no_leak(cols)
    S2.assert_no_s2_leak(cols)
    for c in tuple(W.NEVER_FEATURES) + W.RULE_OUTPUT_COLUMNS:
        if c in cols:
            raise SystemExit(f"fence failure: {c} reached the trainer")
    return cols


def event_groups(df: pd.DataFrame) -> np.ndarray:
    """One id per physical fire. cluster_id if present; else a 0.05 deg grid cell,
    which can only merge events (stricter), never split one across train and test."""
    if "cluster_id" in df.columns and df["cluster_id"].notna().any():
        return df["cluster_id"].astype(str).to_numpy()
    print("  (no cluster_id -- grouping by a 0.05 deg grid cell)")
    return (df["latitude"].round(2).astype(str) + "_" +
            df["longitude"].round(2).astype(str)).to_numpy()


def grouped_split(X, y, groups, test_size=0.2, seed=42):
    idx = np.arange(len(X))
    try:
        tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True,
                                           random_state=seed).split(X, y, groups))
        how = "StratifiedGroupKFold (grouped by event, balanced by class)"
    except ValueError as e:
        try:
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                            random_state=seed).split(X, y, groups))
            how = f"GroupShuffleSplit (StratifiedGroupKFold failed: {e})"
        except ValueError as e2:
            tr, te = train_test_split(idx, test_size=test_size,
                                      random_state=seed, stratify=y)
            how = f"plain stratified 80/20 (grouped splits failed: {e2})"
    return tr, te, how


def run(features=DEFAULT_FEATURES, manifest=DEFAULT_MANIFEST, model_path=DEFAULT_MODEL,
        encoder_path=DEFAULT_ENCODER, test_size=0.2, seed=42, limit=None,
        verbose=True) -> dict:
    if not os.path.exists(features):
        raise SystemExit(f"{features} not found -- run scripts/weaklabels.py first")
    df = W.load_geojson(features)
    if limit:
        df = df.head(int(limit))
    if verbose:
        print(f"loaded {len(df):,} labelled rows from {features}")
    if "final_label" not in df.columns:
        raise SystemExit("no final_label column -- run scripts/weaklabels.py first")

    keep = (df["final_label"].notna()
            & (df["final_label"].astype(str).str.strip() != "")
            & (df["final_label"].astype(str).str.lower() != ABSTAIN))
    n_abstain = int((~keep).sum())
    df = df[keep].copy()
    if verbose:
        print(f"training on {len(df):,} rows; {n_abstain:,} abstains excluded "
              f"(abstain = no evidence, never a class)")
    if df.empty:
        raise SystemExit("every row abstained -- nothing to train on")

    cols = feature_columns(df, manifest)
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if verbose:
        print(f"features: {len(cols)} (from {'manifest' if os.path.exists(manifest) else 'fence resolver'})")

    enc = LabelEncoder()
    y = enc.fit_transform(df["final_label"].astype(str))
    if verbose:
        print("classes: " + ", ".join(f"{i}={c}" for i, c in enumerate(enc.classes_)))

    groups = event_groups(df)
    tr, te, how = grouped_split(X, y, groups, test_size, seed)
    if verbose:
        print(f"split: {how}\n  train {len(tr):,} / test {len(te):,}")

    clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                            eval_metric="mlogloss", random_state=seed)
    clf.fit(X.iloc[tr], y[tr])
    pred = clf.predict(X.iloc[te])

    acc = float(accuracy_score(y[te], pred))
    macro = float(f1_score(y[te], pred, average="macro"))
    share = pd.Series(y[te]).value_counts(normalize=True)
    baseline = float(share.iloc[0])
    if verbose:
        print("\nEvaluating on the held-out group split...")
        print(f"Accuracy: {acc:.4f}")
        print(f"always-say-{enc.classes_[int(share.index[0])]} baseline: {baseline:.4f} "
              f"-> lift {acc - baseline:+.4f}")
        print(f"macro F1: {macro:.4f}   <- quote this and per-class P/R, not accuracy alone")
        print("\nClassification report:")
        print(classification_report(y[te], pred, labels=np.arange(len(enc.classes_)),
                                    target_names=enc.classes_, zero_division=0))
        print("Feature importances:")
        for name, imp in sorted(zip(cols, clf.feature_importances_), key=lambda t: -t[1]):
            print(f"  {name}: {imp:.4f}")

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    clf.save_model(model_path)
    with open(encoder_path, "w", encoding="utf-8") as f:
        json.dump({"classes": enc.classes_.tolist(), "feature_columns": cols}, f, indent=2)
    if verbose:
        print(f"\nSaved model to {model_path}\nSaved encoder + feature order to {encoder_path}")
    return {"rows_total": int(len(df)) + n_abstain, "rows_trained": int(len(df)),
            "abstains_excluded": n_abstain, "classes": enc.classes_.tolist(),
            "feature_count": len(cols), "accuracy": acc,
            "baseline": baseline, "macro_f1": macro}


def main():
    ap = argparse.ArgumentParser(
        description="train the XGBoost classifier on weaklabels output")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="first N rows (smoke test)")
    a = ap.parse_args()
    run(features=a.features, manifest=a.manifest, model_path=a.model,
        encoder_path=a.encoder, test_size=a.test_size, seed=a.seed, limit=a.limit)


if __name__ == "__main__":
    main()
