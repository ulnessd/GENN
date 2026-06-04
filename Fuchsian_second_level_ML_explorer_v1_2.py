#!/usr/bin/env python3
"""
Fuchsian Second-Level ML Explorer v1.2
=====================================

Terminal-based exploratory ML for the Fuchsian/GINN surface dataset.

Purpose
-------
This script reads a dataset produced by Fuchsian_dataset_builder_v1_1.py and
performs modest, honest second-level ML on the surface-level feature table.
It is designed for small-to-medium datasets, including the current ~35-surface
pilot dataset.

It produces:
  - cleaned feature tables
  - feature-block tables
  - PCA coordinates and plots
  - optional UMAP coordinates and plots, if umap-learn is installed
  - K-means and agglomerative clustering diagnostics
  - supervised leave-one-out / stratified CV sanity checks
  - feature-block ablation results
  - feature importance summaries where supported
  - a readable report_summary.txt

Important interpretation
------------------------
With only ~35 surfaces, this is an exploratory structure/audit tool, not a
publication-grade predictive benchmark. The central question is whether exact
and GINN-learned downstairs features organize surfaces into meaningful
computational families.

Example usage
-------------
From the GENN project directory:

  python Fuchsian_second_level_ML_explorer_v1_2.py \
      --dataset-root fuchsian_dataset_runs/fuchsian_dataset_20260530_181315_curated_ml

Or let it find the newest dataset automatically:

  python Fuchsian_second_level_ML_explorer_v1_2.py --dataset-root auto

Authoring note
--------------
This script deliberately avoids a GUI. The outputs are stable files suitable
for later reports, slides, or a PyQt6 explorer after the analysis flow settles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib is only used for saved plots, not interactive GUI use.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_slug(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_") or "item"


def find_newest_dataset(base: Path) -> Path:
    if not base.exists():
        raise FileNotFoundError(f"Dataset base directory not found: {base}")
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("fuchsian_dataset_")]
    if not candidates:
        raise FileNotFoundError(f"No fuchsian_dataset_* directories found in {base}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV if present and nonempty.

    Dataset runs often create placeholder files such as failures.csv.
    When there are no failures, that file may be zero bytes. Pandas raises
    EmptyDataError in that case; for this exploratory loader, an empty table
    should simply be skipped rather than crashing the whole analysis.
    """
    if not path.exists():
        return None
    try:
        if path.stat().st_size == 0:
            print(f"[load] skipping empty CSV: {path}")
            return None
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[load] skipping empty CSV with no columns: {path}")
        return None


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def boolish_to_int_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    lowered = s.astype(str).str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
    }
    return lowered.map(mapping)


def infer_family_column(df: pd.DataFrame) -> str:
    for c in ["family", "surface_family", "group_family", "broad_family", "category"]:
        if c in df.columns:
            return c
    # Try to derive from surface_id/name.
    if "surface_id" in df.columns:
        return "surface_id"
    if "surface_name" in df.columns:
        return "surface_name"
    raise ValueError("Could not find a family or surface label column.")


def derive_broad_family(row: pd.Series) -> str:
    raw_parts = []
    for c in ["family", "surface_family", "group_family", "surface_id", "surface_name", "name"]:
        if c in row.index and pd.notna(row[c]):
            raw_parts.append(str(row[c]).lower())
    raw = " ".join(raw_parts)

    if "hurwitz" in raw or "klein" in raw:
        return "hurwitz_klein"
    if "regular" in raw or "compact_regular" in raw:
        return "compact_regular"
    if "gamma1" in raw or "gamma_1" in raw or "gamma(1" in raw:
        return "gamma1"
    if "gamma" in raw or "principal" in raw:
        return "gamma"
    if "hecke_ab" in raw or "abelian" in raw:
        return "hecke_abelian"
    if "hecke_d" in raw or "dihedral" in raw or "nonabelian" in raw:
        return "hecke_dihedral"
    if "hecke" in raw:
        return "hecke_other"
    if "schottky" in raw:
        return "schottky"
    return "other"


def derive_compact_label(df: pd.DataFrame) -> pd.Series:
    for c in ["compact", "is_compact", "surface_compact"]:
        if c in df.columns:
            vals = boolish_to_int_series(df[c])
            if vals.notna().any():
                return vals.fillna(0).astype(int).map({1: "compact", 0: "noncompact"})
    # Use domain_type or family as fallback.
    raw = pd.Series("", index=df.index)
    for c in ["domain_type", "family", "surface_family", "surface_id", "surface_name", "name"]:
        if c in df.columns:
            raw = raw + " " + df[c].astype(str).str.lower()
    is_compact = raw.str.contains("compact_polygon|regular|hurwitz|klein|compact", regex=True)
    is_noncompact = raw.str.contains("gamma|hecke|cusp|ford|noncompact", regex=True)
    label = np.where(is_compact & ~is_noncompact, "compact", "noncompact")
    # Hurwitz can contain no explicit noncompact markers.
    label = np.where(raw.str.contains("hurwitz|klein"), "compact", label)
    return pd.Series(label, index=df.index)


def pick_primary_metric_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "top10_pruned_rmse",
        "top10_rmse",
        "ginn_top10_rmse",
        "top20_pruned_rmse",
        "top20_rmse",
        "hard_selected_distance_rmse",
        "hard_rmse",
        "test_rmse",
    ]
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    # fuzzy search
    for c in df.columns:
        lc = c.lower()
        if "top10" in lc and "rmse" in lc:
            return c
    for c in df.columns:
        lc = c.lower()
        if "rmse" in lc and ("hard" in lc or "test" in lc):
            return c
    return None


def derive_difficulty_label(df: pd.DataFrame) -> pd.Series:
    metric = pick_primary_metric_column(df)
    if metric is None:
        return pd.Series(["unknown"] * len(df), index=df.index)
    x = pd.to_numeric(df[metric], errors="coerce")
    if x.notna().sum() < 4:
        return pd.Series(["unknown"] * len(df), index=df.index)
    # Use upper tertile as hard; lower/middle as easy/moderate.
    q33 = x.quantile(1 / 3)
    q66 = x.quantile(2 / 3)
    labels = []
    for val in x:
        if pd.isna(val):
            labels.append("unknown")
        elif val <= q33:
            labels.append("easy")
        elif val <= q66:
            labels.append("moderate")
        else:
            labels.append("hard")
    return pd.Series(labels, index=df.index)


def numeric_feature_columns(df: pd.DataFrame, exclude: Sequence[str]) -> List[str]:
    exclude_set = set(exclude)
    cols = []
    for c in df.columns:
        if c in exclude_set:
            continue
        if c.startswith("Unnamed"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # Keep if at least 3 valid values and nonzero variance.
        if s.notna().sum() >= 3 and s.nunique(dropna=True) > 1:
            cols.append(c)
    return cols


def feature_block_for_column(c: str) -> str:
    lc = c.lower()
    if any(k in lc for k in ["family", "genus", "area", "cusp", "compact", "generator", "word_ball", "index", "surface", "domain", "vertex", "side_count"]):
        return "metadata"
    if any(k in lc for k in ["identity", "quotient", "shortcut", "distance_gap", "near_seam", "winning", "depth", "exact", "branch", "unique_true", "word_length", "injectivity"]):
        return "exact_downstairs"
    if any(k in lc for k in ["top", "recall", "rmse", "mae", "r2", "entropy", "confidence", "predicted", "ginn", "pruned", "speedup", "equivalent"]):
        return "ginn_response"
    if any(k in lc for k in ["epoch", "runtime", "wall", "cpu", "train", "val", "loss", "batch", "hidden", "lr", "chunk", "pairs"]):
        return "training_diagnostics"
    return "other_numeric"


def make_feature_blocks(cols: List[str]) -> Dict[str, List[str]]:
    blocks: Dict[str, List[str]] = {}
    for c in cols:
        blocks.setdefault(feature_block_for_column(c), []).append(c)
    blocks["all"] = list(cols)
    return blocks


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_dataset(dataset_root: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    tables_dir = dataset_root / "tables"
    if not tables_dir.exists():
        raise FileNotFoundError(f"tables/ directory not found under {dataset_root}")

    loaded: Dict[str, pd.DataFrame] = {}
    for name in [
        "combined_surface_features",
        "surface_metadata",
        "exact_downstairs_features",
        "ginn_surface_features",
        "failures",
    ]:
        df = read_csv_if_exists(tables_dir / f"{name}.csv")
        if df is not None:
            loaded[name] = df

    if "combined_surface_features" in loaded:
        df = loaded["combined_surface_features"].copy()
    else:
        # Merge available tables on surface_id.
        frames = [v for k, v in loaded.items() if k != "failures" and "surface_id" in v.columns]
        if not frames:
            raise FileNotFoundError("No combined_surface_features.csv and no mergeable surface_id tables found.")
        df = frames[0].copy()
        for other in frames[1:]:
            # Avoid duplicate columns.
            overlap = [c for c in other.columns if c in df.columns and c != "surface_id"]
            other2 = other.drop(columns=overlap)
            df = df.merge(other2, on="surface_id", how="outer")

    # Derived labels.
    if "broad_family" not in df.columns:
        df["broad_family"] = df.apply(derive_broad_family, axis=1)
    if "compact_label" not in df.columns:
        df["compact_label"] = derive_compact_label(df)
    if "difficulty_label" not in df.columns:
        df["difficulty_label"] = derive_difficulty_label(df)

    # Make a stable surface label.
    if "surface_id" not in df.columns:
        if "surface_name" in df.columns:
            df["surface_id"] = df["surface_name"].map(safe_slug)
        elif "name" in df.columns:
            df["surface_id"] = df["name"].map(safe_slug)
        else:
            df["surface_id"] = [f"surface_{i:04d}" for i in range(len(df))]

    return df, loaded


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_scatter(
    coords: pd.DataFrame,
    xcol: str,
    ycol: str,
    color_values: pd.Series,
    label_values: pd.Series,
    title: str,
    outpath: Path,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    # Categorical vs numeric coloring.
    numeric = pd.to_numeric(color_values, errors="coerce")
    if numeric.notna().sum() >= max(3, len(color_values) // 2):
        sc = plt.scatter(coords[xcol], coords[ycol], c=numeric, s=60, alpha=0.85)
        plt.colorbar(sc, label=str(color_values.name))
    else:
        cats = color_values.astype(str).fillna("NA")
        unique = sorted(cats.unique())
        for cat in unique:
            mask = cats == cat
            plt.scatter(coords.loc[mask, xcol], coords.loc[mask, ycol], s=60, alpha=0.85, label=cat)
        if len(unique) <= 12:
            plt.legend(fontsize=8, loc="best")

    for _, row in coords.iterrows():
        sid = str(label_values.loc[row.name]) if row.name in label_values.index else str(row.name)
        plt.annotate(sid, (row[xcol], row[ycol]), fontsize=7, alpha=0.75)

    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def save_bar(series: pd.Series, title: str, ylabel: str, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# -----------------------------------------------------------------------------
# ML functions
# -----------------------------------------------------------------------------


def make_preprocess_model(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def compute_pca(df: pd.DataFrame, cols: List[str], outdir: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(5, len(cols), len(df))))
    ])
    arr = pipe.fit_transform(X)
    pca: PCA = pipe.named_steps["pca"]
    coord_cols = [f"PC{i+1}" for i in range(arr.shape[1])]
    coords = pd.DataFrame(arr, columns=coord_cols, index=df.index)
    coords.insert(0, "surface_id", df["surface_id"].values)
    coords.insert(1, "broad_family", df["broad_family"].values)
    coords.insert(2, "compact_label", df["compact_label"].values)
    coords.insert(3, "difficulty_label", df["difficulty_label"].values)
    coords.to_csv(outdir / "pca_coordinates.csv", index=False)

    explained = {f"PC{i+1}": float(v) for i, v in enumerate(pca.explained_variance_ratio_)}
    write_json(outdir / "pca_explained_variance.json", explained)
    return coords, explained


def compute_umap_if_available(df: pd.DataFrame, cols: List[str], outdir: Path, random_state: int) -> Optional[pd.DataFrame]:
    if not HAS_UMAP:
        return None
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    n_neighbors = max(3, min(10, len(df) - 2))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=random_state)
    arr = reducer.fit_transform(X_scaled)
    coords = pd.DataFrame({
        "surface_id": df["surface_id"].values,
        "broad_family": df["broad_family"].values,
        "compact_label": df["compact_label"].values,
        "difficulty_label": df["difficulty_label"].values,
        "UMAP1": arr[:, 0],
        "UMAP2": arr[:, 1],
    }, index=df.index)
    coords.to_csv(outdir / "umap_coordinates.csv", index=False)
    return coords


def clustering_analysis(df: pd.DataFrame, cols: List[str], outdir: Path, random_state: int) -> pd.DataFrame:
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)

    true_family = LabelEncoder().fit_transform(df["broad_family"].astype(str))
    true_compact = LabelEncoder().fit_transform(df["compact_label"].astype(str))
    true_difficulty = LabelEncoder().fit_transform(df["difficulty_label"].astype(str))

    rows = []
    max_k = min(8, max(2, len(df) - 1))
    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init=30, random_state=random_state)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 and len(df) > k else np.nan
            rows.append({
                "method": "kmeans",
                "k": k,
                "silhouette": sil,
                "ARI_broad_family": adjusted_rand_score(true_family, labels),
                "ARI_compact": adjusted_rand_score(true_compact, labels),
                "ARI_difficulty": adjusted_rand_score(true_difficulty, labels),
            })
        except Exception as e:
            rows.append({"method": "kmeans", "k": k, "error": str(e)})
        try:
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 and len(df) > k else np.nan
            rows.append({
                "method": "agglomerative",
                "k": k,
                "silhouette": sil,
                "ARI_broad_family": adjusted_rand_score(true_family, labels),
                "ARI_compact": adjusted_rand_score(true_compact, labels),
                "ARI_difficulty": adjusted_rand_score(true_difficulty, labels),
            })
        except Exception as e:
            rows.append({"method": "agglomerative", "k": k, "error": str(e)})
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "clustering_results.csv", index=False)
    return res


def can_classify_target(y: pd.Series, min_classes: int = 2, min_per_class: int = 2) -> bool:
    counts = y.astype(str).value_counts()
    return len(counts) >= min_classes and (counts >= min_per_class).sum() >= min_classes


def supervised_classification(
    df: pd.DataFrame,
    feature_blocks: Dict[str, List[str]],
    outdir: Path,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    targets = {
        "compact_label": df["compact_label"].astype(str),
        "difficulty_label": df["difficulty_label"].astype(str),
        "broad_family": df["broad_family"].astype(str),
    }
    models = {
        "knn3": KNeighborsClassifier(n_neighbors=min(3, max(1, len(df) - 1))),
        "logreg": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced", max_iter=10000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced_subsample"),
    }

    rows = []
    pred_rows = []
    for target_name, y_raw in targets.items():
        if not can_classify_target(y_raw):
            rows.append({"target": target_name, "note": "skipped; insufficient class support"})
            continue
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
        min_count = y_raw.value_counts().min()
        # Prefer LOOCV for tiny datasets, but use stratified k-fold when possible.
        if min_count >= 3 and len(df) >= 12:
            n_splits = min(5, int(min_count))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_name = f"StratifiedKFold({n_splits})"
        else:
            cv = LeaveOneOut()
            cv_name = "LeaveOneOut"

        for block_name, cols in feature_blocks.items():
            if block_name == "all" or len(cols) >= 2:
                X = df[cols].apply(pd.to_numeric, errors="coerce")
            else:
                continue
            for model_name, model in models.items():
                # KNN cannot use k >= train fold size in LOOCV for very small n.
                if model_name == "knn3" and len(df) < 5:
                    continue
                pipe = make_preprocess_model(clone(model))
                try:
                    y_pred = cross_val_predict(pipe, X, y, cv=cv)
                    acc = accuracy_score(y, y_pred)
                    bal = balanced_accuracy_score(y, y_pred)
                    rows.append({
                        "task": "classification",
                        "target": target_name,
                        "model": model_name,
                        "feature_block": block_name,
                        "n_features": len(cols),
                        "cv": cv_name,
                        "accuracy": acc,
                        "balanced_accuracy": bal,
                        "n_classes": n_classes,
                        "classes": "|".join(map(str, le.classes_)),
                    })
                    for i, sid in enumerate(df["surface_id"]):
                        pred_rows.append({
                            "target": target_name,
                            "model": model_name,
                            "feature_block": block_name,
                            "surface_id": sid,
                            "true": le.inverse_transform([y[i]])[0],
                            "pred": le.inverse_transform([y_pred[i]])[0],
                        })
                except Exception as e:
                    rows.append({
                        "task": "classification",
                        "target": target_name,
                        "model": model_name,
                        "feature_block": block_name,
                        "n_features": len(cols),
                        "cv": cv_name,
                        "error": str(e),
                    })

    res = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    res.to_csv(outdir / "supervised_cv_results.csv", index=False)
    preds.to_csv(outdir / "supervised_cv_predictions.csv", index=False)
    return res, preds


def supervised_regression(
    df: pd.DataFrame,
    feature_blocks: Dict[str, List[str]],
    outdir: Path,
    random_state: int,
) -> pd.DataFrame:
    candidate_targets = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["top10", "top20", "hard", "winner"]):
            if any(k in lc for k in ["rmse", "recall", "accuracy", "speedup"]):
                if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 8:
                    candidate_targets.append(c)
    # Add primary metric if not present.
    pm = pick_primary_metric_column(df)
    if pm and pm not in candidate_targets:
        candidate_targets.insert(0, pm)

    models = {
        "ridge": Ridge(alpha=1.0),
        "knn3": KNeighborsRegressor(n_neighbors=min(3, max(1, len(df) - 1))),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=random_state),
    }
    rows = []
    cv = LeaveOneOut()
    for target in candidate_targets[:8]:  # avoid giant output
        y = pd.to_numeric(df[target], errors="coerce")
        valid = y.notna()
        if valid.sum() < 8 or y[valid].nunique() < 4:
            continue
        dfv = df.loc[valid].copy()
        yv = y.loc[valid].values
        for block_name, cols in feature_blocks.items():
            cols2 = [c for c in cols if c != target]
            if len(cols2) < 2:
                continue
            X = dfv[cols2].apply(pd.to_numeric, errors="coerce")
            for model_name, model in models.items():
                pipe = make_preprocess_model(clone(model))
                try:
                    pred = cross_val_predict(pipe, X, yv, cv=cv)
                    rmse = math.sqrt(mean_squared_error(yv, pred))
                    mae = mean_absolute_error(yv, pred)
                    r2 = r2_score(yv, pred)
                    rows.append({
                        "task": "regression",
                        "target": target,
                        "model": model_name,
                        "feature_block": block_name,
                        "n_features": len(cols2),
                        "cv": "LeaveOneOut",
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                    })
                except Exception as e:
                    rows.append({
                        "task": "regression",
                        "target": target,
                        "model": model_name,
                        "feature_block": block_name,
                        "error": str(e),
                    })
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "supervised_regression_results.csv", index=False)
    return res


def random_forest_importance(df: pd.DataFrame, cols: List[str], target: str, outdir: Path, random_state: int) -> Optional[pd.DataFrame]:
    if target not in df.columns:
        return None
    y_raw = df[target].astype(str)
    if not can_classify_target(y_raw):
        return None
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=500, random_state=random_state, class_weight="balanced_subsample")),
    ])
    try:
        pipe.fit(X, y)
        rf: RandomForestClassifier = pipe.named_steps["rf"]
        imp = pd.DataFrame({"feature": cols, "importance": rf.feature_importances_})
        imp = imp.sort_values("importance", ascending=False)
        imp.to_csv(outdir / f"feature_importance_{target}.csv", index=False)
        save_bar(imp.head(20).set_index("feature")["importance"], f"Feature importance for {target}", "importance", outdir / "plots" / f"feature_importance_{target}.png")
        return imp
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Report writing
# -----------------------------------------------------------------------------


def write_report(
    outdir: Path,
    dataset_root: Path,
    df: pd.DataFrame,
    cols: List[str],
    blocks: Dict[str, List[str]],
    pca_explained: Dict[str, float],
    clustering: pd.DataFrame,
    classif: pd.DataFrame,
    regress: pd.DataFrame,
    has_umap: bool,
) -> None:
    lines: List[str] = []
    lines.append("Fuchsian Second-Level ML Explorer v1.2")
    lines.append("=" * 48)
    lines.append("")
    lines.append(f"Dataset root: {dataset_root}")
    lines.append(f"Surface count: {len(df)}")
    lines.append(f"Numeric feature count: {len(cols)}")
    lines.append("")

    lines.append("Family counts")
    lines.append("-------------")
    for k, v in df["broad_family"].value_counts().items():
        lines.append(f"{k:24s} {v}")
    lines.append("")

    lines.append("Compact/noncompact counts")
    lines.append("-------------------------")
    for k, v in df["compact_label"].value_counts().items():
        lines.append(f"{k:24s} {v}")
    lines.append("")

    lines.append("Difficulty counts")
    lines.append("-----------------")
    for k, v in df["difficulty_label"].value_counts().items():
        lines.append(f"{k:24s} {v}")
    lines.append("")

    lines.append("Feature blocks")
    lines.append("--------------")
    for b, cs in blocks.items():
        lines.append(f"{b:24s} {len(cs)}")
    lines.append("")

    lines.append("PCA explained variance")
    lines.append("----------------------")
    for k, v in pca_explained.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("")
    lines.append(f"UMAP available: {has_umap}")
    lines.append("")

    lines.append("Best clustering rows by silhouette")
    lines.append("----------------------------------")
    if not clustering.empty and "silhouette" in clustering.columns:
        sub = clustering.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).head(8)
        for _, r in sub.iterrows():
            lines.append(
                f"{r.get('method','?'):15s} k={int(r.get('k', -1)):2d} "
                f"sil={r.get('silhouette', np.nan):.3f} "
                f"ARI_family={r.get('ARI_broad_family', np.nan):.3f} "
                f"ARI_compact={r.get('ARI_compact', np.nan):.3f}"
            )
    lines.append("")

    lines.append("Best supervised classification rows")
    lines.append("-----------------------------------")
    if not classif.empty and "balanced_accuracy" in classif.columns:
        sub = classif.dropna(subset=["balanced_accuracy"]).sort_values("balanced_accuracy", ascending=False).head(12)
        for _, r in sub.iterrows():
            lines.append(
                f"target={r.get('target','?'):18s} model={r.get('model','?'):13s} "
                f"block={r.get('feature_block','?'):20s} "
                f"acc={r.get('accuracy', np.nan):.3f} bal={r.get('balanced_accuracy', np.nan):.3f}"
            )
    lines.append("")

    lines.append("Best supervised regression rows")
    lines.append("-------------------------------")
    if not regress.empty and "r2" in regress.columns:
        sub = regress.dropna(subset=["r2"]).sort_values("r2", ascending=False).head(12)
        for _, r in sub.iterrows():
            lines.append(
                f"target={str(r.get('target','?'))[:24]:24s} model={r.get('model','?'):13s} "
                f"block={r.get('feature_block','?'):20s} "
                f"rmse={r.get('rmse', np.nan):.4g} r2={r.get('r2', np.nan):.3f}"
            )
    lines.append("")

    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append("This is an exploratory second-level ML audit. With the current small number")
    lines.append("of surfaces, results should be read as evidence of structure and feature")
    lines.append("usefulness, not as a definitive predictive benchmark. The most important")
    lines.append("question is whether exact and GINN-learned downstairs features organize")
    lines.append("Fuchsian/Riemann surfaces into meaningful computational families.")
    lines.append("")

    (outdir / "report_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Second-level exploratory ML for Fuchsian GINN datasets.")
    p.add_argument("--dataset-root", default="auto", help="Dataset root directory, or 'auto' to use newest fuchsian_dataset_runs/fuchsian_dataset_*.")
    p.add_argument("--dataset-base", default="fuchsian_dataset_runs", help="Base directory used when --dataset-root auto.")
    p.add_argument("--out-root", default="second_level_ml_runs", help="Output root directory.")
    p.add_argument("--run-name", default="", help="Optional run name suffix.")
    p.add_argument("--random-state", type=int, default=12345)
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    p.add_argument("--exclude-stress", action="store_true", help="Exclude stress-like surfaces such as gamma7 when identifiable.")
    p.add_argument("--max-features", type=int, default=0, help="Optional cap on numeric features by variance after scaling. 0 means no cap.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset_root == "auto":
        dataset_root = find_newest_dataset(Path(args.dataset_base))
    else:
        dataset_root = Path(args.dataset_root)
    dataset_root = dataset_root.resolve()

    run_suffix = safe_slug(args.run_name) if args.run_name else dataset_root.name
    outdir = Path(args.out_root) / f"run_{now_stamp()}_{run_suffix}"
    plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Fuchsian Second-Level ML Explorer v1.2")
    print(f"dataset_root={dataset_root}")
    print(f"outdir={outdir}")
    print("=" * 78)

    df, loaded = load_dataset(dataset_root)
    if args.exclude_stress:
        raw = df["surface_id"].astype(str).str.lower()
        mask = ~(raw.str.contains("gamma7|gamma_7|gamma8|gamma9|gamma10"))
        before = len(df)
        df = df.loc[mask].reset_index(drop=True)
        print(f"[filter] exclude_stress removed {before-len(df)} surfaces")

    df.to_csv(outdir / "cleaned_surface_table.csv", index=False)

    # Label/family summaries.
    family_summary = df.groupby(["broad_family", "compact_label"], dropna=False).size().reset_index(name="count")
    family_summary.to_csv(outdir / "family_counts.csv", index=False)

    exclude = [
        "surface_id", "surface_name", "name", "family", "surface_family", "group_family",
        "broad_family", "compact_label", "difficulty_label", "category", "domain_type",
        "certification_status", "maker_version", "ginn_version",
    ]
    cols = numeric_feature_columns(df, exclude=exclude)

    # Optional variance-based cap for quick tests.
    if args.max_features and len(cols) > args.max_features:
        Xtmp = df[cols].apply(pd.to_numeric, errors="coerce")
        Ximp = SimpleImputer(strategy="median").fit_transform(Xtmp)
        Xs = StandardScaler().fit_transform(Ximp)
        variances = pd.Series(np.nanvar(Xs, axis=0), index=cols).sort_values(ascending=False)
        cols = list(variances.head(args.max_features).index)
        print(f"[features] capped to {len(cols)} features by scaled variance")

    feature_info = pd.DataFrame({"feature": cols, "block": [feature_block_for_column(c) for c in cols]})
    feature_info.to_csv(outdir / "feature_blocks.csv", index=False)
    blocks = make_feature_blocks(cols)
    write_json(outdir / "feature_block_columns.json", blocks)

    # Correlation matrix for numeric features.
    if cols:
        corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
        corr.to_csv(outdir / "feature_correlation.csv")

    print(f"[data] surfaces={len(df)} numeric_features={len(cols)}")
    print("[data] families:")
    for k, v in df["broad_family"].value_counts().items():
        print(f"  {k:20s} {v}")

    # PCA / UMAP.
    pca_coords, pca_explained = compute_pca(df, cols, outdir)
    print(f"[pca] PC1={pca_explained.get('PC1', np.nan):.3f} PC2={pca_explained.get('PC2', np.nan):.3f}")

    if not args.no_plots and "PC1" in pca_coords.columns and "PC2" in pca_coords.columns:
        pcoords = pca_coords.set_index(df.index)
        plot_targets = ["broad_family", "compact_label", "difficulty_label"]
        primary_metric = pick_primary_metric_column(df)
        if primary_metric:
            plot_targets.append(primary_metric)
        for target in plot_targets:
            if target in df.columns:
                save_scatter(pcoords, "PC1", "PC2", df[target], df["surface_id"], f"PCA colored by {target}", plots_dir / f"pca_by_{safe_slug(target)}.png")

    umap_coords = compute_umap_if_available(df, cols, outdir, args.random_state)
    if umap_coords is not None and not args.no_plots:
        ucoords = umap_coords.set_index(df.index)
        for target in ["broad_family", "compact_label", "difficulty_label"]:
            save_scatter(ucoords, "UMAP1", "UMAP2", df[target], df["surface_id"], f"UMAP colored by {target}", plots_dir / f"umap_by_{safe_slug(target)}.png")
        print("[umap] completed")
    else:
        print("[umap] skipped (umap-learn not installed or plotting skipped)")

    # Clustering / supervised.
    clustering = clustering_analysis(df, cols, outdir, args.random_state)
    print("[clustering] completed")
    classif, classif_preds = supervised_classification(df, blocks, outdir, args.random_state)
    print("[supervised classification] completed")
    regress = supervised_regression(df, blocks, outdir, args.random_state)
    print("[supervised regression] completed")

    # Feature importance for selected targets.
    for target in ["compact_label", "difficulty_label", "broad_family"]:
        random_forest_importance(df, cols, target, outdir, args.random_state)

    # Save manifest.
    manifest = {
        "script": "Fuchsian_second_level_ML_explorer_v1_2.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_root": str(dataset_root),
        "outdir": str(outdir),
        "n_surfaces": int(len(df)),
        "n_numeric_features": int(len(cols)),
        "feature_blocks": {k: len(v) for k, v in blocks.items()},
        "has_umap": HAS_UMAP,
        "exclude_stress": bool(args.exclude_stress),
        "random_state": int(args.random_state),
    }
    write_json(outdir / "ml_explorer_manifest.json", manifest)

    write_report(outdir, dataset_root, df, cols, blocks, pca_explained, clustering, classif, regress, HAS_UMAP)

    print("=" * 78)
    print("DONE")
    print(f"Output folder: {outdir}")
    print(f"Summary report: {outdir / 'report_summary.txt'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
