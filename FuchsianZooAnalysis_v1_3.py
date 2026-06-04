#!/usr/bin/env python3
"""
FuchsianZooAnalysis_v1_3.py

Comprehensive post-run analysis for GENN/Fuchsian zoo outputs.

This program does not build surfaces and does not train GINNs.  It ingests one
or more completed ZooBuilder/MasterBuilder table directories, performs classical
(non-ML) analysis, second-level ML analysis, replicate/stability analysis, and
writes a standalone LaTeX report plus CSV/PNG artifacts.

Designed for outputs from FuchsianMasterDatasetBuilder_v3/v3.1, but intentionally
robust to missing columns and partially harvested runs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False


try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


PROGRAM = "FuchsianZooAnalysis_v1_3.py"
VERSION = "1.3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def boolish_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "pass", "passed"])


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def first_existing_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def latex_escape(s: Any) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def fmt_float(x: Any, nd: int = 4) -> str:
    try:
        if x is None or pd.isna(x):
            return ""
        return f"{float(x):.{nd}g}"
    except Exception:
        return str(x)


def dataframe_to_latex_tabular(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "\\emph{No rows.}"
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    out = []
    out.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    out.append("\\toprule")
    out.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    out.append("\\midrule")
    for _, row in d.iterrows():
        out.append(" & ".join(latex_escape(fmt_float(row[c])) for c in cols) + r" \\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


@dataclass
class TableSource:
    label: str
    root: Path
    tables: Path
    zoo_manifest: Optional[Path] = None
    zoo_summary: Optional[Path] = None


def find_table_dirs_from_run(run: Path) -> List[Path]:
    run = Path(run)
    candidates = []
    if (run / "tables").is_dir():
        candidates.append(run / "tables")
    for p in sorted((run / "master_dataset_runs").glob("run_*/tables")):
        candidates.append(p)
    # Also support being handed a nested master run root.
    if run.name.startswith("run_") and (run / "tables").is_dir():
        candidates.append(run / "tables")
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for c in candidates:
        rp = c.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(c)
    return out


def locate_sources(runs: Sequence[str], table_dirs: Sequence[str]) -> List[TableSource]:
    sources: List[TableSource] = []
    for td in table_dirs:
        p = Path(td)
        if not p.is_dir():
            raise FileNotFoundError(f"Table directory not found: {td}")
        root = p.parent
        label = root.name
        sources.append(TableSource(label=label, root=root, tables=p))

    for r in runs:
        rp = Path(r)
        if not rp.exists():
            raise FileNotFoundError(f"Run path not found: {r}")
        table_paths = find_table_dirs_from_run(rp)
        if not table_paths:
            raise FileNotFoundError(f"No tables directory found under run path: {r}")
        # Use the latest table dir by lexical order unless multiple requested explicitly.
        p = table_paths[-1]
        zm = rp / "zoo_manifest.json" if (rp / "zoo_manifest.json").exists() else None
        zs = rp / "zoo_summary.csv" if (rp / "zoo_summary.csv").exists() else None
        label = rp.name
        sources.append(TableSource(label=label, root=rp, tables=p, zoo_manifest=zm, zoo_summary=zs))
    return sources


CSV_NAMES = [
    "second_level_ml_features.csv",
    "learned_downstairs_geometry_catalog.csv",
    "ginn_training_metrics_catalog.csv",
    "full_ginn_second_level_records.csv",
    "analytic_geometry_catalog.csv",
    "upstairs_group_geometry_catalog.csv",
    "exact_downstairs_geometry_catalog.csv",
    "word_ball_budget_report.csv",
    "family_summary.csv",
    "source_runs.csv",
    "ingest_warnings.csv",
    "train_ready_surface_catalog.csv",
    "catalog_only_surface_catalog.csv",
    "artifact_index.csv",
    "master_surface_catalog.csv",
    "eligible_surface_catalog.csv",
    "eligible_unique_surface_catalog.csv",
    "excluded_orbifold_records.csv",
    "duplicate_fingerprint_table.csv",
]


def load_tables(sources: Sequence[TableSource]) -> Dict[str, pd.DataFrame]:
    store: Dict[str, List[pd.DataFrame]] = {n: [] for n in CSV_NAMES}
    metadata_rows = []
    for src in sources:
        metadata_rows.append({
            "input_label": src.label,
            "input_root": str(src.root),
            "tables_dir": str(src.tables),
            "zoo_manifest": str(src.zoo_manifest) if src.zoo_manifest else "",
            "zoo_summary": str(src.zoo_summary) if src.zoo_summary else "",
        })
        for name in CSV_NAMES:
            path = src.tables / name
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    # Avoid empty sentinel tables making downstream awkward.
                    if list(df.columns) == ["empty"]:
                        df = pd.DataFrame()
                    if not df.empty:
                        df.insert(0, "analysis_input_label", src.label)
                        df.insert(1, "analysis_tables_dir", str(src.tables))
                    store[name].append(df)
                except Exception as e:
                    store[name].append(pd.DataFrame({
                        "analysis_input_label": [src.label],
                        "analysis_tables_dir": [str(src.tables)],
                        "__load_error__": [str(e)],
                        "__csv_name__": [name],
                    }))
            else:
                pass
    out: Dict[str, pd.DataFrame] = {"input_sources": pd.DataFrame(metadata_rows)}
    for name, frames in store.items():
        if frames:
            try:
                out[name] = pd.concat(frames, ignore_index=True, sort=False)
            except Exception:
                out[name] = pd.concat([f.astype(str) for f in frames], ignore_index=True, sort=False)
        else:
            out[name] = pd.DataFrame()
    return out




# ---------------------------------------------------------------------------
# Family normalization and feature-block definitions
# ---------------------------------------------------------------------------


def normalize_family_label(row: pd.Series) -> str:
    fam = str(row.get("surface_family", "")).lower()
    sub = str(row.get("surface_subfamily", "")).lower()
    sid = str(row.get("surface_id", "")).lower()
    combo = " ".join([fam, sub, sid])
    if "orbifold" in combo or bool(row.get("orbifold_excluded", False)):
        if "elliptic" in combo:
            return "orbifold_cyclic_elliptic_reference"
        return "orbifold_reference"
    if "schottky" in combo:
        return "schottky_free_fuchsian"
    if "cyclic" in combo and "hyperbolic" in combo:
        return "elementary_cyclic_hyperbolic"
    if "cyclic" in combo and "parabolic" in combo:
        return "elementary_cyclic_parabolic"
    if "gamma2" in combo or "thrice" in combo:
        return "elementary_gamma2_thrice_punctured_sphere"
    if "commutator" in combo or "once_punctured_torus" in combo:
        return "elementary_modular_commutator_once_punctured_torus"
    if "hurwitz_triangle_kernel" in combo or "psl2" in combo:
        return "hurwitz_psl_triangle_kernel"
    if "klein" in combo or "hurwitz_klein" in combo or "klein_quartic" in combo:
        return "compact_hurwitz_klein"
    if "regular" in combo or "regular_genus" in combo:
        return "compact_regular_genus"
    if "hecke" in combo and ("dihedral" in combo or "dq" in combo or "nonabelian" in combo):
        return "hecke_dihedral_nonabelian_cover"
    if "hecke" in combo and ("abelian" in combo or "c2xcq" in combo):
        return "hecke_abelian_cover"
    if "gamma0" in combo or "gamma 0" in combo:
        return "modular_gamma0"
    if "gamma1" in combo or "gamma 1" in combo:
        return "modular_gamma1"
    if "principal" in combo or "gamma" == sub.strip() or "gamma " in combo:
        return "modular_principal_gamma"
    if fam:
        return fam.replace(" ", "_")
    return "unknown"


def add_normalized_family_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["family_normalized"] = out.apply(normalize_family_label, axis=1)
    def coarse(row: pd.Series) -> str:
        f = str(row.get("family_normalized", ""))
        if f.startswith("compact"):
            return "compact_closed"
        if f.startswith("modular") or f.startswith("hecke") or f.startswith("elementary_gamma") or f.startswith("elementary_modular"):
            return "finite_area_cusped"
        if f.startswith("schottky") or f.startswith("elementary_cyclic_hyperbolic") or f.startswith("elementary_cyclic_parabolic"):
            return "infinite_area_noncompact"
        if f.startswith("hurwitz"):
            return "compact_high_symmetry"
        if f.startswith("orbifold"):
            return "orbifold_reference"
        return "other"
    out["family_coarse"] = out.apply(coarse, axis=1)
    return out


ANALYTIC_NUMERIC_FEATURES = [
    "genus", "compactified_genus", "cusp_count", "area", "euler_characteristic",
    "generator_count", "word_ball_depth", "word_ball_size", "word_ball_risk_depth2",
    "word_ball_risk_depth3", "finite_quotient_order", "N", "level_N", "q", "rank",
    "gap", "rotation_deg", "sample_radius", "audit_su11_max_det_error", "audit_bad_generator_count",
    "audit_sampling_tile_count", "audit_polygon_vertex_count", "audit_side_pairing_count",
    "audit_rank", "audit_free_rank", "audit_index_in_psl2z", "audit_exported_generator_count",
    "audit_underlying_schreier_generator_count",
]
ANALYTIC_CATEGORICAL_FEATURES = [
    "surface_family", "surface_subfamily", "family_normalized", "family_coarse", "geometry_regime",
    "compact", "finite_area", "infinite_area", "cusped", "domain_type", "subdomain_type",
    "surface_area_type", "dataset_role", "word_ball_budget_status", "torsion_free", "parent_group",
    "subgroup", "riemann_surface_status", "kahler_status",
]
EXACT_DOWNSTAIRS_FEATURES = [
    "shortcut_fraction_true_test", "shortcut_fraction", "mean_winner_depth", "distance_gap",
    "exact_distance_gap_mean", "exact_distance_gap_median", "near_seam_002", "near_seam_005",
    "near_seam_fraction_gap_lt_0p02", "near_seam_fraction_gap_lt_0p05", "unique_true_branches",
    "unique_true_branches_test", "max_shortest_lift_depth",
]
TRAINING_PROTOCOL_FEATURES = [
    "run_manifest_effective_hyperparameters_pairs", "run_manifest_effective_hyperparameters_epochs",
    "run_manifest_effective_hyperparameters_pair_hidden", "run_manifest_effective_hyperparameters_score_hidden",
    "run_manifest_effective_hyperparameters_batch_size", "run_manifest_effective_hyperparameters_lr",
    "candidate_chunk_size", "estimated_unchunked_activation_mb", "run_manifest_word_depth",
    "run_manifest_word_ball_size",
]
LEARNED_OUTPUT_FEATURES = [
    "winner_acc", "winner_exact_equiv_acc", "top3_acc", "top5_acc", "top10_acc", "top20_acc",
    "depth_acc", "shortcut_fraction_pred_test", "hard_rmse", "hard_mae", "hard_r2",
    "baseline_improvement_fraction", "top1_pruned_rmse", "top3_pruned_rmse", "top5_pruned_rmse",
    "top10_pruned_rmse", "top20_pruned_rmse", "top3_speedup", "top5_speedup", "top10_speedup",
    "top20_speedup", "branch_entropy", "score_margin", "unique_pred_branches",
]
LEAKAGE_PATTERNS = [
    r"^run_manifest_metrics_", r"^predictions_", r"^artifact_", r"^training_table_stdout",
    r"^training_table_stderr", r"^selected_artifact_dir", r"^ginn_metrics_present", r"^full_ginn",
    r"^second_level_ml_role", r"^learned_", r"^test_", r"^val_", r"^train_hard",
    r"^best_val_loss", r"^best_epoch", r"^epochs_ran", r"winning_lift", r"top\d+_recall",
    r"top\d+_selected_word_accuracy", r"top\d+_pruned_exact_distance", r"shortcut_auc",
]


def is_leakage_column(col: str, target: Optional[str] = None) -> bool:
    if target and col == target:
        return True
    if col in LEARNED_OUTPUT_FEATURES:
        return True
    return any(re.search(pat, col) for pat in LEAKAGE_PATTERNS)


def columns_for_feature_block(df: pd.DataFrame, block: str, target: Optional[str] = None) -> Tuple[List[str], List[str]]:
    if block == "analytic_upstairs":
        nums = [c for c in ANALYTIC_NUMERIC_FEATURES if c in df.columns]
        cats = [c for c in ANALYTIC_CATEGORICAL_FEATURES if c in df.columns]
    elif block == "analytic_plus_exact_downstairs":
        nums = [c for c in ANALYTIC_NUMERIC_FEATURES + EXACT_DOWNSTAIRS_FEATURES if c in df.columns]
        cats = [c for c in ANALYTIC_CATEGORICAL_FEATURES if c in df.columns]
    elif block == "analytic_exact_training_protocol":
        nums = [c for c in ANALYTIC_NUMERIC_FEATURES + EXACT_DOWNSTAIRS_FEATURES + TRAINING_PROTOCOL_FEATURES if c in df.columns]
        cats = [c for c in ANALYTIC_CATEGORICAL_FEATURES if c in df.columns]
    elif block == "learned_downstairs_descriptive":
        nums = [c for c in LEARNED_OUTPUT_FEATURES if c in df.columns and c != target]
        cats = [c for c in ["family_normalized", "family_coarse", "geometry_regime"] if c in df.columns]
    else:
        nums = [c for c in ANALYTIC_NUMERIC_FEATURES if c in df.columns]
        cats = [c for c in ANALYTIC_CATEGORICAL_FEATURES if c in df.columns]
    nums = [c for c in nums if not is_leakage_column(c, target) or block == "learned_downstairs_descriptive"]
    cats = [c for c in cats if c != target and not c.startswith("analysis_")]
    return nums, cats


def prune_empty_constant_columns(df: pd.DataFrame, candidate_cols: List[str]) -> List[str]:
    cleaned = []
    for c in candidate_cols:
        if c not in df.columns:
            continue
        s = df[c].replace("", np.nan) if hasattr(df[c], "replace") else df[c]
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        cleaned.append(c)
    return cleaned

# ---------------------------------------------------------------------------
# Metric normalization
# ---------------------------------------------------------------------------


def derive_analysis_frame(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = tables.get("second_level_ml_features.csv", pd.DataFrame()).copy()
    if base.empty:
        # Fall back to eligible/master catalog if needed.
        for alt in ["eligible_surface_catalog.csv", "master_surface_catalog.csv", "analytic_geometry_catalog.csv"]:
            if not tables.get(alt, pd.DataFrame()).empty:
                base = tables[alt].copy()
                break
    if base.empty:
        return base

    # Ensure identifiers exist.
    if "record_uid" not in base.columns:
        if "surface_id" in base.columns:
            base["record_uid"] = base.get("analysis_input_label", "run") .astype(str) + "__" + base["surface_id"].astype(str)
        else:
            base["record_uid"] = [f"record_{i}" for i in range(len(base))]
    if "surface_id" not in base.columns:
        base["surface_id"] = base["record_uid"].astype(str)
    if "unique_surface_key" not in base.columns:
        # Prefer generator hash/fingerprint if available.
        key_col = first_existing_col(base, ["generator_hash", "fingerprint_key", "construction_key", "surface_id"])
        base["unique_surface_key"] = base[key_col].astype(str) if key_col else base["surface_id"].astype(str)

    # Convert booleans/numerics for common columns.
    bool_cols = [
        "mainline_dataset_eligible", "torsion_free", "orbifold_excluded", "compact", "finite_area",
        "infinite_area", "cusped", "pass_geometry_audit", "pass_ginn_preflight",
        "training_candidate_by_word_ball", "catalog_only_by_word_ball", "full_ginn_second_level_ready",
        "ginn_metrics_present", "artifact_predictions_test_csv_present", "artifact_branch_atlas_summary_json_present",
        "artifact_train_log_csv_present", "artifact_metrics_json_present",
    ]
    for c in bool_cols:
        if c in base.columns:
            base[c] = boolish_series(base[c])

    # Learned artifact flags.
    pred_col = first_existing_col(base, ["artifact_predictions_test_csv_present", "predictions_test_summary_present"])
    branch_col = first_existing_col(base, ["artifact_branch_atlas_summary_json_present", "branch_atlas_summary_present"])
    metrics_present_col = first_existing_col(base, ["ginn_metrics_present", "artifact_metrics_json_present"])
    base["learned_artifact_present_any"] = False
    for c in [pred_col, branch_col, metrics_present_col, "full_ginn_second_level_ready"]:
        if c and c in base.columns:
            base["learned_artifact_present_any"] |= boolish_series(base[c])
    if "prediction_test_rows" in base.columns:
        base["learned_artifact_present_any"] |= pd.to_numeric(base["prediction_test_rows"], errors="coerce").fillna(0).gt(0)

    # Preferred targets from strict metrics first, prediction summaries second.
    metric_aliases = {
        "winner_acc": [
            "run_manifest_metrics_winning_lift_accuracy_test",
            "winning_lift_accuracy_test",
            "predictions_winner_correct_mean",
        ],
        "winner_exact_equiv_acc": [
            "run_manifest_metrics_winning_lift_exact_equivalent_accuracy_test_tol_1e_5",
            "winning_lift_exact_equivalent_accuracy_test_tol_1e_5",
            "predictions_pred_exact_equivalent_tol_1e_5_mean",
        ],
        "top3_acc": [
            "run_manifest_metrics_winning_lift_top3_accuracy_test",
            "winning_lift_top3_accuracy_test",
            "predictions_top3_contains_true_mean",
        ],
        "top5_acc": [
            "run_manifest_metrics_winning_lift_top5_accuracy_test",
            "winning_lift_top5_accuracy_test",
            "predictions_top5_contains_true_mean",
        ],
        "depth_acc": [
            "run_manifest_metrics_depth_accuracy_test",
            "depth_accuracy_test",
        ],
        "shortcut_fraction_true_test": [
            "run_manifest_metrics_shortcut_fraction_test",
            "shortcut_fraction_test",
            "exact_shortcut_fraction_test",
            "shortcut_fraction",
        ],
        "shortcut_fraction_pred_test": [
            "run_manifest_metrics_predicted_shortcut_fraction_test",
            "predicted_shortcut_fraction_test",
            "predictions_shortcut_probability_mean",
        ],
        "hard_rmse": [
            "run_manifest_metrics_test_hard_selected_distance_rmse",
            "test_hard_distance_rmse",
        ],
        "hard_mae": [
            "run_manifest_metrics_test_hard_selected_distance_mae",
            "test_hard_distance_mae",
        ],
        "hard_r2": [
            "run_manifest_metrics_test_hard_selected_distance_r2",
            "test_hard_distance_r2",
        ],
        "identity_baseline_rmse": [
            "run_manifest_metrics_baseline_identity_test_rmse",
            "baseline_identity_test_rmse",
        ],
        "baseline_improvement_fraction": [
            "identity_baseline_hard_rmse_improvement_fraction",
        ],
        "top10_acc": [
            "top10_recall_true_winner_test",
            "run_manifest_metrics_topk_pruned_search_top10_recall_true_winner_test",
        ],
        "top20_acc": [
            "top20_recall_true_winner_test",
            "run_manifest_metrics_topk_pruned_search_top20_recall_true_winner_test",
        ],
        "top1_pruned_rmse": [
            "top1_pruned_exact_distance_rmse_test",
            "run_manifest_metrics_topk_pruned_search_top1_pruned_exact_distance_test_rmse",
        ],
        "top3_pruned_rmse": [
            "top3_pruned_exact_distance_rmse_test",
            "run_manifest_metrics_topk_pruned_search_top3_pruned_exact_distance_test_rmse",
        ],
        "top5_pruned_rmse": [
            "top5_pruned_exact_distance_rmse_test",
            "run_manifest_metrics_topk_pruned_search_top5_pruned_exact_distance_test_rmse",
        ],
        "top10_pruned_rmse": [
            "top10_pruned_exact_distance_rmse_test",
            "run_manifest_metrics_topk_pruned_search_top10_pruned_exact_distance_test_rmse",
        ],
        "top20_pruned_rmse": [
            "top20_pruned_exact_distance_rmse_test",
            "run_manifest_metrics_topk_pruned_search_top20_pruned_exact_distance_test_rmse",
        ],
        "top3_pruned_distance_mean": ["predictions_top3_pruned_distance_mean"],
        "top5_pruned_distance_mean": ["predictions_top5_pruned_distance_mean"],
        "top10_pruned_distance_mean": ["predictions_top10_pruned_distance_mean"],
        "top3_speedup": [
            "top3_speedup_factor_vs_full_word_ball",
            "run_manifest_metrics_topk_pruned_search_top3_speedup_factor_vs_full_word_ball",
        ],
        "top5_speedup": [
            "top5_speedup_factor_vs_full_word_ball",
            "run_manifest_metrics_topk_pruned_search_top5_speedup_factor_vs_full_word_ball",
        ],
        "top10_speedup": [
            "top10_speedup_factor_vs_full_word_ball",
            "run_manifest_metrics_topk_pruned_search_top10_speedup_factor_vs_full_word_ball",
        ],
        "top20_speedup": [
            "top20_speedup_factor_vs_full_word_ball",
            "run_manifest_metrics_topk_pruned_search_top20_speedup_factor_vs_full_word_ball",
        ],
        "branch_entropy": [
            "run_manifest_metrics_branch_atlas_test_learned_branch_entropy_mean",
            "learned_branch_entropy_mean",
            "predictions_branch_entropy_mean",
        ],
        "score_margin": [
            "run_manifest_metrics_branch_atlas_test_learned_top1_score_margin_mean",
            "learned_top1_score_margin_mean",
            "predictions_score_margin_top1_top2_mean",
        ],
        "distance_gap": [
            "exact_distance_gap_mean",
            "predictions_exact_distance_gap_top1_top2_mean",
        ],
        "near_seam_002": ["near_seam_fraction_gap_lt_0p02"],
        "near_seam_005": ["near_seam_fraction_gap_lt_0p05"],
        "unique_true_branches": ["unique_true_branches_test"],
        "unique_pred_branches": ["unique_predicted_branches_test", "predictions_unique_pred_winning_lift_word"],
    }
    for new, names in metric_aliases.items():
        vals = None
        for c in names:
            if c in base.columns:
                s = pd.to_numeric(base[c], errors="coerce")
                vals = s if vals is None else vals.combine_first(s)
        if vals is not None:
            base[new] = vals

    numeric_candidates = [
        "genus", "compactified_genus", "cusp_count", "area", "euler_characteristic",
        "generator_count", "word_ball_depth", "word_ball_size", "word_ball_risk_depth2",
        "word_ball_risk_depth3", "shortcut_fraction", "mean_winner_depth", "q", "N", "level_N",
        "rank", "gap", "rotation_deg", "finite_quotient_order", "training_table_wall_seconds",
        "best_val_loss", "best_epoch", "epochs_ran", "candidate_chunk_size", "estimated_unchunked_activation_mb",
        "run_manifest_effective_hyperparameters_pairs", "run_manifest_effective_hyperparameters_epochs",
        "run_manifest_effective_hyperparameters_pair_hidden", "run_manifest_effective_hyperparameters_score_hidden",
        "run_manifest_effective_hyperparameters_batch_size", "run_manifest_effective_hyperparameters_lr",
        "run_manifest_word_ball_size", "run_manifest_pairs",
    ] + list(metric_aliases.keys())
    base = coerce_numeric(base, numeric_candidates)

    # Fill key analytic columns from run-manifest or smoke aliases when the
    # surface-level catalog column is missing.
    fill_aliases = {
        "word_ball_size": ["run_manifest_word_ball_size", "run_manifest_metrics_word_ball_size", "smoke_word_ball_size", "word_ball_size_metrics"],
        "word_ball_depth": ["run_manifest_word_depth", "smoke_word_depth"],
        "generator_count": ["audit_generator_count", "audit_exported_generator_count", "audit_underlying_schreier_generator_count"],
    }
    for dst, aliases in fill_aliases.items():
        if dst not in base.columns:
            base[dst] = np.nan
        cur = pd.to_numeric(base[dst], errors="coerce")
        for a in aliases:
            if a in base.columns:
                cur = cur.combine_first(pd.to_numeric(base[a], errors="coerce"))
        base[dst] = cur

    # Main analysis eligibility flags.
    if "mainline_dataset_eligible" in base.columns:
        base["analysis_eligible"] = boolish_series(base["mainline_dataset_eligible"])
    else:
        base["analysis_eligible"] = ~boolish_series(base.get("orbifold_excluded", pd.Series(False, index=base.index)))
    base["analysis_learned_any"] = base["analysis_eligible"] & base["learned_artifact_present_any"]
    base["analysis_full_ginn"] = base["analysis_eligible"] & (
        boolish_series(base["full_ginn_second_level_ready"]) if "full_ginn_second_level_ready" in base.columns else False
    )

    # Difficulty labels.
    if "top5_acc" in base.columns:
        base["high_top5_learnability"] = pd.to_numeric(base["top5_acc"], errors="coerce").ge(0.95)
    if "winner_acc" in base.columns:
        base["high_top1_learnability"] = pd.to_numeric(base["winner_acc"], errors="coerce").ge(0.85)
    if "hard_rmse" in base.columns:
        base["low_rmse_learnability"] = pd.to_numeric(base["hard_rmse"], errors="coerce").le(0.05)

    # Derived scientific regimes.
    def regime(row: pd.Series) -> str:
        if bool(row.get("orbifold_excluded", False)) or not bool(row.get("analysis_eligible", True)):
            return "excluded_orbifold_reference"
        if bool(row.get("compact", False)):
            return "compact_closed"
        if bool(row.get("finite_area", False)):
            return "noncompact_finite_area_cusped"
        if bool(row.get("infinite_area", False)):
            return "noncompact_infinite_area"
        return "unknown"
    base["geometry_regime"] = base.apply(regime, axis=1)
    base = add_normalized_family_columns(base)
    return base


# ---------------------------------------------------------------------------
# Classical analysis
# ---------------------------------------------------------------------------


def write_data_audit(df: pd.DataFrame, tables: Dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    rows = []
    rows.append({"item": "records_total", "value": len(df)})
    rows.append({"item": "eligible_records", "value": int(df.get("analysis_eligible", pd.Series(False)).sum())})
    rows.append({"item": "unique_eligible_surface_keys", "value": int(df.loc[df.get("analysis_eligible", False), "unique_surface_key"].nunique() if "unique_surface_key" in df.columns else 0)})
    rows.append({"item": "learned_any_records", "value": int(df.get("analysis_learned_any", pd.Series(False)).sum())})
    rows.append({"item": "unique_learned_any_surface_keys", "value": int(df.loc[df.get("analysis_learned_any", False), "unique_surface_key"].nunique() if "unique_surface_key" in df.columns else 0)})
    rows.append({"item": "full_ginn_records", "value": int(df.get("analysis_full_ginn", pd.Series(False)).sum())})
    rows.append({"item": "unique_full_ginn_surface_keys", "value": int(df.loc[df.get("analysis_full_ginn", False), "unique_surface_key"].nunique() if "unique_surface_key" in df.columns else 0)})
    rows.append({"item": "excluded_orbifold_records", "value": int(boolish_series(df.get("orbifold_excluded", pd.Series(False, index=df.index))).sum() if "orbifold_excluded" in df.columns else 0)})
    for name in CSV_NAMES:
        t = tables.get(name, pd.DataFrame())
        rows.append({"item": f"table_rows:{name}", "value": len(t)})
    important = [
        "winner_acc", "top5_acc", "hard_rmse", "hard_r2", "top5_pruned_rmse", "top5_speedup", "topk_auc_1_20", "topk_gain5_minus_top1", "topk_k95", "topk_k99",
        "branch_entropy", "score_margin", "near_seam_002", "word_ball_size", "generator_count",
    ]
    for c in important:
        if c in df.columns:
            rows.append({"item": f"nonmissing:{c}", "value": int(pd.to_numeric(df[c], errors="coerce").notna().sum())})
        else:
            rows.append({"item": f"missing_column:{c}", "value": 1})
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "data_audit.csv", index=False)
    return audit


def family_summary(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Write raw and main eligible family summaries.

    v1.2 separates all/raw records from the main Riemann-surface analysis so
    excluded orbifold references and alias rows do not silently dominate the
    interpretation.
    """
    def summarize(frame: pd.DataFrame, filename: str, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        if frame.empty:
            fs = pd.DataFrame()
        else:
            if group_cols is None:
                group_cols = [c for c in ["family_normalized", "surface_family", "surface_subfamily"] if c in frame.columns]
            if not group_cols:
                group_cols = ["geometry_regime"] if "geometry_regime" in frame.columns else []
            if group_cols:
                agg = {
                    "record_uid": "count",
                    "analysis_eligible": "sum",
                    "analysis_learned_any": "sum",
                    "analysis_full_ginn": "sum",
                    "unique_surface_key": pd.Series.nunique,
                }
                for c in ["winner_acc", "top3_acc", "top5_acc", "top10_acc", "top20_acc", "hard_rmse", "top5_pruned_rmse", "top5_speedup", "word_ball_size", "generator_count", "near_seam_002", "branch_entropy"]:
                    if c in frame.columns:
                        agg[c] = "median"
                fs = frame.groupby(group_cols, dropna=False).agg(agg).reset_index()
                fs = fs.rename(columns={
                    "record_uid": "records_total",
                    "analysis_eligible": "eligible_total",
                    "analysis_learned_any": "learned_any_total",
                    "analysis_full_ginn": "full_ginn_total",
                    "unique_surface_key": "unique_surface_keys",
                })
            else:
                fs = pd.DataFrame()
        fs.to_csv(out / filename, index=False)
        return fs

    raw = summarize(df, "family_summary_analysis_raw.csv", [c for c in ["family_normalized", "surface_family", "surface_subfamily"] if c in df.columns])
    main = df[df.get("analysis_eligible", False)].copy() if "analysis_eligible" in df.columns else df.copy()
    main = summarize(main, "family_summary_analysis.csv", [c for c in ["family_normalized"] if c in main.columns])
    regime_cols = [c for c in ["family_coarse", "geometry_regime"] if c in df.columns]
    if regime_cols:
        summarize(df[df.get("analysis_eligible", False)].copy(), "regime_summary_analysis.csv", regime_cols)
    if "orbifold_excluded" in df.columns:
        orb = df[boolish_series(df["orbifold_excluded"])].copy()
        summarize(orb, "orbifold_reference_summary.csv")
    return main

def metric_summary(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    metrics = [
        "winner_acc", "winner_exact_equiv_acc", "top3_acc", "top5_acc", "top10_acc", "top20_acc", "depth_acc",
        "hard_rmse", "hard_mae", "hard_r2", "baseline_improvement_fraction",
        "top3_pruned_rmse", "top5_pruned_rmse", "top10_pruned_rmse", "top20_pruned_rmse", "top3_speedup", "top5_speedup", "top10_speedup", "top20_speedup",
        "branch_entropy", "score_margin", "distance_gap", "near_seam_002", "near_seam_005",
        "unique_true_branches", "unique_pred_branches", "word_ball_size", "generator_count",
    ]
    rows = []
    for m in metrics:
        if m in df.columns:
            s = pd.to_numeric(df.loc[df.get("analysis_eligible", True), m], errors="coerce").dropna()
            rows.append({
                "metric": m,
                "n": len(s),
                "mean": s.mean() if len(s) else np.nan,
                "median": s.median() if len(s) else np.nan,
                "std": s.std() if len(s) > 1 else np.nan,
                "min": s.min() if len(s) else np.nan,
                "q25": s.quantile(0.25) if len(s) else np.nan,
                "q75": s.quantile(0.75) if len(s) else np.nan,
                "max": s.max() if len(s) else np.nan,
            })
    ms = pd.DataFrame(rows)
    ms.to_csv(out / "learned_metric_summary.csv", index=False)
    return ms


def correlation_analysis(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Leakage-controlled Spearman associations.

    v1.1 correlated targets with all numeric columns, which correctly exposed
    duplicated metric columns but was not scientifically useful.  v1.2 reports
    associations by feature block and excludes learned-output / target-leakage
    columns for predictive questions.
    """
    targets = [c for c in ["winner_acc", "top5_acc", "hard_rmse", "top5_pruned_rmse", "top5_speedup", "branch_entropy"] if c in df.columns]
    rows = []
    dbase = df[df.get("analysis_learned_any", True)].copy()
    blocks = ["analytic_upstairs", "analytic_plus_exact_downstairs", "analytic_exact_training_protocol"]
    for block in blocks:
        nums, cats = columns_for_feature_block(dbase, block)
        # Only numeric variables get Spearman correlations; categorical effects are handled by ML.
        feature_cols = prune_empty_constant_columns(dbase, nums)
        for t in targets:
            if t not in dbase.columns:
                continue
            y = pd.to_numeric(dbase[t], errors="coerce")
            for c in feature_cols:
                if c == t or is_leakage_column(c, t):
                    continue
                x = pd.to_numeric(dbase[c], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() < 10 or x[valid].nunique() <= 1 or y[valid].nunique() <= 1:
                    continue
                r = x[valid].corr(y[valid], method="spearman")
                if pd.notna(r):
                    rows.append({"feature_block": block, "target": t, "feature": c, "spearman_r": r, "abs_spearman_r": abs(r), "n": int(valid.sum())})
    corr_long = pd.DataFrame(rows)
    if not corr_long.empty:
        corr_long = corr_long.sort_values(["feature_block", "target", "abs_spearman_r"], ascending=[True, True, False])
    corr_long.to_csv(out / "correlation_table.csv", index=False)

    # A separate diagnostic leakage table is useful for checking that duplicate
    # learned metrics exist but are intentionally excluded from the predictive tables.
    leak_rows = []
    numeric_cols = []
    for c in dbase.columns:
        s = pd.to_numeric(dbase[c], errors="coerce")
        if s.notna().sum() >= 10 and s.nunique(dropna=True) > 1:
            numeric_cols.append(c)
    for t in targets:
        y = pd.to_numeric(dbase[t], errors="coerce")
        for c in numeric_cols:
            if c == t:
                continue
            if not is_leakage_column(c, t):
                continue
            x = pd.to_numeric(dbase[c], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 10 or x[valid].nunique() <= 1:
                continue
            r = x[valid].corr(y[valid], method="spearman")
            if pd.notna(r):
                leak_rows.append({"target": t, "leakage_like_feature": c, "spearman_r": r, "abs_spearman_r": abs(r), "n": int(valid.sum())})
    leak = pd.DataFrame(leak_rows)
    if not leak.empty:
        leak = leak.sort_values(["target", "abs_spearman_r"], ascending=[True, False])
    leak.to_csv(out / "leakage_diagnostic_correlations.csv", index=False)
    return corr_long

def best_worst_anomalies(df: pd.DataFrame, out: Path, top_n: int = 25) -> Dict[str, pd.DataFrame]:
    learned = df[df.get("analysis_learned_any", False)].copy() if "analysis_learned_any" in df.columns else df.copy()
    id_cols = [c for c in ["analysis_input_label", "surface_id", "surface_family", "surface_subfamily", "geometry_regime", "unique_surface_key"] if c in learned.columns]
    metric_cols = [c for c in ["winner_acc", "top5_acc", "hard_rmse", "hard_r2", "top5_pruned_rmse", "top5_speedup", "topk_auc_1_20", "topk_gain5_minus_top1", "topk_k95", "topk_k99", "branch_entropy", "word_ball_size", "generator_count", "near_seam_002"] if c in learned.columns]
    outputs: Dict[str, pd.DataFrame] = {}
    def emit(name: str, frame: pd.DataFrame):
        frame.to_csv(out / f"{name}.csv", index=False)
        outputs[name] = frame
    if not learned.empty and metric_cols:
        if "top5_acc" in learned.columns:
            emit("best_surfaces_top5_accuracy", learned.sort_values("top5_acc", ascending=False)[id_cols + metric_cols].head(top_n))
            emit("worst_surfaces_top5_accuracy", learned.sort_values("top5_acc", ascending=True)[id_cols + metric_cols].head(top_n))
        if "hard_rmse" in learned.columns:
            emit("best_surfaces_low_rmse", learned.sort_values("hard_rmse", ascending=True)[id_cols + metric_cols].head(top_n))
            emit("worst_surfaces_high_rmse", learned.sort_values("hard_rmse", ascending=False)[id_cols + metric_cols].head(top_n))
        if "top5_speedup" in learned.columns:
            emit("best_surfaces_top5_speedup", learned.sort_values("top5_speedup", ascending=False)[id_cols + metric_cols].head(top_n))
    # Composite anomaly score using robust z-scores.
    score = pd.Series(0.0, index=learned.index)
    components = []
    for c, sign in [("word_ball_size", 1), ("generator_count", 1), ("near_seam_002", 1), ("branch_entropy", 1), ("winner_acc", -1), ("top5_acc", -1), ("hard_rmse", 1)]:
        if c in learned.columns:
            s = pd.to_numeric(learned[c], errors="coerce")
            med = s.median()
            mad = (s - med).abs().median()
            if pd.notna(mad) and mad > 0:
                z = ((s - med) / (1.4826 * mad)).clip(-8, 8)
                score = score.add(sign * z.fillna(0), fill_value=0)
                components.append(c)
    if len(learned):
        anom = learned.copy()
        anom["anomaly_score"] = score
        anom["anomaly_components"] = ";".join(components)
        emit("anomaly_table", anom.sort_values("anomaly_score", ascending=False)[id_cols + ["anomaly_score", "anomaly_components"] + metric_cols].head(max(top_n, 50)))
    # Training failures or partials.
    fail = df.copy()
    cond = pd.Series(False, index=fail.index)
    if "analysis_eligible" in fail.columns:
        cond |= boolish_series(fail["analysis_eligible"]) & ~boolish_series(fail.get("learned_artifact_present_any", pd.Series(False, index=fail.index)))
    if "training_table_returncode" in fail.columns:
        cond |= pd.to_numeric(fail["training_table_returncode"], errors="coerce").fillna(0).ne(0)
    if "training_table_pass_ginn_training" in fail.columns:
        col = fail["training_table_pass_ginn_training"]
        cond |= col.notna() & ~boolish_series(col)
    fail_cols = id_cols + [c for c in ["training_table_returncode", "training_table_pass_ginn_training", "training_table_wall_seconds", "training_table_stdout_tail", "training_table_stderr_tail", "word_ball_size", "generator_count", "word_ball_budget_status"] if c in fail.columns]
    emit("training_failure_or_missing_learned_artifact_table", fail.loc[cond, fail_cols].head(500))
    return outputs


def replicate_analysis(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    if df.empty or "unique_surface_key" not in df.columns or "analysis_input_label" not in df.columns:
        rep = pd.DataFrame()
    else:
        metric_cols = [c for c in ["winner_acc", "top5_acc", "hard_rmse", "top5_pruned_rmse", "top5_speedup", "branch_entropy"] if c in df.columns]
        learned = df[df.get("analysis_learned_any", True)].copy()
        rows = []
        for key, g in learned.groupby("unique_surface_key"):
            run_count = g["analysis_input_label"].nunique()
            if run_count < 2:
                continue
            row = {
                "unique_surface_key": key,
                "surface_ids": ";".join(sorted(set(g.get("surface_id", pd.Series()).astype(str))))[:500],
                "families": ";".join(sorted(set(g.get("surface_family", pd.Series()).astype(str))))[:500],
                "replicate_runs": run_count,
                "replicate_records": len(g),
            }
            for c in metric_cols:
                s = pd.to_numeric(g[c], errors="coerce").dropna()
                row[f"{c}_mean"] = s.mean() if len(s) else np.nan
                row[f"{c}_std"] = s.std() if len(s) > 1 else np.nan
                row[f"{c}_range"] = s.max() - s.min() if len(s) else np.nan
            rows.append(row)
        rep = pd.DataFrame(rows)
        if not rep.empty:
            sort_col = "top5_acc_std" if "top5_acc_std" in rep.columns else rep.columns[-1]
            rep = rep.sort_values(sort_col, ascending=False)
    rep.to_csv(out / "replicate_stability.csv", index=False)
    return rep




# ---------------------------------------------------------------------------
# Unique-surface and top-k curve analysis
# ---------------------------------------------------------------------------


def unique_surface_summary(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    if df.empty or "unique_surface_key" not in df.columns:
        u = pd.DataFrame()
        u.to_csv(out / "unique_surface_summary.csv", index=False)
        return u
    d = df[df.get("analysis_eligible", True)].copy()
    metrics = [c for c in [
        "winner_acc", "top3_acc", "top5_acc", "top10_acc", "top20_acc", "hard_rmse", "hard_r2",
        "top5_pruned_rmse", "top5_speedup", "branch_entropy", "score_margin", "distance_gap",
        "near_seam_002", "word_ball_size", "generator_count", "cusp_count", "area", "genus",
    ] if c in d.columns]
    rows = []
    for key, g in d.groupby("unique_surface_key", dropna=False):
        row = {
            "unique_surface_key": key,
            "records": len(g),
            "learned_any_records": int(g.get("analysis_learned_any", pd.Series(False, index=g.index)).sum()),
            "full_ginn_records": int(g.get("analysis_full_ginn", pd.Series(False, index=g.index)).sum()),
            "surface_ids": ";".join(sorted(set(g.get("surface_id", pd.Series(dtype=str)).astype(str))))[:500],
            "family_normalized": g.get("family_normalized", pd.Series([""])).dropna().astype(str).iloc[0] if "family_normalized" in g.columns and g["family_normalized"].notna().any() else "",
            "family_coarse": g.get("family_coarse", pd.Series([""])).dropna().astype(str).iloc[0] if "family_coarse" in g.columns and g["family_coarse"].notna().any() else "",
            "geometry_regime": g.get("geometry_regime", pd.Series([""])).dropna().astype(str).iloc[0] if "geometry_regime" in g.columns and g["geometry_regime"].notna().any() else "",
        }
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{m}_median"] = vals.median() if len(vals) else np.nan
            row[f"{m}_std"] = vals.std() if len(vals) > 1 else np.nan
        rows.append(row)
    u = pd.DataFrame(rows)
    if not u.empty:
        sort_col = "top5_acc_mean" if "top5_acc_mean" in u.columns else "records"
        u = u.sort_values(sort_col, ascending=False)
    u.to_csv(out / "unique_surface_summary.csv", index=False)
    return u


def _topk_anchor_from_row(row: pd.Series, k: int) -> Optional[float]:
    candidates = []
    if k == 1:
        candidates += ["winner_acc", "top1_recall_true_winner_test", "run_manifest_metrics_topk_pruned_search_top1_recall_true_winner_test"]
    candidates += [
        f"top{k}_acc",
        f"top{k}_recall_true_winner_test",
        f"run_manifest_metrics_topk_pruned_search_top{k}_recall_true_winner_test",
        f"winning_lift_top{k}_accuracy_test",
        f"run_manifest_metrics_winning_lift_top{k}_accuracy_test",
        f"predictions_top{k}_contains_true_mean",
    ]
    for c in candidates:
        if c in row.index:
            try:
                v = float(row[c])
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    return None


def compute_topk_accuracy_curves(df: pd.DataFrame, out: Path, max_k: int = 20, try_raw_predictions: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute accuracy-vs-k curves.

    If raw prediction files are available at selected_artifact_dir/predictions_test.csv,
    v1.2 computes exact empirical top-k accuracy for every k=1..max_k from
    true_word_model_rank.  If those files are not present, it falls back to the
    master-table anchors k=1,3,5,10,20.
    """
    rows = []
    learned = df[df.get("analysis_learned_any", False)].copy() if "analysis_learned_any" in df.columns else df.copy()
    learned = learned[learned.get("analysis_eligible", True)].copy() if "analysis_eligible" in learned.columns else learned
    for idx, row in learned.iterrows():
        base = {
            "analysis_input_label": row.get("analysis_input_label", ""),
            "surface_id": row.get("surface_id", ""),
            "unique_surface_key": row.get("unique_surface_key", row.get("surface_id", "")),
            "surface_family": row.get("surface_family", ""),
            "surface_subfamily": row.get("surface_subfamily", ""),
            "family_normalized": row.get("family_normalized", row.get("surface_family", "")),
            "family_coarse": row.get("family_coarse", ""),
            "geometry_regime": row.get("geometry_regime", ""),
        }
        raw_done = False
        if try_raw_predictions and "selected_artifact_dir" in row.index:
            pred_path = Path(str(row.get("selected_artifact_dir", ""))) / "predictions_test.csv"
            if pred_path.exists():
                try:
                    pr = pd.read_csv(pred_path, usecols=lambda c: c in ["true_word_model_rank"])
                    ranks = pd.to_numeric(pr.get("true_word_model_rank"), errors="coerce").dropna()
                    ranks = ranks[ranks > 0]
                    if len(ranks):
                        for k in range(1, max_k + 1):
                            rr = dict(base)
                            rr.update({"k": k, "topk_accuracy": float((ranks <= k).mean()), "n_test_pairs": int(len(ranks)), "source": "raw_predictions_true_rank"})
                            rows.append(rr)
                        raw_done = True
                except Exception:
                    raw_done = False
        if not raw_done:
            # Master-table anchor fallback.  We record only ks supported by the table.
            for k in [1, 2, 3, 4, 5, 10, 20]:
                v = _topk_anchor_from_row(row, k)
                if v is not None:
                    rr = dict(base)
                    rr.update({"k": k, "topk_accuracy": float(v), "n_test_pairs": np.nan, "source": "master_table_anchor"})
                    rows.append(rr)
    surf = pd.DataFrame(rows)
    if surf.empty:
        fam = pd.DataFrame()
    else:
        surf.to_csv(out / "topk_accuracy_by_surface.csv", index=False)
        fam = surf.groupby(["family_normalized", "k"], dropna=False).agg(
            topk_accuracy_mean=("topk_accuracy", "mean"),
            topk_accuracy_median=("topk_accuracy", "median"),
            surface_records=("unique_surface_key", "nunique"),
            raw_prediction_records=("source", lambda x: int((x == "raw_predictions_true_rank").sum())),
        ).reset_index().sort_values(["family_normalized", "k"])
        fam.to_csv(out / "topk_accuracy_by_family.csv", index=False)
    if surf.empty:
        surf.to_csv(out / "topk_accuracy_by_surface.csv", index=False)
        fam.to_csv(out / "topk_accuracy_by_family.csv", index=False)
    return surf, fam


def plot_topk_accuracy_by_family(topk_family: pd.DataFrame, path: Path) -> None:
    if not HAVE_MATPLOTLIB or topk_family is None or topk_family.empty:
        return
    if "family_normalized" not in topk_family.columns or "k" not in topk_family.columns:
        return
    plt.figure(figsize=(8.5, 5.5))
    for fam, g in topk_family.groupby("family_normalized"):
        g = g.sort_values("k")
        if len(g) == 0:
            continue
        plt.plot(g["k"], g["topk_accuracy_mean"], marker="o", linewidth=1.8, label=str(fam))
    plt.xlabel("k in top-k candidate list")
    plt.ylabel("Mean top-k contains true winner accuracy")
    plt.title("Top-k branch-recall accuracy by family")
    plt.xlim(1, 20)
    plt.ylim(0, 1.02)
    plt.xticks(list(range(1, 21)))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, loc="lower right", ncol=1)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()



# ---------------------------------------------------------------------------
# Top-k descriptor and advanced second-level ML helpers
# ---------------------------------------------------------------------------


def add_topk_descriptor_columns(df: pd.DataFrame, out: Optional[Path] = None, max_k: int = 20) -> pd.DataFrame:
    """Add compact descriptors of the top-k accuracy curve.

    These descriptors let the second-level ML model the whole ranking curve,
    not just top-1 or top-5 accuracy.  They use whatever anchors are available
    in the master tables: k=1,3,5,10,20.  Raw per-pair top-k curves are still
    written separately by compute_topk_accuracy_curves when predictions_test.csv
    files are available.
    """
    if df.empty:
        return df
    outdf = df.copy()
    anchors = [1, 3, 5, 10, 20]
    for k in anchors:
        col = f"top{k}_curve_anchor"
        if col not in outdf.columns:
            vals = []
            for _, row in outdf.iterrows():
                vals.append(_topk_anchor_from_row(row, k))
            outdf[col] = pd.to_numeric(pd.Series(vals, index=outdf.index), errors="coerce")
    k_grid = np.arange(1, max_k + 1)
    aucs, gains5, gains20, k95s, k99s, slopes = [], [], [], [], [], []
    for _, row in outdf.iterrows():
        pts = []
        for k in anchors:
            v = row.get(f"top{k}_curve_anchor", np.nan)
            try:
                v = float(v)
                if math.isfinite(v):
                    pts.append((k, max(0.0, min(1.0, v))))
            except Exception:
                pass
        if len(pts) < 2:
            aucs.append(np.nan); gains5.append(np.nan); gains20.append(np.nan); k95s.append(np.nan); k99s.append(np.nan); slopes.append(np.nan); continue
        pts = sorted(set(pts))
        ks = np.array([p[0] for p in pts], dtype=float)
        vs = np.array([p[1] for p in pts], dtype=float)
        interp = np.interp(k_grid, ks, vs)
        aucs.append(float(np.mean(interp)))
        v1 = float(np.interp(1, ks, vs)); v5 = float(np.interp(5, ks, vs)); v20 = float(np.interp(20, ks, vs))
        gains5.append(v5 - v1)
        gains20.append(v20 - v1)
        ge95 = k_grid[interp >= 0.95]
        ge99 = k_grid[interp >= 0.99]
        k95s.append(float(ge95[0]) if len(ge95) else np.nan)
        k99s.append(float(ge99[0]) if len(ge99) else np.nan)
        # early slope: accuracy gain per rank from k=1 to k=5
        slopes.append((v5 - v1) / 4.0)
    outdf["topk_auc_1_20"] = aucs
    outdf["topk_gain5_minus_top1"] = gains5
    outdf["topk_gain20_minus_top1"] = gains20
    outdf["topk_k95"] = k95s
    outdf["topk_k99"] = k99s
    outdf["topk_early_slope_1_to_5"] = slopes
    if out is not None:
        cols = [c for c in ["analysis_input_label", "surface_id", "unique_surface_key", "family_normalized", "winner_acc", "top3_acc", "top5_acc", "top10_acc", "top20_acc", "topk_auc_1_20", "topk_gain5_minus_top1", "topk_gain20_minus_top1", "topk_k95", "topk_k99", "topk_early_slope_1_to_5"] if c in outdf.columns]
        outdf[cols].to_csv(out / "topk_curve_descriptors.csv", index=False)
    return outdf


def create_feature_block_comparison(reg: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Summarize model performance by target and feature block."""
    if reg is None or reg.empty or "fit_status" not in reg.columns or "r2_mean" not in reg.columns:
        comp = pd.DataFrame()
    else:
        d = reg[reg["fit_status"].astype(str).eq("ok")].copy()
        if d.empty:
            comp = pd.DataFrame()
        else:
            idx = d.groupby(["target", "feature_block"])["r2_mean"].idxmax()
            comp = d.loc[idx, [c for c in ["target", "feature_block", "model", "n", "cv", "mae_mean", "rmse_mean", "r2_mean", "r2_std"] if c in d.columns]].sort_values(["target", "r2_mean"], ascending=[True, False])
    comp.to_csv(out / "feature_block_model_comparison.csv", index=False)
    return comp


def run_leave_one_family_out(df: pd.DataFrame, out: Path, seed: int = 123, ml_level: str = "standard") -> pd.DataFrame:
    """Train on all but one normalized family and test on the held-out family.

    This is intentionally stricter than random holdout. It asks whether an
    analytic/upstairs rule generalizes to families not seen during training.
    """
    if not HAVE_SKLEARN or df.empty or "family_normalized" not in df.columns:
        res = pd.DataFrame()
        res.to_csv(out / "leave_one_family_out_scores.csv", index=False)
        return res
    if ml_level == "heavy":
        targets = [c for c in ["winner_acc", "top5_acc", "hard_rmse", "top5_pruned_rmse", "topk_auc_1_20", "topk_gain5_minus_top1"] if c in df.columns]
        blocks = ["analytic_upstairs", "analytic_plus_exact_downstairs", "analytic_exact_training_protocol"]
        models = {
            "ridge": Ridge(alpha=1.0, random_state=seed),
            "extra_trees": ExtraTreesRegressor(n_estimators=180, random_state=seed, n_jobs=1, min_samples_leaf=2),
        }
    else:
        targets = [c for c in ["winner_acc", "top5_acc", "hard_rmse", "topk_auc_1_20"] if c in df.columns]
        blocks = ["analytic_upstairs", "analytic_plus_exact_downstairs"]
        models = {"ridge": Ridge(alpha=1.0, random_state=seed)}
    rows = []
    for block in blocks:
        for target in targets:
            X, y, groups, num_cols, cat_cols = make_ml_dataset(df, target, feature_block=block)
            if len(y) < 30 or groups is None or groups.nunique() < 3:
                continue
            X = X.copy(); y = y.copy(); groups = groups.loc[X.index] if hasattr(groups, 'loc') else pd.Series(groups, index=X.index)
            for held in sorted(groups.dropna().unique()):
                test_mask = groups.astype(str).eq(str(held))
                if test_mask.sum() < 3 or (~test_mask).sum() < 20:
                    continue
                for mname, model in models.items():
                    pre = make_preprocessor(num_cols, cat_cols)
                    pipe = Pipeline([("pre", pre), ("model", model)])
                    try:
                        pipe.fit(X.loc[~test_mask], y.loc[~test_mask])
                        pred = pipe.predict(X.loc[test_mask])
                        yy = y.loc[test_mask]
                        rmse = mean_squared_error(yy, pred, squared=False) if "squared" in mean_squared_error.__code__.co_varnames else float(np.sqrt(mean_squared_error(yy, pred)))
                        rows.append({
                            "feature_block": block,
                            "target": target,
                            "model": mname,
                            "heldout_family": held,
                            "n_train": int((~test_mask).sum()),
                            "n_test": int(test_mask.sum()),
                            "mae": mean_absolute_error(yy, pred),
                            "rmse": rmse,
                            "r2": r2_score(yy, pred) if len(yy) >= 2 and np.nanvar(yy) > 0 else np.nan,
                            "fit_status": "ok",
                        })
                    except Exception as e:
                        rows.append({"feature_block": block, "target": target, "model": mname, "heldout_family": held, "fit_status": "failed", "error": str(e)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "leave_one_family_out_scores.csv", index=False)
    return res


def run_regime_classification_from_learned(df: pd.DataFrame, out: Path, seed: int = 123, ml_level: str = "standard") -> pd.DataFrame:
    """Classify surface regimes/families from learned-downstairs metrics only.

    This reverses the usual direction: the GINN becomes a probe of geometry.
    """
    if not HAVE_SKLEARN or df.empty:
        res = pd.DataFrame()
        res.to_csv(out / "regime_classification_from_learned_metrics.csv", index=False)
        return res
    learned = df[df.get("analysis_learned_any", True)].copy()
    features = [c for c in ["winner_acc", "top3_acc", "top5_acc", "top10_acc", "top20_acc", "hard_rmse", "top5_pruned_rmse", "top5_speedup", "branch_entropy", "score_margin", "topk_auc_1_20", "topk_gain5_minus_top1", "topk_k95", "topk_k99"] if c in learned.columns]
    rows = []
    targets = [c for c in ["geometry_regime", "family_coarse", "family_normalized"] if c in learned.columns]
    models = {
        "logistic": LogisticRegression(max_iter=2500, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=160 if ml_level != "heavy" else 320, random_state=seed, n_jobs=1, min_samples_leaf=2, class_weight="balanced"),
    }
    for target in targets:
        d = learned[[target] + features].copy()
        d[target] = d[target].astype(str)
        counts = d[target].value_counts()
        keep_classes = counts[counts >= 5].index
        d = d[d[target].isin(keep_classes)].copy()
        if len(d) < 30 or d[target].nunique() < 2 or len(features) < 2:
            continue
        X = d[features].apply(pd.to_numeric, errors="coerce")
        y = d[target]
        pipe_pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        n_splits = min(5 if ml_level == "heavy" else 3, int(y.value_counts().min()))
        n_splits = max(2, n_splits)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for mname, model in models.items():
            pipe = Pipeline([("pre", pipe_pre), ("model", model)])
            try:
                res = cross_validate(pipe, X, y, cv=cv, scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}, n_jobs=1, error_score=np.nan)
                rows.append({
                    "target": target,
                    "model": mname,
                    "n": len(y),
                    "classes": int(y.nunique()),
                    "cv": f"StratifiedKFold_{n_splits}",
                    "accuracy_mean": np.nanmean(res["test_accuracy"]),
                    "balanced_accuracy_mean": np.nanmean(res["test_balanced_accuracy"]),
                    "f1_macro_mean": np.nanmean(res["test_f1_macro"]),
                    "fit_status": "ok",
                })
            except Exception as e:
                rows.append({"target": target, "model": mname, "n": len(y), "fit_status": "failed", "error": str(e)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "regime_classification_from_learned_metrics.csv", index=False)
    return res

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def scatter_plot(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    if not HAVE_MATPLOTLIB or x not in df.columns or y not in df.columns:
        return
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(d[x], d[y], s=18, alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    if d[x].min() > 0 and d[x].max() / max(d[x].min(), 1e-12) > 50:
        plt.xscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()




def scatter_plot_by_family(df: pd.DataFrame, x: str, y: str, path: Path, title: str, family_col: str = "family_normalized") -> None:
    if not HAVE_MATPLOTLIB or x not in df.columns or y not in df.columns:
        return
    cols = [x, y]
    if family_col in df.columns:
        cols.append(family_col)
    d = df[cols].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if len(d) < 5:
        return
    plt.figure(figsize=(7.5, 5.5))
    if family_col in d.columns and d[family_col].nunique() > 1:
        for fam, g in d.groupby(family_col):
            plt.scatter(g[x], g[y], s=20, alpha=0.75, label=str(fam))
        plt.legend(fontsize=7, loc="best", framealpha=0.8)
    else:
        plt.scatter(d[x], d[y], s=20, alpha=0.75)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    if d[x].min() > 0 and d[x].max() / max(d[x].min(), 1e-12) > 50:
        plt.xscale("log")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def histogram_plot(df: pd.DataFrame, col: str, path: Path, title: str) -> None:
    if not HAVE_MATPLOTLIB or col not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 3:
        return
    plt.figure(figsize=(7, 5))
    plt.hist(s, bins=min(30, max(8, int(math.sqrt(len(s))))))
    plt.xlabel(col)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def boxplot_by_family(df: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    if not HAVE_MATPLOTLIB or metric not in df.columns or "surface_family" not in df.columns:
        return
    d = df[["surface_family", metric]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna()
    if len(d) < 5 or d["surface_family"].nunique() < 2:
        return
    groups = []
    labels = []
    for fam, g in d.groupby("surface_family"):
        if len(g) >= 2:
            groups.append(g[metric].values)
            labels.append(str(fam))
    if len(groups) < 2:
        return
    plt.figure(figsize=(max(7, 0.9 * len(groups)), 5))
    plt.boxplot(groups, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_figures(df: pd.DataFrame, figdir: Path) -> List[str]:
    safe_mkdir(figdir)
    figures = []
    if df.empty or not HAVE_MATPLOTLIB:
        return figures
    learned = df[df.get("analysis_learned_any", True)].copy()
    specs = [
        ("word_ball_size", "winner_acc", "winner_accuracy_vs_word_ball_size.png", "Winner accuracy vs word-ball size"),
        ("word_ball_size", "top5_acc", "top5_accuracy_vs_word_ball_size.png", "Top-5 accuracy vs word-ball size"),
        ("word_ball_size", "hard_rmse", "hard_rmse_vs_word_ball_size.png", "Hard-distance RMSE vs word-ball size"),
        ("shortcut_fraction_true_test", "hard_rmse", "hard_rmse_vs_shortcut_fraction.png", "RMSE vs shortcut fraction"),
        ("near_seam_002", "winner_acc", "winner_accuracy_vs_near_seam_fraction.png", "Winner accuracy vs near-seam fraction"),
        ("generator_count", "branch_entropy", "branch_entropy_vs_generator_count.png", "Branch entropy vs generator count"),
        ("top5_speedup", "top5_pruned_rmse", "top5_pruned_rmse_vs_speedup.png", "Top-5 pruned RMSE vs speedup"),
    ]
    for x, y, fn, title in specs:
        p = figdir / fn
        scatter_plot(learned, x, y, p, title)
        if p.exists():
            figures.append(str(p))
        pf = figdir / fn.replace(".png", "_by_family.png")
        scatter_plot_by_family(learned, x, y, pf, title + " by family")
        if pf.exists():
            figures.append(str(pf))
    for col in ["winner_acc", "top5_acc", "hard_rmse", "top5_pruned_rmse", "top5_speedup", "branch_entropy", "word_ball_size"]:
        p = figdir / f"hist_{col}.png"
        histogram_plot(learned, col, p, f"Distribution of {col}")
        if p.exists():
            figures.append(str(p))
    for metric in ["winner_acc", "top5_acc", "hard_rmse", "top5_pruned_rmse"]:
        p = figdir / f"boxplot_{metric}_by_family.png"
        boxplot_by_family(learned, metric, p, f"{metric} by family")
        if p.exists():
            figures.append(str(p))
    return figures


# ---------------------------------------------------------------------------
# Second-level ML
# ---------------------------------------------------------------------------


def make_ml_dataset(df: pd.DataFrame, target: str, feature_block: str = "analytic_upstairs") -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], List[str], List[str]]:
    if target not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), None, [], []
    d = df[df.get("analysis_learned_any", True)].copy()
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d[d[target].notna()].copy()
    if len(d) < 20:
        return pd.DataFrame(), pd.Series(dtype=float), None, [], []

    # Choose leakage-controlled feature blocks.
    num_candidates, cat_candidates = columns_for_feature_block(d, feature_block, target=target)
    candidate_cols = prune_empty_constant_columns(d, num_candidates + cat_candidates)
    if not candidate_cols:
        return pd.DataFrame(), pd.Series(dtype=float), None, [], []

    X = d[candidate_cols].copy()
    y = d[target].copy()
    groups = d["family_normalized"].astype(str) if "family_normalized" in d.columns else (d["surface_family"].astype(str) if "surface_family" in d.columns else None)

    numeric_cols = []
    categorical_cols = []
    for c in candidate_cols:
        s_num = pd.to_numeric(X[c], errors="coerce")
        # Use numeric if enough values are numeric; otherwise categorical.
        if c in num_candidates and s_num.notna().sum() >= max(5, int(0.20 * len(X))):
            X[c] = s_num
            numeric_cols.append(c)
        else:
            X[c] = X[c].astype(str).replace("nan", np.nan).replace("None", np.nan)
            categorical_cols.append(c)
    return X, y, groups, numeric_cols, categorical_cols

def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols))
    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def cv_for_regression(y: pd.Series, groups: Optional[pd.Series], max_splits: int = 3) -> Tuple[Any, Optional[pd.Series], str]:
    if groups is not None and groups.nunique() >= 3 and len(y) >= groups.nunique() * 3:
        n = min(max_splits, int(groups.nunique()))
        return GroupKFold(n_splits=n), groups, f"GroupKFold_by_family_{n}"
    n = min(max_splits, max(2, len(y) // 10))
    return KFold(n_splits=n, shuffle=True, random_state=123), None, f"KFold_{n}"


def cv_for_classification(y: pd.Series, groups: Optional[pd.Series], max_splits: int = 3) -> Tuple[Any, Optional[pd.Series], str]:
    if groups is not None and groups.nunique() >= 3 and len(y) >= groups.nunique() * 3:
        n = min(max_splits, int(groups.nunique()))
        return GroupKFold(n_splits=n), groups, f"GroupKFold_by_family_{n}"
    n = min(max_splits, int(y.value_counts().min()))
    n = max(2, n)
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=123), None, f"StratifiedKFold_{n}"


def run_second_level_ml(df: pd.DataFrame, out: Path, seed: int = 123, ml_level: str = "standard") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not HAVE_SKLEARN:
        msg = pd.DataFrame([{"status": "skipped", "reason": "scikit-learn not available"}])
        msg.to_csv(out / "regression_model_scores.csv", index=False)
        msg.to_csv(out / "classification_model_scores.csv", index=False)
        msg.to_csv(out / "feature_importance.csv", index=False)
        return msg, msg, msg

    if ml_level == "light":
        target_list = ["top5_acc", "hard_rmse", "winner_acc", "topk_auc_1_20", "topk_gain5_minus_top1"]
        n_estimators = 80
        hgb_iter = 80
        max_splits = 3
        include_hgb = False
        feature_blocks = ["analytic_upstairs", "analytic_plus_exact_downstairs"]
    elif ml_level == "heavy":
        target_list = ["winner_acc", "top5_acc", "top10_acc", "top20_acc", "hard_rmse", "top5_pruned_rmse", "top5_speedup", "branch_entropy", "topk_auc_1_20", "topk_gain5_minus_top1", "topk_k95", "topk_k99"]
        n_estimators = 400
        hgb_iter = 300
        max_splits = 5
        include_hgb = True
        feature_blocks = ["analytic_upstairs", "analytic_plus_exact_downstairs", "analytic_exact_training_protocol"]
    else:
        target_list = ["winner_acc", "top5_acc", "top20_acc", "hard_rmse", "top5_pruned_rmse", "top5_speedup", "branch_entropy", "topk_auc_1_20", "topk_gain5_minus_top1"]
        n_estimators = 100
        hgb_iter = 80
        max_splits = 3
        include_hgb = False
        feature_blocks = ["analytic_upstairs", "analytic_plus_exact_downstairs"]
    reg_targets = [c for c in target_list if c in df.columns]
    reg_models = {
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1, min_samples_leaf=2),
        "extra_trees": ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1, min_samples_leaf=2),
    }
    if include_hgb:
        reg_models["hist_gradient_boosting"] = HistGradientBoostingRegressor(random_state=seed, max_iter=hgb_iter, l2_regularization=0.01)
    reg_rows = []
    importance_rows = []
    for feature_block in feature_blocks:
        for target in reg_targets:
            X, y, groups, num_cols, cat_cols = make_ml_dataset(df, target, feature_block=feature_block)
            if len(y) < 20 or (not num_cols and not cat_cols):
                continue
            pre = make_preprocessor(num_cols, cat_cols)
            cv, cv_groups, cv_name = cv_for_regression(y, groups, max_splits=max_splits)
            for name, model in reg_models.items():
                pipe = Pipeline([("pre", pre), ("model", model)])
                try:
                    if ml_level == "heavy":
                        res = cross_validate(
                            pipe, X, y,
                            cv=cv,
                            groups=cv_groups,
                            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
                            n_jobs=1,
                            error_score=np.nan,
                        )
                        row_scores = {
                            "cv": cv_name,
                            "mae_mean": -np.nanmean(res["test_mae"]),
                            "mae_std": np.nanstd(-res["test_mae"]),
                            "rmse_mean": -np.nanmean(res["test_rmse"]),
                            "rmse_std": np.nanstd(-res["test_rmse"]),
                            "r2_mean": np.nanmean(res["test_r2"]),
                            "r2_std": np.nanstd(res["test_r2"]),
                        }
                    else:
                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
                        pipe.fit(Xtr, ytr)
                        pred = pipe.predict(Xte)
                        rmse = mean_squared_error(yte, pred, squared=False) if "squared" in mean_squared_error.__code__.co_varnames else float(np.sqrt(mean_squared_error(yte, pred)))
                        row_scores = {
                            "cv": "single_holdout_75_25_fast",
                            "mae_mean": mean_absolute_error(yte, pred),
                            "mae_std": np.nan,
                            "rmse_mean": rmse,
                            "rmse_std": np.nan,
                            "r2_mean": r2_score(yte, pred),
                            "r2_std": np.nan,
                        }
                    reg_rows.append({
                        "feature_block": feature_block,
                        "target": target,
                        "model": name,
                        "n": len(y),
                        "n_numeric_features": len(num_cols),
                        "n_categorical_features": len(cat_cols),
                        **row_scores,
                        "fit_status": "ok",
                    })
                    if name in ["random_forest", "extra_trees"]:
                        pipe.fit(X, y)
                        try:
                            feat_names = []
                            if num_cols:
                                feat_names.extend(num_cols)
                            if cat_cols:
                                enc = pipe.named_steps["pre"].named_transformers_.get("cat")
                                if enc is not None:
                                    oh = enc.named_steps["onehot"]
                                    try:
                                        feat_names.extend(list(oh.get_feature_names_out(cat_cols)))
                                    except Exception:
                                        feat_names.extend([f"cat_{i}" for i in range(len(pipe.named_steps["model"].feature_importances_) - len(feat_names))])
                            imps = pipe.named_steps["model"].feature_importances_
                            for f, imp in sorted(zip(feat_names, imps), key=lambda x: x[1], reverse=True)[:60]:
                                importance_rows.append({"task": "regression", "feature_block": feature_block, "target": target, "model": name, "feature": f, "importance": imp})
                        except Exception:
                            pass
                except Exception as e:
                    reg_rows.append({"feature_block": feature_block, "target": target, "model": name, "n": len(y), "cv": cv_name, "fit_status": "failed", "error": str(e)})

    # Classification targets.  v1.2 avoids finite_area as a target with analytic
    # predictors because compact/finite_area flags are direct leakage.
    class_df = df.copy()
    if "top5_acc" in class_df.columns:
        class_df["class_high_top5"] = pd.to_numeric(class_df["top5_acc"], errors="coerce").ge(0.95)
    if "winner_acc" in class_df.columns:
        class_df["class_high_top1"] = pd.to_numeric(class_df["winner_acc"], errors="coerce").ge(0.85)
    if "hard_rmse" in class_df.columns:
        class_df["class_low_rmse"] = pd.to_numeric(class_df["hard_rmse"], errors="coerce").le(0.05)
    class_targets = [c for c in ["class_high_top5", "class_high_top1", "class_low_rmse"] if c in class_df.columns]
    clf_models = {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1, min_samples_leaf=2, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1, min_samples_leaf=2, class_weight="balanced"),
    }
    if include_hgb:
        clf_models["hist_gradient_boosting"] = HistGradientBoostingClassifier(random_state=seed, max_iter=hgb_iter, l2_regularization=0.01)
    clf_rows = []
    for feature_block in feature_blocks:
        for target in class_targets:
            d = class_df[class_df.get("analysis_learned_any", True)].copy()
            d = d[d[target].notna()].copy()
            if len(d) < 25 or d[target].nunique() < 2 or d[target].value_counts().min() < 5:
                continue
            tmp = d.copy()
            tmp[target] = tmp[target].astype(int)
            X, ynum, groups, num_cols, cat_cols = make_ml_dataset(tmp, target, feature_block=feature_block)
            y = ynum.astype(int)
            if len(y) < 25 or y.nunique() < 2 or (not num_cols and not cat_cols):
                continue
            pre = make_preprocessor(num_cols, cat_cols)
            cv, cv_groups, cv_name = cv_for_classification(y, groups, max_splits=max_splits)
            for name, model in clf_models.items():
                pipe = Pipeline([("pre", pre), ("model", model)])
                try:
                    if ml_level == "heavy":
                        res = cross_validate(
                            pipe, X, y,
                            cv=cv,
                            groups=cv_groups,
                            scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1": "f1"},
                            n_jobs=1,
                            error_score=np.nan,
                        )
                        row_scores = {
                            "cv": cv_name,
                            "accuracy_mean": np.nanmean(res["test_accuracy"]),
                            "balanced_accuracy_mean": np.nanmean(res["test_balanced_accuracy"]),
                            "f1_mean": np.nanmean(res["test_f1"]),
                        }
                    else:
                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y if y.value_counts().min() >= 2 else None)
                        pipe.fit(Xtr, ytr)
                        pred = pipe.predict(Xte)
                        row_scores = {
                            "cv": "single_holdout_75_25_fast",
                            "accuracy_mean": accuracy_score(yte, pred),
                            "balanced_accuracy_mean": balanced_accuracy_score(yte, pred),
                            "f1_mean": f1_score(yte, pred, zero_division=0),
                        }
                    clf_rows.append({
                        "feature_block": feature_block,
                        "target": target,
                        "model": name,
                        "n": len(y),
                        "positive_fraction": y.mean(),
                        "n_numeric_features": len(num_cols),
                        "n_categorical_features": len(cat_cols),
                        **row_scores,
                        "fit_status": "ok",
                    })
                except Exception as e:
                    clf_rows.append({"feature_block": feature_block, "target": target, "model": name, "n": len(y), "cv": cv_name, "fit_status": "failed", "error": str(e)})

    reg = pd.DataFrame(reg_rows)
    clf = pd.DataFrame(clf_rows)
    imp = pd.DataFrame(importance_rows)
    reg.to_csv(out / "regression_model_scores.csv", index=False)
    clf.to_csv(out / "classification_model_scores.csv", index=False)
    imp.to_csv(out / "feature_importance.csv", index=False)
    return reg, clf, imp


# ---------------------------------------------------------------------------
# Clustering / unsupervised
# ---------------------------------------------------------------------------


def run_unsupervised(df: pd.DataFrame, out: Path, seed: int = 123) -> pd.DataFrame:
    if not HAVE_SKLEARN or df.empty:
        cl = pd.DataFrame()
        cl.to_csv(out / "cluster_assignments.csv", index=False)
        return cl
    try:
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.decomposition import PCA
    except Exception:
        cl = pd.DataFrame()
        cl.to_csv(out / "cluster_assignments.csv", index=False)
        return cl
    d = df[df.get("analysis_eligible", True)].copy()
    # Use analytic plus learned summary metrics, but only rows with some numeric data.
    cols = [c for c in [
        "genus", "compactified_genus", "cusp_count", "area", "euler_characteristic", "generator_count", "word_ball_size",
        "word_ball_risk_depth2", "shortcut_fraction_true_test", "mean_winner_depth", "winner_acc", "top5_acc",
        "hard_rmse", "top5_pruned_rmse", "branch_entropy", "near_seam_002", "unique_true_branches",
    ] if c in d.columns]
    if len(cols) < 3 or len(d) < 10:
        cl = pd.DataFrame()
        cl.to_csv(out / "cluster_assignments.csv", index=False)
        return cl
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    keep = X.notna().sum(axis=1) >= 3
    d = d.loc[keep].copy()
    X = X.loc[keep]
    if len(d) < 10:
        cl = pd.DataFrame()
        cl.to_csv(out / "cluster_assignments.csv", index=False)
        return cl
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    Xs = pipe.fit_transform(X)
    ncomp = min(5, Xs.shape[1], len(d) - 1)
    pca = PCA(n_components=ncomp, random_state=seed)
    Z = pca.fit_transform(Xs)
    k = min(6, max(2, int(math.sqrt(len(d) / 2))))
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(Z[:, :min(3, ncomp)])
    try:
        agg = AgglomerativeClustering(n_clusters=k).fit_predict(Z[:, :min(3, ncomp)])
    except Exception:
        agg = np.full(len(d), -1)
    outdf = pd.DataFrame({
        "analysis_input_label": d.get("analysis_input_label", "").values,
        "record_uid": d.get("record_uid", d.index.astype(str)).values,
        "surface_id": d.get("surface_id", d.index.astype(str)).values,
        "surface_family": d.get("surface_family", "").values,
        "surface_subfamily": d.get("surface_subfamily", "").values,
        "unique_surface_key": d.get("unique_surface_key", d.index.astype(str)).values,
        "pca1": Z[:, 0],
        "pca2": Z[:, 1] if ncomp > 1 else np.nan,
        "pca3": Z[:, 2] if ncomp > 2 else np.nan,
        "kmeans_cluster": labels,
        "agglomerative_cluster": agg,
    })
    outdf.to_csv(out / "cluster_assignments.csv", index=False)

    # PCA scatter figure.
    if HAVE_MATPLOTLIB and ncomp >= 2:
        plt.figure(figsize=(7, 5))
        plt.scatter(outdf["pca1"], outdf["pca2"], s=18, alpha=0.75)
        plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.title("PCA of analytic + learned surface features")
        plt.tight_layout()
        plt.savefig(out.parent / "figures" / "pca_combined_features.png", dpi=160)
        plt.close()
    return outdf


# ---------------------------------------------------------------------------
# Scientific question synthesis and report
# ---------------------------------------------------------------------------


def scientific_question_answers(df: pd.DataFrame, reg: pd.DataFrame, clf: pd.DataFrame, corr: pd.DataFrame, out: Path) -> str:
    lines = []
    def add(s=""):
        lines.append(s)
    learned = df[df.get("analysis_learned_any", False)].copy() if "analysis_learned_any" in df.columns else df.copy()
    eligible = df[df.get("analysis_eligible", True)].copy()
    add("# Scientific question summaries")
    add("")
    add("## Q1. What did the GINN learn downstairs?")
    add("")
    n_learned = len(learned)
    n_unique = learned["unique_surface_key"].nunique() if "unique_surface_key" in learned.columns else np.nan
    add(f"The analysis set contains {n_learned} learned-downstairs records and {n_unique} unique learned surface keys.")
    for m in ["winner_acc", "top3_acc", "top5_acc", "top10_acc", "top20_acc", "hard_rmse", "top5_pruned_rmse", "top5_speedup", "branch_entropy"]:
        if m in learned.columns:
            s = pd.to_numeric(learned[m], errors="coerce").dropna()
            if len(s):
                add(f"- {m}: n={len(s)}, median={s.median():.4g}, mean={s.mean():.4g}, IQR=({s.quantile(.25):.4g},{s.quantile(.75):.4g}).")
    add("")
    add("## Q2. Which analytic/upstairs features predict learnability?")
    add("")
    if not corr.empty:
        add("Associations below are leakage-controlled and grouped by feature block; duplicate learned-output columns are excluded from this table.")
        for target in ["winner_acc", "top5_acc", "hard_rmse", "top5_pruned_rmse"]:
            sub = corr[corr["target"] == target].head(8)
            if not sub.empty:
                bits = ", ".join(f"[{getattr(r, 'feature_block', '')}] {r.feature} ({r.spearman_r:.2f})" for r in sub.itertuples())
                add(f"- Strongest leakage-controlled Spearman associations for {target}: {bits}.")
    if not reg.empty and "r2_mean" in reg.columns:
        good = reg[reg.get("fit_status", "ok") == "ok"].sort_values("r2_mean", ascending=False).head(10)
        if not good.empty:
            add("- Best cross-validated regression results suggest which targets are predictable from analytic/upstairs features; see `regression_model_scores.csv`.")
            for r in good.itertuples():
                add(f"  - {getattr(r, 'feature_block', '')} / {r.target} / {r.model}: R2={getattr(r, 'r2_mean', np.nan):.3g}, RMSE={getattr(r, 'rmse_mean', np.nan):.3g}.")
    add("")
    add("## Q3. Are compact, cusped, Schottky, and elementary examples distinct regimes?")
    add("")
    if "geometry_regime" in eligible.columns:
        counts = eligible["geometry_regime"].value_counts(dropna=False)
        for k, v in counts.items():
            add(f"- {k}: {v} eligible records.")
    add("Use the family boxplots and PCA/cluster assignments to assess whether analytic regimes remain distinct in learned-downstairs feature space.")
    add("")
    add("## Q4. Is the GINN more useful as an exact branch predictor or as a top-k search-pruning engine?")
    add("")
    if "winner_acc" in learned.columns and "top5_acc" in learned.columns:
        w = pd.to_numeric(learned["winner_acc"], errors="coerce")
        t5 = pd.to_numeric(learned["top5_acc"], errors="coerce")
        gap = (t5 - w).dropna()
        if len(gap):
            add(f"The median top5-minus-top1 accuracy gap is {gap.median():.4g}. A positive gap indicates that the GINN is often especially valuable as a pruning/ranking tool.")
    if "top5_speedup" in learned.columns:
        s = pd.to_numeric(learned["top5_speedup"], errors="coerce").dropna()
        if len(s):
            add(f"The median top-5 speedup factor is {s.median():.4g} over full word-ball search.")
    add("")
    add("## Q5. Which surfaces are scientifically interesting anomalies?")
    add("")
    add("See `anomaly_table.csv`, `worst_surfaces_high_rmse.csv`, and `worst_surfaces_top5_accuracy.csv` for surfaces worth inspecting individually.")
    text = "\n".join(lines) + "\n"
    (out / "scientific_question_answers.md").write_text(text, encoding="utf-8")
    return text



def _wrap_table_tex(df: pd.DataFrame, max_rows: int = 20, tiny: bool = False) -> str:
    body = dataframe_to_latex_tabular(df, max_rows=max_rows)
    size = r"\scriptsize" if tiny else r"\small"
    if df is not None and not df.empty and len(df.columns) > 4:
        inner = r"\resizebox{\linewidth}{!}{%" + "\n" + body + "\n}"
    else:
        inner = body
    return "\n" + r"\begin{center}" + "\n" + size + "\n" + inner + "\n" + r"\normalsize" + "\n" + r"\end{center}" + "\n"


def _short_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 20) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[[c for c in cols if c in df.columns]].head(max_rows).copy()


def _create_fallback_pdf(pdf_path: Path, title: str, audit_small: pd.DataFrame, metric_small: pd.DataFrame, question_text: str, figures: List[str]) -> None:
    """Create a simple PDF report when pdflatex is unavailable or fails."""
    if not HAVE_REPORTLAB:
        return
    try:
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Executive summary", styles["Heading1"]))
        for _, r in audit_small.iterrows():
            story.append(Paragraph(f"<b>{latex_escape(r.get('item',''))}</b>: {latex_escape(r.get('value',''))}", styles["BodyText"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Learned metric summary", styles["Heading1"]))
        if metric_small is not None and not metric_small.empty:
            for _, r in metric_small.iterrows():
                story.append(Paragraph(f"<b>{latex_escape(r.get('metric',''))}</b>: n={latex_escape(r.get('n',''))}, median={fmt_float(r.get('median'))}, mean={fmt_float(r.get('mean'))}", styles["BodyText"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Scientific questions", styles["Heading1"]))
        for line in question_text.splitlines():
            if not line.strip():
                story.append(Spacer(1, 4)); continue
            if line.startswith("## "):
                story.append(Paragraph(latex_escape(line[3:]), styles["Heading2"]))
            elif line.startswith("# "):
                story.append(Paragraph(latex_escape(line[2:]), styles["Heading1"]))
            else:
                story.append(Paragraph(latex_escape(line), styles["BodyText"]))
        for f in figures[:8]:
            fp = Path(f)
            if fp.exists():
                story.append(PageBreak())
                story.append(Paragraph(fp.stem.replace("_", " "), styles["Heading2"]))
                try:
                    img = Image(str(fp), width=480, height=320)
                    img.hAlign = "CENTER"
                    story.append(img)
                except Exception:
                    pass
        doc.build(story)
    except Exception:
        pass


def write_latex_report(outroot: Path, tables: Dict[str, pd.DataFrame], df: pd.DataFrame, audit: pd.DataFrame, fam: pd.DataFrame, metric: pd.DataFrame, reg: pd.DataFrame, clf: pd.DataFrame, figures: List[str], question_text: str, compile_pdf: bool = True) -> Path:
    report_dir = safe_mkdir(outroot / "report")
    tex_path = report_dir / "FuchsianZooAnalysis_Report.tex"

    audit_small = audit[audit["item"].isin([
        "records_total", "eligible_records", "unique_eligible_surface_keys", "learned_any_records",
        "unique_learned_any_surface_keys", "full_ginn_records", "unique_full_ginn_surface_keys", "excluded_orbifold_records",
    ])].copy()

    fam_cols = ["family_normalized", "unique_surface_keys", "learned_any_total", "winner_acc", "top5_acc", "hard_rmse", "top5_speedup", "branch_entropy"]
    fam_small = _short_table(fam, fam_cols, max_rows=25)
    if not fam_small.empty:
        fam_small = fam_small.rename(columns={"family_normalized":"family", "unique_surface_keys":"unique", "learned_any_total":"learned", "winner_acc":"top1", "top5_acc":"top5", "hard_rmse":"rmse", "top5_speedup":"speedup", "branch_entropy":"entropy"})
    metric_small = metric[metric["metric"].isin(["winner_acc", "top3_acc", "top5_acc", "top10_acc", "top20_acc", "hard_rmse", "top5_pruned_rmse", "top5_speedup", "topk_auc_1_20", "topk_gain5_minus_top1", "branch_entropy"])] if metric is not None and not metric.empty else pd.DataFrame()
    reg_cols = ["feature_block", "target", "model", "n", "cv", "rmse_mean", "r2_mean", "fit_status"]
    reg_small = _short_table(reg.sort_values("r2_mean", ascending=False) if reg is not None and not reg.empty and "r2_mean" in reg.columns else reg, reg_cols, max_rows=18)
    clf_cols = ["feature_block", "target", "model", "n", "positive_fraction", "cv", "balanced_accuracy_mean", "f1_mean", "fit_status"]
    clf_small = _short_table(clf.sort_values("balanced_accuracy_mean", ascending=False) if clf is not None and not clf.empty and "balanced_accuracy_mean" in clf.columns else clf, clf_cols, max_rows=15)

    anomaly = pd.DataFrame()
    anom_path = outroot / "tables" / "anomaly_table.csv"
    if anom_path.exists():
        try:
            anomaly = pd.read_csv(anom_path)
        except Exception:
            anomaly = pd.DataFrame()
    anomaly_small = _short_table(anomaly, ["surface_id", "family_normalized", "geometry_regime", "anomaly_score", "winner_acc", "top5_acc", "hard_rmse", "word_ball_size", "generator_count"], max_rows=12)

    fig_rel = []
    # Put the top-k plot and family-colored plots first when present.
    for f in figures:
        if "topk_accuracy" in str(f):
            fig_rel.append(os.path.relpath(f, report_dir))
    for f in figures:
        if "by_family" in str(f) or "colored" in str(f):
            rel = os.path.relpath(f, report_dir)
            if rel not in fig_rel:
                fig_rel.append(rel)
    for f in figures:
        rel = os.path.relpath(f, report_dir)
        if rel not in fig_rel:
            fig_rel.append(rel)
    fig_rel = fig_rel[:10]

    # Executive summary bullets from available metrics.
    def audit_value(name: str) -> str:
        sub = audit_small[audit_small["item"].astype(str).eq(name)]
        return str(sub["value"].iloc[0]) if not sub.empty else ""
    learned_records = audit_value("learned_any_records")
    unique_learned = audit_value("unique_learned_any_surface_keys")
    med = {}
    if metric_small is not None and not metric_small.empty:
        for _, r in metric_small.iterrows():
            med[str(r.get("metric"))] = fmt_float(r.get("median"), 4)

    q_latex = latex_escape(question_text).replace("\n", "\n\n")
    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{enumitem}
\usepackage{float}
\usepackage{caption}
\usepackage{microtype}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\title{Fuchsian Zoo Analysis Report}
\author{Generated by FuchsianZooAnalysis\_v1\_3}
\date{\today}
\begin{document}
\maketitle

\section{Executive summary}
This report analyzes completed GENN/Fuchsian zoo runs.  The central question is whether analytic/upstairs Fuchsian data can explain learned downstairs branch geometry.

\begin{itemize}[leftmargin=1.5em]
"""
    if learned_records:
        tex += f"\n\\item The analysis contains {latex_escape(learned_records)} learned-downstairs records and {latex_escape(unique_learned)} unique learned surface keys."
    if med.get("winner_acc") and med.get("top5_acc"):
        tex += f"\n\\item Median top-1 winner accuracy is {med.get('winner_acc')}, while median top-5 branch recall is {med.get('top5_acc')}."
    if med.get("top20_acc"):
        tex += f"\n\\item Median top-20 branch recall is {med.get('top20_acc')}, supporting the use of the GINN as a branch-pruning engine."
    if med.get("top5_pruned_rmse") and med.get("top5_speedup"):
        tex += f"\n\\item Median top-5 pruned RMSE is {med.get('top5_pruned_rmse')} with median top-5 speedup {med.get('top5_speedup')}."
    tex += r"""
\end{itemize}

\section{Data audit}
""" + _wrap_table_tex(audit_small, max_rows=30) + r"""

\section{Main family summary}
This table is intentionally compact. Full raw and alias-resolved tables are written to \texttt{tables/}.
""" + _wrap_table_tex(fam_small, max_rows=30, tiny=True) + r"""

\section{Learned metric summary}
""" + _wrap_table_tex(metric_small, max_rows=30) + r"""

\section{Scientific question summaries}
\small
\begin{verbatim}
""" + question_text[:7000] + r"""
\end{verbatim}
\normalsize

\section{Leakage-controlled second-level regression}
""" + _wrap_table_tex(reg_small, max_rows=20, tiny=True) + r"""

\section{Leakage-controlled second-level classification}
""" + _wrap_table_tex(clf_small, max_rows=20, tiny=True) + r"""

\section{Scientifically interesting anomalies}
""" + _wrap_table_tex(anomaly_small, max_rows=12, tiny=True) + r"""

\section{Selected figures}
"""
    for f in fig_rel:
        tex += "\n\\begin{figure}[H]\n\\centering\n"
        tex += f"\\includegraphics[width=0.86\\linewidth]{{{latex_escape(f)}}}\n"
        tex += f"\\caption{{{latex_escape(Path(f).stem.replace('_', ' '))}}}\n"
        tex += "\\end{figure}\n"
    tex += r"""

\section{Output files}
The most important CSV outputs are in the \texttt{tables/} directory. Key files include \texttt{topk\_curve\_descriptors.csv}, \texttt{feature\_block\_model\_comparison.csv}, \texttt{leave\_one\_family\_out\_scores.csv}, \texttt{regime\_classification\_from\_learned\_metrics.csv}, \texttt{anomaly\_table.csv}, \texttt{regression\_model\_scores.csv}, \texttt{classification\_model\_scores.csv}, and \texttt{feature\_importance.csv}.

\end{document}
"""
    tex_path.write_text(tex, encoding="utf-8")

    pdf_path = tex_path.with_suffix(".pdf")
    if compile_pdf:
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            try:
                subprocess.run([pdflatex, "-interaction=nonstopmode", tex_path.name], cwd=str(report_dir), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
                subprocess.run([pdflatex, "-interaction=nonstopmode", tex_path.name], cwd=str(report_dir), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
            except Exception:
                pass
        if not pdf_path.exists():
            _create_fallback_pdf(pdf_path, "Fuchsian Zoo Analysis Report", audit_small, metric_small, question_text, [str(report_dir / f) if not Path(f).is_absolute() else f for f in fig_rel])
    return tex_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Comprehensive analysis for completed Fuchsian Zoo / GENN training runs.")
    ap.add_argument("--runs", nargs="*", default=[], help="Completed zoo run directories or master run directories.")
    ap.add_argument("--table-dirs", nargs="*", default=[], help="Direct paths to MasterBuilder tables directories.")
    ap.add_argument("--outroot", default="analysis_runs", help="Output root directory.")
    ap.add_argument("--label", default="analysis", help="Label for this analysis run.")
    ap.add_argument("--dedupe", choices=["none", "primary", "unique_mean"], default="none", help="Dedupe mode for exported model-ready table. ML uses all learned rows unless --dedupe=unique_mean.")
    ap.add_argument("--top-n", type=int, default=25, help="Number of best/worst/anomaly rows to report.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for ML analysis.")
    ap.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    ap.add_argument("--no-ml", action="store_true", help="Skip second-level ML modeling.")
    ap.add_argument("--ml-level", choices=["light", "standard", "heavy"], default="standard", help="How much second-level ML to run. Use heavy for the longest analysis.")
    ap.add_argument("--compile-pdf", action="store_true", help="Compatibility flag: try to compile the LaTeX report. In v1.3 PDF generation is attempted by default unless --no-pdf is used.")
    ap.add_argument("--no-pdf", action="store_true", help="Do not try to create a PDF report; write only LaTeX and Markdown summaries.")
    ap.add_argument("--topk-max", type=int, default=20, help="Maximum k for top-k accuracy curves.")
    ap.add_argument("--no-raw-prediction-topk", action="store_true", help="Do not load raw predictions_test.csv files for exact top-k curves; use master-table anchors only.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.runs and not args.table_dirs:
        print("ERROR: provide at least one --runs path or --table-dirs path", file=sys.stderr)
        return 2

    stamp = now_stamp()
    outroot = safe_mkdir(Path(args.outroot) / f"run_{stamp}_{args.label}")
    tables_out = safe_mkdir(outroot / "tables")
    fig_out = safe_mkdir(outroot / "figures")
    report_out = safe_mkdir(outroot / "report")

    manifest: Dict[str, Any] = {
        "program": PROGRAM,
        "version": VERSION,
        "created": stamp,
        "args": vars(args),
        "outroot": str(outroot),
        "python": sys.version,
        "have_sklearn": HAVE_SKLEARN,
        "have_matplotlib": HAVE_MATPLOTLIB,
        "have_reportlab": HAVE_REPORTLAB,
    }

    try:
        sources = locate_sources(args.runs, args.table_dirs)
        manifest["sources"] = [s.__dict__ | {"root": str(s.root), "tables": str(s.tables), "zoo_manifest": str(s.zoo_manifest) if s.zoo_manifest else "", "zoo_summary": str(s.zoo_summary) if s.zoo_summary else ""} for s in sources]
        tables = load_tables(sources)
        tables["input_sources"].to_csv(tables_out / "input_sources.csv", index=False)
        # Copy zoo summaries/manifests if present for convenience.
        for s in sources:
            if s.zoo_manifest and s.zoo_manifest.exists():
                shutil.copy2(s.zoo_manifest, outroot / f"{s.label}_zoo_manifest.json")
            if s.zoo_summary and s.zoo_summary.exists():
                shutil.copy2(s.zoo_summary, outroot / f"{s.label}_zoo_summary.csv")

        df = derive_analysis_frame(tables)
        if df.empty:
            raise RuntimeError("No analyzable records found in input tables.")

        # Add top-k curve descriptors before exporting/modeling.
        df = add_topk_descriptor_columns(df, tables_out, max_k=args.topk_max)

        # Export raw combined analysis frame.
        df.to_csv(tables_out / "combined_second_level_features_raw.csv", index=False)

        # Optional deduped model-ready table.
        model_df = df.copy()
        if args.dedupe == "primary" and "primary_unique_record" in model_df.columns:
            model_df = model_df[boolish_series(model_df["primary_unique_record"])]
        elif args.dedupe == "unique_mean" and "unique_surface_key" in model_df.columns:
            # Numeric mean + first categorical per unique key.
            num = model_df.select_dtypes(include=[np.number]).columns.tolist()
            non = [c for c in model_df.columns if c not in num and c != "unique_surface_key"]
            agg = {c: "mean" for c in num}
            for c in non:
                agg[c] = "first"
            model_df = model_df.groupby("unique_surface_key", dropna=False).agg(agg).reset_index()
        model_df.to_csv(tables_out / "model_ready_records.csv", index=False)

        audit = write_data_audit(df, tables, tables_out)
        fam = family_summary(df, tables_out)
        metric = metric_summary(df, tables_out)
        corr = correlation_analysis(df, tables_out)
        anomaly_outputs = best_worst_anomalies(df, tables_out, args.top_n)
        rep = replicate_analysis(df, tables_out)
        unique = unique_surface_summary(df, tables_out)
        topk_surface, topk_family = compute_topk_accuracy_curves(df, tables_out, max_k=args.topk_max, try_raw_predictions=not args.no_raw_prediction_topk)
        cl = run_unsupervised(df, tables_out, args.seed)
        figures = [] if args.no_figures else make_figures(df, fig_out)
        if not args.no_figures and not topk_family.empty:
            topk_fig = fig_out / "topk_accuracy_vs_k_by_family.png"
            plot_topk_accuracy_by_family(topk_family, topk_fig)
            if topk_fig.exists():
                figures.insert(0, str(topk_fig))

        if args.no_ml:
            reg = pd.DataFrame([{"status": "skipped", "reason": "--no-ml"}])
            clf = pd.DataFrame([{"status": "skipped", "reason": "--no-ml"}])
            imp = pd.DataFrame([{"status": "skipped", "reason": "--no-ml"}])
            reg.to_csv(tables_out / "regression_model_scores.csv", index=False)
            clf.to_csv(tables_out / "classification_model_scores.csv", index=False)
            imp.to_csv(tables_out / "feature_importance.csv", index=False)
        else:
            ml_df = model_df if args.dedupe == "unique_mean" else df
            reg, clf, imp = run_second_level_ml(ml_df, tables_out, args.seed, ml_level=args.ml_level)
            feature_block_comp = create_feature_block_comparison(reg, tables_out)
            if args.ml_level == "light":
                pd.DataFrame([{"status":"skipped", "reason":"ml-level light"}]).to_csv(tables_out / "leave_one_family_out_scores.csv", index=False)
                pd.DataFrame([{"status":"skipped", "reason":"ml-level light"}]).to_csv(tables_out / "regime_classification_from_learned_metrics.csv", index=False)
            elif args.ml_level == "standard":
                lofo = run_leave_one_family_out(ml_df, tables_out, args.seed, ml_level=args.ml_level)
                pd.DataFrame([{"status":"skipped", "reason":"use --ml-level heavy for learned-metric regime classification"}]).to_csv(tables_out / "regime_classification_from_learned_metrics.csv", index=False)
            else:
                lofo = run_leave_one_family_out(ml_df, tables_out, args.seed, ml_level=args.ml_level)
                regime_clf = run_regime_classification_from_learned(ml_df, tables_out, args.seed, ml_level=args.ml_level)

        qtext = scientific_question_answers(df, reg, clf, corr, tables_out)
        attempt_pdf = (args.compile_pdf or not args.no_pdf)
        tex_path = write_latex_report(outroot, tables, df, audit, fam, metric, reg, clf, figures, qtext, compile_pdf=attempt_pdf)
        manifest["outputs"] = {
            "tables": str(tables_out),
            "figures": str(fig_out),
            "report_tex": str(tex_path),
            "report_pdf": str(tex_path.with_suffix(".pdf")) if tex_path.with_suffix(".pdf").exists() else "",
        }
        manifest["summary"] = {r["item"]: r["value"] for _, r in audit.iterrows()}
        write_json(outroot / "manifest.json", manifest)
        print("FuchsianZooAnalysis v1.3 complete")
        print(f"outroot={outroot}")
        for k, v in manifest["summary"].items():
            if k in ["records_total", "eligible_records", "unique_eligible_surface_keys", "learned_any_records", "unique_learned_any_surface_keys", "full_ginn_records", "unique_full_ginn_surface_keys"]:
                print(f"{k}={v}")
        print(f"report={tex_path}")
        return 0
    except Exception as e:
        manifest["error"] = str(e)
        manifest["traceback"] = traceback.format_exc()
        write_json(outroot / "manifest.json", manifest)
        print("ERROR:", e, file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
