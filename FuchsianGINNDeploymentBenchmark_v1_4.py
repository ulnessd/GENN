#!/usr/bin/env python3
"""
FuchsianGINNDeploymentBenchmark_v1_4.py

Deployment-style benchmark for trained Fuchsian Downstairs GINNs.

This program does not train new models.  It reads a completed ZooBuilder run
and evaluates the already-saved held-out prediction artifacts as a deployment
proxy:

    train once -> use on held-out p,q pairs -> prune the finite word-ball search.

For every predictions_test.csv it can find, it computes recall@k, nominal
exact-check speedup W/k, miss counts, rank statistics, and pruned-distance
error metrics when the prediction file exposes the needed distance columns.

The output is designed to answer the practical question:

    If we exact-check only the GINN top-k candidates, how often does the true
    finite-word winning lift survive, and how much word-ball search is avoided?

v1.4 supports replay/fresh deployment benchmarking, cleaned reporting, baseline comparisons, timing estimates, and stronger exact-search baselines:

    replay: read saved held-out predictions_test.csv artifacts.
    fresh:  load saved PyTorch checkpoints, sample brand-new p,q pairs, score the
            finite word ball, exact-check only the learned top-k candidates,
            and compare against full finite-word brute-force labels.

Fresh mode is the direct train-once/use-many-times deployment test. It requires
the original run directories to contain surface.json and downstairs_ginn_v2_4.pt beside predictions_test.csv; run_manifest.json is used when available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib is only used for report generation.  We use the noninteractive
# backend so this can run on a headless workstation or over SSH.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROGRAM = "FuchsianGINNDeploymentBenchmark_v1_4.py"
VERSION = "1.4"

K_LIST_DEFAULT = [1, 3, 5, 10, 20]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default



def df_to_markdown_safe(df: pd.DataFrame, index: bool = False, max_rows: int = 50) -> str:
    """Small dependency-free markdown table writer.

    pandas.DataFrame.to_markdown requires the optional tabulate package.  This
    function keeps reports portable on bare venvs.
    """
    if df is None or df.empty:
        return "_(empty)_"
    show = df.head(max_rows).copy()
    if index:
        show = show.reset_index()
    cols = [str(c) for c in show.columns]
    rows = []
    for _, r in show.iterrows():
        vals = []
        for c in show.columns:
            v = r[c]
            if isinstance(v, float):
                if math.isfinite(v):
                    vals.append(f"{v:.5g}")
                else:
                    vals.append("")
            else:
                txt = str(v)
                if len(txt) > 80:
                    txt = txt[:77] + "..."
                vals.append(txt.replace("|", "/"))
        rows.append(vals)
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for vals in rows:
        out.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        out.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(out)

def truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return bool(x) and not pd.isna(x)
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive fallback
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None



FAMILY_KEY_MAP = {
    "compact_regular_genus": "C-Reg",
    "regular_genus_g": "C-Reg",
    "compact_hurwitz_klein": "C-Klein",
    "hurwitz_psl_kernel": "H-PSL",
    "modular_principal_gamma": "M-Gamma",
    "modular_gamma0": "M-G0",
    "modular_gamma1": "M-G1",
    "hecke_abelian": "He-Ab",
    "hecke_dihedral_nonabelian": "He-Dih",
    "schottky_free_fuchsian": "Sch",
    "elementary_once_punctured_torus": "E-Torus",
    "elementary_gamma2_trinion": "E-Pants",
    "elementary_cyclic_hyperbolic": "E-CycH",
    "elementary_cyclic_parabolic": "E-CycP",
    "orbifold_cyclic_elliptic_reference": "O-CycE",
    "unknown": "UNK",
    "nan": "UNK",
    "": "UNK",
}

FAMILY_KEY_DESCRIPTION = {
    "C-Reg": "compact regular genus-g surfaces",
    "C-Klein": "compact Klein/Hurwitz quartic surface representation",
    "H-PSL": "Hurwitz triangle-kernel PSL(2,q) surface",
    "M-Gamma": "principal modular congruence subgroup Gamma(N)",
    "M-G0": "modular Gamma_0(N) torsion-audited subgroup",
    "M-G1": "modular Gamma_1(N) torsion-free subgroup",
    "He-Ab": "torsion-free Hecke abelian cover",
    "He-Dih": "torsion-free Hecke nonabelian/dihedral cover",
    "Sch": "Schottky/free-Fuchsian infinite-area example",
    "E-Torus": "elementary once-punctured torus / modular commutator",
    "E-Pants": "elementary thrice-punctured sphere / Gamma(2)",
    "E-CycH": "elementary cyclic hyperbolic quotient",
    "E-CycP": "elementary cyclic parabolic quotient",
    "O-CycE": "excluded cyclic elliptic orbifold reference",
    "UNK": "unclassified or metadata-missing family",
}


def clean_str_value(x: object, default: str = "") -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return default
    return s


def family_key_from_name(name: object) -> str:
    s = clean_str_value(name, "unknown")
    return FAMILY_KEY_MAP.get(s, FAMILY_KEY_MAP.get(s.lower(), "UNK"))


def family_key_description_table() -> pd.DataFrame:
    rows = []
    for key, desc in FAMILY_KEY_DESCRIPTION.items():
        canonical = [k for k, v in FAMILY_KEY_MAP.items() if v == key]
        rows.append({"family_key": key, "canonical_family_names": ", ".join(sorted(set(canonical))), "description": desc})
    return pd.DataFrame(rows).sort_values("family_key").reset_index(drop=True)


def is_main_riemann_record(row: pd.Series) -> bool:
    """True for the main deployment report: eligible non-orbifold Riemann-surface records."""
    fam = clean_str_value(row.get("family_normalized", ""), "unknown")
    regime = clean_str_value(row.get("geometry_regime", row.get("geometry_regime_benchmark", "")), "unknown")
    if fam == "orbifold_cyclic_elliptic_reference" or family_key_from_name(fam).startswith("O-"):
        return False
    if regime == "excluded_orbifold_or_ineligible":
        return False
    if "orbifold" in regime.lower() or "ineligible" in regime.lower():
        return False
    if "eligible" in row.index and not truthy(row.get("eligible", True)):
        return False
    if "mainline_dataset_eligible" in row.index and not truthy(row.get("mainline_dataset_eligible", True)):
        return False
    return True

def normalize_family(row: pd.Series) -> str:
    sf = str(row.get("surface_family", "") or "").lower()
    ss = str(row.get("surface_subfamily", "") or "").lower()
    sid = str(row.get("surface_id", "") or "").lower()
    combo = " ".join([sf, ss, sid])
    if "compact" in combo and "regular" in combo:
        return "compact_regular_genus"
    if "hurwitz" in combo and ("psl" in combo or "triangle" in combo or "kernel" in combo):
        return "hurwitz_psl_kernel"
    if "hurwitz" in combo or "klein" in combo:
        return "compact_hurwitz_klein"
    if "gamma_0" in combo or "gamma0" in combo:
        return "modular_gamma0"
    if "gamma_1" in combo or "gamma1" in combo:
        return "modular_gamma1"
    if "principal" in combo or re.search(r"\bgamma\b", combo):
        return "modular_principal_gamma"
    if "hecke" in combo and ("dihedral" in combo or "nonabelian" in combo or " d" in combo):
        return "hecke_dihedral_nonabelian"
    if "hecke" in combo and ("abelian" in combo or "ab" in combo):
        return "hecke_abelian"
    if "schottky" in combo:
        return "schottky_free_fuchsian"
    if "commutator" in combo or "punctured_torus" in combo or "once" in combo:
        return "elementary_once_punctured_torus"
    if "gamma2" in combo or "thrice" in combo or "trinion" in combo:
        return "elementary_gamma2_trinion"
    if "cyclic" in combo and "parabolic" in combo:
        return "elementary_cyclic_parabolic"
    if "cyclic" in combo and "hyperbolic" in combo:
        return "elementary_cyclic_hyperbolic"
    if "cyclic" in combo and "elliptic" in combo:
        return "orbifold_cyclic_elliptic_reference"
    return sf or "unknown"


def geometry_regime(row: pd.Series) -> str:
    if truthy(row.get("orbifold_excluded")) or not truthy(row.get("mainline_dataset_eligible", row.get("analysis_eligible", True))):
        return "excluded_orbifold_or_ineligible"
    if truthy(row.get("compact")):
        return "compact_closed"
    if truthy(row.get("finite_area")):
        return "noncompact_finite_area_cusped"
    if truthy(row.get("infinite_area")) or "infinite" in str(row.get("surface_area_type", "")).lower():
        return "noncompact_infinite_area"
    return "unknown_regime"


def find_latest_master_run(run_root: Path) -> Optional[Path]:
    # Accept either a ZooBuilder run root or a master run root or a tables dir.
    if (run_root / "tables" / "second_level_ml_features.csv").exists():
        return run_root
    if (run_root / "second_level_ml_features.csv").exists():
        return run_root.parent
    mroot = run_root / "master_dataset_runs"
    if mroot.exists():
        candidates = [p for p in mroot.iterdir() if p.is_dir() and (p / "tables").exists()]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    return None


def table_path(master_run: Optional[Path], name: str) -> Optional[Path]:
    if master_run is None:
        return None
    for p in [master_run / "tables" / name, master_run / name]:
        if p.exists():
            return p
    return None


def read_csv_if_exists(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed reading {path}: {e}", file=sys.stderr)
        return pd.DataFrame()


def discover_prediction_artifacts(run_root: Path, master_run: Optional[Path]) -> pd.DataFrame:
    """Return rows with prediction paths and surface identifiers.

    Priority:
      1. master artifact_index.csv
      2. recursive search under run_root/family_runs
    """
    rows = []
    ai = read_csv_if_exists(table_path(master_run, "artifact_index.csv"))
    if not ai.empty:
        name_col = first_existing_col(ai, ["artifact_name"])
        path_col = first_existing_col(ai, ["artifact_path", "path"])
        if name_col and path_col:
            sub = ai[ai[name_col].astype(str).str.contains("predictions_test\\.csv", regex=True, na=False)].copy()
            for _, r in sub.iterrows():
                rows.append({
                    "record_uid": r.get("record_uid", ""),
                    "surface_id": r.get("surface_id", ""),
                    "predictions_path": str(r.get(path_col, "")),
                    "artifact_source": "artifact_index",
                })
    # Recursive fallback.  This also helps if artifact_index has absolute paths
    # from another machine but the run root is present locally.
    search_roots = [run_root]
    for p in search_roots:
        if p.exists():
            for f in p.rglob("predictions_test.csv"):
                # Infer surface id from parent directory, usually ginn_runs/<surface_id>/predictions_test.csv
                sid = f.parent.name
                rows.append({
                    "record_uid": "",
                    "surface_id": sid,
                    "predictions_path": str(f),
                    "artifact_source": "recursive",
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Prefer recursive local paths when both exist, because artifact_index can
    # contain absolute paths from the original training machine.
    df["path_exists"] = df["predictions_path"].map(lambda s: Path(str(s)).exists())
    df["path_parent"] = df["predictions_path"].map(lambda s: str(Path(str(s)).parent))
    df["artifact_priority"] = df["artifact_source"].map({"artifact_index": 0, "recursive": 1}).fillna(2)
    df = df.sort_values(["surface_id", "path_exists", "artifact_priority"], ascending=[True, False, True])
    df = df.drop_duplicates(subset=["surface_id", "path_parent"], keep="first")
    # If there are two paths for the same surface and one exists, keep both only if parent differs.
    return df.reset_index(drop=True)


def build_surface_metadata(run_root: Path, master_run: Optional[Path]) -> pd.DataFrame:
    # second_level_ml_features has the richest metadata.
    meta = read_csv_if_exists(table_path(master_run, "second_level_ml_features.csv"))
    if meta.empty:
        meta = read_csv_if_exists(table_path(master_run, "master_surface_catalog.csv"))
    if meta.empty:
        return pd.DataFrame()
    # De-fragment after reading very wide master tables; avoids pandas PerformanceWarnings
    # when adding benchmark-specific columns.
    meta = meta.copy()

    # Keep one metadata row per record_uid if possible, and a separate surface_id lookup.
    if "record_uid" not in meta.columns:
        # v3 tables sometimes only have surface-level rows; synthesize a stable UID.
        sid = meta.get("surface_id", pd.Series(range(len(meta)))).astype(str)
        meta["record_uid"] = sid + "__" + meta.index.astype(str)
    if "surface_id" not in meta.columns:
        meta["surface_id"] = meta["record_uid"].astype(str)

    meta["family_normalized"] = meta.apply(normalize_family, axis=1)
    meta["family_key"] = meta["family_normalized"].map(family_key_from_name)
    meta["geometry_regime_benchmark"] = meta.apply(geometry_regime, axis=1)
    if "exact_duplicate_hash" in meta.columns:
        meta["unique_surface_key_benchmark"] = meta["exact_duplicate_hash"].fillna("").astype(str)
        mask = meta["unique_surface_key_benchmark"].str.len() < 4
        meta.loc[mask, "unique_surface_key_benchmark"] = meta.loc[mask, "surface_id"].astype(str)
    elif "generator_hash" in meta.columns:
        meta["unique_surface_key_benchmark"] = meta["generator_hash"].fillna("").astype(str)
        mask = meta["unique_surface_key_benchmark"].str.len() < 4
        meta.loc[mask, "unique_surface_key_benchmark"] = meta.loc[mask, "surface_id"].astype(str)
    else:
        meta["unique_surface_key_benchmark"] = meta["surface_id"].astype(str)
    return meta


def resolve_prediction_path(path_str: str, run_root: Path) -> Optional[Path]:
    p = Path(str(path_str))
    if p.exists():
        return p
    # Try suffix under run root if absolute path came from original machine.
    parts = list(p.parts)
    # Look for a suffix beginning at family_runs or ginn_runs.
    for marker in ["family_runs", "ginn_runs"]:
        if marker in parts:
            idx = parts.index(marker)
            candidate = run_root.joinpath(*parts[idx:])
            if candidate.exists():
                return candidate
    # Last resort: search for parent dir surface_id/predictions_test.csv
    surface_dir = p.parent.name
    matches = list(run_root.rglob(f"{surface_dir}/predictions_test.csv"))
    if matches:
        return matches[0]
    return None


def read_predictions(path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        return pd.read_csv(path), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def rank_column(pr: pd.DataFrame) -> Optional[str]:
    return first_existing_col(pr, [
        "true_word_model_rank",
        "true_winner_model_rank",
        "true_branch_model_rank",
        "true_word_rank",
        "true_rank",
        "rank_true_word",
    ])


def true_distance_column(pr: pd.DataFrame) -> Optional[str]:
    return first_existing_col(pr, [
        "true_exact_distance", "true_distance", "exact_winner_distance", "winner_distance",
        "quotient_distance", "exact_quotient_distance", "hard_selected_distance", "d_true",
    ])


def pruned_distance_column(pr: pd.DataFrame, k: int) -> Optional[str]:
    return first_existing_col(pr, [
        f"top{k}_pruned_exact_distance", f"top{k}_pruned_distance", f"pruned_top{k}_distance",
        f"top{k}_selected_distance", f"pred_top{k}_distance", f"distance_top{k}_pruned",
        f"top{k}_distance_after_pruned_min",
    ])


def topk_contains_column(pr: pd.DataFrame, k: int) -> Optional[str]:
    return first_existing_col(pr, [
        f"top{k}_contains_true", f"top{k}_contains_true_word", f"top{k}_recall_true_winner",
        f"true_in_top{k}", f"contains_true_top{k}",
    ])


def selected_word_correct_column(pr: pd.DataFrame) -> Optional[str]:
    return first_existing_col(pr, [
        "winner_correct", "pred_winner_correct", "selected_word_correct", "top1_contains_true",
        "pred_exact_equivalent_tol_1e_5", "top1_recall_true_winner",
    ])


def benchmark_predictions(pr: pd.DataFrame, k_list: Sequence[int], word_ball_size: float) -> Dict[str, float]:
    out: Dict[str, float] = {"n_pairs": len(pr)}
    rcol = rank_column(pr)
    ranks = None
    if rcol:
        ranks = pd.to_numeric(pr[rcol], errors="coerce")
        ranks = ranks.replace([np.inf, -np.inf], np.nan)
        ranks_valid = ranks.dropna()
        out["rank_column_present"] = 1.0
        out["true_rank_mean"] = float(ranks_valid.mean()) if len(ranks_valid) else np.nan
        out["true_rank_median"] = float(ranks_valid.median()) if len(ranks_valid) else np.nan
        out["true_rank_p90"] = float(ranks_valid.quantile(0.90)) if len(ranks_valid) else np.nan
        out["true_rank_max"] = float(ranks_valid.max()) if len(ranks_valid) else np.nan
    else:
        out["rank_column_present"] = 0.0

    scol = selected_word_correct_column(pr)
    if scol:
        vals = pd.to_numeric(pr[scol], errors="coerce")
        out["top1_recall"] = float(vals.mean())
    elif ranks is not None:
        out["top1_recall"] = float((ranks <= 1).mean())
    else:
        out["top1_recall"] = np.nan

    for k in k_list:
        ccol = topk_contains_column(pr, k)
        if ccol:
            vals = pd.to_numeric(pr[ccol], errors="coerce")
            recall = float(vals.mean())
        elif ranks is not None:
            recall = float((ranks <= k).mean())
        else:
            recall = np.nan
        out[f"recall_at_{k}"] = recall
        out[f"miss_rate_at_{k}"] = 1.0 - recall if math.isfinite(recall) else np.nan
        out[f"miss_count_at_{k}"] = int((1.0 - recall) * len(pr)) if math.isfinite(recall) else np.nan
        if word_ball_size and math.isfinite(word_ball_size) and word_ball_size > 0:
            out[f"nominal_speedup_at_{k}"] = float(word_ball_size / max(k, 1))
            out[f"candidate_reduction_fraction_at_{k}"] = float(1.0 - min(k, word_ball_size) / word_ball_size)
        else:
            out[f"nominal_speedup_at_{k}"] = np.nan
            out[f"candidate_reduction_fraction_at_{k}"] = np.nan

    tdcol = true_distance_column(pr)
    true_d = pd.to_numeric(pr[tdcol], errors="coerce") if tdcol else None
    out["true_distance_column_present"] = 1.0 if tdcol else 0.0
    for k in k_list:
        pdcol = pruned_distance_column(pr, k)
        out[f"top{k}_distance_column_present"] = 1.0 if pdcol else 0.0
        if true_d is not None and pdcol:
            pdist = pd.to_numeric(pr[pdcol], errors="coerce")
            err = (pdist - true_d).replace([np.inf, -np.inf], np.nan).dropna()
            if len(err):
                out[f"top{k}_pruned_mae"] = float(np.mean(np.abs(err)))
                out[f"top{k}_pruned_rmse"] = float(np.sqrt(np.mean(err * err)))
                out[f"top{k}_pruned_max_abs_error"] = float(np.max(np.abs(err)))
            else:
                out[f"top{k}_pruned_mae"] = np.nan
                out[f"top{k}_pruned_rmse"] = np.nan
                out[f"top{k}_pruned_max_abs_error"] = np.nan
        else:
            out[f"top{k}_pruned_mae"] = np.nan
            out[f"top{k}_pruned_rmse"] = np.nan
            out[f"top{k}_pruned_max_abs_error"] = np.nan

    # AUC-like average recall over k=1..20 if ranks are present.
    if ranks is not None:
        max_k = max([k for k in k_list if k <= 20] + [20])
        auc_vals = [(ranks <= k).mean() for k in range(1, max_k + 1)]
        out[f"recall_auc_1_{max_k}"] = float(np.mean(auc_vals))
        # smallest k for thresholds
        for thresh in [0.90, 0.95, 0.99]:
            kval = np.nan
            for k in range(1, max_k + 1):
                if float((ranks <= k).mean()) >= thresh:
                    kval = k
                    break
            out[f"k_for_recall_{int(thresh*100)}"] = kval
    return out


def infer_word_ball_size(meta_row: Optional[pd.Series]) -> float:
    if meta_row is None:
        return np.nan
    for c in [
        "word_ball_size", "run_manifest_word_ball_size", "word_ball_size_metrics",
        "run_manifest_metrics_word_ball_size", "smoke_word_ball_size",
    ]:
        if c in meta_row.index:
            v = safe_float(meta_row.get(c))
            if math.isfinite(v) and v > 0:
                return v
    return np.nan


def merge_artifacts_with_metadata(art: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if art.empty:
        return art
    df = art.copy()
    if meta.empty:
        if "family_normalized" not in df.columns:
            df["family_normalized"] = df.apply(normalize_family, axis=1)
        if "geometry_regime_benchmark" not in df.columns:
            df["geometry_regime_benchmark"] = df.apply(geometry_regime, axis=1)
        if "unique_surface_key_benchmark" not in df.columns:
            df["unique_surface_key_benchmark"] = df.get("surface_id", pd.Series(range(len(df)))).astype(str)
        return df

    if "record_uid" not in df.columns:
        df["record_uid"] = ""
    if "surface_id" not in df.columns:
        df["surface_id"] = ""

    # Merge twice: by record_uid when it matches, and by surface_id as a fallback
    # for recursively discovered artifacts.  Then coalesce metadata columns.
    m_uid = df.merge(meta, on="record_uid", how="left", suffixes=("", "_uid")) if "record_uid" in meta.columns else df.copy()
    m_sid = df.merge(meta, on="surface_id", how="left", suffixes=("", "_sid")) if "surface_id" in meta.columns else df.copy()
    merged = m_uid.copy()

    wanted = [
        "surface_id", "surface_family", "surface_subfamily", "mainline_dataset_eligible",
        "compact", "finite_area", "infinite_area", "word_ball_size", "generator_count",
        "exact_duplicate_hash", "generator_hash", "unique_surface_key_benchmark",
        "family_normalized", "family_key", "geometry_regime_benchmark",
    ]
    for col in wanted:
        if col not in merged.columns:
            merged[col] = np.nan
        # Coalesce variants from UID merge first.
        for alt in [col + "_uid", col + "_meta"]:
            if alt in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[alt])
        # Coalesce surface-id merge values.
        sid_candidates = [col]
        if col + "_sid" in m_sid.columns:
            sid_candidates.append(col + "_sid")
        for alt in sid_candidates:
            if alt in m_sid.columns:
                merged[col] = merged[col].where(merged[col].notna(), m_sid[alt])

    mask = merged["family_normalized"].isna() | (merged["family_normalized"].astype(str).str.lower() == "nan") | (merged["family_normalized"].astype(str).str.len() == 0)
    merged.loc[mask, "family_normalized"] = merged.loc[mask].apply(normalize_family, axis=1)
    merged["family_normalized"] = merged["family_normalized"].map(lambda x: clean_str_value(x, "unknown"))
    merged["family_key"] = merged["family_normalized"].map(family_key_from_name)
    mask = merged["geometry_regime_benchmark"].isna() | (merged["geometry_regime_benchmark"].astype(str).str.lower() == "nan") | (merged["geometry_regime_benchmark"].astype(str).str.len() == 0)
    merged.loc[mask, "geometry_regime_benchmark"] = merged.loc[mask].apply(geometry_regime, axis=1)
    mask = merged["unique_surface_key_benchmark"].isna() | (merged["unique_surface_key_benchmark"].astype(str).str.lower() == "nan") | (merged["unique_surface_key_benchmark"].astype(str).str.len() < 3)
    merged.loc[mask, "unique_surface_key_benchmark"] = merged.loc[mask, "surface_id"].astype(str)
    return merged


def build_benchmark(run_root: Path, k_list: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    master_run = find_latest_master_run(run_root)
    meta = build_surface_metadata(run_root, master_run)
    art = discover_prediction_artifacts(run_root, master_run)
    art_meta = merge_artifacts_with_metadata(art, meta)

    rows = []
    failures = []
    if art_meta.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"master_run": str(master_run) if master_run else "", "message": "no prediction artifacts found"}

    for i, r in art_meta.iterrows():
        path = resolve_prediction_path(str(r.get("predictions_path", "")), run_root)
        surface_id = str(r.get("surface_id", "") or r.get("surface_id_meta", "") or Path(str(r.get("predictions_path", ""))).parent.name)
        fam = clean_str_value(r.get("family_normalized", ""), "") or normalize_family(r)
        fam_key = family_key_from_name(fam)
        regime = clean_str_value(r.get("geometry_regime_benchmark", ""), "") or geometry_regime(r)
        uid = str(r.get("record_uid", "") or f"{surface_id}__{i}")
        unique_key = str(r.get("unique_surface_key_benchmark", "") or surface_id)
        if path is None:
            failures.append({
                "record_uid": uid, "surface_id": surface_id, "family_normalized": fam,
                "predictions_path": str(r.get("predictions_path", "")), "error": "predictions_test.csv not found",
            })
            continue
        pr, err = read_predictions(path)
        if err:
            failures.append({
                "record_uid": uid, "surface_id": surface_id, "family_normalized": fam,
                "predictions_path": str(path), "error": err,
            })
            continue
        wb = infer_word_ball_size(r)
        metrics = benchmark_predictions(pr, k_list, wb)
        row = {
            "record_uid": uid,
            "surface_id": surface_id,
            "unique_surface_key": unique_key,
            "family_normalized": fam,
            "family_key": fam_key,
            "geometry_regime": regime,
            "predictions_path": str(path),
            "word_ball_size": wb,
            "generator_count": safe_float(r.get("generator_count", np.nan)),
            "surface_family": r.get("surface_family", ""),
            "surface_subfamily": r.get("surface_subfamily", ""),
            "eligible": truthy(r.get("mainline_dataset_eligible", True)),
            "compact": truthy(r.get("compact", False)),
            "finite_area": truthy(r.get("finite_area", False)),
            "infinite_area": truthy(r.get("infinite_area", False)),
            "source_run_root": str(run_root),
        }
        row.update(metrics)
        rows.append(row)

    surface_perf = pd.DataFrame(rows)
    if not surface_perf.empty:
        surface_perf["family_key"] = surface_perf.get("family_key", surface_perf["family_normalized"].map(family_key_from_name))
        surface_perf["main_riemann_record"] = surface_perf.apply(is_main_riemann_record, axis=1)
    failure_df = pd.DataFrame(failures)
    if surface_perf.empty:
        family_perf = pd.DataFrame()
    else:
        agg_cols = {"surface_id": "count", "unique_surface_key": pd.Series.nunique, "n_pairs": "sum"}
        metric_cols = [c for c in surface_perf.columns if c.startswith("recall_at_") or c.startswith("nominal_speedup_at_") or c.endswith("_rmse") or c.endswith("_ms_per_pair") or c.startswith("estimated_wall_speedup_at_") or c in ["true_rank_mean", "true_rank_median", "recall_auc_1_20", "full_exact_seconds", "candidate_feature_seconds", "ginn_score_seconds"]]
        for c in metric_cols:
            agg_cols[c] = "median"
        family_perf = surface_perf.groupby(["family_key", "geometry_regime"], dropna=False).agg(agg_cols).reset_index()
        family_perf = family_perf.rename(columns={"surface_id": "records", "unique_surface_key": "unique_surfaces", "n_pairs": "test_pairs_total"})

    meta_info = {
        "master_run": str(master_run) if master_run else "",
        "prediction_artifacts_discovered": int(len(art_meta)),
        "prediction_artifacts_benchmarked": int(len(surface_perf)),
        "prediction_artifact_failures": int(len(failure_df)),
    }
    return surface_perf, family_perf, failure_df, meta_info



def import_ginn_module(module_name: str):
    """Import the saved-training GINN module lazily for fresh mode."""
    import importlib
    return importlib.import_module(module_name)


def find_fresh_artifact_paths(predictions_path: Path) -> Dict[str, Path]:
    d = predictions_path.parent
    return {
        "predictions": predictions_path,
        "model": d / "downstairs_ginn_v2_4.pt",
        "surface": d / "surface.json",
        "run_manifest": d / "run_manifest.json",
        "metrics": d / "metrics.json",
    }


def _choose_torch_device(torch_mod, requested: str):
    if requested == "auto":
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        return torch_mod.device("cpu")
    return torch_mod.device(requested)


def _score_model_numpy(ginn, torch_mod, model, Xn: np.ndarray, Cn: np.ndarray, device, batch_size: int, candidate_chunk_size: int) -> np.ndarray:
    model.eval()
    n, W, _ = Cn.shape
    scores_all = np.empty((n, W), dtype=np.float32)
    with torch_mod.no_grad():
        for i0 in range(0, n, batch_size):
            i1 = min(n, i0 + batch_size)
            xb = torch_mod.tensor(Xn[i0:i1], dtype=torch_mod.float32, device=device)
            parts = []
            if candidate_chunk_size <= 0 or candidate_chunk_size >= W:
                cb = torch_mod.tensor(Cn[i0:i1], dtype=torch_mod.float32, device=device)
                sj, _, _ = model(xb, cb)
                scores = sj.detach().cpu().numpy()
            else:
                for j0 in range(0, W, candidate_chunk_size):
                    j1 = min(W, j0 + candidate_chunk_size)
                    cb = torch_mod.tensor(Cn[i0:i1, j0:j1], dtype=torch_mod.float32, device=device)
                    sj, _, _ = model(xb, cb)
                    parts.append(sj.detach().cpu())
                scores = torch_mod.cat(parts, dim=1).numpy()
            scores_all[i0:i1] = scores.astype(np.float32, copy=False)
    return scores_all


def _fresh_metrics_from_scores(scores: np.ndarray, D: np.ndarray, y_idx: np.ndarray, k_list: Sequence[int], word_ball_size: int) -> Dict[str, float]:
    n, W = scores.shape
    # Rank by learned score.  Rank 1 means no candidate had a higher score than the true winner.
    true_scores = scores[np.arange(n), y_idx]
    ranks = 1 + np.sum(scores > true_scores[:, None], axis=1)
    out: Dict[str, float] = {
        "n_pairs": int(n),
        "rank_column_present": 1.0,
        "true_rank_mean": float(np.mean(ranks)),
        "true_rank_median": float(np.median(ranks)),
        "true_rank_p90": float(np.quantile(ranks, 0.90)),
        "true_rank_max": float(np.max(ranks)),
        "top1_recall": float(np.mean(ranks <= 1)),
        "true_distance_column_present": 1.0,
        "fresh_inference": 1.0,
    }
    true_d = D[np.arange(n), y_idx]
    for k in k_list:
        kk = min(int(k), W)
        if kk <= 0:
            continue
        # Candidate pool by learned score, then exact-check distances only within that pool.
        # argpartition is enough for unordered membership and pruned min.
        top = np.argpartition(-scores, kth=kk-1, axis=1)[:, :kk]
        contains = np.array([y_idx[i] in set(top[i].tolist()) for i in range(n)], dtype=bool)
        vals = np.take_along_axis(D, top, axis=1)
        pruned_d = np.min(vals, axis=1)
        err = pruned_d - true_d
        recall = float(np.mean(contains))
        out[f"recall_at_{k}"] = recall
        out[f"miss_rate_at_{k}"] = 1.0 - recall
        out[f"miss_count_at_{k}"] = int(np.sum(~contains))
        out[f"nominal_speedup_at_{k}"] = float(word_ball_size / max(kk, 1))
        out[f"candidate_reduction_fraction_at_{k}"] = float(1.0 - kk / max(word_ball_size, 1))
        out[f"top{k}_distance_column_present"] = 1.0
        out[f"top{k}_pruned_mae"] = float(np.mean(np.abs(err)))
        out[f"top{k}_pruned_rmse"] = float(np.sqrt(np.mean(err * err)))
        out[f"top{k}_pruned_max_abs_error"] = float(np.max(np.abs(err)))
    max_k = max([k for k in k_list if k <= 20] + [20])
    auc_vals = [float(np.mean(ranks <= k)) for k in range(1, max_k + 1)]
    out[f"recall_auc_1_{max_k}"] = float(np.mean(auc_vals))
    for thresh in [0.90, 0.95, 0.99]:
        kval = np.nan
        for k in range(1, max_k + 1):
            if float(np.mean(ranks <= k)) >= thresh:
                kval = k
                break
        out[f"k_for_recall_{int(thresh*100)}"] = kval
    return out



def _word_depths_from_words(words: Sequence[object]) -> np.ndarray:
    depths = []
    for w in words:
        sw = str(w).strip()
        if sw.lower() in {"", "identity", "id", "e"}:
            depths.append(0)
        else:
            # Existing checkpoints use whitespace-separated word tokens.
            depths.append(max(1, len(sw.split())))
    return np.asarray(depths, dtype=np.int64)


def _identity_index(words: Sequence[object], depths: Optional[np.ndarray] = None) -> int:
    for i, w in enumerate(words):
        if str(w).strip().lower() in {"", "identity", "id", "e"}:
            return i
    if depths is not None and len(depths):
        return int(np.argmin(depths))
    return 0


def _true_index_from_predictions_csv(pred_path: Path, gen_words: Sequence[str], W: int) -> Optional[np.ndarray]:
    """Try to recover historical true-winner indices for the frequency-prior baseline.

    This is only used to build a global prior from the saved training/test artifact.
    If the artifact does not expose a true-index/true-word column, return None.
    """
    try:
        pr = pd.read_csv(pred_path)
    except Exception:
        return None
    idx_col = first_existing_col(pr, [
        "shortest_lift_index", "true_word_index", "true_winner_index", "true_lift_index",
        "label_index", "y_idx", "target_index", "true_candidate_index",
    ])
    if idx_col:
        vals = pd.to_numeric(pr[idx_col], errors="coerce").dropna().astype(int).to_numpy()
        vals = vals[(vals >= 0) & (vals < W)]
        if len(vals):
            return vals
    word_col = first_existing_col(pr, [
        "true_word", "shortest_lift_word", "true_winner_word", "winner_word", "label_word",
    ])
    if word_col:
        word_to_idx = {str(w): i for i, w in enumerate(gen_words)}
        vals = []
        for w in pr[word_col].dropna().astype(str):
            if w in word_to_idx:
                vals.append(word_to_idx[w])
        if vals:
            return np.asarray(vals, dtype=np.int64)
    return None


def _frequency_prior_order(pred_path: Path, gen_words: Sequence[str], W: int) -> Optional[np.ndarray]:
    vals = _true_index_from_predictions_csv(pred_path, gen_words, W)
    if vals is None or len(vals) == 0:
        return None
    counts = np.bincount(vals, minlength=W).astype(float)
    # Highest frequency first; stable tie-break by index.
    return np.lexsort((np.arange(W), -counts)).astype(np.int64)


def _static_order_metrics(order: np.ndarray, D: np.ndarray, y_idx: np.ndarray, k_list: Sequence[int], W: int, prefix: str, exact_check: bool = False) -> Dict[str, float]:
    """Recall/pruned-distance metrics for a fixed candidate ranking independent of (p,q)."""
    out: Dict[str, float] = {}
    n = int(len(y_idx))
    true_d = D[np.arange(n), y_idx]
    order = np.asarray(order, dtype=np.int64)
    order = order[(order >= 0) & (order < W)]
    seen = set()
    order = np.asarray([int(i) for i in order if not (int(i) in seen or seen.add(int(i)))], dtype=np.int64)
    if len(order) == 0:
        for k in k_list:
            out[f"{prefix}_recall_at_{k}"] = np.nan
        return out
    for k in k_list:
        kk = min(int(k), len(order))
        pool = order[:kk]
        contains = np.isin(y_idx, pool)
        out[f"{prefix}_recall_at_{k}"] = float(np.mean(contains))
        out[f"{prefix}_miss_rate_at_{k}"] = 1.0 - out[f"{prefix}_recall_at_{k}"]
        vals = D[:, pool]
        pruned_d = np.min(vals, axis=1)
        err = pruned_d - true_d
        out[f"{prefix}_top{k}_pruned_rmse"] = float(np.sqrt(np.mean(err * err)))
        out[f"{prefix}_candidate_pool_size_at_{k}"] = int(kk)
        out[f"{prefix}_nominal_speedup_at_{k}"] = float(W / max(kk, 1))
    return out


def _random_baseline_metrics(rng: np.random.Generator, D: np.ndarray, y_idx: np.ndarray, k_list: Sequence[int], W: int, prefix: str = "base_rand") -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = int(len(y_idx))
    true_d = D[np.arange(n), y_idx]
    for k in k_list:
        kk = min(int(k), W)
        if kk <= 0:
            continue
        # Efficient enough for current benchmark sizes; sample without replacement per pair.
        pools = np.empty((n, kk), dtype=np.int64)
        for i in range(n):
            pools[i] = rng.choice(W, size=kk, replace=False)
        contains = np.array([y_idx[i] in pools[i] for i in range(n)], dtype=bool)
        vals = np.take_along_axis(D, pools, axis=1)
        err = np.min(vals, axis=1) - true_d
        out[f"{prefix}_recall_at_{k}"] = float(np.mean(contains))
        out[f"{prefix}_miss_rate_at_{k}"] = 1.0 - out[f"{prefix}_recall_at_{k}"]
        out[f"{prefix}_top{k}_pruned_rmse"] = float(np.sqrt(np.mean(err * err)))
        out[f"{prefix}_candidate_pool_size_at_{k}"] = int(kk)
        out[f"{prefix}_nominal_speedup_at_{k}"] = float(W / max(kk, 1))
        out[f"{prefix}_expected_recall_at_{k}"] = float(kk / max(W, 1))
    return out


def _exact_shell_metrics(D: np.ndarray, y_idx: np.ndarray, shell_indices: np.ndarray, k_list: Sequence[int], W: int, prefix: str = "base_shell1") -> Dict[str, float]:
    """Recall for exact-checking a limited short-word shell, then keeping the k closest within that shell."""
    out: Dict[str, float] = {}
    n = int(len(y_idx))
    shell = np.asarray(shell_indices, dtype=np.int64)
    shell = shell[(shell >= 0) & (shell < W)]
    if len(shell) == 0:
        for k in k_list:
            out[f"{prefix}_recall_at_{k}"] = np.nan
        return out
    true_d = D[np.arange(n), y_idx]
    shell_D = D[:, shell]
    for k in k_list:
        kk = min(int(k), len(shell))
        if kk <= 0:
            continue
        if kk == len(shell):
            local_top = np.tile(np.arange(len(shell)), (n, 1))
        else:
            local_top = np.argpartition(shell_D, kth=kk-1, axis=1)[:, :kk]
        pools = shell[local_top]
        contains = np.array([y_idx[i] in set(pools[i].tolist()) for i in range(n)], dtype=bool)
        vals = np.take_along_axis(D, pools, axis=1)
        err = np.min(vals, axis=1) - true_d
        out[f"{prefix}_recall_at_{k}"] = float(np.mean(contains))
        out[f"{prefix}_miss_rate_at_{k}"] = 1.0 - out[f"{prefix}_recall_at_{k}"]
        out[f"{prefix}_top{k}_pruned_rmse"] = float(np.sqrt(np.mean(err * err)))
        out[f"{prefix}_candidate_pool_size_at_{k}"] = int(kk)
        out[f"{prefix}_precheck_shell_size"] = int(len(shell))
        out[f"{prefix}_nominal_speedup_at_{k}"] = float(W / max(kk, 1))
    return out



def _inverse_token(tok: str, gen_set: set) -> str:
    """Best-effort inverse token for word strings written by the GINN exporter."""
    tok = str(tok)
    candidates = []
    if tok.endswith("^-1"):
        candidates.append(tok[:-3])
    if tok.endswith("^{-1}"):
        candidates.append(tok[:-5])
    if tok.endswith("-1"):
        candidates.append(tok[:-2])
    candidates.append(tok + "^-1")
    candidates.append(tok + "^{-1}")
    candidates.append(tok + "-1")
    for c in candidates:
        if c in gen_set:
            return c
    # Fallback still gives a useful cancellation rule for common token formats.
    if tok.endswith("^-1"):
        return tok[:-3]
    if tok.endswith("^{-1}"):
        return tok[:-5]
    return tok + "^-1"


def _parse_word_tokens(w: object) -> Tuple[str, ...]:
    sw = str(w).strip()
    if sw.lower() in {"", "identity", "id", "e"}:
        return tuple()
    return tuple(sw.split())


def _format_word_tokens(tokens: Sequence[str]) -> str:
    return " ".join(tokens) if tokens else "identity"


def _build_word_neighbors(gen_words: Sequence[str]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    """Build a reduced-word neighbor map from the exported finite word ball.

    This is deliberately conservative.  It uses the word strings already stored
    in the checkpoint.  A neighbor is accepted only if appending a depth-1 token,
    with adjacent inverse cancellation, lands on a word that actually exists in
    the exported word ball.  If the word exporter used an unfamiliar compact
    notation, the map may be sparse; the baseline then gracefully degrades.
    """
    W = len(gen_words)
    tokens_by_idx = [_parse_word_tokens(w) for w in gen_words]
    depths = np.asarray([len(t) for t in tokens_by_idx], dtype=np.int64)
    gen_tokens = [t[0] for t in tokens_by_idx if len(t) == 1]
    # Stable unique ordering.
    seen = set()
    gen_tokens = [g for g in gen_tokens if not (g in seen or seen.add(g))]
    gen_set = set(gen_tokens)
    # Include both original and stripped identity-style keys for lookup.
    word_to_idx = {str(w).strip(): i for i, w in enumerate(gen_words)}
    word_to_idx["identity"] = _identity_index(gen_words, depths)
    word_to_idx[""] = _identity_index(gen_words, depths)
    tok_to_idx = {tokens_by_idx[i]: i for i in range(W)}

    neighbors: List[np.ndarray] = []
    for toks in tokens_by_idx:
        neigh = []
        for g in gen_tokens:
            new = list(toks)
            if new and _inverse_token(g, gen_set) == new[-1]:
                new = new[:-1]
            else:
                new.append(g)
            tup = tuple(new)
            idx = tok_to_idx.get(tup)
            if idx is None:
                idx = word_to_idx.get(_format_word_tokens(tup))
            if idx is not None:
                neigh.append(int(idx))
        if neigh:
            neighbors.append(np.asarray(sorted(set(neigh)), dtype=np.int64))
        else:
            neighbors.append(np.asarray([], dtype=np.int64))
    return neighbors, depths, np.asarray(gen_tokens, dtype=object), gen_tokens


def _pool_metrics_from_per_pair_pools(D: np.ndarray, y_idx: np.ndarray, pools: List[np.ndarray], k_list: Sequence[int], W: int, prefix: str, elapsed_seconds: float) -> Dict[str, float]:
    """Compute recall/RMSE for adaptive baselines that return per-pair ranked pools."""
    out: Dict[str, float] = {}
    n = int(len(y_idx))
    true_d = D[np.arange(n), y_idx]
    eval_sizes = np.asarray([len(p) for p in pools], dtype=float) if pools else np.asarray([], dtype=float)
    out[f"{prefix}_evaluated_candidates_mean"] = float(np.mean(eval_sizes)) if len(eval_sizes) else np.nan
    out[f"{prefix}_evaluated_candidates_median"] = float(np.median(eval_sizes)) if len(eval_sizes) else np.nan
    out[f"{prefix}_method_seconds"] = float(elapsed_seconds)
    out[f"{prefix}_method_ms_per_pair"] = float(1000.0 * elapsed_seconds / max(n, 1))
    for k in k_list:
        kk = int(k)
        contains = np.zeros(n, dtype=bool)
        pruned = np.full(n, np.inf, dtype=float)
        pool_sizes = []
        for i, pool in enumerate(pools):
            if pool is None or len(pool) == 0:
                continue
            pool = np.asarray(pool, dtype=np.int64)
            # Pool is expected to be exact-distance ranked already.  Re-rank defensively.
            vals = D[i, pool]
            order = np.argsort(vals)
            use = pool[order[:min(kk, len(order))]]
            pool_sizes.append(len(use))
            contains[i] = bool(np.any(use == y_idx[i]))
            pruned[i] = float(np.min(D[i, use])) if len(use) else np.inf
        err = pruned - true_d
        err = err[np.isfinite(err)]
        out[f"{prefix}_recall_at_{k}"] = float(np.mean(contains)) if n else np.nan
        out[f"{prefix}_miss_rate_at_{k}"] = 1.0 - out[f"{prefix}_recall_at_{k}"] if n else np.nan
        out[f"{prefix}_top{k}_pruned_rmse"] = float(np.sqrt(np.mean(err * err))) if len(err) else np.nan
        out[f"{prefix}_candidate_pool_size_at_{k}"] = float(np.mean(pool_sizes)) if pool_sizes else np.nan
        out[f"{prefix}_nominal_speedup_at_{k}"] = float(W / max(kk, 1))
    return out


def _greedy_exact_metrics(D: np.ndarray, y_idx: np.ndarray, neighbors: List[np.ndarray], ident: int, k_list: Sequence[int], W: int, prefix: str = "base_greedy") -> Dict[str, float]:
    """Pair-dependent greedy exact search baseline.

    For each fresh pair, start at identity.  Repeatedly exact-check one-step
    neighbors of the current best candidate and move if a neighbor improves the
    exact disk distance.  The returned top-k pool is the best k candidates among
    all exact-checked candidates during the greedy walk.
    """
    import time
    t0 = time.perf_counter()
    n = D.shape[0]
    pools: List[np.ndarray] = []
    max_steps = max(1, max(len(str(i)) for i in [0]))  # harmless placeholder; loop below stops naturally
    # Use a safe cap to avoid pathological cycling if an unusual neighbor map appears.
    step_cap = max(1, min(20, int(math.ceil(math.log(max(W, 2), 2))) + 5))
    for i in range(n):
        current = int(ident)
        evaluated = {current}
        best_d = float(D[i, current])
        for _ in range(step_cap):
            neigh = neighbors[current] if 0 <= current < len(neighbors) else np.asarray([], dtype=np.int64)
            if len(neigh) == 0:
                break
            for j in neigh:
                evaluated.add(int(j))
            vals = D[i, neigh]
            best_local_pos = int(np.argmin(vals))
            candidate = int(neigh[best_local_pos])
            cand_d = float(vals[best_local_pos])
            if cand_d + 1e-12 < best_d:
                current = candidate
                best_d = cand_d
            else:
                break
        ev = np.asarray(sorted(evaluated), dtype=np.int64)
        ranked = ev[np.argsort(D[i, ev])]
        pools.append(ranked)
    return _pool_metrics_from_per_pair_pools(D, y_idx, pools, k_list, W, prefix, time.perf_counter() - t0)


def _beam_exact_metrics(D: np.ndarray, y_idx: np.ndarray, neighbors: List[np.ndarray], ident: int, k_list: Sequence[int], W: int, beam_width: int, prefix: str) -> Dict[str, float]:
    """Pair-dependent exact beam-search baseline.

    At each depth, expand the current beam by one generator step, exact-check the
    children, and keep the best `beam_width` children by exact disk distance.
    The candidate pool is the best k among all candidates exact-checked during
    the beam search.  This is a strong non-ML competitor because it is allowed
    to use exact distances adaptively for each pair.
    """
    import time
    t0 = time.perf_counter()
    n = D.shape[0]
    pools: List[np.ndarray] = []
    # Word balls are usually depth 2 here.  This cap keeps the baseline bounded
    # even for future deeper artifacts.
    step_cap = max(1, min(4, int(math.ceil(math.log(max(W, 2), 2)))))
    for i in range(n):
        frontier = np.asarray([int(ident)], dtype=np.int64)
        evaluated = {int(ident)}
        for _ in range(step_cap):
            children_list = []
            for f in frontier:
                if 0 <= int(f) < len(neighbors) and len(neighbors[int(f)]):
                    children_list.append(neighbors[int(f)])
            if not children_list:
                break
            children = np.unique(np.concatenate(children_list).astype(np.int64))
            if len(children) == 0:
                break
            for j in children:
                evaluated.add(int(j))
            vals = D[i, children]
            bw = min(int(beam_width), len(children))
            if bw <= 0:
                break
            local = np.argpartition(vals, kth=bw-1)[:bw] if bw < len(vals) else np.arange(len(vals))
            # Sort selected beam for deterministic behavior.
            local = local[np.argsort(vals[local])]
            frontier = children[local]
        ev = np.asarray(sorted(evaluated), dtype=np.int64)
        ranked = ev[np.argsort(D[i, ev])]
        pools.append(ranked)
    return _pool_metrics_from_per_pair_pools(D, y_idx, pools, k_list, W, prefix, time.perf_counter() - t0)


def _baseline_metrics_for_fresh(pred_path: Path, gen_words: Sequence[str], D: np.ndarray, y_idx: np.ndarray, k_list: Sequence[int], rng_seed: int) -> Dict[str, float]:
    """Compute learning and non-learning baseline pools on the same fresh pairs.

    v1.4 includes simple static baselines plus stronger pair-dependent exact
    search baselines (greedy and beam search).  These baselines are evaluated
    against the same finite-word exact winner used for the GINN benchmark.
    """
    W = int(D.shape[1])
    depths = _word_depths_from_words(gen_words)
    ident = _identity_index(gen_words, depths)
    short_order = np.lexsort((np.arange(W), depths)).astype(np.int64)
    # Force identity first in the identity+short baseline.
    id_short_order = np.asarray([ident] + [i for i in short_order.tolist() if i != ident], dtype=np.int64)
    shell1 = np.where(depths <= 1)[0]
    out: Dict[str, float] = {}
    out.update(_static_order_metrics(np.asarray([ident], dtype=np.int64), D, y_idx, k_list, W, "base_identity"))
    out.update(_static_order_metrics(short_order, D, y_idx, k_list, W, "base_shortword"))
    out.update(_static_order_metrics(id_short_order, D, y_idx, k_list, W, "base_id_short"))
    freq_order = _frequency_prior_order(pred_path, gen_words, W)
    if freq_order is not None:
        out.update(_static_order_metrics(freq_order, D, y_idx, k_list, W, "base_freq"))
        out["base_freq_available"] = 1.0
    else:
        for k in k_list:
            out[f"base_freq_recall_at_{k}"] = np.nan
        out["base_freq_available"] = 0.0
    out.update(_exact_shell_metrics(D, y_idx, shell1, k_list, W, "base_shell1"))
    out.update(_random_baseline_metrics(np.random.default_rng(rng_seed), D, y_idx, k_list, W, "base_rand"))

    # Stronger non-ML competitors.  These use exact distance evaluations
    # adaptively per pair but avoid evaluating the full word ball when possible.
    try:
        neighbors, parsed_depths, gen_tokens_arr, gen_tokens = _build_word_neighbors(gen_words)
        out["advanced_baselines_available"] = 1.0 if len(gen_tokens) else 0.0
        out["advanced_generator_token_count"] = float(len(gen_tokens))
        if len(gen_tokens):
            out.update(_greedy_exact_metrics(D, y_idx, neighbors, ident, k_list, W, "base_greedy"))
            for bw in [5, 10, 20]:
                out.update(_beam_exact_metrics(D, y_idx, neighbors, ident, k_list, W, bw, f"base_beam{bw}"))
    except Exception as e:
        # Do not let a baseline parser failure invalidate the GINN deployment test.
        out["advanced_baselines_available"] = 0.0
        out["advanced_baseline_error"] = str(e)[:200]
        for method_prefix in ["base_greedy", "base_beam5", "base_beam10", "base_beam20"]:
            for k in k_list:
                out[f"{method_prefix}_recall_at_{k}"] = np.nan
    return out

def baseline_long_table(surface_perf: pd.DataFrame, k_list: Sequence[int]) -> pd.DataFrame:
    """Convert baseline columns into a tidy method comparison table."""
    if surface_perf is None or surface_perf.empty:
        return pd.DataFrame()
    methods = [
        ("GINN", "recall_at_", "top", ""),
        ("Random", "base_rand_recall_at_", "base_rand_top", "base_rand"),
        ("Identity", "base_identity_recall_at_", "base_identity_top", "base_identity"),
        ("ShortWord", "base_shortword_recall_at_", "base_shortword_top", "base_shortword"),
        ("Id+Short", "base_id_short_recall_at_", "base_id_short_top", "base_id_short"),
        ("FreqPrior", "base_freq_recall_at_", "base_freq_top", "base_freq"),
        ("Shell1Exact", "base_shell1_recall_at_", "base_shell1_top", "base_shell1"),
        ("GreedyExact", "base_greedy_recall_at_", "base_greedy_top", "base_greedy"),
        ("Beam5Exact", "base_beam5_recall_at_", "base_beam5_top", "base_beam5"),
        ("Beam10Exact", "base_beam10_recall_at_", "base_beam10_top", "base_beam10"),
        ("Beam20Exact", "base_beam20_recall_at_", "base_beam20_top", "base_beam20"),
    ]
    rows = []
    for _, r in surface_perf.iterrows():
        base = {
            "record_uid": r.get("record_uid", ""),
            "surface_id": r.get("surface_id", ""),
            "unique_surface_key": r.get("unique_surface_key", ""),
            "family_key": r.get("family_key", ""),
            "family_normalized": r.get("family_normalized", ""),
            "geometry_regime": r.get("geometry_regime", ""),
            "word_ball_size": safe_float(r.get("word_ball_size", np.nan)),
            "n_pairs": safe_int(r.get("n_pairs", 0)),
        }
        for method, prefix, rmse_prefix, aux_prefix in methods:
            # Skip unavailable FreqPrior rows if no frequencies were found.
            if method == "FreqPrior" and not truthy(r.get("base_freq_available", False)):
                continue
            # Skip advanced baseline rows when the word parser could not build a graph.
            if method in {"GreedyExact", "Beam5Exact", "Beam10Exact", "Beam20Exact"} and not math.isfinite(safe_float(r.get(f"{aux_prefix}_recall_at_{k_list[0]}", np.nan))):
                continue
            row = dict(base)
            row["method"] = method
            if method == "GINN":
                row["method_ms_per_pair"] = safe_float(r.get("estimated_ginn_top5_ms_per_pair", np.nan))
                row["eval_candidates_mean"] = 5.0
            elif aux_prefix:
                row["method_ms_per_pair"] = safe_float(r.get(f"{aux_prefix}_method_ms_per_pair", np.nan))
                row["eval_candidates_mean"] = safe_float(r.get(f"{aux_prefix}_evaluated_candidates_mean", np.nan))
            for k in k_list:
                c = f"{prefix}{k}"
                row[f"recall_at_{k}"] = safe_float(r.get(c, np.nan))
                rc = f"{rmse_prefix}{k}_pruned_rmse"
                # GINN uses top{k}_pruned_rmse, baselines use base_*_top{k}_pruned_rmse.
                if method == "GINN":
                    rc = f"top{k}_pruned_rmse"
                row[f"top{k}_pruned_rmse"] = safe_float(r.get(rc, np.nan))
            rows.append(row)
    return pd.DataFrame(rows)


def baseline_family_summary(surface_perf: pd.DataFrame, k_list: Sequence[int]) -> pd.DataFrame:
    bl = baseline_long_table(surface_perf, k_list)
    if bl.empty:
        return bl
    agg = {"surface_id": "count", "unique_surface_key": pd.Series.nunique, "n_pairs": "sum"}
    if "method_ms_per_pair" in bl.columns:
        agg["method_ms_per_pair"] = "median"
    if "eval_candidates_mean" in bl.columns:
        agg["eval_candidates_mean"] = "median"
    for k in k_list:
        agg[f"recall_at_{k}"] = "median"
        agg[f"top{k}_pruned_rmse"] = "median"
    out = bl.groupby(["method", "family_key", "geometry_regime"], dropna=False).agg(agg).reset_index()
    out = out.rename(columns={"surface_id": "records", "unique_surface_key": "unique_surfaces", "n_pairs": "test_pairs_total"})
    return out


def baseline_method_summary(surface_perf: pd.DataFrame, k_list: Sequence[int]) -> pd.DataFrame:
    bl = baseline_long_table(surface_perf, k_list)
    if bl.empty:
        return bl
    agg = {"surface_id": "count", "unique_surface_key": pd.Series.nunique, "n_pairs": "sum"}
    if "method_ms_per_pair" in bl.columns:
        agg["method_ms_per_pair"] = "median"
    if "eval_candidates_mean" in bl.columns:
        agg["eval_candidates_mean"] = "median"
    for k in k_list:
        agg[f"recall_at_{k}"] = "median"
        agg[f"top{k}_pruned_rmse"] = "median"
    out = bl.groupby(["method"], dropna=False).agg(agg).reset_index()
    out = out.rename(columns={"surface_id": "records", "unique_surface_key": "unique_surfaces", "n_pairs": "test_pairs_total"})
    return out

def benchmark_fresh_one_artifact(r: pd.Series, run_root: Path, k_list: Sequence[int], fresh_pairs: int, fresh_seed: int, fresh_device: str, fresh_batch_size: int, fresh_candidate_chunk_size: int, ginn_module_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Fresh deployment benchmark for one trained artifact.

    Loads surface.json and downstairs_ginn_v2_4.pt, samples brand-new p,q pairs,
    recomputes exact finite-word labels, then scores the word ball with the saved
    model and exact-checks only top-k learned candidates.
    """
    pred_path = resolve_prediction_path(str(r.get("predictions_path", "")), run_root)
    surface_id = str(r.get("surface_id", "") or Path(str(r.get("predictions_path", ""))).parent.name)
    fam = clean_str_value(r.get("family_normalized", ""), "") or normalize_family(r)
    fam_key = family_key_from_name(fam)
    regime = clean_str_value(r.get("geometry_regime_benchmark", ""), "") or geometry_regime(r)
    uid = str(r.get("record_uid", "") or surface_id)
    unique_key = str(r.get("unique_surface_key_benchmark", "") or surface_id)
    if pred_path is None:
        return None, {"record_uid": uid, "surface_id": surface_id, "family_normalized": fam, "family_key": fam_key, "error": "predictions path not found for fresh mode"}
    paths = find_fresh_artifact_paths(pred_path)
    # v1.2: run_manifest is preferred but no longer mandatory.  Some
    # elementary/Schottky family testers wrote surface.json and the model
    # checkpoint but omitted run_manifest.json; in that case we infer depth
    # and word-ball size from the checkpoint/prediction metadata.
    missing = [name for name, path in paths.items() if name in {"model", "surface"} and not path.exists()]
    if missing:
        return None, {"record_uid": uid, "surface_id": surface_id, "family_normalized": fam, "family_key": fam_key, "predictions_path": str(pred_path), "error": "fresh mode missing required artifacts: " + ",".join(missing)}
    try:
        ginn = import_ginn_module(ginn_module_name)
        torch_mod = ginn.torch
        if torch_mod is None:
            raise RuntimeError("PyTorch is not available through GINN module")
        device = _choose_torch_device(torch_mod, fresh_device)
        surface_json = json.loads(paths["surface"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8")) if paths["run_manifest"].exists() else {}
        ckpt = torch_mod.load(paths["model"], map_location=device)
        ckpt_words = ckpt.get("word_ball_words", [])
        def _word_depth_from_checkpoint(words):
            best = 0
            for w in words or []:
                sw = str(w).strip()
                if sw.lower() in {"", "identity", "id", "e"}:
                    d = 0
                else:
                    # Existing GINN word strings use whitespace-separated tokens.
                    # If a future run writes compact words, this still gives a
                    # safe lower bound and the checkpoint-order check below will
                    # catch mismatches.
                    d = len(sw.split())
                best = max(best, d)
            return best or 2
        depth = int(manifest.get("word_depth", manifest.get("depth", ckpt.get("word_depth", ckpt.get("depth", _word_depth_from_checkpoint(ckpt_words))))))
        max_word_ball = int(manifest.get("word_ball_size", len(ckpt_words) if ckpt_words else r.get("word_ball_size", 0)) or 0)
        import time
        # Allow same word ball size as training. If missing, disable cap.
        t_exact0 = time.perf_counter()
        rows, X, D, word_ball, label_meta, feature_names = ginn.generate_ginn_dataset(surface_json, fresh_pairs, depth, fresh_seed, max_word_ball=max_word_ball)
        t_exact1 = time.perf_counter()
        gen_words = [m.word if m.word else "identity" for m in word_ball]
        if ckpt_words and list(ckpt_words) != gen_words:
            raise RuntimeError(f"word-ball order mismatch: checkpoint W={len(ckpt_words)} generated W={len(gen_words)}")
        t_feat0 = time.perf_counter()
        C, cand_names = ginn.build_candidate_feature_cube(rows, word_ball, depth)
        t_feat1 = time.perf_counter()
        metrics = ckpt.get("metrics", {})
        norm = metrics.get("normalization", {})
        x_mean = np.asarray(norm.get("x_mean"), dtype=np.float32)
        x_std = np.asarray(norm.get("x_std"), dtype=np.float32)
        c_mean = np.asarray(norm.get("candidate_mean"), dtype=np.float32)
        c_std = np.asarray(norm.get("candidate_std"), dtype=np.float32)
        if x_mean.size != X.shape[1] or x_std.size != X.shape[1] or c_mean.size != C.shape[2] or c_std.size != C.shape[2]:
            raise RuntimeError("normalization dimensions do not match fresh feature dimensions")
        Xn = ((X.astype(np.float32) - x_mean) / np.maximum(x_std, 1e-6)).astype(np.float32)
        Cn = ((C.astype(np.float32) - c_mean) / np.maximum(c_std, 1e-6)).astype(np.float32)
        pair_hidden = int(ckpt.get("pair_hidden", manifest.get("pair_hidden", 256)))
        context_dim = int(ckpt.get("context_dim", max(32, int(manifest.get("score_hidden", 128)))))
        model = ginn.CandidateRankGINN(pair_dim=Xn.shape[1], cand_dim=Cn.shape[2], hidden=pair_hidden, context_dim=context_dim).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        chunk = int(fresh_candidate_chunk_size or ckpt.get("candidate_chunk_size", manifest.get("candidate_chunk_size", 0)) or 0)
        t_score0 = time.perf_counter()
        scores = _score_model_numpy(ginn, torch_mod, model, Xn, Cn, device, fresh_batch_size, chunk)
        t_score1 = time.perf_counter()
        y_idx = np.asarray([int(rw["shortest_lift_index"]) for rw in rows], dtype=np.int64)
        perf = _fresh_metrics_from_scores(scores, D, y_idx, k_list, len(word_ball))
        perf.update(_baseline_metrics_for_fresh(pred_path, gen_words, D, y_idx, k_list, rng_seed=fresh_seed + 177))
        exact_seconds = max(0.0, t_exact1 - t_exact0)
        feature_seconds = max(0.0, t_feat1 - t_feat0)
        score_seconds = max(0.0, t_score1 - t_score0)
        n_pairs_eff = max(1, int(fresh_pairs))
        W_eff = max(1, int(len(word_ball)))
        perf.update({
            "full_exact_seconds": exact_seconds,
            "candidate_feature_seconds": feature_seconds,
            "ginn_score_seconds": score_seconds,
            "full_exact_ms_per_pair": 1000.0 * exact_seconds / n_pairs_eff,
            "candidate_feature_ms_per_pair": 1000.0 * feature_seconds / n_pairs_eff,
            "ginn_score_ms_per_pair": 1000.0 * score_seconds / n_pairs_eff,
        })
        # Estimated deployment time assumes exact-checking k candidates costs k/W of the full exact-distance pass.
        for kk_est in k_list:
            kk_eff = min(int(kk_est), W_eff)
            est = feature_seconds + score_seconds + exact_seconds * (kk_eff / W_eff)
            perf[f"estimated_ginn_top{kk_est}_seconds"] = float(est)
            perf[f"estimated_ginn_top{kk_est}_ms_per_pair"] = float(1000.0 * est / n_pairs_eff)
            perf[f"estimated_wall_speedup_at_{kk_est}"] = float(exact_seconds / est) if est > 0 else np.nan
        row = {
            "record_uid": uid,
            "surface_id": surface_id,
            "unique_surface_key": unique_key,
            "family_normalized": fam,
            "family_key": fam_key,
            "geometry_regime": regime,
            "predictions_path": str(pred_path),
            "model_path": str(paths["model"]),
            "word_ball_size": float(len(word_ball)),
            "generator_count": safe_float(r.get("generator_count", np.nan)),
            "surface_family": r.get("surface_family", ""),
            "surface_subfamily": r.get("surface_subfamily", ""),
            "eligible": truthy(r.get("mainline_dataset_eligible", True)),
            "compact": truthy(r.get("compact", False)),
            "finite_area": truthy(r.get("finite_area", False)),
            "infinite_area": truthy(r.get("infinite_area", False)),
            "source_run_root": str(run_root),
            "benchmark_mode": "fresh",
            "fresh_pairs": int(fresh_pairs),
            "fresh_seed": int(fresh_seed),
            "fresh_device": str(device),
            "fresh_candidate_chunk_size": int(chunk),
            "sampler_kind": label_meta.get("sampler_kind", ""),
        }
        row.update(perf)
        return row, None
    except Exception as exc:
        return None, {"record_uid": uid, "surface_id": surface_id, "family_normalized": fam, "family_key": fam_key, "predictions_path": str(pred_path), "error": f"fresh benchmark failed: {type(exc).__name__}: {exc}"}


def build_fresh_benchmark(run_root: Path, k_list: Sequence[int], fresh_pairs: int, fresh_seed: int, fresh_device: str, fresh_batch_size: int, fresh_candidate_chunk_size: int, ginn_module_name: str, max_surfaces: int, dedupe_artifacts: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    master_run = find_latest_master_run(run_root)
    meta = build_surface_metadata(run_root, master_run)
    art = discover_prediction_artifacts(run_root, master_run)
    art_meta = merge_artifacts_with_metadata(art, meta)
    if art_meta.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"master_run": str(master_run) if master_run else "", "message": "no prediction artifacts found"}
    # Fresh inference is expensive; by default use one artifact per unique surface key.
    if dedupe_artifacts:
        key_col = "unique_surface_key_benchmark" if "unique_surface_key_benchmark" in art_meta.columns else "surface_id"
        art_meta = art_meta.sort_values([key_col, "surface_id"]).drop_duplicates(subset=[key_col], keep="first").reset_index(drop=True)
    if max_surfaces and max_surfaces > 0:
        art_meta = art_meta.head(max_surfaces).copy()
    rows = []
    failures = []
    import time
    t0 = time.time()
    for i, r in art_meta.iterrows():
        sid = str(r.get("surface_id", "") or Path(str(r.get("predictions_path", ""))).parent.name)
        print(f"[fresh] {i+1}/{len(art_meta)} {sid}", flush=True)
        row, fail = benchmark_fresh_one_artifact(r, run_root, k_list, fresh_pairs, fresh_seed + i, fresh_device, fresh_batch_size, fresh_candidate_chunk_size, ginn_module_name)
        if row is not None:
            rows.append(row)
        if fail is not None:
            failures.append(fail)
    surface_perf = pd.DataFrame(rows)
    if not surface_perf.empty:
        surface_perf["family_key"] = surface_perf.get("family_key", surface_perf["family_normalized"].map(family_key_from_name))
        surface_perf["main_riemann_record"] = surface_perf.apply(is_main_riemann_record, axis=1)
    failure_df = pd.DataFrame(failures)
    if not surface_perf.empty:
        agg_cols = {"surface_id": "count", "unique_surface_key": pd.Series.nunique, "n_pairs": "sum"}
        metric_cols = [c for c in surface_perf.columns if c.startswith("recall_at_") or c.startswith("nominal_speedup_at_") or c.endswith("_rmse") or c.endswith("_ms_per_pair") or c.startswith("estimated_wall_speedup_at_") or c in ["true_rank_mean", "true_rank_median", "recall_auc_1_20", "full_exact_seconds", "candidate_feature_seconds", "ginn_score_seconds"]]
        for c in metric_cols:
            agg_cols[c] = "median"
        family_perf = surface_perf.groupby(["family_key", "geometry_regime"], dropna=False).agg(agg_cols).reset_index()
        family_perf = family_perf.rename(columns={"surface_id": "records", "unique_surface_key": "unique_surfaces", "n_pairs": "test_pairs_total"})
    else:
        family_perf = pd.DataFrame()
    meta_info = {
        "master_run": str(master_run) if master_run else "",
        "fresh_prediction_artifacts_discovered": int(len(art_meta)),
        "fresh_artifacts_benchmarked": int(len(surface_perf)),
        "fresh_artifact_failures": int(len(failure_df)),
        "fresh_pairs": int(fresh_pairs),
        "fresh_seconds": float(time.time() - t0),
        "fresh_mode_note": "fresh p,q pairs sampled after training; saved PyTorch model scores finite word ball; exact distances checked only inside top-k pools",
    }
    return surface_perf, family_perf, failure_df, meta_info

def dedupe_surface_perf(surface_perf: pd.DataFrame, mode: str = "unique_mean") -> pd.DataFrame:
    if surface_perf.empty or mode == "raw":
        return surface_perf
    if "unique_surface_key" not in surface_perf.columns:
        return surface_perf
    numeric = surface_perf.select_dtypes(include=[np.number]).columns.tolist()
    nonnum = [c for c in surface_perf.columns if c not in numeric]
    if mode == "unique_best_top5":
        if "recall_at_5" in surface_perf.columns:
            idx = surface_perf.sort_values("recall_at_5", ascending=False).groupby("unique_surface_key").head(1).index
            return surface_perf.loc[idx].reset_index(drop=True)
    # unique_mean: average numeric, take first nonnumeric.
    agg = {c: "mean" for c in numeric}
    for c in nonnum:
        if c != "unique_surface_key":
            agg[c] = "first"
    return surface_perf.groupby("unique_surface_key", dropna=False).agg(agg).reset_index()


def make_topk_curve(surface_perf: pd.DataFrame, k_list: Sequence[int]) -> pd.DataFrame:
    rows = []
    if surface_perf.empty:
        return pd.DataFrame()
    for _, r in surface_perf.iterrows():
        for k in k_list:
            c = f"recall_at_{k}"
            if c in surface_perf.columns and math.isfinite(safe_float(r.get(c))):
                rows.append({
                    "surface_id": r.get("surface_id", ""),
                    "unique_surface_key": r.get("unique_surface_key", ""),
                    "family_normalized": r.get("family_normalized", ""),
                    "family_key": r.get("family_key", family_key_from_name(r.get("family_normalized", ""))),
                    "geometry_regime": r.get("geometry_regime", ""),
                    "k": k,
                    "recall": safe_float(r.get(c)),
                    "word_ball_size": safe_float(r.get("word_ball_size")),
                    "nominal_speedup": safe_float(r.get(f"nominal_speedup_at_{k}")),
                })
    return pd.DataFrame(rows)


def hardest_surfaces(surface_perf: pd.DataFrame, k: int = 5, n: int = 25) -> pd.DataFrame:
    if surface_perf.empty:
        return pd.DataFrame()
    df = surface_perf.copy()
    recall_col = f"recall_at_{k}"
    miss_col = f"miss_rate_at_{k}"
    sort_cols = []
    if miss_col in df.columns:
        sort_cols.append(miss_col)
    if "true_rank_median" in df.columns:
        sort_cols.append("true_rank_median")
    if "word_ball_size" in df.columns:
        sort_cols.append("word_ball_size")
    if not sort_cols:
        return df.head(n)
    out = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(n)
    # Deliberately omit true_rank_* from the PDF/table defaults.  In fresh mode
    # ties in neural scores can make strict-rank summaries confusing; recall@k
    # is the authoritative deployment metric.
    keep = [c for c in ["surface_id", "family_key", "geometry_regime", "word_ball_size", "n_pairs", "recall_at_1", recall_col, "recall_at_10", "recall_at_20", "top5_pruned_rmse", "nominal_speedup_at_5", "estimated_wall_speedup_at_5"] if c in out.columns]
    return out[keep]



def filter_main_report_records(surface_perf: pd.DataFrame, include_ineligible: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (main_records, excluded_records) for report summaries.

    By default, the main deployment report is restricted to eligible, non-orbifold
    Riemann-surface records.  Excluded/orbifold/metadata-ineligible records are
    still written to CSV and summarized in an appendix.
    """
    if surface_perf.empty or include_ineligible:
        return surface_perf.copy(), pd.DataFrame()
    df = surface_perf.copy()
    if "main_riemann_record" not in df.columns:
        df["main_riemann_record"] = df.apply(is_main_riemann_record, axis=1)
    main = df[df["main_riemann_record"].astype(bool)].copy()
    excluded = df[~df["main_riemann_record"].astype(bool)].copy()
    return main.reset_index(drop=True), excluded.reset_index(drop=True)


def family_key_table_for_report(surface_perf: pd.DataFrame) -> pd.DataFrame:
    base = family_key_description_table()
    if surface_perf.empty or "family_key" not in surface_perf.columns:
        return base
    used = set(surface_perf["family_key"].dropna().astype(str))
    return base[base["family_key"].isin(used)].reset_index(drop=True)

def write_markdown_report(out_dir: Path, surface_perf: pd.DataFrame, family_perf: pd.DataFrame, failures: pd.DataFrame, meta: Dict, k_list: Sequence[int], excluded_records: Optional[pd.DataFrame] = None) -> Path:
    p = out_dir / "deployment_benchmark_report.md"
    lines = []
    lines.append(f"# Fuchsian GINN Deployment Benchmark Report\n")
    lines.append(f"Generated by `{PROGRAM}` v{VERSION} on {datetime.now().isoformat(timespec='seconds')}.\n")
    lines.append("## Interpretation\n")
    mode = str(meta.get("benchmark_mode_requested", "replay"))
    if mode == "fresh":
        lines.append("This is a **fresh deployment benchmark**. It reloads saved PyTorch checkpoints, samples brand-new `(p,q)` pairs after training, uses the trained GINN to score the finite word ball, and then exact-checks only the learned top-k pool. This directly tests the train-once/use-many-times deployment claim.\n")
    elif mode == "both":
        lines.append("This report combines **replay** and **fresh** deployment checks. Replay uses saved held-out `predictions_test.csv`; fresh mode reloads PyTorch checkpoints and samples brand-new `(p,q)` pairs after training.\n")
    else:
        lines.append("This is a replay deployment benchmark over saved held-out `predictions_test.csv` files. It tests the train-once/use-many-times claim in the held-out sense: for each saved test pair, how small a GINN top-k pool is needed to retain the exact finite-word winning lift?\n")
    lines.append("The main tables below are restricted to eligible non-orbifold Riemann-surface records unless `--include-ineligible-in-main` was used. Excluded/orbifold/reference records are written separately. Speedups are nominal candidate-reduction factors `W/k`, not measured wall-clock speedups.\n")
    lines.append("## Audit\n")
    for k, v in meta.items():
        lines.append(f"- **{k}**: `{v}`")
    if not surface_perf.empty:
        unique_n = surface_perf["unique_surface_key"].nunique() if "unique_surface_key" in surface_perf.columns else len(surface_perf)
        lines.append(f"- **benchmarked records**: `{len(surface_perf)}`")
        lines.append(f"- **unique surfaces**: `{unique_n}`")
        lines.append(f"- **total held-out pairs**: `{int(surface_perf['n_pairs'].sum()) if 'n_pairs' in surface_perf.columns else 'NA'}`")
        for k in k_list:
            c = f"recall_at_{k}"
            if c in surface_perf.columns:
                lines.append(f"- **median recall@{k}**: `{surface_perf[c].median():.4g}`")
        if "nominal_speedup_at_5" in surface_perf.columns:
            lines.append(f"- **median nominal speedup@5**: `{surface_perf['nominal_speedup_at_5'].median():.4g}`")
    lines.append("\n## Family key\n")
    key_df = family_key_table_for_report(surface_perf)
    lines.append(df_to_markdown_safe(key_df, index=False, max_rows=40))
    lines.append("\n## Main family summary\n")
    if family_perf.empty:
        lines.append("No family performance table was produced.\n")
    else:
        show_cols = [c for c in ["family_key", "geometry_regime", "records", "unique_surfaces", "test_pairs_total", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "nominal_speedup_at_5", "top5_pruned_rmse"] if c in family_perf.columns]
        lines.append(df_to_markdown_safe(family_perf[show_cols], index=False))
    # Baseline comparison: only populated in fresh mode, where the exact D matrix
    # and candidate word list are available.
    base_method = baseline_method_summary(surface_perf, k_list)
    if not base_method.empty:
        lines.append("\n## Baseline comparison, same fresh pairs\n")
        lines.append("Baselines are evaluated on the same fresh `(p,q)` pairs. `GINN` is the trained model. `Random` samples candidates uniformly. `Identity` only checks the identity lift. `ShortWord` ranks candidates by word depth. `Id+Short` forces identity first, then short words. `Shell1Exact` exact-checks the depth-0/1 shell and keeps the closest k candidates in that shell. `FreqPrior`, when available, ranks by historical winner frequency in the saved prediction artifact. v1.4 also adds pair-dependent exact-search baselines: `GreedyExact` performs local descent in the word graph, and `Beam5/10/20Exact` performs exact-distance beam search with the indicated beam width. These stronger baselines are useful competitors because they use exact distances adaptively without training.\n")
        show_cols = [c for c in ["method", "records", "unique_surfaces", "test_pairs_total", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "top5_pruned_rmse", "eval_candidates_mean", "method_ms_per_pair"] if c in base_method.columns]
        lines.append(df_to_markdown_safe(base_method[show_cols].sort_values("recall_at_5", ascending=False), index=False, max_rows=30))
    timing_cols = [c for c in ["family_key", "records", "full_exact_ms_per_pair", "candidate_feature_ms_per_pair", "ginn_score_ms_per_pair", "estimated_ginn_top5_ms_per_pair", "estimated_wall_speedup_at_5"] if c in family_perf.columns]
    if timing_cols:
        lines.append("\n## Timing estimates\n")
        lines.append("`full_exact_ms_per_pair` is the measured cost of the full finite-word exact label/brute-force pass used for benchmarking. The estimated top-k deployment time adds candidate-feature construction, GINN scoring, and an estimated exact check of k candidates. These are approximate but useful for the train-once/use-many-times argument.\n")
        lines.append(df_to_markdown_safe(family_perf[timing_cols], index=False, max_rows=30))

    lines.append("\n## Hardest surfaces by miss-rate@5\n")
    hard = hardest_surfaces(surface_perf, k=5, n=20)
    if hard.empty:
        lines.append("No hardest-surface table was produced.\n")
    else:
        lines.append(df_to_markdown_safe(hard, index=False))
    if excluded_records is not None and not excluded_records.empty:
        lines.append("\n## Records excluded from the main Riemann-surface summary\n")
        keep = [c for c in ["surface_id", "family_key", "family_normalized", "geometry_regime", "word_ball_size", "recall_at_1", "recall_at_5"] if c in excluded_records.columns]
        lines.append(df_to_markdown_safe(excluded_records[keep].head(50), index=False))
    if not failures.empty:
        lines.append("\n## Missing or unreadable prediction/model artifacts\n")
        lines.append(df_to_markdown_safe(failures.head(50), index=False))
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def add_text_page(pdf: PdfPages, title: str, lines: Sequence[str], fontsize: int = 10):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.05, 0.95, title, fontsize=18, fontweight="bold", va="top")
    y = 0.90
    for line in lines:
        wrapped = textwrap.wrap(str(line), width=120) or [""]
        for w in wrapped:
            ax.text(0.05, y, w, fontsize=fontsize, va="top", family="monospace" if line.startswith("    ") else None)
            y -= 0.026
            if y < 0.05:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                y = 0.95
        y -= 0.010
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


PDF_COL_LABELS = {
    "family_key": "fam",
    "geometry_regime": "regime",
    "records": "n rec",
    "unique_surfaces": "n surf",
    "test_pairs_total": "pairs",
    "recall_at_1": "R@1",
    "recall_at_3": "R@3",
    "recall_at_5": "R@5",
    "recall_at_10": "R@10",
    "recall_at_20": "R@20",
    "nominal_speedup_at_5": "W/5",
    "estimated_wall_speedup_at_5": "est speed",
    "top5_pruned_rmse": "RMSE@5",
    "word_ball_size": "W",
    "n_pairs": "pairs",
    "surface_id": "surface",
    "method": "method",
    "full_exact_ms_per_pair": "exact ms/pair",
    "ginn_score_ms_per_pair": "GINN ms/pair",
    "estimated_ginn_top5_ms_per_pair": "est top5 ms/pair",
}


def _shorten_cell(v, max_len: int = 26):
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.4g}"
    s = str(v)
    if len(s) > max_len:
        return s[:max_len-1] + "…"
    return s


def table_page(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows: int = 25, max_cell_len: int = 26):
    fig, ax = plt.subplots(figsize=(14, 8.0))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        show = df.head(max_rows).copy()
        for c in show.columns:
            show[c] = show[c].map(lambda v: _shorten_cell(v, max_cell_len))
        col_labels = [PDF_COL_LABELS.get(str(c), str(c).replace("_", " ")) for c in show.columns]
        tbl = ax.table(cellText=show.values, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        # Dynamic font size: smaller for wide tables.
        fs = 7 if len(show.columns) <= 9 else 5.5
        tbl.set_fontsize(fs)
        tbl.scale(1, 1.15)
        # Slightly emphasize header.
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(out_dir: Path, surface_perf: pd.DataFrame, family_perf: pd.DataFrame, failures: pd.DataFrame, meta: Dict, k_list: Sequence[int], excluded_records: Optional[pd.DataFrame] = None) -> Path:
    p = out_dir / "deployment_benchmark_report.pdf"
    with PdfPages(p) as pdf:
        mode = str(meta.get("benchmark_mode_requested", "replay"))
        if mode == "fresh":
            intro = "Fresh deployment mode: reloads saved PyTorch checkpoints, samples brand-new p,q pairs after training, scores the finite word ball with the trained GINN, and exact-checks only top-k learned candidates."
        elif mode == "both":
            intro = "Combined replay+fresh mode: replay uses saved held-out predictions_test.csv; fresh mode reloads PyTorch models and samples brand-new p,q pairs."
        else:
            intro = "Replay mode: evaluates saved held-out predictions_test.csv artifacts produced by training."
        lines = [
            intro,
            "For each pair, the benchmark asks whether the exact finite-word winning lift is contained in the GINN-predicted top-k pool.",
            "If yes, then exact-checking only that pool recovers the true finite-word quotient distance for that pair.",
            "Main summaries are restricted to eligible non-orbifold Riemann-surface records unless --include-ineligible-in-main was used.",
            "Reported speedups are nominal W/k candidate-reduction factors, not measured wall-clock speedups.",
            "",
        ]
        for k, v in meta.items():
            lines.append(f"{k}: {v}")
        if not surface_perf.empty:
            unique_n = surface_perf["unique_surface_key"].nunique() if "unique_surface_key" in surface_perf.columns else len(surface_perf)
            lines += [
                f"benchmarked records: {len(surface_perf)}",
                f"unique surfaces: {unique_n}",
                f"total held-out pairs: {int(surface_perf['n_pairs'].sum()) if 'n_pairs' in surface_perf.columns else 'NA'}",
            ]
            for k in k_list:
                c = f"recall_at_{k}"
                if c in surface_perf.columns:
                    lines.append(f"median recall@{k}: {surface_perf[c].median():.4g}")
            if "nominal_speedup_at_5" in surface_perf.columns:
                lines.append(f"median nominal speedup@5: {surface_perf['nominal_speedup_at_5'].median():.4g}")
        add_text_page(pdf, "Fuchsian GINN Deployment Benchmark", lines)

        if not family_perf.empty:
            show_cols = [c for c in ["family_key", "geometry_regime", "records", "unique_surfaces", "test_pairs_total", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "nominal_speedup_at_5", "top5_pruned_rmse"] if c in family_perf.columns]
            table_page(pdf, "Family-level GINN deployment performance", family_perf[show_cols].sort_values("recall_at_5" if "recall_at_5" in show_cols else show_cols[0]), max_rows=30)

        # Baseline comparison on the same fresh pairs.
        base_method = baseline_method_summary(surface_perf, k_list)
        if not base_method.empty:
            show_cols = [c for c in ["method", "records", "unique_surfaces", "test_pairs_total", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "top5_pruned_rmse", "eval_candidates_mean", "method_ms_per_pair"] if c in base_method.columns]
            table_page(pdf, "Baseline comparison by method", base_method[show_cols].sort_values("recall_at_5", ascending=False), max_rows=20)
            fig, ax = plt.subplots(figsize=(10, 6))
            for _, rr in base_method.iterrows():
                xs, ys = [], []
                for kk in k_list:
                    c = f"recall_at_{kk}"
                    if c in base_method.columns and math.isfinite(safe_float(rr.get(c))):
                        xs.append(kk); ys.append(safe_float(rr.get(c)))
                if xs:
                    ax.plot(xs, ys, marker="o", label=str(rr.get("method", "")))
            ax.set_xlabel("k")
            ax.set_ylabel("median recall@k")
            ax.set_title("GINN vs non-learning baselines")
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not family_perf.empty and "full_exact_ms_per_pair" in family_perf.columns:
            show_cols = [c for c in ["family_key", "records", "full_exact_ms_per_pair", "candidate_feature_ms_per_pair", "ginn_score_ms_per_pair", "estimated_ginn_top5_ms_per_pair", "estimated_wall_speedup_at_5"] if c in family_perf.columns]
            table_page(pdf, "Per-pair timing estimates by family", family_perf[show_cols].sort_values("full_exact_ms_per_pair"), max_rows=30)

        topk = make_topk_curve(surface_perf, k_list)
        if not topk.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            fam = topk.groupby(["family_key", "k"], dropna=False)["recall"].median().reset_index()
            for family, sub in fam.groupby("family_key"):
                ax.plot(sub["k"], sub["recall"], marker="o", label=str(family))
            ax.set_xlabel("k")
            ax.set_ylabel("median recall@k")
            ax.set_title("GINN deployment recall vs candidate-pool size")
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="lower right")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 6))
            if "word_ball_size" in surface_perf.columns and "recall_at_5" in surface_perf.columns:
                for family, sub in surface_perf.groupby("family_key"):
                    ax.scatter(sub["word_ball_size"], sub["recall_at_5"], label=str(family), alpha=0.75)
                ax.set_xscale("log")
                ax.set_xlabel("word-ball size W")
                ax.set_ylabel("recall@5")
                ax.set_title("Recall@5 vs word-ball size")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc="best")
                pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 6))
            if "nominal_speedup_at_5" in surface_perf.columns and "recall_at_5" in surface_perf.columns:
                for family, sub in surface_perf.groupby("family_key"):
                    ax.scatter(sub["nominal_speedup_at_5"], sub["recall_at_5"], label=str(family), alpha=0.75)
                ax.set_xscale("log")
                ax.set_xlabel("nominal speedup W/5")
                ax.set_ylabel("recall@5")
                ax.set_title("Recall@5 vs nominal speedup")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc="best")
                pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        hard = hardest_surfaces(surface_perf, k=5, n=25)
        if not hard.empty:
            table_page(pdf, "Hardest surfaces by miss-rate@5", hard, max_rows=25)
        if excluded_records is not None and not excluded_records.empty:
            keep = [c for c in ["surface_id", "family_key", "family_normalized", "geometry_regime", "word_ball_size", "recall_at_1", "recall_at_5"] if c in excluded_records.columns]
            table_page(pdf, "Records excluded from main Riemann-surface summary", excluded_records[keep].head(25), max_rows=25)
        if not failures.empty:
            table_page(pdf, "Missing/unreadable prediction or model artifacts", failures.head(25), max_rows=25)
    return p


def write_outputs(out_dir: Path, surface_perf: pd.DataFrame, family_perf: pd.DataFrame, failures: pd.DataFrame, meta: Dict, k_list: Sequence[int], excluded_records: Optional[pd.DataFrame] = None, all_surface_perf: Optional[pd.DataFrame] = None):
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    surface_perf.to_csv(tables / "deployment_surface_performance.csv", index=False)
    if all_surface_perf is not None and not all_surface_perf.empty:
        all_surface_perf.to_csv(tables / "deployment_surface_performance_all_records.csv", index=False)
    if excluded_records is not None and not excluded_records.empty:
        excluded_records.to_csv(tables / "deployment_surface_performance_excluded_from_main.csv", index=False)
    family_key_description_table().to_csv(tables / "deployment_family_key.csv", index=False)
    family_perf.to_csv(tables / "deployment_family_performance.csv", index=False)
    baseline_long = baseline_long_table(surface_perf, k_list)
    baseline_method = baseline_method_summary(surface_perf, k_list)
    baseline_family = baseline_family_summary(surface_perf, k_list)
    baseline_long.to_csv(tables / "deployment_baseline_surface_long.csv", index=False)
    baseline_method.to_csv(tables / "deployment_baseline_method_summary.csv", index=False)
    baseline_family.to_csv(tables / "deployment_baseline_family_summary.csv", index=False)
    failures.to_csv(tables / "deployment_prediction_artifact_failures.csv", index=False)
    make_topk_curve(surface_perf, k_list).to_csv(tables / "deployment_topk_curve.csv", index=False)
    hardest_surfaces(surface_perf, k=5, n=100).to_csv(tables / "deployment_hardest_surfaces_top5.csv", index=False)

    manifest = dict(meta)
    manifest.update({
        "program": PROGRAM,
        "version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "k_list": list(k_list),
        "outputs": {
            "surface_performance": str(tables / "deployment_surface_performance.csv"),
            "family_performance": str(tables / "deployment_family_performance.csv"),
            "family_key": str(tables / "deployment_family_key.csv"),
            "all_surface_performance": str(tables / "deployment_surface_performance_all_records.csv") if all_surface_perf is not None and not all_surface_perf.empty else "",
            "excluded_from_main": str(tables / "deployment_surface_performance_excluded_from_main.csv") if excluded_records is not None and not excluded_records.empty else "",
            "topk_curve": str(tables / "deployment_topk_curve.csv"),
            "hardest_surfaces_top5": str(tables / "deployment_hardest_surfaces_top5.csv"),
            "baseline_surface_long": str(tables / "deployment_baseline_surface_long.csv"),
            "baseline_method_summary": str(tables / "deployment_baseline_method_summary.csv"),
            "baseline_family_summary": str(tables / "deployment_baseline_family_summary.csv"),
        },
        "interpretation_note": "replay mode uses saved held-out predictions_test.csv; fresh mode loads saved PyTorch models and samples brand-new p,q pairs for deployment inference",
    })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md = write_markdown_report(out_dir, surface_perf, family_perf, failures, meta, k_list, excluded_records=excluded_records)
    pdf = write_pdf_report(out_dir, surface_perf, family_perf, failures, meta, k_list, excluded_records=excluded_records)
    return md, pdf


def parse_k_list(s: str) -> List[int]:
    vals = []
    for part in str(s).replace(",", " ").split():
        try:
            v = int(part)
            if v > 0:
                vals.append(v)
        except Exception:
            pass
    return sorted(set(vals)) or K_LIST_DEFAULT


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Deployment-style benchmark for trained Fuchsian Downstairs GINNs")
    ap.add_argument("--run", "--runs", nargs="+", required=True, help="Completed ZooBuilder run root(s), e.g. zoo_runs/run_..._training_overnight_training_v13")
    ap.add_argument("--outroot", default="deployment_benchmark_runs", help="Output root directory")
    ap.add_argument("--label", default="deployment_benchmark", help="Label for this benchmark run")
    ap.add_argument("--k-list", default="1,3,5,10,20", help="Comma/space-separated top-k values")
    ap.add_argument("--dedupe", choices=["raw", "unique_mean", "unique_best_top5"], default="raw", help="Whether to deduplicate repeated/alias records for report summaries")
    ap.add_argument("--mode", choices=["replay", "fresh", "both"], default="replay", help="replay saved predictions_test.csv, run fresh model inference, or both")
    ap.add_argument("--fresh-pairs", type=int, default=0, help="Number of brand-new p,q pairs per surface for fresh mode. If 0 and mode=fresh/both, uses 500.")
    ap.add_argument("--fresh-seed", type=int, default=97531, help="Base RNG seed for fresh p,q sampling")
    ap.add_argument("--fresh-device", default="auto", help="auto, cpu, cuda for fresh PyTorch inference")
    ap.add_argument("--fresh-batch-size", type=int, default=64, help="Batch size for fresh inference scoring")
    ap.add_argument("--fresh-candidate-chunk-size", type=int, default=0, help="Candidate chunk size for fresh inference; 0 uses checkpoint value")
    ap.add_argument("--fresh-max-surfaces", type=int, default=0, help="Limit number of unique surfaces in fresh mode; 0 means all discovered unique surfaces")
    ap.add_argument("--fresh-raw-artifacts", action="store_true", help="Fresh mode normally benchmarks one artifact per unique surface; this uses all prediction artifacts and can be very expensive")
    ap.add_argument("--include-ineligible-in-main", action="store_true", help="Include excluded/orbifold/ineligible records in the main report tables instead of moving them to an appendix")
    ap.add_argument("--ginn-module", default="FuchsianDownstairsGINN_v2_4", help="Import name for the GINN training module used to reconstruct model/features in fresh mode")
    args = ap.parse_args(argv)

    k_list = parse_k_list(args.k_list)
    stamp = now_stamp()
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.label).strip("_") or "deployment_benchmark"
    out_dir = Path(args.outroot) / f"run_{stamp}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_surface = []
    all_failures = []
    fresh_pairs = int(args.fresh_pairs or (500 if args.mode in {"fresh", "both"} else 0))
    meta_all = {
        "run_roots": [str(Path(r)) for r in args.run],
        "benchmark_mode_requested": args.mode,
        "fresh_pairs_requested": int(fresh_pairs),
        "fresh_pairs_status": "fresh model inference enabled" if args.mode in {"fresh", "both"} else "not requested; using saved held-out predictions_test.csv artifact replay",
    }
    for run in args.run:
        root = Path(run)
        if args.mode in {"replay", "both"}:
            print(f"[benchmark-replay] run={root}")
            sp, fp, fail, meta = build_benchmark(root, k_list)
            if not sp.empty:
                sp["benchmark_input_run"] = str(root)
                sp["benchmark_mode"] = "replay"
                all_surface.append(sp)
            if not fail.empty:
                fail["benchmark_input_run"] = str(root)
                fail["benchmark_mode"] = "replay"
                all_failures.append(fail)
            meta_all[f"replay_meta_{root.name}"] = meta
            print(f"[benchmark-replay] artifacts={meta.get('prediction_artifacts_discovered', 0)} benchmarked={meta.get('prediction_artifacts_benchmarked', 0)} failures={meta.get('prediction_artifact_failures', 0)}")
        if args.mode in {"fresh", "both"}:
            print(f"[benchmark-fresh] run={root} fresh_pairs={fresh_pairs}")
            sp, fp, fail, meta = build_fresh_benchmark(
                root, k_list, fresh_pairs=fresh_pairs, fresh_seed=args.fresh_seed,
                fresh_device=args.fresh_device, fresh_batch_size=args.fresh_batch_size,
                fresh_candidate_chunk_size=args.fresh_candidate_chunk_size,
                ginn_module_name=args.ginn_module, max_surfaces=args.fresh_max_surfaces,
                dedupe_artifacts=(not args.fresh_raw_artifacts),
            )
            if not sp.empty:
                sp["benchmark_input_run"] = str(root)
                sp["benchmark_mode"] = "fresh"
                all_surface.append(sp)
            if not fail.empty:
                fail["benchmark_input_run"] = str(root)
                fail["benchmark_mode"] = "fresh"
                all_failures.append(fail)
            meta_all[f"fresh_meta_{root.name}"] = meta
            print(f"[benchmark-fresh] artifacts={meta.get('fresh_prediction_artifacts_discovered', 0)} benchmarked={meta.get('fresh_artifacts_benchmarked', 0)} failures={meta.get('fresh_artifact_failures', 0)}")

    surface_perf = pd.concat(all_surface, ignore_index=True) if all_surface else pd.DataFrame()
    failures = pd.concat(all_failures, ignore_index=True) if all_failures else pd.DataFrame()
    if args.dedupe != "raw" and not surface_perf.empty:
        surface_deduped = dedupe_surface_perf(surface_perf, mode=args.dedupe)
    else:
        surface_deduped = surface_perf
    surface_for_summary, excluded_from_main = filter_main_report_records(surface_deduped, include_ineligible=args.include_ineligible_in_main)
    if not surface_for_summary.empty:
        # Recompute family table on selected summary level and eligible-main filter.
        metric_cols = [c for c in surface_for_summary.columns if c.startswith("recall_at_") or c.startswith("nominal_speedup_at_") or c.endswith("_rmse") or c.endswith("_ms_per_pair") or c.startswith("estimated_wall_speedup_at_") or c in ["true_rank_mean", "true_rank_median", "recall_auc_1_20", "full_exact_seconds", "candidate_feature_seconds", "ginn_score_seconds"]]
        agg_cols = {"surface_id": "count", "unique_surface_key": pd.Series.nunique, "n_pairs": "sum"}
        for c in metric_cols:
            agg_cols[c] = "median"
        group_cols = [c for c in ["family_key", "geometry_regime"] if c in surface_for_summary.columns]
        family_perf = surface_for_summary.groupby(group_cols, dropna=False).agg(agg_cols).reset_index().rename(columns={"surface_id": "records", "unique_surface_key": "unique_surfaces", "n_pairs": "test_pairs_total"})
    else:
        family_perf = pd.DataFrame()

    meta_all.update({
        "dedupe_mode": args.dedupe,
        "benchmarked_records_raw": int(len(surface_perf)),
        "benchmarked_records_after_dedupe": int(len(surface_deduped)),
        "benchmarked_records_reported_main": int(len(surface_for_summary)),
        "records_excluded_from_main_report": int(len(excluded_from_main)),
        "unique_surfaces_raw": int(surface_perf["unique_surface_key"].nunique()) if "unique_surface_key" in surface_perf.columns else 0,
        "unique_surfaces_reported_main": int(surface_for_summary["unique_surface_key"].nunique()) if "unique_surface_key" in surface_for_summary.columns and not surface_for_summary.empty else 0,
        "main_report_filter": "eligible non-orbifold Riemann-surface records" if not args.include_ineligible_in_main else "all records including excluded/orbifold/ineligible",
        "prediction_artifact_failures_total": int(len(failures)),
    })

    md, pdf = write_outputs(out_dir, surface_for_summary, family_perf, failures, meta_all, k_list, excluded_records=excluded_from_main, all_surface_perf=surface_perf)
    print("=" * 78)
    print(f"[done] out_dir={out_dir}")
    print(f"[done] surface_records={len(surface_for_summary)} family_rows={len(family_perf)} failures={len(failures)}")
    print(f"[done] markdown_report={md}")
    print(f"[done] pdf_report={pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
