#!/usr/bin/env python3
"""FuchsianMasterDatasetBuilder_v3_1.py

Second-level-ML-aware master catalog builder for the Fuchsian GENN project.

This program ingests family-tester and GINN run directories, consolidates their
surface records, and writes a rich set of catalogs organized around the three
scientific layers of the project:

  1. analytic/topological/upstairs Fuchsian geometry;
  2. exact downstairs label geometry from finite word-ball quotient searches;
  3. learned downstairs branch geometry from the geometry-informed neural net.

Version 3.1 adds prediction-artifact-level learned-downstairs harvesting for family testers
that produce predictions_test.csv + branch_atlas_summary.json but not metrics.json.

It deliberately does not generate surfaces or train models.  ZooBuilder should
control what gets attempted; this master builder classifies and preserves what
was actually produced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROGRAM = "FuchsianMasterDatasetBuilder_v3_1.py"
VERSION = "3.1"
CONTRACT_VERSION = "fuchsian_master_dataset_contract_v3"

DEFAULT_INPUT_ROOTS = [
    "compact_polygon_tester_runs",
    "modular_congruence_tester_runs",
    "hecke_tester_runs",
    "hurwitz_tester_runs",
    "fuchsian_dataset_runs",
    "schottky_tester_runs",
    "elementary_tester_runs",
]

SURFACE_DIR_NAMES = [
    "surfaces",
    "kernel_surfaces",
    "excluded_surfaces",
    "eligible_surfaces_all",
    "surfaces_all",
]

LIGHT_ARTIFACT_NAMES = {
    "metrics.json",
    "run_manifest.json",
    "branch_atlas_summary.json",
    "train_log.csv",
    "predictions_test.csv",
    "word_ball.json",
}
HEAVY_ARTIFACT_NAMES = {
    "candidate_distances.npz",
    "learned_branch_context_embeddings.npz",
    "downstairs_ginn_v2_4.pt",
    "pair_dataset.csv",
}


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or x == "":
        return default
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "pass", "passed", "ok"}:
        return True
    if s in {"0", "false", "no", "n", "fail", "failed"}:
        return False
    return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def csv_value(x: Any) -> Any:
    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, sort_keys=True, ensure_ascii=True, default=str)
    if isinstance(x, bool):
        return "true" if x else "false"
    if x is None:
        return ""
    return x


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def short_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(stable_json_dumps(obj).encode("utf-8")).hexdigest()[:n]


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def read_csv_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        out: List[Dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                out.append(dict(row))
                if max_rows is not None and i + 1 >= max_rows:
                    break
        return out
    except Exception:
        return []


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    if fields:
        fieldnames.extend(fields)
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: csv_value(r.get(k)) for k in fieldnames})


def flatten(prefix: str, obj: Any, out: Optional[Dict[str, Any]] = None, max_depth: int = 5) -> Dict[str, Any]:
    if out is None:
        out = {}
    if max_depth < 0:
        out[prefix] = obj
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}_{k}" if prefix else str(k)
            flatten(key, v, out, max_depth - 1)
    elif isinstance(obj, list):
        # Keep long arrays out of CSVs but preserve compact scalar lists.
        if len(obj) <= 12 and all(not isinstance(v, (dict, list)) for v in obj):
            out[prefix] = obj
        else:
            out[f"{prefix}_len"] = len(obj)
            out[prefix] = obj[:12]
    else:
        out[prefix] = obj
    return out


def get_nested(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def latest_run_dirs(root: Path, all_runs: bool = False) -> List[Path]:
    if not root.exists():
        return []
    if (root / "manifest.json").exists() or (root / "run_summary.json").exists() or (root / "tables").exists():
        return [root]
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("fuchsian_dataset_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs if all_runs else runs[:1]


def discover_run_dirs(inputs: Sequence[str], all_runs: bool = False) -> Tuple[List[Path], List[Dict[str, Any]]]:
    run_dirs: List[Path] = []
    warnings: List[Dict[str, Any]] = []
    seen = set()
    for item in inputs:
        root = Path(item).expanduser().resolve()
        if not root.exists():
            warnings.append({"level": "warning", "path": str(root), "message": "input path does not exist"})
            continue
        found = latest_run_dirs(root, all_runs=all_runs)
        if not found and root.is_dir():
            found = [root]
        for rd in found:
            key = str(rd.resolve())
            if key not in seen:
                seen.add(key)
                run_dirs.append(rd.resolve())
    return run_dirs, warnings


def looks_like_surface_json(obj: Dict[str, Any], path: Path) -> bool:
    if not isinstance(obj, dict):
        return False
    if path.name.lower() in {"manifest.json", "run_manifest.json", "metrics.json", "branch_atlas_summary.json", "word_ball.json"}:
        return False
    if obj.get("master_record") and (obj.get("surface_id") or obj.get("surface_spec")):
        return True
    gens = obj.get("generators")
    if isinstance(gens, dict) and (obj.get("surface_id") or obj.get("surface_spec") or obj.get("domain_type")):
        return True
    if obj.get("surface_family") and (obj.get("surface_id") or obj.get("domain_type")):
        return True
    return False


def discover_surface_files(run_dir: Path) -> List[Tuple[Path, str]]:
    candidates: List[Tuple[Path, str]] = []
    for sub in SURFACE_DIR_NAMES:
        d = run_dir / sub
        if d.exists():
            for p in sorted(d.glob("*.json")):
                candidates.append((p, sub))
    raw = run_dir / "raw_ginn_runs"
    if raw.exists():
        for p in sorted(raw.glob("*/run_*/surface.json")):
            candidates.append((p, "dataset_builder_raw_ginn_surface"))
    for root_name in ["ginn_runs", "ginn_preflight", "downstairs_ginn_v2_4_runs"]:
        d = run_dir / root_name
        if d.exists():
            for p in sorted(d.glob("**/surface.json")):
                candidates.append((p, root_name))
    if (run_dir / "surface.json").exists():
        candidates.append((run_dir / "surface.json", "direct_ginn_surface"))
    # Fallback: surface-like jsons at shallow depth, but avoid ingesting every artifact.
    for p in sorted(run_dir.glob("*.json")):
        if p.name not in {"manifest.json", "run_manifest.json", "metrics.json", "run_summary.json"}:
            candidates.append((p, "root_surface_json"))
    out: List[Tuple[Path, str]] = []
    seen = set()
    for p, k in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append((p.resolve(), k))
    return out


def canonical_surface_id(surface: Dict[str, Any], path: Path) -> str:
    sid = surface.get("surface_id") or surface.get("surface_spec") or surface.get("name") or path.stem
    return str(sid).replace(" ", "_").replace("/", "_")


def infer_family(surface: Dict[str, Any]) -> str:
    for k in ["surface_family", "family", "category"]:
        if surface.get(k):
            return str(surface[k])
    dom = str(surface.get("domain_type") or "")
    if "schottky" in dom:
        return "schottky"
    if "compact_polygon" in dom:
        return "compact_polygon"
    if surface.get("level_N") is not None or surface.get("subgroup"):
        return "modular_congruence"
    if surface.get("q") is not None and "hecke" in str(surface.get("surface_type", "")).lower():
        return "hecke"
    if "hurwitz" in str(surface.get("surface_type", "")).lower():
        return "hurwitz"
    return "unknown"


def inferred_torsion_free(surface: Dict[str, Any]) -> bool:
    if surface.get("torsion_free") is not None:
        return safe_bool(surface.get("torsion_free"), default=False)
    if str(surface.get("domain_type")) == "compact_polygon" and (surface.get("generators") or {}):
        return True
    if str(surface.get("surface_type", "")).lower().endswith("kernel_surface"):
        return True
    return False


def is_eligible(surface: Dict[str, Any]) -> Tuple[bool, str, str]:
    explicit = surface.get("mainline_dataset_eligible")
    torsion_free = inferred_torsion_free(surface)
    orbifold_excluded = safe_bool(surface.get("orbifold_excluded"), default=not torsion_free)
    generators = surface.get("generators") or {}
    sampling = str(surface.get("sampling_status") or "")
    audit = surface.get("geometry_audit_pass")
    if explicit is not None:
        eligible = safe_bool(explicit)
        reason = "explicit mainline_dataset_eligible field"
    else:
        eligible = bool(torsion_free and not orbifold_excluded and isinstance(generators, dict) and len(generators) > 0)
        if sampling == "not supported by current GINN sampler":
            eligible = False
        if audit is False or str(audit).lower() == "false":
            eligible = False
        reason = "inferred from torsion_free/orbifold/generator/sampler/audit fields"
    excl = str(surface.get("exclusion_reason") or "")
    if not eligible and not excl:
        if not torsion_free:
            excl = "torsion-free status absent or false; orbifold/excluded record"
        elif orbifold_excluded:
            excl = "orbifold_excluded true"
        elif not generators:
            excl = "no exported generators"
        elif sampling == "not supported by current GINN sampler":
            excl = "no supported sampling model"
        else:
            excl = "not selected for mainline dataset"
    return eligible, reason, excl


def generator_hash(surface: Dict[str, Any]) -> str:
    gens = surface.get("generators") or {}
    compact: Dict[str, Any] = {}
    if isinstance(gens, dict):
        for label, g in sorted(gens.items(), key=lambda kv: str(kv[0])):
            if not isinstance(g, dict):
                compact[str(label)] = g
                continue
            a, b = g.get("alpha"), g.get("beta")
            try:
                compact[str(label)] = {
                    "alpha": [round(float(a[0]), 12), round(float(a[1]), 12)],
                    "beta": [round(float(b[0]), 12), round(float(b[1]), 12)],
                }
            except Exception:
                compact[str(label)] = g
    return short_hash(compact, 16)


def strip_volatile(obj: Any) -> Any:
    volatile = {"created", "created_at", "timestamp", "run_id", "run_root", "source_path", "source_run", "master_record_ingest", "ingested_at"}
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in volatile}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def word_ball_size_formula(generator_count: Optional[int], depth: int) -> Optional[int]:
    if generator_count is None or generator_count < 0:
        return None
    m = int(generator_count)
    if depth <= 0:
        return 1
    letters = 2 * m
    if letters == 0:
        return 1
    total = 1
    level = letters
    for k in range(1, depth + 1):
        if k == 1:
            level = letters
        else:
            level *= max(letters - 1, 0)
        total += level
    return total


def fingerprint_key(row: Dict[str, Any]) -> str:
    area = safe_float(row.get("area"), None)
    fp = {
        "surface_family": row.get("surface_family"),
        "surface_subfamily": row.get("surface_subfamily"),
        "compact": row.get("compact"),
        "finite_area": row.get("finite_area"),
        "genus": row.get("genus"),
        "compactified_genus": row.get("compactified_genus"),
        "cusp_count": row.get("cusp_count"),
        "area": None if area is None else round(area, 8),
        "generator_count": row.get("generator_count"),
        "domain_type": row.get("domain_type"),
        "subdomain_type": row.get("subdomain_type"),
    }
    return short_hash(fp, 16)


def construction_key(surface: Dict[str, Any], row: Dict[str, Any]) -> str:
    params = surface.get("construction_parameters") or {}
    key = {
        "surface_id": row.get("surface_id"),
        "surface_spec": row.get("surface_spec"),
        "family": row.get("surface_family"),
        "subfamily": row.get("surface_subfamily"),
        "level_N": surface.get("level_N") or params.get("level_N") or params.get("N"),
        "q": surface.get("q") or params.get("q"),
        "genus": row.get("genus"),
        "rank": surface.get("rank") or params.get("rank"),
        "gap": surface.get("gap") or params.get("gap"),
        "subgroup": surface.get("subgroup"),
        "parent_group": surface.get("parent_group"),
        "surface_type": surface.get("surface_type"),
    }
    return short_hash(key, 16)


def table_index(run_dir: Path) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[Dict[str, Any]]]:
    idx: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    table_rows: List[Dict[str, Any]] = []
    tdir = run_dir / "tables"
    if not tdir.exists():
        return idx, table_rows
    for csv_path in sorted(tdir.glob("*.csv")):
        rows = read_csv_rows(csv_path)
        stem = csv_path.stem
        for row in rows:
            annotated = dict(row)
            annotated["_table_name"] = stem
            table_rows.append(annotated)
            keys = []
            for k in ["surface_id", "surface", "surface_spec", "name", "surface_name"]:
                if row.get(k):
                    keys.append(str(row[k]))
            for key in keys:
                idx[stem][key] = row
                # Also expose some canonical table categories.
                if stem == "geometry_audit" or stem.endswith("geometry_audit"):
                    idx["geometry"][key] = {**idx["geometry"].get(key, {}), **row}
                if stem == "ginn_smoke_summary":
                    idx["smoke"][key] = row
                if stem == "ginn_training_summary":
                    idx["training"][key] = row
                if stem == "combined_surface_features":
                    idx["combined_surface_features"][key] = row
                    idx["geometry"][key] = {**idx["geometry"].get(key, {}), **row}
    return idx, table_rows


def lookup(idx: Dict[str, Dict[str, Dict[str, Any]]], kind: str, keys: Sequence[str]) -> Dict[str, Any]:
    for key in keys:
        if key and key in idx.get(kind, {}):
            return idx[kind][key]
    return {}


def find_ginn_artifact_dirs(surface_path: Path, source_run: Path, sid: str, spec: str) -> List[Path]:
    dirs: List[Path] = []
    # Direct GINN run surfaces have artifacts beside surface.json.
    if surface_path.name == "surface.json":
        dirs.append(surface_path.parent)
    # Family-testers and dataset builders place GINN runs in standard folders.
    for root_name in ["ginn_runs", "ginn_preflight", "raw_ginn_runs", "downstairs_ginn_v2_4_runs"]:
        root = source_run / root_name
        if root.exists():
            # v3.1: include prediction-only GINN runs. Some auxiliary family testers
            # emit branch_atlas_summary.json + predictions_test.csv + train_log.csv
            # but not metrics.json or run_manifest.json.
            for pattern in [
                "**/metrics.json",
                "**/run_manifest.json",
                "**/surface.json",
                "**/branch_atlas_summary.json",
                "**/predictions_test.csv",
                "**/train_log.csv",
            ]:
                for p in root.glob(pattern):
                    dirs.append(p.parent)
    # Keep directories with matching surface if possible; otherwise retain all nearby candidates.
    matched: List[Path] = []
    tokens = {sid, spec, surface_path.stem}
    for d in dirs:
        name_blob = str(d)
        surf = read_json(d / "surface.json") if (d / "surface.json").exists() else None
        surf_sid = canonical_surface_id(surf, d / "surface.json") if surf else ""
        if surf_sid in tokens or any(t and t in name_blob for t in tokens):
            matched.append(d)
    use = matched if matched else dirs
    out: List[Path] = []
    seen = set()
    for d in use:
        key = str(d.resolve())
        if key not in seen and d.exists():
            seen.add(key); out.append(d.resolve())
    return out


def select_best_artifact_dir(dirs: List[Path]) -> Optional[Path]:
    if not dirs:
        return None
    def score(d: Path) -> Tuple[int, float]:
        s = 0
        if (d / "metrics.json").exists(): s += 10
        if (d / "branch_atlas_summary.json").exists(): s += 5
        if (d / "predictions_test.csv").exists(): s += 4
        if (d / "train_log.csv").exists(): s += 3
        if (d / "run_manifest.json").exists(): s += 2
        return (s, d.stat().st_mtime)
    return sorted(dirs, key=score, reverse=True)[0]


def prediction_summary(pred_path: Path) -> Dict[str, Any]:
    rows = read_csv_rows(pred_path)
    if not rows:
        return {}
    out: Dict[str, Any] = {"prediction_test_rows": len(rows)}
    numeric_cols = [
        "winner_correct", "top3_contains_true", "top5_contains_true", "shortcut_probability",
        "branch_entropy", "score_margin_top1_top2", "exact_distance_gap_top1_top2",
        "top3_pruned_distance", "top5_pruned_distance", "top10_pruned_distance",
        "true_word_model_rank", "pred_exact_equivalent_tol_1e_5",
    ]
    for col in numeric_cols:
        vals = [safe_float(r.get(col), None) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            out[f"predictions_{col}_mean"] = statistics.fmean(vals)
            out[f"predictions_{col}_median"] = statistics.median(vals)
    for col in ["true_winning_lift_word", "pred_winning_lift_word", "pred_winning_lift_depth"]:
        vals = [str(r.get(col, "")) for r in rows if r.get(col, "") != ""]
        if vals:
            c = Counter(vals)
            out[f"predictions_unique_{col}"] = len(c)
            out[f"predictions_top_{col}"] = c.most_common(1)[0][0]
    return out


def artifact_index_for_dirs(record_uid: str, sid: str, dirs: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in dirs:
        for p in sorted(d.iterdir()) if d.exists() else []:
            if not p.is_file():
                continue
            name = p.name
            if name in LIGHT_ARTIFACT_NAMES:
                kind = "light"
            elif name in HEAVY_ARTIFACT_NAMES:
                kind = "heavy"
            else:
                continue
            rows.append({
                "record_uid": record_uid,
                "surface_id": sid,
                "artifact_kind": kind,
                "artifact_name": name,
                "artifact_path": str(p),
                "size_bytes": p.stat().st_size,
                "copy_default": name in LIGHT_ARTIFACT_NAMES and name not in {"word_ball.json", "predictions_test.csv"},
            })
    return rows


def harvest_metrics(artifact_dir: Optional[Path]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return exact_downstairs, training, learned_downstairs, artifact_summary."""
    exact: Dict[str, Any] = {}
    training: Dict[str, Any] = {}
    learned: Dict[str, Any] = {}
    art: Dict[str, Any] = {}
    if artifact_dir is None:
        return exact, training, learned, art
    metrics = read_json(artifact_dir / "metrics.json") or {}
    manifest = read_json(artifact_dir / "run_manifest.json") or {}
    branch = read_json(artifact_dir / "branch_atlas_summary.json") or {}
    if metrics:
        training.update({
            "ginn_metrics_present": True,
            "model_type": metrics.get("model_type"),
            "ginn_interpretation": metrics.get("interpretation"),
            "word_ball_size_metrics": metrics.get("word_ball_size"),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_epoch": metrics.get("best_epoch"),
            "epochs_ran": metrics.get("epochs_ran"),
            "candidate_chunk_size": metrics.get("candidate_chunk_size"),
            "estimated_unchunked_activation_mb": metrics.get("estimated_unchunked_activation_mb"),
            "identity_baseline_hard_rmse_improvement_fraction": metrics.get("identity_baseline_hard_rmse_improvement_fraction"),
            "test_hard_distance_rmse": get_nested(metrics, "test.hard_selected_distance.rmse"),
            "test_hard_distance_mae": get_nested(metrics, "test.hard_selected_distance.mae"),
            "test_hard_distance_r2": get_nested(metrics, "test.hard_selected_distance.r2"),
            "val_hard_distance_rmse": get_nested(metrics, "val.hard_selected_distance.rmse"),
            "train_hard_distance_rmse": get_nested(metrics, "train.hard_selected_distance.rmse"),
            "baseline_identity_test_rmse": get_nested(metrics, "baseline_identity_test.rmse"),
            "baseline_mean_test_rmse": get_nested(metrics, "baseline_mean_test.rmse"),
        })
        learned.update({
            "winning_lift_accuracy_test": metrics.get("winning_lift_accuracy_test"),
            "winning_lift_exact_equivalent_accuracy_test_tol_1e_5": metrics.get("winning_lift_exact_equivalent_accuracy_test_tol_1e_5"),
            "winning_lift_top3_accuracy_test": metrics.get("winning_lift_top3_accuracy_test"),
            "winning_lift_top5_accuracy_test": metrics.get("winning_lift_top5_accuracy_test"),
            "depth_accuracy_test": metrics.get("depth_accuracy_test"),
            "shortcut_fraction_test": metrics.get("shortcut_fraction_test"),
            "predicted_shortcut_fraction_test": metrics.get("predicted_shortcut_fraction_test"),
            "shortcut_auc_proxy_accuracy_test": metrics.get("shortcut_auc_proxy_accuracy_test"),
        })
        exact.update({
            "exact_downstairs_metrics_present": True,
            "exact_shortcut_fraction_test": metrics.get("shortcut_fraction_test"),
        })
        for k, v in (metrics.get("branch_atlas_test") or {}).items():
            learned[k] = v
            if k.startswith("exact_") or k.startswith("near_seam_") or k.startswith("unique_true_"):
                exact[k] = v
        for topk_name, topk in (metrics.get("topk_pruned_search") or {}).items():
            prefix = f"{topk_name}_"
            learned[prefix + "recall_true_winner_test"] = topk.get("recall_true_winner_test")
            learned[prefix + "candidate_fraction_examined"] = topk.get("candidate_fraction_examined")
            learned[prefix + "speedup_factor_vs_full_word_ball"] = topk.get("speedup_factor_vs_full_word_ball")
            learned[prefix + "selected_word_accuracy_after_pruned_min_test"] = topk.get("selected_word_accuracy_after_pruned_min_test")
            learned[prefix + "pruned_exact_distance_rmse_test"] = get_nested(topk, "pruned_exact_distance_test.rmse")
            learned[prefix + "pruned_exact_distance_mae_test"] = get_nested(topk, "pruned_exact_distance_test.mae")
    if branch:
        learned["branch_atlas_summary_present"] = True
        for k, v in branch.items():
            learned.setdefault(k, v)
            if k.startswith("exact_") or k.startswith("near_seam_") or k.startswith("unique_true_"):
                exact.setdefault(k, v)
    if manifest:
        flat_m = flatten("run_manifest", manifest, max_depth=3)
        for k, v in flat_m.items():
            if k not in training and len(str(v)) < 500:
                training[k] = v
    pred_sum = prediction_summary(artifact_dir / "predictions_test.csv")
    learned.update(pred_sum)
    if pred_sum:
        learned["predictions_test_summary_present"] = True
        learned["learned_downstairs_prediction_artifacts_present"] = True
        # v3.1: family testers for elementary/Schottky surfaces may not export
        # metrics.json, but predictions_test.csv already contains learned branch
        # behavior. Promote prediction summaries into the canonical learned-downstairs
        # fields when metrics.json did not already set them. These are test-set
        # empirical summaries, not training-loop metrics.
        alias_map = {
            "winning_lift_accuracy_test": "predictions_winner_correct_mean",
            "winning_lift_exact_equivalent_accuracy_test_tol_1e_5": "predictions_pred_exact_equivalent_tol_1e_5_mean",
            "winning_lift_top3_accuracy_test": "predictions_top3_contains_true_mean",
            "winning_lift_top5_accuracy_test": "predictions_top5_contains_true_mean",
            "learned_branch_entropy_mean": "predictions_branch_entropy_mean",
            "learned_branch_entropy_median": "predictions_branch_entropy_median",
            "learned_top1_score_margin_mean": "predictions_score_margin_top1_top2_mean",
            "learned_top1_score_margin_median": "predictions_score_margin_top1_top2_median",
            "exact_distance_gap_mean": "predictions_exact_distance_gap_top1_top2_mean",
            "exact_distance_gap_median": "predictions_exact_distance_gap_top1_top2_median",
            "unique_predicted_branches_test": "predictions_unique_pred_winning_lift_word",
        }
        for dst, src in alias_map.items():
            if learned.get(dst) in (None, "") and pred_sum.get(src) not in (None, ""):
                learned[dst] = pred_sum.get(src)
        if pred_sum.get("predictions_unique_pred_winning_lift_word") not in (None, ""):
            learned["unique_predicted_branches_test"] = pred_sum.get("predictions_unique_pred_winning_lift_word")
        exact.setdefault("exact_downstairs_metrics_present", True)
        exact.setdefault("exact_distance_gap_mean", pred_sum.get("predictions_exact_distance_gap_top1_top2_mean"))
        exact.setdefault("exact_distance_gap_median", pred_sum.get("predictions_exact_distance_gap_top1_top2_median"))
    for name in list(LIGHT_ARTIFACT_NAMES | HEAVY_ARTIFACT_NAMES):
        p = artifact_dir / name
        if p.exists():
            art[f"artifact_{name.replace('.', '_')}_present"] = True
            art[f"artifact_{name.replace('.', '_')}_size_bytes"] = p.stat().st_size
    art["selected_artifact_dir"] = str(artifact_dir)
    return exact, training, learned, art


def build_base_row(surface: Dict[str, Any], path: Path, source_run: Path, source_kind: str,
                   tables: Dict[str, Dict[str, Dict[str, Any]]], args: argparse.Namespace) -> Dict[str, Any]:
    sid = canonical_surface_id(surface, path)
    spec = str(surface.get("surface_spec") or sid)
    family = str(surface.get("surface_family") or infer_family(surface))
    subfamily = str(surface.get("surface_subfamily") or surface.get("subdomain_type") or "")
    keys = [sid, spec, path.stem]
    geom = lookup(tables, "geometry", keys)
    smoke = lookup(tables, "smoke", keys)
    train_tbl = lookup(tables, "training", keys)
    combo = lookup(tables, "combined_surface_features", keys)
    eligible, eligible_reason, excl = is_eligible(surface)
    gen_count = safe_int(surface.get("generator_count"), None)
    if gen_count is None:
        gen_count = len(surface.get("generators") or {}) if isinstance(surface.get("generators"), dict) else None
    if geom.get("generator_count") not in (None, ""):
        gen_count = safe_int(geom.get("generator_count"), gen_count)
    depth = safe_int(smoke.get("word_depth"), safe_int(surface.get("word_ball_recommended_depth"), 2))
    observed_w = safe_int(smoke.get("word_ball_size"), safe_int(combo.get("word_ball_size"), None))
    if observed_w is None:
        observed_w = safe_int(train_tbl.get("word_ball_size"), None)
    wb2 = word_ball_size_formula(gen_count, 2)
    wb3 = word_ball_size_formula(gen_count, 3)
    pass_geom = surface.get("geometry_audit_pass")
    if pass_geom is None and geom.get("pass_geometry_audit") not in (None, ""):
        pass_geom = safe_bool(geom.get("pass_geometry_audit"))
    pass_smoke = None
    if smoke:
        pass_smoke = safe_bool(smoke.get("pass_ginn_preflight"), default=False)
    elif train_tbl:
        pass_smoke = safe_bool(train_tbl.get("pass_ginn_training"), default=False)
    finite_area_default = surface.get("area") is not None and not safe_bool(surface.get("orbifold_excluded"), False)
    compact = safe_bool(surface.get("compact"), default=str(surface.get("domain_type")) == "compact_polygon")
    finite_area = safe_bool(surface.get("finite_area"), default=finite_area_default)
    row: Dict[str, Any] = {
        "surface_id": sid,
        "surface_spec": spec,
        "surface_family": family,
        "surface_subfamily": subfamily,
        "mainline_dataset_eligible": eligible,
        "eligibility_reason": eligible_reason,
        "exclusion_reason": excl,
        "riemann_surface_status": surface.get("riemann_surface_status") or (surface.get("master_record") or {}).get("riemann_surface_status"),
        "kahler_status": surface.get("kahler_status") or (surface.get("master_record") or {}).get("kahler_status"),
        "torsion_free": inferred_torsion_free(surface),
        "orbifold_excluded": safe_bool(surface.get("orbifold_excluded"), default=not inferred_torsion_free(surface)),
        "compact": compact,
        "finite_area": finite_area,
        "infinite_area": bool(not finite_area and eligible),
        "cusped": bool((safe_int(surface.get("cusp_count"), 0) or 0) > 0),
        "genus": surface.get("genus"),
        "compactified_genus": surface.get("compactified_genus"),
        "cusp_count": surface.get("cusp_count"),
        "cusp_widths": surface.get("cusp_widths"),
        "area": surface.get("area"),
        "euler_characteristic": surface.get("euler_characteristic"),
        "surface_area_type": surface.get("surface_area_type"),
        "dataset_role": surface.get("dataset_role"),
        "domain_type": surface.get("domain_type"),
        "subdomain_type": surface.get("subdomain_type"),
        "fundamental_domain_status": surface.get("fundamental_domain_status"),
        "sampling_status": surface.get("sampling_status"),
        "generator_count": gen_count,
        "generator_truncated": safe_bool(surface.get("generator_truncated"), default=safe_bool((surface.get("generator_export_audit") or {}).get("generator_truncated_by_cli_max_generators"), False)),
        "source_program": surface.get("source_program") or (surface.get("master_record") or {}).get("source_program") or geom.get("source_program"),
        "source_version": surface.get("source_version") or (surface.get("master_record") or {}).get("source_version") or geom.get("source_version"),
        "source_run": str(source_run),
        "source_path": str(path),
        "source_kind": source_kind,
        "pass_geometry_audit": pass_geom,
        "pass_ginn_preflight": pass_smoke,
        "word_ball_depth": depth,
        "word_ball_size": observed_w,
        "word_ball_risk_depth2": wb2,
        "word_ball_risk_depth3": wb3,
        "shortcut_fraction": smoke.get("shortcut_fraction") or combo.get("shortcut_fraction"),
        "mean_winner_depth": smoke.get("mean_winner_depth") or smoke.get("mean_shortest_lift_depth") or combo.get("mean_winner_depth"),
        "max_word_ball": smoke.get("max_word_ball") or combo.get("max_word_ball"),
        "exact_duplicate_hash": short_hash(strip_volatile(surface), 16),
        "generator_hash": generator_hash(surface),
        "isomorphism_status": "unresolved; hashes/fingerprints are not proof of isomorphism",
    }
    # Preserve common analytic/upstairs construction parameters when present.
    params = surface.get("construction_parameters") or {}
    for key in ["level_N", "N", "q", "rank", "gap", "rotation_deg", "sample_radius", "subgroup", "parent_group", "finite_quotient_order", "psl_q", "triangle_signature", "surface_type"]:
        if surface.get(key) is not None:
            row[key] = surface.get(key)
        elif params.get(key) is not None:
            row[key] = params.get(key)
    # Preserve geometry audit columns under audit_*.
    for k, v in geom.items():
        row.setdefault(f"audit_{k}", v)
    for k, v in smoke.items():
        row.setdefault(f"smoke_{k}", v)
    for k, v in train_tbl.items():
        row.setdefault(f"training_table_{k}", v)
    row["construction_key"] = construction_key(surface, row)
    row["fingerprint_key"] = fingerprint_key(row)
    row["unique_surface_key"] = row["generator_hash"] or row["exact_duplicate_hash"]
    row["primary_unique_record"] = False
    uid_payload = {"surface_id": sid, "source_path": str(path), "exact": row["exact_duplicate_hash"]}
    row["record_uid"] = f"{sid}__{short_hash(uid_payload, 10)}"
    # Budget classifications.
    budget_w = observed_w if observed_w is not None else wb2
    if budget_w is None:
        wb_status = "missing"
    elif budget_w <= args.training_word_ball_cap:
        wb_status = "training_cap_ok"
    elif budget_w <= args.catalog_word_ball_cap:
        wb_status = "catalog_cap_only"
    else:
        wb_status = "exceeds_catalog_cap"
    row["word_ball_budget_status"] = wb_status
    row["training_candidate_by_word_ball"] = bool(eligible and wb_status == "training_cap_ok")
    row["catalog_only_by_word_ball"] = bool(eligible and wb_status in {"catalog_cap_only", "exceeds_catalog_cap", "missing"})
    return row


def classify_duplicates(rows: List[Dict[str, Any]]) -> None:
    by_exact: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_fp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_exact[str(r.get("exact_duplicate_hash"))].append(r)
        by_fp[str(r.get("fingerprint_key"))].append(r)
    for h, members in by_exact.items():
        group = "exact_unique" if len(members) == 1 else f"exact_duplicate_{h}"
        for r in members:
            r["duplicate_group"] = group
    for h, members in by_fp.items():
        group = "fingerprint_unique" if len(members) == 1 else f"possible_fingerprint_match_{h}"
        for r in members:
            r["possible_isomorphic_group"] = group


def mark_primary_unique_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pref = {
        "surfaces": 0,
        "kernel_surfaces": 1,
        "family_surfaces": 2,
        "ginn_runs": 3,
        "direct_ginn_surface": 4,
        "dataset_builder_raw_ginn_surface": 5,
        "raw_ginn_runs": 6,
    }
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if safe_bool(r.get("mainline_dataset_eligible")):
            groups[str(r.get("unique_surface_key"))].append(r)
    primary: List[Dict[str, Any]] = []
    for key, members in groups.items():
        members.sort(key=lambda r: (pref.get(str(r.get("source_kind")), 99), str(r.get("surface_id")), str(r.get("source_path"))))
        chosen = members[0]
        chosen["primary_unique_record"] = True
        for other in members[1:]:
            other["primary_unique_record"] = False
            other["duplicate_note"] = f"non-primary duplicate of {chosen.get('record_uid')} by unique_surface_key={key}"
        primary.append(chosen)
    return primary


def duplicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key_name in ["exact_duplicate_hash", "fingerprint_key", "generator_hash"]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[str(r.get(key_name))].append(r)
        for h, members in groups.items():
            if len(members) > 1:
                out.append({
                    "match_type": key_name,
                    "hash": h,
                    "member_count": len(members),
                    "surface_ids": ";".join(str(m.get("surface_id")) for m in members),
                    "record_uids": ";".join(str(m.get("record_uid")) for m in members),
                    "interpretation": "audit duplicate/fingerprint only; not an isomorphism proof",
                })
    return out


def summarize_family(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("surface_family")), str(r.get("surface_subfamily") or ""))].append(r)
    out: List[Dict[str, Any]] = []
    for (fam, sub), members in sorted(groups.items()):
        elig = [r for r in members if safe_bool(r.get("mainline_dataset_eligible"))]
        trained = [r for r in elig if safe_bool(r.get("ginn_metrics_present"))]
        out.append({
            "surface_family": fam,
            "surface_subfamily": sub,
            "records_total": len(members),
            "eligible_total": len(elig),
            "primary_unique_eligible": sum(1 for r in elig if safe_bool(r.get("primary_unique_record"))),
            "compact_eligible": sum(1 for r in elig if safe_bool(r.get("compact"))),
            "finite_area_eligible": sum(1 for r in elig if safe_bool(r.get("finite_area"))),
            "infinite_area_eligible": sum(1 for r in elig if safe_bool(r.get("infinite_area"))),
            "trained_ginn_records": len(trained),
            "training_candidate_by_word_ball": sum(1 for r in elig if safe_bool(r.get("training_candidate_by_word_ball"))),
            "catalog_only_by_word_ball": sum(1 for r in elig if safe_bool(r.get("catalog_only_by_word_ball"))),
            "orbifold_or_excluded": sum(1 for r in members if safe_bool(r.get("orbifold_excluded")) or not safe_bool(r.get("mainline_dataset_eligible"))),
            "max_depth2_word_ball_estimate": max([safe_int(r.get("word_ball_risk_depth2"), 0) or 0 for r in members] or [0]),
            "max_observed_word_ball_size": max([safe_int(r.get("word_ball_size"), 0) or 0 for r in members] or [0]),
        })
    return out


def source_run_rows(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    out = []
    for rd in run_dirs:
        manifest = read_json(rd / "manifest.json") or read_json(rd / "run_summary.json") or {}
        out.append({
            "source_run": str(rd),
            "program": manifest.get("program"),
            "version": manifest.get("version"),
            "contract_version": manifest.get("contract_version"),
            "completed": manifest.get("completed") or manifest.get("groups") or manifest.get("surfaces"),
            "failures": manifest.get("failures"),
            "mtime": rd.stat().st_mtime,
        })
    return out


def word_ball_budget_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "record_uid": r.get("record_uid"),
            "surface_id": r.get("surface_id"),
            "surface_family": r.get("surface_family"),
            "eligible": r.get("mainline_dataset_eligible"),
            "generator_count": r.get("generator_count"),
            "observed_word_ball_size": r.get("word_ball_size"),
            "observed_word_depth": r.get("word_ball_depth"),
            "depth2_word_ball_estimate": r.get("word_ball_risk_depth2"),
            "depth3_word_ball_estimate": r.get("word_ball_risk_depth3"),
            "word_ball_budget_status": r.get("word_ball_budget_status"),
            "training_candidate_by_word_ball": r.get("training_candidate_by_word_ball"),
            "catalog_only_by_word_ball": r.get("catalog_only_by_word_ball"),
        })
    return sorted(out, key=lambda x: str(x.get("word_ball_budget_status")))


def table_select(rows: List[Dict[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    return [{k: r.get(k) for k in fields if k in r or True} for r in rows]


ANALYTIC_FIELDS = [
    "record_uid", "surface_id", "surface_family", "surface_subfamily", "mainline_dataset_eligible",
    "riemann_surface_status", "kahler_status", "torsion_free", "orbifold_excluded",
    "compact", "finite_area", "infinite_area", "cusped", "genus", "compactified_genus",
    "cusp_count", "cusp_widths", "area", "euler_characteristic", "surface_area_type",
    "dataset_role", "domain_type", "subdomain_type", "fundamental_domain_status", "sampling_status",
]
UPSTAIRS_FIELDS = [
    "record_uid", "surface_id", "surface_family", "surface_subfamily", "generator_count",
    "generator_truncated", "generator_hash", "construction_key", "fingerprint_key", "unique_surface_key",
    "word_ball_depth", "word_ball_size", "word_ball_risk_depth2", "word_ball_risk_depth3",
    "word_ball_budget_status", "level_N", "N", "q", "rank", "gap", "rotation_deg",
    "sample_radius", "subgroup", "parent_group", "finite_quotient_order", "psl_q", "triangle_signature",
    "surface_type", "pass_geometry_audit",
]
EXACT_DOWNSTAIRS_FIELDS = [
    "record_uid", "surface_id", "surface_family", "word_ball_depth", "word_ball_size", "shortcut_fraction",
    "mean_winner_depth", "exact_downstairs_metrics_present", "exact_shortcut_fraction_test",
    "exact_distance_gap_mean", "exact_distance_gap_median", "near_seam_fraction_gap_lt_0p02",
    "near_seam_fraction_gap_lt_0p05", "unique_true_branches_test",
]
TRAINING_FIELDS = [
    "record_uid", "surface_id", "surface_family", "ginn_metrics_present", "strict_ginn_metrics_ready",
    "learned_downstairs_prediction_level_ready", "learned_downstairs_any_ready", "model_type", "ginn_interpretation",
    "word_ball_size_metrics", "best_val_loss", "best_epoch", "epochs_ran", "candidate_chunk_size",
    "estimated_unchunked_activation_mb", "test_hard_distance_rmse", "test_hard_distance_mae",
    "test_hard_distance_r2", "val_hard_distance_rmse", "train_hard_distance_rmse",
    "baseline_identity_test_rmse", "baseline_mean_test_rmse", "identity_baseline_hard_rmse_improvement_fraction",
]
LEARNED_FIELDS = [
    "record_uid", "surface_id", "surface_family", "learned_downstairs_any_ready",
    "learned_downstairs_prediction_level_ready", "learned_downstairs_prediction_artifacts_present",
    "strict_ginn_metrics_ready", "winning_lift_accuracy_test",
    "winning_lift_exact_equivalent_accuracy_test_tol_1e_5", "winning_lift_top3_accuracy_test",
    "winning_lift_top5_accuracy_test", "depth_accuracy_test", "shortcut_fraction_test",
    "predicted_shortcut_fraction_test", "shortcut_auc_proxy_accuracy_test",
    "learned_branch_entropy_mean", "learned_branch_entropy_median", "learned_top1_score_margin_mean",
    "learned_top1_score_margin_median", "unique_true_branches_test", "unique_predicted_branches_test",
    "top3_recall_true_winner_test", "top5_recall_true_winner_test", "top10_recall_true_winner_test",
    "top5_pruned_exact_distance_rmse_test", "top10_pruned_exact_distance_rmse_test",
    "top5_speedup_factor_vs_full_word_ball", "top10_speedup_factor_vs_full_word_ball",
    "predictions_test_summary_present", "predictions_winner_correct_mean",
    "predictions_top3_contains_true_mean", "predictions_top5_contains_true_mean",
    "predictions_branch_entropy_mean", "predictions_score_margin_top1_top2_mean",
    "predictions_exact_distance_gap_top1_top2_mean",
    "predictions_true_word_model_rank_mean", "predictions_true_word_model_rank_median",
]


def copy_surface_with_ingest_metadata(dst: Path, surface: Dict[str, Any], row: Dict[str, Any]) -> None:
    out = dict(surface)
    out["master_record_ingest"] = {k: row.get(k) for k in [
        "record_uid", "source_run", "source_path", "source_kind", "exact_duplicate_hash",
        "construction_key", "generator_hash", "fingerprint_key", "duplicate_group",
        "possible_isomorphic_group", "isomorphism_status", "word_ball_budget_status",
        "second_level_ml_role",
    ]}
    write_json(dst, out)


def copy_artifacts(out_root: Path, row: Dict[str, Any], artifact_rows: List[Dict[str, Any]], copy_heavy: bool = False) -> None:
    uid = str(row.get("record_uid"))
    for ar in artifact_rows:
        if ar.get("record_uid") != uid:
            continue
        p = Path(str(ar.get("artifact_path")))
        if not p.exists():
            continue
        kind = str(ar.get("artifact_kind"))
        if kind == "heavy" and not copy_heavy:
            continue
        if kind == "light" and p.name == "word_ball.json":
            # word balls may be large; index by default, copy only in heavy mode.
            if not copy_heavy:
                continue
        dst = out_root / "artifacts" / uid / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, dst)
        except Exception:
            pass


def assign_second_level_role(row: Dict[str, Any]) -> None:
    metrics_ready = bool(safe_bool(row.get("ginn_metrics_present")) and safe_bool(row.get("branch_atlas_summary_present"), False))
    learned_metric_present = row.get("winning_lift_accuracy_test") not in (None, "")
    prediction_ready = bool(safe_bool(row.get("predictions_test_summary_present"), False) and safe_bool(row.get("branch_atlas_summary_present"), False))
    row["strict_ginn_metrics_ready"] = metrics_ready
    row["learned_downstairs_prediction_level_ready"] = prediction_ready
    row["learned_downstairs_any_ready"] = bool(metrics_ready or learned_metric_present or prediction_ready)
    if not safe_bool(row.get("mainline_dataset_eligible")):
        row["second_level_ml_role"] = "orbifold_or_excluded_reference"
    elif metrics_ready or learned_metric_present:
        row["second_level_ml_role"] = "full_ginn_learned_downstairs_record"
    elif prediction_ready:
        row["second_level_ml_role"] = "prediction_artifact_learned_downstairs_record"
    elif safe_bool(row.get("pass_ginn_preflight"), False) or row.get("shortcut_fraction") not in (None, ""):
        row["second_level_ml_role"] = "exact_label_or_smoke_record"
    elif safe_bool(row.get("training_candidate_by_word_ball"), False):
        row["second_level_ml_role"] = "geometry_catalog_training_candidate"
    else:
        row["second_level_ml_role"] = "geometry_catalog_only"
    # v3.1: full_ginn_second_level_ready means suitable for second-level learned-downstairs
    # ML. strict_ginn_metrics_ready remains available when one wants only metrics.json records.
    row["full_ginn_second_level_ready"] = row["second_level_ml_role"] in {
        "full_ginn_learned_downstairs_record",
        "prediction_artifact_learned_downstairs_record",
    }


def build_master(args: argparse.Namespace) -> int:
    inputs = args.inputs or DEFAULT_INPUT_ROOTS
    run_dirs, warnings = discover_run_dirs(inputs, all_runs=args.all_runs)
    out_root = Path(args.outroot) / f"run_{now_stamp()}{('_' + args.label) if args.label else ''}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{PROGRAM} v{VERSION}")
    print(f"out_root={out_root}")
    print(f"run_dirs={len(run_dirs)}")

    all_rows: List[Dict[str, Any]] = []
    surface_blobs: Dict[str, Tuple[Dict[str, Any], Path]] = {}
    artifact_rows: List[Dict[str, Any]] = []
    rejected_jsons: List[Dict[str, Any]] = []
    discovered_files = 0

    for rd in run_dirs:
        print(f"[ingest] {rd}")
        idx, _table_rows = table_index(rd)
        for path, kind in discover_surface_files(rd):
            discovered_files += 1
            obj = read_json(path)
            if obj is None or not looks_like_surface_json(obj, path):
                rejected_jsons.append({"level": "info", "path": str(path), "source_run": str(rd), "message": "not recognized as surface JSON"})
                continue
            row = build_base_row(obj, path, rd, kind, idx, args)
            sid = str(row["surface_id"]); spec = str(row.get("surface_spec") or sid)
            artifact_dirs = find_ginn_artifact_dirs(path, rd, sid, spec)
            selected_dir = select_best_artifact_dir(artifact_dirs)
            exact, training, learned, art = harvest_metrics(selected_dir)
            row.update(exact); row.update(training); row.update(learned); row.update(art)
            assign_second_level_role(row)
            all_rows.append(row)
            surface_blobs[str(row["record_uid"])] = (obj, path)
            artifact_rows.extend(artifact_index_for_dirs(str(row["record_uid"]), sid, artifact_dirs))

    classify_duplicates(all_rows)
    primary_eligible_rows = mark_primary_unique_records(all_rows)
    eligible_rows = [r for r in all_rows if safe_bool(r.get("mainline_dataset_eligible"))]
    excluded_rows = [r for r in all_rows if not safe_bool(r.get("mainline_dataset_eligible"))]
    finite_area_rows = [r for r in eligible_rows if safe_bool(r.get("finite_area"))]
    infinite_area_rows = [r for r in eligible_rows if not safe_bool(r.get("finite_area"))]
    full_ginn_rows = [r for r in eligible_rows if safe_bool(r.get("full_ginn_second_level_ready"))]
    train_candidate_rows = [r for r in eligible_rows if safe_bool(r.get("training_candidate_by_word_ball"))]
    catalog_only_rows = [r for r in eligible_rows if not safe_bool(r.get("training_candidate_by_word_ball"))]

    # Copy surface records.
    for r in all_rows:
        obj, src = surface_blobs[str(r["record_uid"])]
        safe_name = str(r["record_uid"]).replace("/", "_").replace(" ", "_") + ".json"
        copy_surface_with_ingest_metadata(out_root / "surfaces_all" / safe_name, obj, r)
        if safe_bool(r.get("mainline_dataset_eligible")):
            copy_surface_with_ingest_metadata(out_root / "eligible_surfaces_all" / safe_name, obj, r)
            if safe_bool(r.get("primary_unique_record")):
                copy_surface_with_ingest_metadata(out_root / "surfaces" / safe_name, obj, r)
        else:
            copy_surface_with_ingest_metadata(out_root / "excluded_surfaces" / safe_name, obj, r)
        if args.copy_light_artifacts or args.copy_heavy_artifacts:
            copy_artifacts(out_root, r, artifact_rows, copy_heavy=args.copy_heavy_artifacts)

    # Tables.
    tables = out_root / "tables"
    write_csv(tables / "master_surface_catalog.csv", all_rows)
    write_csv(tables / "eligible_surface_catalog.csv", eligible_rows)
    write_csv(tables / "eligible_unique_surface_catalog.csv", primary_eligible_rows)
    write_csv(tables / "excluded_orbifold_records.csv", excluded_rows)
    write_csv(tables / "analytic_geometry_catalog.csv", table_select(all_rows, ANALYTIC_FIELDS), list(ANALYTIC_FIELDS))
    write_csv(tables / "upstairs_group_geometry_catalog.csv", table_select(all_rows, UPSTAIRS_FIELDS), list(UPSTAIRS_FIELDS))
    write_csv(tables / "exact_downstairs_geometry_catalog.csv", table_select(all_rows, EXACT_DOWNSTAIRS_FIELDS), list(EXACT_DOWNSTAIRS_FIELDS))
    write_csv(tables / "ginn_training_metrics_catalog.csv", table_select(all_rows, TRAINING_FIELDS), list(TRAINING_FIELDS))
    write_csv(tables / "learned_downstairs_geometry_catalog.csv", table_select(all_rows, LEARNED_FIELDS), list(LEARNED_FIELDS))
    write_csv(tables / "second_level_ml_features.csv", all_rows)
    write_csv(tables / "full_ginn_second_level_records.csv", full_ginn_rows)
    write_csv(tables / "learned_downstairs_second_level_records.csv", full_ginn_rows)
    write_csv(tables / "strict_ginn_metrics_records.csv", [r for r in eligible_rows if safe_bool(r.get("strict_ginn_metrics_ready"))])
    write_csv(tables / "prediction_artifact_learned_records.csv", [r for r in eligible_rows if safe_bool(r.get("learned_downstairs_prediction_level_ready")) and not safe_bool(r.get("strict_ginn_metrics_ready"))])
    write_csv(tables / "train_ready_surface_catalog.csv", train_candidate_rows)
    write_csv(tables / "catalog_only_surface_catalog.csv", catalog_only_rows)
    write_csv(tables / "finite_area_surface_catalog.csv", finite_area_rows)
    write_csv(tables / "infinite_area_surface_catalog.csv", infinite_area_rows)
    write_csv(tables / "orbifold_reference_catalog.csv", excluded_rows)
    write_csv(tables / "word_ball_budget_report.csv", word_ball_budget_rows(all_rows))
    write_csv(tables / "duplicate_fingerprint_table.csv", duplicate_rows(all_rows))
    write_csv(tables / "family_summary.csv", summarize_family(all_rows))
    write_csv(tables / "source_runs.csv", source_run_rows(run_dirs))
    write_csv(tables / "artifact_index.csv", artifact_rows)
    write_csv(tables / "ingest_warnings.csv", warnings + rejected_jsons)

    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "contract_version": CONTRACT_VERSION,
        "created": now_stamp(),
        "args": vars(args),
        "input_roots": inputs,
        "source_run_count": len(run_dirs),
        "discovered_surface_files": discovered_files,
        "records_total": len(all_rows),
        "eligible_total": len(eligible_rows),
        "eligible_unique_total": len(primary_eligible_rows),
        "excluded_total": len(excluded_rows),
        "finite_area_eligible_total": len(finite_area_rows),
        "infinite_area_eligible_total": len(infinite_area_rows),
        "full_ginn_second_level_records": len(full_ginn_rows),
        "train_candidate_records_by_word_ball": len(train_candidate_rows),
        "catalog_only_records_by_word_ball": len(catalog_only_rows),
        "artifact_index_rows": len(artifact_rows),
        "warning_count": len(warnings) + len(rejected_jsons),
        "isomorphism_note": "duplicate/fingerprint fields are audit aids only; they are not proofs of Riemann-surface isomorphism",
        "scientific_note": "v3 separates analytic/upstairs geometry, exact downstairs label geometry, and learned downstairs branch geometry for second-level ML.",
    }
    write_json(out_root / "manifest.json", manifest)
    print("-" * 78)
    print(f"[done] records_total={len(all_rows)} eligible={len(eligible_rows)} unique_eligible={len(primary_eligible_rows)} excluded={len(excluded_rows)} warnings={manifest['warning_count']}")
    print(f"[done] full_ginn_second_level_records={len(full_ginn_rows)} train_candidates={len(train_candidate_rows)} catalog_only={len(catalog_only_rows)}")
    print(f"[done] second-level ML features: {tables / 'second_level_ml_features.csv'}")
    print(f"[done] master catalog: {tables / 'master_surface_catalog.csv'}")
    return 0 if all_rows else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Second-level-ML-aware master dataset builder for Fuchsian GENN outputs")
    ap.add_argument("--inputs", nargs="*", default=None, help="Run directories or roots containing run_* directories. Defaults to known tester roots in cwd.")
    ap.add_argument("--all-runs", action="store_true", help="Ingest every run_* directory under each input root instead of only the latest")
    ap.add_argument("--outroot", default="master_dataset_runs", help="Output root for master run")
    ap.add_argument("--label", default="", help="Optional output run label suffix")
    ap.add_argument("--training-word-ball-cap", type=int, default=50000, help="Word-ball cap for ordinary training-candidate classification")
    ap.add_argument("--catalog-word-ball-cap", type=int, default=1000000, help="Larger cap for catalog/label-only classification")
    ap.add_argument("--copy-light-artifacts", action="store_true", help="Copy small GINN artifacts such as metrics.json and branch_atlas_summary.json into the master run")
    ap.add_argument("--copy-heavy-artifacts", action="store_true", help="Also copy heavy artifacts such as candidate_distances.npz and model checkpoints")
    args = ap.parse_args(argv)
    return build_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
