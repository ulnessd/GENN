#!/usr/bin/env python3
"""
Fuchsian_dataset_builder_v1_3.py

Curated batch dataset builder (v1.2 compatibility refresh) for the Fuchsian GENN / downstairs GINN project.

Purpose
-------
Build a structured dataset of Fuchsian/Riemann-surface examples by running
FuchsianDownstairsGINN_v2_4.py on a controlled family set and collecting:

  1. surface metadata and certification/audit data,
  2. exact finite-word downstairs geometry labels,
  3. GINN learned branch-ranking / top-k pruning metrics,
  4. training/runtime diagnostics,
  5. per-surface pair-level tables for second-level ML.

The dataset is organized so a later second-level ML script can load either:

  tables/combined_surface_features.csv       # one row per surface
  pairs/<surface_id>_predictions_test.csv    # one row per held-out test pair
  pairs/<surface_id>_pair_dataset.csv        # all labeled pairs

Important scientific convention
-------------------------------
The exact labels are finite word-ball labels, not global mathematical proofs.
The GINN features are learned branch-atlas / top-k pruning diagnostics, not exact
invariants.  The dataset keeps these categories separate.

Typical use
-----------
    python Fuchsian_dataset_builder_v1_3.py --suite pilot
    python Fuchsian_dataset_builder_v1_3.py --suite curated_ml
    python Fuchsian_dataset_builder_v1_3.py --suite curated_ml --skip-existing

For a small robustness/stress extension:
    python Fuchsian_dataset_builder_v1_3.py --suite curated_ml --include-stress

Dependencies
------------
Requires FuchsianDownstairsGINN_v2_4.py and FuchsianDomainMaker_v13.py to be in
the working directory or available at the default /mnt/data paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Surface suites
# -----------------------------------------------------------------------------

def suite_surfaces(name: str, include_stress: bool = False) -> List[str]:
    """Controlled surface sets for preliminary second-level ML."""
    name = name.lower().strip()

    pilot = [
        "regular_g2",
        "regular_g4",
        "hurwitz",
        "gamma3",
        "gamma5",
        "gamma1_5",
        "hecke_ab5",
        "hecke_d5",
    ]

    compact_core = [f"regular_g{g}" for g in range(2, 7)] + ["hurwitz"]

    modular_clean = [
        # Principal Gamma(N): stop at Gamma(5) for the clean mainline because
        # Gamma(6)+ is larger and should be treated as stress/hard-case data.
        "gamma3", "gamma4", "gamma5",
        # Gamma_1 examples are not truncated in the tested range and provide
        # noncompact variation without very large word balls.
        "gamma1_4", "gamma1_5", "gamma1_6", "gamma1_7", "gamma1_8", "gamma1_10",
    ]

    hecke_core = (
        [f"hecke_ab{q}" for q in [3, 4, 5, 6, 7, 8, 9, 10, 12]] +
        [f"hecke_d{q}" for q in [3, 4, 5, 6, 7, 8, 9, 10, 12]]
    )

    # v1.3: explicit standard suite.  In v1.2, --suite standard fell
    # through to the custom-list branch and was accidentally passed to
    # FuchsianDownstairsGINN as the literal spec "standard", where the
    # GINN engine expanded it internally.  That confused the dataset
    # builder because it expects one surface per subprocess call.
    standard_core = (
        [f"regular_g{g}" for g in range(2, 6)] +
        ["hurwitz"] +
        [f"gamma{N}" for N in range(3, 7)] +
        [f"gamma1_{N}" for N in range(4, 8)] +
        [f"hecke_ab{q}" for q in range(3, 8)] +
        [f"hecke_d{q}" for q in range(3, 8)]
    )

    stress = [
        "gamma6",       # full tokenized generator set; hard but scientifically useful
        "gamma7",       # may be expensive; explicit max-word-ball will decide
        "gamma1_10",    # included in clean set too, but useful as a hard noncompact example
    ]

    if name == "pilot":
        out = pilot
    elif name == "compact":
        out = compact_core
    elif name == "modular":
        out = modular_clean
    elif name == "hecke":
        out = hecke_core
    elif name == "standard":
        out = standard_core
    elif name in {"curated", "curated_ml", "main"}:
        out = compact_core + modular_clean + hecke_core
    elif name in {"all", "mainline"}:
        out = compact_core + modular_clean + hecke_core
    elif name == "stress":
        out = stress
    elif name in {"smoke", "test"}:
        out = ["regular_g2", "hurwitz", "gamma5", "hecke_ab7", "hecke_d7"]
    else:
        # comma-separated custom list
        out = [s.strip() for s in name.split(",") if s.strip()]
        if not out:
            raise ValueError(f"Unknown or empty suite: {name!r}")

    if include_stress and name not in {"stress"}:
        for s in stress:
            if s not in out:
                out.append(s)

    # Preserve order, remove duplicates.
    seen = set()
    deduped = []
    for s in out:
        if s not in seen:
            seen.add(s); deduped.append(s)
    return deduped


def surface_id(spec: str) -> str:
    return spec.lower().replace(" ", "_").replace("/", "_")


# -----------------------------------------------------------------------------
# File and JSON helpers
# -----------------------------------------------------------------------------

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def default_ginn_path() -> str:
    local = Path("FuchsianDownstairsGINN_v2_4.py")
    if local.exists():
        return str(local)
    return "/mnt/data/FuchsianDownstairsGINN_v2_4.py"


def default_maker_path() -> str:
    local = Path("FuchsianDomainMaker_v13.py")
    if local.exists():
        return str(local)
    return "/mnt/data/FuchsianDomainMaker_v13.py"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def as_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def as_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Stable but inclusive field ordering.
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def latest_run_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def zip_or_copy_witnesses(src_run: Path, dst_dir: Path) -> None:
    src = src_run / "geodesic_witnesses"
    if not src.exists():
        return
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src, dst_dir)


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def infer_family(spec: str) -> Tuple[str, str]:
    s = spec.lower()
    if s.startswith("regular_g"):
        return "compact_regular", s.replace("regular_g", "g")
    if s in {"hurwitz", "klein", "klein_quartic"}:
        return "hurwitz_klein", "psl27"
    if s.startswith("gamma1_"):
        return "modular_gamma1", s.split("_", 1)[1]
    if s.startswith("gamma"):
        return "modular_principal_gamma", s.replace("gamma", "")
    if s.startswith("hecke_ab"):
        return "hecke_abelian", s.replace("hecke_ab", "")
    if s.startswith("hecke_d"):
        return "hecke_dihedral", s.replace("hecke_d", "")
    return "custom", s


def extract_surface_rows(spec: str, run_dir: Path, status: str, note: str = "") -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    sid = surface_id(spec)
    fam, param = infer_family(spec)

    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    surface_path = run_dir / "surface.json"
    branch_path = run_dir / "branch_atlas_summary.json"

    manifest: Dict[str, Any] = read_json(manifest_path) if manifest_path.exists() else {}
    metrics: Dict[str, Any] = read_json(metrics_path) if metrics_path.exists() else {}
    surf: Dict[str, Any] = read_json(surface_path) if surface_path.exists() else {}
    branch: Dict[str, Any] = read_json(branch_path) if branch_path.exists() else safe_get(metrics, "branch_atlas_test", {}) or {}

    # Metadata/certification row.
    compact = surf.get("compact")
    if compact is None:
        compact = surf.get("domain_type") == "compact_polygon"

    meta = {
        "surface_id": sid,
        "surface_spec": spec,
        "family": fam,
        "family_parameter": param,
        "status": status,
        "note": note,
        "surface_name": manifest.get("surface_name") or surf.get("name"),
        "domain_type": manifest.get("domain_type") or surf.get("domain_type"),
        "compact": bool(compact),
        "genus": surf.get("genus", manifest.get("genus")),
        "compactified_genus": surf.get("compactified_genus", surf.get("genus", manifest.get("genus"))),
        "area": surf.get("area", manifest.get("area")),
        "gauss_bonnet_area": surf.get("gauss_bonnet_area"),
        "cusp_count": surf.get("cusp_count"),
        "cusp_widths": json.dumps(surf.get("cusp_widths", [])),
        "generator_count": surf.get("generators_count", len(surf.get("generators", {}) or {})),
        "word_ball_size": manifest.get("word_ball_size", safe_get(metrics, "word_ball_size")),
        "word_depth": manifest.get("word_depth"),
        "tokenized_words": True,
        "max_generators": manifest.get("max_generators"),
        "generator_truncated": bool((manifest.get("max_generators") not in [None, 0]) and as_int(manifest.get("max_generators"), 0) > 0),
        "certification_status": safe_get(surf, "certification.status", surf.get("certification_level")),
        "riemann_surface_status": surf.get("riemann_surface_status"),
        "kahler_status": surf.get("kahler_status"),
        "maker_version": surf.get("maker_version"),
        "ginn_program": manifest.get("program"),
    }

    # Exact/algorithmic downstairs geometry row.
    exact = {
        "surface_id": sid,
        "family": fam,
        "word_ball_size": meta["word_ball_size"],
        "word_depth": meta["word_depth"],
        "pairs": manifest.get("pairs"),
        "shortcut_fraction_test": safe_get(metrics, "shortcut_fraction_test"),
        "identity_distance_rmse_vs_quotient": safe_get(metrics, "baseline_identity_test.rmse"),
        "mean_baseline_rmse": safe_get(metrics, "baseline_mean_test.rmse"),
        "exact_distance_gap_mean": branch.get("exact_distance_gap_mean"),
        "exact_distance_gap_median": branch.get("exact_distance_gap_median"),
        "near_seam_fraction_gap_lt_0p02": branch.get("near_seam_fraction_gap_lt_0p02"),
        "near_seam_fraction_gap_lt_0p05": branch.get("near_seam_fraction_gap_lt_0p05"),
        "unique_true_branches_test": branch.get("unique_true_branches_test"),
    }

    # GINN-learned downstairs geometry row.
    ginn = {
        "surface_id": sid,
        "family": fam,
        "profile": manifest.get("profile"),
        "pair_hidden": manifest.get("pair_hidden"),
        "score_hidden": manifest.get("score_hidden"),
        "batch_size": manifest.get("batch_size"),
        "lr": manifest.get("lr"),
        "epochs_requested": manifest.get("epochs"),
        "epochs_ran": metrics.get("epochs_ran"),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_loss": metrics.get("best_val_loss"),
        "candidate_chunk_size": metrics.get("candidate_chunk_size", manifest.get("effective_hyperparameters", {}).get("candidate_chunk_size")),
        "estimated_unchunked_activation_mb": metrics.get("estimated_unchunked_activation_mb"),
        "hard_rmse_test": safe_get(metrics, "test.hard_selected_distance.rmse"),
        "hard_mae_test": safe_get(metrics, "test.hard_selected_distance.mae"),
        "hard_r2_test": safe_get(metrics, "test.hard_selected_distance.r2"),
        "identity_baseline_improvement_fraction": metrics.get("identity_baseline_hard_rmse_improvement_fraction"),
        "winning_lift_accuracy_test": metrics.get("winning_lift_accuracy_test"),
        "winning_lift_exact_equivalent_accuracy_test_tol_1e_5": metrics.get("winning_lift_exact_equivalent_accuracy_test_tol_1e_5"),
        "winning_lift_top3_accuracy_test": metrics.get("winning_lift_top3_accuracy_test"),
        "winning_lift_top5_accuracy_test": metrics.get("winning_lift_top5_accuracy_test"),
        "depth_accuracy_test": metrics.get("depth_accuracy_test"),
        "predicted_shortcut_fraction_test": metrics.get("predicted_shortcut_fraction_test"),
        "learned_branch_entropy_mean": branch.get("learned_branch_entropy_mean"),
        "learned_branch_entropy_median": branch.get("learned_branch_entropy_median"),
        "learned_top1_score_margin_mean": branch.get("learned_top1_score_margin_mean"),
        "learned_top1_score_margin_median": branch.get("learned_top1_score_margin_median"),
        "unique_predicted_branches_test": branch.get("unique_predicted_branches_test"),
        "wall_seconds_total": manifest.get("wall_seconds_total"),
        "cpu_seconds_total": manifest.get("cpu_seconds_total"),
        "rss_mb_final": manifest.get("rss_mb_final"),
    }

    for k in [1, 3, 5, 10, 20]:
        base = f"top{k}"
        tk = safe_get(metrics, f"topk_pruned_search.{base}", {}) or {}
        ginn[f"{base}_recall_true_winner_test"] = tk.get("recall_true_winner_test")
        ginn[f"{base}_pruned_rmse_test"] = safe_get(tk, "pruned_exact_distance_test.rmse")
        ginn[f"{base}_pruned_mae_test"] = safe_get(tk, "pruned_exact_distance_test.mae")
        ginn[f"{base}_pruned_r2_test"] = safe_get(tk, "pruned_exact_distance_test.r2")
        ginn[f"{base}_candidate_fraction_examined"] = tk.get("candidate_fraction_examined")
        ginn[f"{base}_speedup_factor_vs_full_word_ball"] = tk.get("speedup_factor_vs_full_word_ball")
        ginn[f"{base}_selected_word_accuracy_after_pruned_min_test"] = tk.get("selected_word_accuracy_after_pruned_min_test")

    # Combined row is flat for second-level ML.
    combined = {}
    for row in (meta, exact, ginn):
        for key, value in row.items():
            if key in combined and combined[key] == value:
                continue
            if key in combined and combined[key] != value:
                combined[f"ginn_{key}" if row is ginn else f"extra_{key}"] = value
            else:
                combined[key] = value

    return meta, exact, ginn, combined


# -----------------------------------------------------------------------------
# Running GINN and dataset assembly
# -----------------------------------------------------------------------------

def run_ginn_for_surface(args: argparse.Namespace, run_root: Path, spec: str) -> Tuple[str, Optional[Path], int, str]:
    sid = surface_id(spec)
    raw_root = run_root / "raw_ginn_runs" / sid
    raw_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "logs" / f"{sid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        args.ginn,
        "--surface", spec,
        "--maker", args.maker,
        "--outdir", str(raw_root),
        "--profile", args.profile,
        "--depth", str(args.depth),
        "--max-word-ball", str(args.max_word_ball),
        "--max-generators", str(args.max_generators),
        "--max-cosets", str(args.max_cosets),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    # Let profile choose pairs/hyperparameters by default; allow explicit override.
    if args.pairs is not None:
        cmd += ["--pairs", str(args.pairs)]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.candidate_chunk_size is not None:
        cmd += ["--candidate-chunk-size", str(args.candidate_chunk_size)]
    if args.no_train:
        cmd.append("--no-train")

    if args.dry_run:
        return spec, None, 0, "DRY_RUN: " + " ".join(cmd)

    print(f"[dataset] START {spec} -> {raw_root}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("COMMAND: " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    wall = time.time() - t0
    rd = latest_run_dir(raw_root)
    if proc.returncode == 0 and rd is not None:
        print(f"[dataset] PASS  {spec:12s} wall={wall:8.1f}s run={rd.name}", flush=True)
        return spec, rd, 0, "PASS"
    print(f"[dataset] FAIL  {spec:12s} wall={wall:8.1f}s returncode={proc.returncode}; see {log_path}", flush=True)
    return spec, rd, proc.returncode, f"FAIL returncode={proc.returncode}; log={log_path}"


def assemble_surface_outputs(run_root: Path, spec: str, run_dir: Path, status: str, note: str) -> Dict[str, Any]:
    sid = surface_id(spec)
    # Stable dataset-facing copies for easy loading.
    surf_dir = run_root / "surfaces" / sid
    pair_dir = run_root / "pairs"
    pred_dir = run_root / "predictions"
    witness_dir = run_root / "witnesses" / sid
    train_dir = run_root / "training_logs"
    model_dir = run_root / "models"

    for d in [surf_dir, pair_dir, pred_dir, train_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    copy_if_exists(run_dir / "surface.json", surf_dir / "surface.json")
    copy_if_exists(run_dir / "run_manifest.json", surf_dir / "run_manifest.json")
    copy_if_exists(run_dir / "metrics.json", surf_dir / "metrics.json")
    copy_if_exists(run_dir / "branch_atlas_summary.json", surf_dir / "branch_atlas_summary.json")
    copy_if_exists(run_dir / "word_ball.json", surf_dir / "word_ball.json")
    copy_if_exists(run_dir / "candidate_distances.npz", surf_dir / "candidate_distances.npz")
    copy_if_exists(run_dir / "pair_dataset.csv", pair_dir / f"{sid}_pair_dataset.csv")
    copy_if_exists(run_dir / "predictions_test.csv", pred_dir / f"{sid}_predictions_test.csv")
    copy_if_exists(run_dir / "train_log.csv", train_dir / f"{sid}_train_log.csv")
    copy_if_exists(run_dir / "downstairs_ginn_v2_4.pt", model_dir / f"{sid}_downstairs_ginn_v2_4.pt")
    zip_or_copy_witnesses(run_dir, witness_dir)

    meta, exact, ginn, combined = extract_surface_rows(spec, run_dir, status=status, note=note)
    return {"metadata": meta, "exact": exact, "ginn": ginn, "combined": combined}


def write_schema(run_root: Path) -> None:
    schema = {
        "dataset_builder": "Fuchsian_dataset_builder_v1_3.py",
        "principle": "Separate exact finite-word geometry, GINN-learned branch-atlas features, and training diagnostics.",
        "main_tables": {
            "tables/surface_metadata.csv": "One row per surface: family, topology, area, cusps, certification, generator/word-ball metadata.",
            "tables/exact_downstairs_features.csv": "One row per surface: finite-search quotient labels summarized, shortcut fraction, seam/gap statistics.",
            "tables/ginn_surface_features.csv": "One row per surface: GINN branch-ranking, top-k pruning, entropy, training/runtime metrics.",
            "tables/combined_surface_features.csv": "Flat one-row-per-surface table for second-level ML.",
            "predictions/<surface>_predictions_test.csv": "Pair-level held-out test rows with exact labels and GINN predictions/top-k sets.",
            "pairs/<surface>_pair_dataset.csv": "All generated point-pair labels for the surface.",
        },
        "caveats": [
            "Distances are finite word-ball labels unless otherwise certified.",
            "Graphical/GUI Explorer may not parse tokenized generator labels; v2.4 is the batch/GINN backend.",
            "Principal Gamma(N) for large N can exceed the comfortable word-ball range; explicit max-word-ball controls this.",
            "Do not interpret top-1 word accuracy alone for noncompact/Ford domains with many exact-equivalent branches; top-k pruned exact distance is often the correct headline metric.",
        ],
    }
    with (run_root / "feature_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build a curated Fuchsian/Riemann-surface dataset using the v2.4 downstairs GINN engine.")
    p.add_argument("--suite", default="curated_ml", help="pilot, smoke, standard, compact, modular, hecke, curated_ml, stress, or comma-separated surface list")
    p.add_argument("--include-stress", action="store_true", help="Add stress cases such as full-tokenized Gamma(6)/Gamma(7)")
    p.add_argument("--outdir", default="fuchsian_dataset_runs", help="Dataset run root directory")
    p.add_argument("--name", default="", help="Optional dataset name suffix")
    p.add_argument("--ginn", default=default_ginn_path(), help="Path to FuchsianDownstairsGINN_v2_4.py")
    p.add_argument("--maker", default=default_maker_path(), help="Path to FuchsianDomainMaker_v13.py")
    p.add_argument("--profile", choices=["balanced", "fast", "accurate", "manual"], default="balanced")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--max-word-ball", type=int, default=50000)
    p.add_argument("--max-generators", type=int, default=0, help="0 means no generator cap; any cap is explicit and recorded")
    p.add_argument("--max-cosets", type=int, default=20000)
    p.add_argument("--pairs", type=int, default=None, help="Override profile pair count; omit to let v2.4 choose")
    p.add_argument("--epochs", type=int, default=None, help="Override profile epochs; omit to let v2.4 choose")
    p.add_argument("--candidate-chunk-size", type=int, default=None, help="Override profile chunking; omit for auto/profile")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=24680)
    p.add_argument("--skip-existing", action="store_true", help="Skip a surface if its stable copied metrics already exist in this dataset run")
    p.add_argument("--no-train", action="store_true", help="Only generate exact labels/word balls; no GINN training")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    surfaces = suite_surfaces(args.suite, include_stress=args.include_stress)
    stamp = now_stamp()
    suffix = args.name.strip() or args.suite.lower().replace(",", "_")
    run_root = Path(args.outdir) / f"fuchsian_dataset_{stamp}_{suffix}"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "tables").mkdir(exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "Fuchsian_dataset_builder_v1_3.py",
        "suite": args.suite,
        "include_stress": args.include_stress,
        "surfaces": surfaces,
        "num_surfaces": len(surfaces),
        "ginn_path": args.ginn,
        "maker_path": args.maker,
        "profile": args.profile,
        "depth": args.depth,
        "max_word_ball": args.max_word_ball,
        "max_generators": args.max_generators,
        "max_cosets": args.max_cosets,
        "device": args.device,
        "seed": args.seed,
        "folder_structure": {
            "surfaces/": "stable per-surface metadata/metrics/word-ball copies",
            "pairs/": "all pair-label CSVs, one file per surface",
            "predictions/": "held-out test prediction CSVs, one file per surface",
            "training_logs/": "training logs, one file per surface",
            "raw_ginn_runs/": "complete raw v2.4 run folders",
            "witnesses/": "geodesic witness JSON files",
            "tables/": "surface-level ML-ready tables",
        },
    }
    with (run_root / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_schema(run_root)

    print("=" * 78, flush=True)
    print("Fuchsian dataset builder v1.3", flush=True)
    print(f"suite={args.suite} surfaces={len(surfaces)} profile={args.profile} depth={args.depth}", flush=True)
    print(f"run_root={run_root}", flush=True)
    print("Surfaces: " + ", ".join(surfaces), flush=True)
    print("=" * 78, flush=True)

    metadata_rows: List[Dict[str, Any]] = []
    exact_rows: List[Dict[str, Any]] = []
    ginn_rows: List[Dict[str, Any]] = []
    combined_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    for i, spec in enumerate(surfaces, start=1):
        print(f"\n[{i}/{len(surfaces)}] {spec}", flush=True)
        if args.skip_existing and (run_root / "surfaces" / surface_id(spec) / "metrics.json").exists():
            print(f"[dataset] SKIP existing {spec}", flush=True)
            # Not implemented: re-extract skipped rows. Simple explicit skip.
            continue
        spec_name, rd, rc, note = run_ginn_for_surface(args, run_root, spec)
        if args.dry_run:
            print(note, flush=True)
            continue
        if rc != 0 or rd is None or not (rd / "metrics.json").exists():
            failure_rows.append({"surface_id": surface_id(spec), "surface_spec": spec, "returncode": rc, "note": note, "raw_run_dir": str(rd) if rd else ""})
            continue
        try:
            rows = assemble_surface_outputs(run_root, spec, rd, status="PASS", note=note)
            metadata_rows.append(rows["metadata"])
            exact_rows.append(rows["exact"])
            ginn_rows.append(rows["ginn"])
            combined_rows.append(rows["combined"])
            # Incremental writes so partial datasets survive interruption.
            write_csv(run_root / "tables" / "surface_metadata.csv", metadata_rows)
            write_csv(run_root / "tables" / "exact_downstairs_features.csv", exact_rows)
            write_csv(run_root / "tables" / "ginn_surface_features.csv", ginn_rows)
            write_csv(run_root / "tables" / "combined_surface_features.csv", combined_rows)
            write_csv(run_root / "tables" / "failures.csv", failure_rows)
        except Exception as e:
            failure_rows.append({"surface_id": surface_id(spec), "surface_spec": spec, "returncode": rc, "note": f"assembly error: {e}", "raw_run_dir": str(rd)})
            write_csv(run_root / "tables" / "failures.csv", failure_rows)
            print(f"[dataset] ASSEMBLY ERROR {spec}: {e}", flush=True)

    if not args.dry_run:
        write_csv(run_root / "tables" / "surface_metadata.csv", metadata_rows)
        write_csv(run_root / "tables" / "exact_downstairs_features.csv", exact_rows)
        write_csv(run_root / "tables" / "ginn_surface_features.csv", ginn_rows)
        write_csv(run_root / "tables" / "combined_surface_features.csv", combined_rows)
        write_csv(run_root / "tables" / "failures.csv", failure_rows)
        completion = {
            "completed_surfaces": [r["surface_spec"] for r in metadata_rows],
            "failed_surfaces": failure_rows,
            "num_completed": len(metadata_rows),
            "num_failed": len(failure_rows),
            "tables": [
                "tables/surface_metadata.csv",
                "tables/exact_downstairs_features.csv",
                "tables/ginn_surface_features.csv",
                "tables/combined_surface_features.csv",
                "tables/failures.csv",
            ],
        }
        with (run_root / "completion_summary.json").open("w", encoding="utf-8") as f:
            json.dump(completion, f, indent=2)
        print("\n" + "=" * 78, flush=True)
        print(f"[dataset] completed={len(metadata_rows)} failed={len(failure_rows)}", flush=True)
        print(f"[dataset] ML-ready table: {run_root / 'tables' / 'combined_surface_features.csv'}", flush=True)
        print(f"[dataset] pair-level data: {run_root / 'predictions'}", flush=True)
        print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
