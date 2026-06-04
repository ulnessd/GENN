#!/usr/bin/env python3
"""
FuchsianGINNParamExplorer_v1_2.py

Parameter explorer v1.2 for FuchsianDownstairsGINN_v2_1.py.

Purpose
-------
Runs a representative compact/noncompact parameter study for the geometry-informed
branch-ranker GINN with word-search depth fixed at 2.  The goal is not just to find a single best RMSE, but to
measure time/accuracy/precision tradeoffs for the learned downstairs quotient
branch structure.

It calls FuchsianDownstairsGINN_v2_1.py as an engine and summarizes:
  - hard selected downstairs-distance RMSE/R2
  - winning lift accuracy and exact-equivalent accuracy
  - top-3/top-5 recall
  - top-k pruned-search RMSE and speedup vs full word ball
  - near-seam fraction and branch entropy
  - wall/cpu/runtime diagnostics

Typical use
-----------
Dry run:
    python FuchsianGINNParamExplorer_v1_2.py --preset standard --dry-run

Run two trials at a time after the v1.1 output-isolation fix:
    python FuchsianGINNParamExplorer_v1_2.py --preset standard --parallel-trials 2

More compact/fast:
    python FuchsianGINNParamExplorer_v1_2.py --preset quick

Larger overnight:
    python FuchsianGINNParamExplorer_v1_2.py --preset overnight --parallel-trials 2

Outputs
-------
ginn_param_explorer_runs/run_YYYYMMDD_HHMMSS_<preset>/
    explorer_manifest.json
    trial_summary.csv
    best_by_surface.csv
    best_by_config.csv
    compact_vs_noncompact_summary.csv
    commands.sh
    logs/
    raw_runs/

Notes
-----
This explorer intentionally includes both compact and noncompact surfaces.
The GINN v2.1 engine is per-known-surface: the model is surface-aware through
its candidate deck transformations, lifted endpoints, and word metadata rather
than through a pooled compact/noncompact flag.

v1.2 keeps the v1.1 stability fixes and removes all depth-3 configurations:
  1. every trial gets its own unique output root, preventing parallel workers
     for the same surface from writing into the same timestamp/surface folder;
  2. metrics are collected from the exact trial output root, not by guessing
     a suffix pattern such as *_surface;
  3. depth is fixed at 2, so the sweep now focuses on width, score network
     size, pair count, learning rate, and batch size. Wide settings use smaller
     batches to avoid VRAM spikes on larger noncompact word balls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
import resource
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def slugify(text: str, max_len: int = 96) -> str:
    """Filesystem-safe short label for trial directories and log files."""
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return (out or "x")[:max_len]


def trial_output_root(raw_root: Path, idx: int, surface: str, config: str) -> Path:
    # Unique per trial. This prevents parallel child processes for the same
    # surface from colliding in timestamp/surface-named engine folders.
    return raw_root / f"trial_{idx:04d}_{slugify(surface)}_{slugify(config, 72)}"


def rss_mb() -> float:
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 2)
        except Exception:
            pass
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def cpu_seconds() -> float:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return float(ru.ru_utime + ru.ru_stime)
    except Exception:
        return float("nan")


def default_engine_path() -> str:
    candidates = [
        Path("FuchsianDownstairsGINN_v2_1.py"),
        Path("/mnt/data/FuchsianDownstairsGINN_v2_1.py"),
        Path.home() / "PycharmProjects" / "GENN" / "FuchsianDownstairsGINN_v2_1.py",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "FuchsianDownstairsGINN_v2_1.py"


def default_maker_path() -> str:
    candidates = [
        Path("FuchsianDomainMaker_v13.py"),
        Path("/mnt/data/FuchsianDomainMaker_v13.py"),
        Path.home() / "PycharmProjects" / "GENN" / "FuchsianDomainMaker_v13.py",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "FuchsianDomainMaker_v13.py"


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    kind: str       # compact or noncompact
    family: str
    reason: str


@dataclass(frozen=True)
class TrialConfig:
    name: str
    pairs: int
    depth: int
    epochs: int
    patience: int
    pair_hidden: int
    score_hidden: int
    batch_size: int
    lr: float
    ce_weight: float = 1.0
    soft_distance_weight: float = 0.2
    temperature: float = 1.0


REPRESENTATIVE_SURFACES: List[SurfaceSpec] = [
    SurfaceSpec("regular_g2", "compact", "regular_polygon", "small compact; previously among harder compact cases"),
    SurfaceSpec("regular_g4", "compact", "regular_polygon", "medium compact regular polygon"),
    SurfaceSpec("hurwitz", "compact", "hurwitz_klein", "certified PSL(2,7) genus-3 special compact case"),
    SurfaceSpec("gamma4", "noncompact", "principal_congruence", "moderate modular cusped case"),
    SurfaceSpec("gamma6", "noncompact", "principal_congruence", "harder modular cusped case"),
    SurfaceSpec("gamma1_5", "noncompact", "gamma1", "representative Gamma_1 cusped case"),
    SurfaceSpec("hecke_ab7", "noncompact", "hecke_abelian", "harder abelian Hecke cover"),
    SurfaceSpec("hecke_d5", "noncompact", "hecke_dihedral", "representative nonabelian/dihedral Hecke cover"),
    SurfaceSpec("hecke_d7", "noncompact", "hecke_dihedral", "harder nonabelian/dihedral Hecke cover"),
]

QUICK_SURFACES = [s for s in REPRESENTATIVE_SURFACES if s.name in {"regular_g2", "hurwitz", "gamma6", "hecke_ab7", "hecke_d7"}]
OVERNIGHT_EXTRA_SURFACES: List[SurfaceSpec] = [
    SurfaceSpec("regular_g5", "compact", "regular_polygon", "larger compact regular polygon"),
    SurfaceSpec("gamma5", "noncompact", "principal_congruence", "diagnostic modular cusped case"),
    SurfaceSpec("hecke_ab6", "noncompact", "hecke_abelian", "diagnostic abelian Hecke cover"),
    SurfaceSpec("hecke_d6", "noncompact", "hecke_dihedral", "diagnostic dihedral Hecke cover"),
]


def standard_configs() -> List[TrialConfig]:
    """Depth-2-only grid around the successful GINN v2.1 settings.

    Depth 3 was removed after the quick experiments showed much larger word
    balls, runtime, and VRAM pressure without clear accuracy improvement. This
    grid focuses on the remaining tradeoffs: pair count, context width,
    candidate-score width, learning rate, and batch size.
    """
    return [
        TrialConfig("base_p9000_d2_ph256_sh128_b256_lr1e-3", 9000, 2, 220, 40, 256, 128, 256, 1.0e-3),
        TrialConfig("widepair_p9000_d2_ph384_sh128_b256_lr7e-4", 9000, 2, 240, 45, 384, 128, 256, 7.0e-4),
        TrialConfig("widescore_p9000_d2_ph256_sh192_b256_lr1e-3", 9000, 2, 240, 45, 256, 192, 256, 1.0e-3),
        # Wide-both can hit VRAM limits on large noncompact word balls at batch 256.
        TrialConfig("wideboth_p9000_d2_ph384_sh192_b128_lr7e-4", 9000, 2, 250, 45, 384, 192, 128, 7.0e-4),
        TrialConfig("smallfast_p9000_d2_ph192_sh96_b256_lr1p5e-3", 9000, 2, 220, 40, 192, 96, 256, 1.5e-3),
        TrialConfig("xwide_p9000_d2_ph512_sh192_b128_lr5e-4", 9000, 2, 280, 55, 512, 192, 128, 5.0e-4),
        TrialConfig("morepairs_p14000_d2_ph256_sh128_b256_lr1e-3", 14000, 2, 240, 45, 256, 128, 256, 1.0e-3),
        TrialConfig("morepairs_wide_p14000_d2_ph384_sh192_b128_lr7e-4", 14000, 2, 270, 55, 384, 192, 128, 7.0e-4),
        TrialConfig("highpairs_p20000_d2_ph256_sh128_b256_lr8e-4", 20000, 2, 270, 55, 256, 128, 256, 8.0e-4),
    ]

def quick_configs() -> List[TrialConfig]:
    # Four depth-2 configs: a fast sanity check for base, width, and pair-count effects.
    return [
        TrialConfig("base_p6000_d2_ph256_sh128_b256_lr1e-3", 6000, 2, 140, 25, 256, 128, 256, 1.0e-3),
        TrialConfig("widepair_p6000_d2_ph384_sh128_b256_lr7e-4", 6000, 2, 160, 30, 384, 128, 256, 7.0e-4),
        TrialConfig("wideboth_p6000_d2_ph384_sh192_b128_lr7e-4", 6000, 2, 170, 30, 384, 192, 128, 7.0e-4),
        TrialConfig("morepairs_p10000_d2_ph256_sh128_b256_lr1e-3", 10000, 2, 170, 35, 256, 128, 256, 1.0e-3),
    ]

def overnight_configs() -> List[TrialConfig]:
    # Larger depth-2-only sweep. Includes pair-count and width sweeps while
    # avoiding depth-3 word-ball explosions.
    cfgs = standard_configs()
    cfgs += [
        TrialConfig("lowpairs_p4000_d2_ph256_sh128_b256_lr1e-3", 4000, 2, 180, 35, 256, 128, 256, 1.0e-3),
        TrialConfig("midpairs_p6000_d2_ph256_sh128_b256_lr1e-3", 6000, 2, 200, 40, 256, 128, 256, 1.0e-3),
        TrialConfig("pair320_p12000_d2_ph320_sh128_b256_lr8e-4", 12000, 2, 250, 50, 320, 128, 256, 8.0e-4),
        TrialConfig("pair448_p12000_d2_ph448_sh160_b128_lr6e-4", 12000, 2, 280, 55, 448, 160, 128, 6.0e-4),
        TrialConfig("score256_p12000_d2_ph384_sh256_b96_lr5e-4", 12000, 2, 300, 60, 384, 256, 96, 5.0e-4),
        TrialConfig("veryhighpairs_p30000_d2_ph256_sh128_b256_lr6e-4", 30000, 2, 300, 65, 256, 128, 256, 6.0e-4),
    ]
    return cfgs

def choose_surfaces(preset: str, surfaces_arg: str) -> List[SurfaceSpec]:
    all_known = {s.name: s for s in REPRESENTATIVE_SURFACES + OVERNIGHT_EXTRA_SURFACES}
    if surfaces_arg.strip():
        out = []
        for name in [x.strip() for x in surfaces_arg.split(',') if x.strip()]:
            if name in all_known:
                out.append(all_known[name])
            else:
                # Infer compactness/family for custom surface names.
                kind = "compact" if name.startswith("regular_g") or name in {"hurwitz", "klein"} else "noncompact"
                fam = "custom"
                if name.startswith("gamma1_"):
                    fam = "gamma1"
                elif name.startswith("gamma"):
                    fam = "principal_congruence"
                elif name.startswith("hecke_ab"):
                    fam = "hecke_abelian"
                elif name.startswith("hecke_d"):
                    fam = "hecke_dihedral"
                elif name.startswith("regular_g"):
                    fam = "regular_polygon"
                elif name == "hurwitz":
                    fam = "hurwitz_klein"
                out.append(SurfaceSpec(name, kind, fam, "custom user-specified surface"))
        return out
    if preset == "quick":
        return QUICK_SURFACES
    if preset == "overnight":
        # Deduplicate while preserving order.
        seen = set(); out = []
        for s in REPRESENTATIVE_SURFACES + OVERNIGHT_EXTRA_SURFACES:
            if s.name not in seen:
                out.append(s); seen.add(s.name)
        return out
    return REPRESENTATIVE_SURFACES


def choose_configs(preset: str) -> List[TrialConfig]:
    if preset == "quick":
        return quick_configs()
    if preset == "overnight":
        return overnight_configs()
    return standard_configs()


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


def nested_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def build_command(engine: str, maker: str, surface: str, cfg: TrialConfig, out_root: Path, seed: int, device: str) -> List[str]:
    return [
        sys.executable, engine,
        "--surface", surface,
        "--maker", maker,
        "--outdir", str(out_root),
        "--pairs", str(cfg.pairs),
        "--depth", str(cfg.depth),
        "--epochs", str(cfg.epochs),
        "--patience", str(cfg.patience),
        "--pair-hidden", str(cfg.pair_hidden),
        "--score-hidden", str(cfg.score_hidden),
        "--batch-size", str(cfg.batch_size),
        "--lr", str(cfg.lr),
        "--ce-weight", str(cfg.ce_weight),
        "--soft-distance-weight", str(cfg.soft_distance_weight),
        "--temperature", str(cfg.temperature),
        "--seed", str(seed),
        "--device", device,
    ]


def find_run_dir(trial_root: Path) -> Optional[Path]:
    """Find the engine-created run directory inside one isolated trial root.

    GINN v2.1 creates a timestamped directory below --outdir. In v1 we tried
    to infer it from suffixes in a shared raw_runs directory; that was fragile
    and failed for names like *_ginn_v2. v1.1 gives each trial its own outdir,
    so the correct run directory is simply the descendant containing
    metrics.json.
    """
    if not trial_root.exists():
        return None
    candidates: List[Tuple[float, Path]] = []
    for m in trial_root.rglob("metrics.json"):
        try:
            candidates.append((m.stat().st_mtime, m.parent))
        except Exception:
            candidates.append((0.0, m.parent))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row["hard_rmse"] = nested_get(metrics, ["test", "hard_selected_distance", "rmse"])
    row["hard_mae"] = nested_get(metrics, ["test", "hard_selected_distance", "mae"])
    row["hard_r2"] = nested_get(metrics, ["test", "hard_selected_distance", "r2"])
    row["identity_rmse"] = nested_get(metrics, ["baseline_identity_test", "rmse"])
    row["mean_baseline_rmse"] = nested_get(metrics, ["baseline_mean_test", "rmse"])
    row["identity_improvement_fraction"] = metrics.get("identity_baseline_hard_rmse_improvement_fraction")
    row["winner_acc"] = metrics.get("winning_lift_accuracy_test")
    row["winner_exact_equiv_acc_tol_1e_5"] = metrics.get("winning_lift_exact_equivalent_accuracy_test_tol_1e_5")
    row["top3_acc"] = metrics.get("winning_lift_top3_accuracy_test")
    row["top5_acc"] = metrics.get("winning_lift_top5_accuracy_test")
    row["depth_acc"] = metrics.get("depth_accuracy_test")
    row["shortcut_fraction"] = metrics.get("shortcut_fraction_test")
    row["predicted_shortcut_fraction"] = metrics.get("predicted_shortcut_fraction_test")
    row["word_ball_size"] = metrics.get("word_ball_size")
    row["epochs_ran"] = metrics.get("epochs_ran")
    row["best_epoch"] = metrics.get("best_epoch")
    row["best_val_loss"] = metrics.get("best_val_loss")
    ba = metrics.get("branch_atlas_test", {}) or {}
    row["branch_entropy_mean"] = ba.get("learned_branch_entropy_mean")
    row["branch_entropy_median"] = ba.get("learned_branch_entropy_median")
    row["score_margin_mean"] = ba.get("learned_top1_score_margin_mean")
    row["score_margin_median"] = ba.get("learned_top1_score_margin_median")
    row["exact_gap_mean"] = ba.get("exact_distance_gap_mean")
    row["exact_gap_median"] = ba.get("exact_distance_gap_median")
    row["near_seam_frac_gap_lt_0p02"] = ba.get("near_seam_fraction_gap_lt_0p02")
    row["near_seam_frac_gap_lt_0p05"] = ba.get("near_seam_fraction_gap_lt_0p05")
    row["unique_true_branches_test"] = ba.get("unique_true_branches_test")
    row["unique_predicted_branches_test"] = ba.get("unique_predicted_branches_test")
    tk = metrics.get("topk_pruned_search", {}) or {}
    for k in [1, 3, 5, 10]:
        kk = f"top{k}"
        if kk in tk:
            row[f"{kk}_pruned_rmse"] = nested_get(tk, [kk, "pruned_exact_distance_test", "rmse"])
            row[f"{kk}_pruned_mae"] = nested_get(tk, [kk, "pruned_exact_distance_test", "mae"])
            row[f"{kk}_contains_true"] = tk[kk].get("contains_true_winner_test")
            row[f"{kk}_candidate_fraction"] = tk[kk].get("candidate_fraction_examined")
            row[f"{kk}_speedup"] = tk[kk].get("speedup_factor_vs_full_word_ball")
            row[f"{kk}_selected_word_acc"] = tk[kk].get("selected_word_accuracy_after_pruned_min_test")
    return row


def run_trial(idx: int, total: int, surface: SurfaceSpec, cfg: TrialConfig, engine: str, maker: str,
              raw_root: Path, log_root: Path, base_seed: int, device: str, dry_run: bool = False) -> Dict[str, Any]:
    seed = base_seed + 100000 * idx + (abs(hash(surface.name + cfg.name)) % 10000)
    trial_root = trial_output_root(raw_root, idx, surface.name, cfg.name)
    trial_root.mkdir(parents=True, exist_ok=True)
    cmd = build_command(engine, maker, surface.name, cfg, trial_root, seed, device)
    row: Dict[str, Any] = {
        "trial_index": idx,
        "total_trials": total,
        "surface": surface.name,
        "surface_kind": surface.kind,
        "surface_family": surface.family,
        "surface_reason": surface.reason,
        "config": cfg.name,
        **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
        "seed": seed,
        "trial_root": str(trial_root),
        "command": " ".join(shlex.quote(x) for x in cmd),
    }
    if dry_run:
        row.update({"status": "DRY_RUN"})
        return row

    log_path = log_root / f"trial_{idx:04d}_{slugify(surface.name)}_{slugify(cfg.name, 96)}.log"
    print(f"[trial {idx}/{total}] START {surface.name:12s} {cfg.name}", flush=True)
    t0 = time.time(); c0 = cpu_seconds(); rss0 = rss_mb()
    status = "unknown"; returncode = -999
    try:
        with log_path.open("w") as log:
            log.write("# Command:\n" + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            returncode = proc.returncode
        status = "PASS" if returncode == 0 else "FAIL"
    except KeyboardInterrupt:
        raise
    except Exception as e:
        status = "ERROR"
        row["error"] = repr(e)

    wall = time.time() - t0; cpu = cpu_seconds() - c0
    row.update({
        "status": status,
        "returncode": returncode,
        "wall_seconds": wall,
        "cpu_seconds_parent": cpu,
        "rss_mb_start_parent": rss0,
        "rss_mb_end_parent": rss_mb(),
        "log_path": str(log_path),
    })
    run_dir = find_run_dir(trial_root)
    row["run_dir"] = str(run_dir) if run_dir else ""
    try:
        if run_dir is not None and (run_dir / "metrics.json").exists():
            metrics = read_json(run_dir / "metrics.json")
            row.update(flatten_metrics(metrics))
            # Prefer engine runtime if present in run_manifest.
            man_path = run_dir / "run_manifest.json"
            if man_path.exists():
                man = read_json(man_path)
                perf = man.get("performance", {}) or {}
                for key in ["wall_seconds", "cpu_seconds", "rss_mb"]:
                    if key in perf:
                        row[f"engine_{key}"] = perf[key]
        else:
            row["metrics_error"] = "metrics.json not found"
    except Exception as e:
        row["metrics_error"] = repr(e)

    rmse = row.get("hard_rmse")
    top5 = row.get("top5_acc")
    speed = row.get("top5_speedup")
    print(f"[trial {idx}/{total}] {status:5s} {surface.name:12s} {cfg.name:42s} wall={wall:7.1f}s rmse={rmse} top5={top5} speedup={speed}", flush=True)
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    # Stable important fields first.
    preferred = [
        "trial_index", "status", "surface", "surface_kind", "surface_family", "config",
        "cfg_pairs", "cfg_depth", "cfg_pair_hidden", "cfg_score_hidden", "cfg_batch_size", "cfg_lr",
        "wall_seconds", "engine_wall_seconds", "hard_rmse", "hard_mae", "hard_r2", "identity_rmse",
        "winner_acc", "winner_exact_equiv_acc_tol_1e_5", "top3_acc", "top5_acc", "depth_acc",
        "top5_pruned_rmse", "top5_speedup", "branch_entropy_mean", "near_seam_frac_gap_lt_0p02",
    ]
    for f in preferred:
        for r in rows:
            if f in r and f not in seen:
                fields.append(f); seen.add(f); break
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def numeric(v: Any) -> Optional[float]:
    return safe_float(v)


def summarize_best(rows: List[Dict[str, Any]], key_field: str, out_path: Path) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") == "PASS" and numeric(r.get("hard_rmse")) is not None:
            groups.setdefault(str(r.get(key_field)), []).append(r)
    best: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items()):
        rs2 = sorted(rs, key=lambda r: (float(r.get("hard_rmse")), -float(r.get("top5_acc") or 0.0), float(r.get("wall_seconds") or 1e99)))
        b = dict(rs2[0])
        b["best_group"] = key
        best.append(b)
    write_csv(out_path, best)
    return best


def summarize_compact_noncompact(rows: List[Dict[str, Any]], out_path: Path) -> List[Dict[str, Any]]:
    # Aggregate by surface_kind and config.
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") == "PASS" and numeric(r.get("hard_rmse")) is not None:
            buckets.setdefault((str(r.get("surface_kind")), str(r.get("config"))), []).append(r)
    summary: List[Dict[str, Any]] = []
    for (kind, cfg), rs in sorted(buckets.items()):
        def vals(field: str) -> List[float]:
            return [float(r[field]) for r in rs if numeric(r.get(field)) is not None]
        rm = vals("hard_rmse"); r2 = vals("hard_r2"); t5 = vals("top5_acc"); wall = vals("wall_seconds")
        row = {
            "surface_kind": kind,
            "config": cfg,
            "n": len(rs),
            "mean_hard_rmse": sum(rm)/len(rm) if rm else None,
            "median_hard_rmse": sorted(rm)[len(rm)//2] if rm else None,
            "mean_hard_r2": sum(r2)/len(r2) if r2 else None,
            "mean_top5_acc": sum(t5)/len(t5) if t5 else None,
            "mean_wall_seconds": sum(wall)/len(wall) if wall else None,
            "mean_top5_speedup": (sum(vals("top5_speedup"))/len(vals("top5_speedup"))) if vals("top5_speedup") else None,
        }
        # A simple Pareto score: accuracy per sqrt time, lower is better.
        if row["mean_hard_rmse"] is not None and row["mean_wall_seconds"]:
            row["rmse_sqrt_time_score"] = row["mean_hard_rmse"] * math.sqrt(row["mean_wall_seconds"])
        summary.append(row)
    write_csv(out_path, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Parameter explorer for FuchsianDownstairsGINN_v2_1.py")
    ap.add_argument("--preset", choices=["quick", "standard", "overnight"], default="standard")
    ap.add_argument("--engine", default=default_engine_path(), help="Path to FuchsianDownstairsGINN_v2_1.py")
    ap.add_argument("--maker", default=default_maker_path(), help="Path to FuchsianDomainMaker_v13.py")
    ap.add_argument("--outdir", default="ginn_param_explorer_runs")
    ap.add_argument("--surfaces", default="", help="Optional comma-separated surface list overriding preset")
    ap.add_argument("--parallel-trials", type=int, default=1, help="Run this many independent trials concurrently")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda passed to GINN engine")
    ap.add_argument("--base-seed", type=int, default=91000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    engine = str(Path(args.engine).expanduser())
    maker = str(Path(args.maker).expanduser())
    surfaces = choose_surfaces(args.preset, args.surfaces)
    configs = choose_configs(args.preset)
    run_root = Path(args.outdir) / f"run_{now_stamp()}_{args.preset}"
    raw_root = run_root / "raw_runs"
    log_root = run_root / "logs"
    raw_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "Representative compact/noncompact parameter explorer for FuchsianDownstairsGINN_v2_1 geometry-informed branch ranker.",
        "preset": args.preset,
        "engine": engine,
        "maker": maker,
        "surfaces": [asdict(s) for s in surfaces],
        "configs": [asdict(c) for c in configs],
        "num_surfaces": len(surfaces),
        "num_configs": len(configs),
        "num_trials": len(surfaces) * len(configs),
        "parallel_trials": args.parallel_trials,
        "device": args.device,
        "base_seed": args.base_seed,
        "python": sys.version,
        "platform": platform.platform(),
        "psutil_available": psutil is not None,
        "run_root": str(run_root),
        "dry_run": bool(args.dry_run),
        "interpretation_note": "Model is geometry-informed because it ranks deck-transformation candidate branches using lifted endpoints and word metadata, not raw scalar distance regression. It is per-known-surface; compact/noncompact awareness enters through the candidate structure rather than an explicit pooled flag.",
        "v1_2_design_notes": [
            "per-trial isolated --outdir roots under raw_runs/trial_####_surface_config",
            "metrics collection searches only the isolated trial root for metrics.json",
            "log filenames are slugified to avoid shell/path surprises",
            "all configs use depth=2; depth-3 was removed after testing showed large runtime/memory cost without accuracy gains",
            "wide/high-capacity configs use batch_size=128 to reduce VRAM pressure on large noncompact word balls"
        ],
    }
    with (run_root / "explorer_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    # Command script for reproducibility.
    commands: List[str] = []
    total = len(surfaces) * len(configs)
    trial_inputs = []
    idx = 0
    for s in surfaces:
        for c in configs:
            idx += 1
            seed = args.base_seed + 100000 * idx + (abs(hash(s.name + c.name)) % 10000)
            troot = trial_output_root(raw_root, idx, s.name, c.name)
            cmd = build_command(engine, maker, s.name, c, troot, seed, args.device)
            commands.append(" ".join(shlex.quote(x) for x in cmd))
            trial_inputs.append((idx, total, s, c))
    with (run_root / "commands.sh").open("w") as f:
        f.write("#!/usr/bin/env bash\nset -e\n\n")
        for cmd in commands:
            f.write(cmd + "\n")

    print("="*78, flush=True)
    print("Fuchsian GINN parameter explorer v1.2 depth-2 sweep", flush=True)
    print(f"preset={args.preset} surfaces={len(surfaces)} configs={len(configs)} trials={total} parallel={args.parallel_trials}", flush=True)
    print(f"run_root={run_root}", flush=True)
    print("Surfaces:", ", ".join(s.name for s in surfaces), flush=True)
    print("Configs:", ", ".join(c.name for c in configs), flush=True)
    if args.dry_run:
        print("DRY RUN: no trials will be executed", flush=True)
    print("="*78, flush=True)

    t0 = time.time(); c0 = cpu_seconds()
    rows: List[Dict[str, Any]] = []
    if args.parallel_trials <= 1 or args.dry_run:
        for idx, total, s, c in trial_inputs:
            rows.append(run_trial(idx, total, s, c, engine, maker, raw_root, log_root, args.base_seed, args.device, dry_run=args.dry_run))
            write_csv(run_root / "trial_summary.csv", rows)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel_trials) as ex:
            futs = [ex.submit(run_trial, idx, total, s, c, engine, maker, raw_root, log_root, args.base_seed, args.device, False)
                    for idx, total, s, c in trial_inputs]
            for fut in as_completed(futs):
                rows.append(fut.result())
                rows.sort(key=lambda r: int(r.get("trial_index", 0)))
                write_csv(run_root / "trial_summary.csv", rows)

    write_csv(run_root / "trial_summary.csv", rows)
    summarize_best(rows, "surface", run_root / "best_by_surface.csv")
    summarize_best(rows, "config", run_root / "best_by_config.csv")
    summarize_compact_noncompact(rows, run_root / "compact_vs_noncompact_summary.csv")

    # Overall best by simple objectives.
    pass_rows = [r for r in rows if r.get("status") == "PASS" and numeric(r.get("hard_rmse")) is not None]
    if pass_rows:
        best_rmse = sorted(pass_rows, key=lambda r: float(r["hard_rmse"]))[0]
        best_top5 = sorted(pass_rows, key=lambda r: (-float(r.get("top5_acc") or 0.0), float(r.get("hard_rmse") or 1e99)))[0]
        fastest_good = sorted(
            [r for r in pass_rows if float(r.get("top5_acc") or 0.0) >= 0.98],
            key=lambda r: float(r.get("wall_seconds") or 1e99)
        )
        summary = {
            "total_wall_seconds": time.time() - t0,
            "total_cpu_seconds_parent": cpu_seconds() - c0,
            "num_pass": len(pass_rows),
            "num_trials": len(rows),
            "best_rmse_trial": {k: best_rmse.get(k) for k in ["surface", "config", "hard_rmse", "top5_acc", "wall_seconds", "top5_speedup"]},
            "best_top5_trial": {k: best_top5.get(k) for k in ["surface", "config", "hard_rmse", "top5_acc", "wall_seconds", "top5_speedup"]},
            "fastest_top5_ge_0p98_trial": ({k: fastest_good[0].get(k) for k in ["surface", "config", "hard_rmse", "top5_acc", "wall_seconds", "top5_speedup"]} if fastest_good else None),
        }
    else:
        summary = {"total_wall_seconds": time.time() - t0, "num_pass": 0, "num_trials": len(rows)}
    with (run_root / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("="*78, flush=True)
    print("DONE", flush=True)
    print(f"trial_summary: {run_root / 'trial_summary.csv'}", flush=True)
    print(f"best_by_surface: {run_root / 'best_by_surface.csv'}", flush=True)
    print(f"compact/noncompact summary: {run_root / 'compact_vs_noncompact_summary.csv'}", flush=True)
    print(f"total wall={time.time()-t0:.1f}s parent_cpu={cpu_seconds()-c0:.1f}s rss={rss_mb():.1f} MB", flush=True)
    print("="*78, flush=True)


if __name__ == "__main__":
    main()
