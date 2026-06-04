#!/usr/bin/env python3
"""FuchsianSchottkyTester_v1_1.py

Focused terminal tester for Schottky/free-Fuchsian ideal-geodesic pairing
examples in the Fuchsian GENN project.

This family is deliberately labeled differently from compact polygon, modular
finite-area, Hecke finite-area, and Hurwitz-kernel records.  A Schottky record
here is a torsion-free free Fuchsian quotient D/Gamma and hence a Riemann
surface, but the supplied domain is an ideal-geodesic/free-group domain with a
bounded sampling scaffold rather than a compact polygon or a finite-area Ford
domain.  In particular, the default records are noncompact and infinite-area.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROGRAM = "FuchsianSchottkyTester_v1_1.py"
VERSION = "v1.1"

try:
    from FuchsianSurfaceRecordTools_v1_0 import (
        GEOMETRY_AUDIT_FIELDS, GINN_SMOKE_FIELDS, GINN_TRAINING_FIELDS, FAILURE_FIELDS,
        generator_su11_audit, normalize_surface_record, write_csv, write_json,
    )
except Exception:  # pragma: no cover - fallback for unusual run locations
    GEOMETRY_AUDIT_FIELDS = None
    GINN_SMOKE_FIELDS = None
    GINN_TRAINING_FIELDS = None
    FAILURE_FIELDS = None

    def write_json(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2))

    def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys: List[str] = list(fieldnames or [])
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        if not keys:
            keys = ["empty"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def generator_su11_audit(sj: Dict[str, Any]) -> Dict[str, Any]:
        gens = sj.get("generators") or {}
        bad: List[str] = []
        maxerr = 0.0
        for name, g in gens.items():
            try:
                a = g["alpha"]; b = g["beta"]
                alpha = complex(a[0], a[1]); beta = complex(b[0], b[1])
                det = abs(alpha)**2 - abs(beta)**2
                maxerr = max(maxerr, abs(det - 1.0))
                if det <= 0:
                    bad.append(str(name))
            except Exception:
                bad.append(str(name))
        return {"generator_count": len(gens), "su11_max_det_error": maxerr, "bad_generator_count": len(bad), "bad_generators": ";".join(bad[:20])}

    def normalize_surface_record(sj: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        out = dict(sj)
        out.update(kwargs)
        out["surface_id"] = kwargs.get("surface_spec", out.get("name", "schottky"))
        out["mainline_dataset_eligible"] = kwargs.get("mainline_dataset_eligible", False)
        out["exclusion_reason"] = kwargs.get("exclusion_reason", "")
        out["master_record"] = {k: out.get(k) for k in out.keys()}
        return out


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


class _DummyQt:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return _DummyQt()
    def __getattr__(self, name):
        return _DummyQt()
    def __iter__(self):
        return iter(())
    def __bool__(self):
        return False


def _install_gui_stubs() -> None:
    """Install tiny PyQt/matplotlib Qt stubs for headless DomainMaker import."""
    import types
    if "PyQt6" not in sys.modules:
        pyqt = types.ModuleType("PyQt6")
        qtwidgets = types.ModuleType("PyQt6.QtWidgets")
        names = [
            "QApplication", "QCheckBox", "QComboBox", "QFileDialog", "QFrame",
            "QGridLayout", "QGroupBox", "QHBoxLayout", "QLabel", "QMainWindow",
            "QMessageBox", "QPushButton", "QSpinBox", "QDoubleSpinBox",
            "QTextEdit", "QVBoxLayout", "QWidget",
        ]
        for name in names:
            setattr(qtwidgets, name, _DummyQt)
        sys.modules["PyQt6"] = pyqt
        sys.modules["PyQt6.QtWidgets"] = qtwidgets
    if "matplotlib.backends.backend_qtagg" not in sys.modules:
        backend = types.ModuleType("matplotlib.backends.backend_qtagg")
        backend.FigureCanvasQTAgg = _DummyQt
        sys.modules["matplotlib.backends.backend_qtagg"] = backend


def load_module(path: str, name: str):
    if "DomainMaker" in Path(path).name:
        _install_gui_stubs()
    p = Path(path).expanduser().resolve()
    if not p.exists():
        alt = Path(__file__).resolve().parent / p.name
        if alt.exists():
            p = alt
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cpair_to_complex(xy: List[float] | Tuple[float, float]) -> complex:
    return complex(float(xy[0]), float(xy[1]))


def angle01(z: complex) -> float:
    a = math.atan2(z.imag, z.real)
    return a if a >= 0 else a + 2.0 * math.pi


def safe_float_label(x: float) -> str:
    return (f"{x:.4g}").replace("-", "m").replace(".", "p")


def schottky_surface_id(rank: int, gap: float, rotation: float) -> str:
    return f"schottky_r{rank}_gap{safe_float_label(gap)}_rot{safe_float_label(rotation)}"


def add_bounded_sampling_scaffold(sj: Dict[str, Any], sample_radius: float) -> Dict[str, Any]:
    """Add a bounded disk-tile scaffold for GINN point sampling.

    The Schottky domain is infinite-area and includes ideal boundary arcs.  The
    current GINN sampler needs compact triangles.  We therefore sample from a
    bounded polygonal scaffold obtained by radially shrinking the ideal endpoints
    and triangulating the resulting cyclic polygon.  This is intentionally
    labeled as a bounded sampling scaffold, not as the full fundamental domain.
    """
    out = dict(sj)
    sides = out.get("geodesic_sides") or []
    pts: List[complex] = []
    for side in sides:
        for xy in side.get("ideal_endpoints", []):
            z = cpair_to_complex(xy)
            if abs(z) > 0:
                pts.append(sample_radius * z / abs(z))
    # remove near-duplicates and sort cyclically
    uniq: Dict[Tuple[int, int], complex] = {}
    for z in pts:
        key = (round(z.real, 12), round(z.imag, 12))
        uniq[key] = z
    pts = sorted(uniq.values(), key=angle01)
    if len(pts) < 3:
        raise ValueError("Schottky sampling scaffold requires at least three ideal endpoints.")
    center = 0.0 + 0.0j
    tiles = []
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        # skip degenerate tiny triangles
        area2 = abs((a.real-center.real)*(b.imag-center.imag) - (a.imag-center.imag)*(b.real-center.real))
        if area2 <= 1.0e-12:
            continue
        tiles.append({
            "tile_index": i,
            "tile_kind": "bounded_schottky_sampling_scaffold_triangle",
            "vertices": [[center.real, center.imag], [a.real, a.imag], [b.real, b.imag]],
        })
    out["fundamental_domain_tiles"] = tiles
    out["sampling_scaffold"] = {
        "type": "bounded_radial_shrink_of_ideal_endpoints",
        "sample_radius": float(sample_radius),
        "ideal_endpoint_count": len(pts),
        "tile_count": len(tiles),
        "warning": "Sampling scaffold only; not the full infinite-area Schottky fundamental domain.",
    }
    out["fundamental_domain_status"] = "Schottky ideal-geodesic free-domain plus bounded sampling scaffold in the Poincare disk"
    out["sampling_status"] = "supported by bounded Schottky sampling scaffold; not a full finite-area sampling measure"
    return out


def make_schottky_surface(maker: Any, rank: int, gap: float, rotation: float, sample_radius: float) -> Dict[str, Any]:
    sj = maker.make_schottky_geodesic_pairing(rank, gap=gap, rotation_deg=rotation)
    sj = add_bounded_sampling_scaffold(sj, sample_radius=sample_radius)
    sid = schottky_surface_id(rank, gap, rotation)
    sj["surface_id"] = sid
    sj["surface_spec"] = sid
    sj["surface_family"] = "schottky_free_fuchsian"
    sj["surface_subfamily"] = "ideal_geodesic_pairing"
    sj["rank"] = int(rank)
    sj["free_rank"] = int(rank)
    sj["compact"] = False
    sj["finite_area"] = False
    sj["cusp_count"] = 0
    sj["torsion_free"] = True
    sj["orbifold_excluded"] = False
    sj["surface_area_type"] = "noncompact_infinite_area_schottky_free_fuchsian"
    sj["riemann_surface_status"] = (
        "smooth noncompact hyperbolic Riemann surface D/Gamma from a torsion-free "
        "free Fuchsian Schottky group; infinite-area/bounded-sampling-scaffold record"
    )
    sj["kahler_status"] = "Riemann surface, hence Kähler in complex dimension one"
    sj["word_ball_recommended_depth"] = 2
    sj["mainline_finite_area_dataset_eligible"] = False
    sj["auxiliary_schottky_dataset_eligible"] = True
    sj["dataset_role"] = "all_riemann_surfaces_zoo_noncompact_infinite_area_branch"
    sj["notes"] = str(sj.get("notes", "")) + " v1.1 adds bounded sampling scaffold and master-builder metadata."
    return sj




def schottky_interval_audit_from_sides(sj: Dict[str, Any]) -> Dict[str, Any]:
    """Audit disjoint ping-pong boundary intervals for Schottky sides.

    DomainMaker stores each ideal-geodesic side as two endpoints on S^1.  The
    Schottky ping-pong intervals are the minor arcs between each endpoint pair.
    This independent audit avoids relying on older DomainMaker interval metadata,
    which can report zero gap for wrapped intervals.
    """
    raw_intervals: List[Tuple[float, float]] = []
    for side in sj.get("geodesic_sides") or []:
        eps = side.get("ideal_endpoints") or []
        if len(eps) != 2:
            continue
        a = angle01(cpair_to_complex(eps[0]))
        b = angle01(cpair_to_complex(eps[1]))
        # choose the minor arc from a to b; represent possibly wrapped as a<=b
        d = (b - a) % (2.0 * math.pi)
        if d <= math.pi:
            start, end = a, a + d
        else:
            start, end = b, b + (2.0 * math.pi - d)
        raw_intervals.append((start, end))
    # Split wrapped intervals into [0, end-2pi] and [start, 2pi].
    segs: List[Tuple[float, float, int]] = []
    for i, (a, b) in enumerate(raw_intervals):
        if b <= 2.0 * math.pi:
            segs.append((a, b, i))
        else:
            segs.append((a, 2.0 * math.pi, i))
            segs.append((0.0, b - 2.0 * math.pi, i))
    disjoint = True
    min_gap = float("inf")
    segs_sorted = sorted(segs, key=lambda x: x[0])
    for k, (a, b, i) in enumerate(segs_sorted):
        a2, b2, j = segs_sorted[(k + 1) % len(segs_sorted)]
        if k + 1 == len(segs_sorted):
            gap = (a2 + 2.0 * math.pi) - b
        else:
            gap = a2 - b
        if i != j and gap < -1.0e-12:
            disjoint = False
        min_gap = min(min_gap, max(0.0, gap))
    if not segs_sorted:
        disjoint = False
        min_gap = 0.0
    return {
        "intervals_disjoint": bool(disjoint),
        "min_boundary_gap_radians": float(min_gap),
        "interval_count": len(raw_intervals),
    }

def schottky_audit(surface: str, sj: Dict[str, Any], endpoint_tol: float) -> Dict[str, Any]:
    su = generator_su11_audit(sj)
    ia_old = sj.get("interval_disjointness_audit") or {}
    ia = schottky_interval_audit_from_sides(sj)
    endpoint_rows = sj.get("side_pairing_endpoint_audit") or []
    endpoint_errors = []
    for r in endpoint_rows:
        try:
            endpoint_errors.append(float(r.get("endpoint_error_near_ideal", 0.0)))
        except Exception:
            pass
    max_ep = max(endpoint_errors) if endpoint_errors else None
    rank = int(sj.get("rank") or sj.get("free_rank") or 0)
    expected_gens = rank
    tile_count = len(sj.get("fundamental_domain_tiles") or [])
    pass_geom = bool(
        sj.get("domain_type") == "schottky_ideal_geodesic_domain"
        and rank >= 1
        and su["generator_count"] == expected_gens
        and su["bad_generator_count"] == 0
        and float(su["su11_max_det_error"]) < 1.0e-8
        and bool(ia.get("intervals_disjoint"))
        and max_ep is not None and max_ep < endpoint_tol
        and tile_count >= 3
        and bool(sj.get("torsion_free"))
    )
    return {
        "surface": surface,
        "surface_id": sj.get("surface_id"),
        "surface_family": "schottky_free_fuchsian",
        "surface_subfamily": "ideal_geodesic_pairing",
        "mainline_dataset_eligible": bool(pass_geom),
        "exclusion_reason": "" if pass_geom else "Schottky geometry audit failed or sampling scaffold missing",
        "riemann_surface_status": sj.get("riemann_surface_status"),
        "domain_type": sj.get("domain_type"),
        "subdomain_type": sj.get("subdomain_type"),
        "torsion_free": sj.get("torsion_free"),
        "orbifold_excluded": sj.get("orbifold_excluded"),
        "compact": sj.get("compact"),
        "finite_area": sj.get("finite_area"),
        "genus": sj.get("genus"),
        "compactified_genus": sj.get("compactified_genus"),
        "area": sj.get("area"),
        "cusp_count": sj.get("cusp_count"),
        "rank": rank,
        "free_rank": rank,
        "intervals_disjoint": ia.get("intervals_disjoint"),
        "min_boundary_gap_radians": ia.get("min_boundary_gap_radians"),
        "domainmaker_intervals_disjoint_raw": ia_old.get("intervals_disjoint"),
        "domainmaker_min_boundary_gap_radians_raw": ia_old.get("min_boundary_gap_radians"),
        "max_endpoint_error_near_ideal": max_ep,
        "sampling_tile_count": tile_count,
        "surface_area_type": sj.get("surface_area_type"),
        "mainline_finite_area_dataset_eligible": sj.get("mainline_finite_area_dataset_eligible"),
        "auxiliary_schottky_dataset_eligible": sj.get("auxiliary_schottky_dataset_eligible"),
        "generator_count": su["generator_count"],
        "generator_truncated": False,
        "su11_max_det_error": su["su11_max_det_error"],
        "bad_generator_count": su["bad_generator_count"],
        "bad_generators": su["bad_generators"],
        "pass_geometry_audit": bool(pass_geom),
        "source_program": PROGRAM,
        "source_version": VERSION,
    }


def run_smoke(ginn: Any, sj: Dict[str, Any], surface: str, pairs: int, depth: int, seed: int, max_word_ball: int) -> Dict[str, Any]:
    rows, X, D, word_ball, label_meta, feature_names = ginn.generate_ginn_dataset(
        sj, pairs, depth, seed, max_word_ball=max_word_ball
    )
    return {
        "surface": surface,
        "surface_id": sj.get("surface_id"),
        "pairs": pairs,
        "word_depth": depth,
        "word_ball_size": len(word_ball),
        "shortcut_fraction": label_meta.get("shortcut_fraction"),
        "mean_winner_depth": label_meta.get("mean_shortest_lift_depth"),
        "max_word_ball": max_word_ball,
        "pass_ginn_preflight": True,
        "error": "",
    }


def run_direct_ginn(ginn: Any, sj: Dict[str, Any], surface: str, run_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Train GINN directly on the Schottky surface JSON.

    The main FuchsianDownstairsGINN CLI does not yet have named Schottky specs,
    so this tester calls the GINN module functions directly.  This keeps the
    training path honest while avoiding a v2.5 GINN change solely for Schottky.
    """
    t0 = time.time()
    outdir = run_root / "ginn_runs" / surface
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        rows, X, D, word_ball, label_meta, feature_names = ginn.generate_ginn_dataset(
            sj, args.ginn_pairs, args.ginn_depth, args.seed, max_word_ball=args.ginn_max_word_ball
        )
        # Save a minimal dataset trace for reproducibility.
        write_json(outdir / "surface.json", sj)
        write_json(outdir / "label_meta.json", label_meta)
        write_json(outdir / "word_ball.json", [
            {"word": m.word if m.word else "identity", "trace_real": m.trace_real()} for m in word_ball
        ])
        metrics = ginn.train_ginn(
            rows, X, D, word_ball, outdir, args.ginn_depth, args.ginn_epochs,
            args.ginn_pair_hidden, args.ginn_score_hidden, args.ginn_lr,
            args.ginn_batch_size, args.seed, args.ginn_device, args.ginn_patience,
            args.ginn_ce_weight, args.ginn_soft_distance_weight, args.ginn_temperature,
            candidate_chunk_size=args.ginn_candidate_chunk_size,
            auto_chunk_threshold_mb=args.ginn_auto_chunk_threshold_mb,
        )
        return {
            "surface": surface,
            "surface_id": sj.get("surface_id"),
            "returncode": 0,
            "wall_seconds": time.time() - t0,
            "pass_ginn_training": True,
            "cmd": "direct_module_call:ginn.train_ginn",
            "stdout_tail": json.dumps({
                "word_ball_size": metrics.get("word_ball_size"),
                "best_val_loss": metrics.get("best_val_loss"),
                "winning_lift_accuracy_test": metrics.get("winning_lift_accuracy_test"),
            })[-3500:],
            "stderr_tail": "",
        }
    except Exception as e:
        return {
            "surface": surface,
            "surface_id": sj.get("surface_id"),
            "returncode": 1,
            "wall_seconds": time.time() - t0,
            "pass_ginn_training": False,
            "cmd": "direct_module_call:ginn.train_ginn",
            "stdout_tail": "",
            "stderr_tail": f"{type(e).__name__}: {e}",
        }


def expand_surfaces(args: argparse.Namespace) -> List[Tuple[int, float, float]]:
    if args.surfaces:
        out: List[Tuple[int, float, float]] = []
        for spec in args.surfaces.split(","):
            spec = spec.strip()
            if not spec:
                continue
            # accepted forms: r2, r2_gap0.18_rot0, schottky_r2_gap0.18_rot0
            rank = None
            gap = args.gap
            rot = args.rotation
            parts = spec.replace("schottky_", "").split("_")
            for part in parts:
                if part.startswith("r") and part[1:].isdigit():
                    rank = int(part[1:])
                elif part.startswith("gap"):
                    try: gap = float(part[3:].replace("p", "."))
                    except Exception: pass
                elif part.startswith("rot"):
                    try: rot = float(part[3:].replace("p", ".").replace("m", "-"))
                    except Exception: pass
            if rank is None:
                raise ValueError(f"Could not parse Schottky surface spec {spec!r}; use r2 or schottky_r2_gap0.18_rot0.")
            out.append((rank, gap, rot))
        return out
    ranks = args.rank if args.rank else [2, 3, 4]
    gaps = args.gaps if args.gaps else [args.gap]
    rotations = args.rotations if args.rotations else [args.rotation]
    return [(int(r), float(g), float(rot)) for r in ranks for g in gaps for rot in rotations]


def main() -> int:
    ap = argparse.ArgumentParser(description="Schottky/free-Fuchsian tester v1.1")
    ap.add_argument("--rank", nargs="*", type=int, default=[2, 3, 4], help="Schottky free ranks to generate")
    ap.add_argument("--gap", type=float, default=0.18, help="Default endpoint gap parameter")
    ap.add_argument("--gaps", nargs="*", type=float, default=None, help="Optional list of gap parameters")
    ap.add_argument("--rotation", type=float, default=0.0, help="Default rotation in degrees")
    ap.add_argument("--rotations", nargs="*", type=float, default=None, help="Optional list of rotations in degrees")
    ap.add_argument("--surfaces", default="", help="Comma-list such as r2,r3 or schottky_r2_gap0p18_rot0")
    ap.add_argument("--sample-radius", type=float, default=0.82, help="Radial shrink for bounded sampling scaffold")
    ap.add_argument("--endpoint-tol", type=float, default=5.0e-5, help="Endpoint-pairing audit tolerance near the ideal boundary")
    ap.add_argument("--maker", default="FuchsianDomainMaker_v13.py")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--outroot", default="schottky_tester_runs")
    ap.add_argument("--label", default="")
    ap.add_argument("--ginn-smoke", action="store_true")
    ap.add_argument("--run-ginn", action="store_true")
    ap.add_argument("--ginn-pairs", type=int, default=400)
    ap.add_argument("--ginn-depth", type=int, default=2)
    ap.add_argument("--ginn-max-word-ball", type=int, default=50000)
    ap.add_argument("--ginn-epochs", type=int, default=60)
    ap.add_argument("--ginn-batch-size", type=int, default=128)
    ap.add_argument("--ginn-device", default="auto")
    ap.add_argument("--ginn-pair-hidden", type=int, default=192)
    ap.add_argument("--ginn-score-hidden", type=int, default=96)
    ap.add_argument("--ginn-lr", type=float, default=1.0e-3)
    ap.add_argument("--ginn-patience", type=int, default=20)
    ap.add_argument("--ginn-ce-weight", type=float, default=1.0)
    ap.add_argument("--ginn-soft-distance-weight", type=float, default=0.2)
    ap.add_argument("--ginn-temperature", type=float, default=1.0)
    ap.add_argument("--ginn-candidate-chunk-size", type=int, default=0)
    ap.add_argument("--ginn-auto-chunk-threshold-mb", type=float, default=2048.0)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    maker = load_module(args.maker, "domain_maker_schottky_v13")
    ginn = load_module(args.ginn_script, "ginn_v24_schottky") if (args.ginn_smoke or args.run_ginn) else None
    specs = expand_surfaces(args)
    run_root = Path(args.outroot) / f"run_{now_stamp()}{('_' + args.label) if args.label else ''}"
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"{PROGRAM}\nrun_root={run_root}\nsurfaces={specs}\n" + "-"*78, flush=True)
    audit_rows: List[Dict[str, Any]] = []
    smoke_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for rank, gap, rot in specs:
        surface = schottky_surface_id(rank, gap, rot)
        print("="*78, flush=True)
        print(f"[surface] {surface}", flush=True)
        try:
            sj0 = make_schottky_surface(maker, rank, gap, rot, sample_radius=args.sample_radius)
            ar0 = schottky_audit(surface, sj0, args.endpoint_tol)
            sj = normalize_surface_record(
                sj0,
                surface_spec=surface,
                surface_family="schottky_free_fuchsian",
                surface_subfamily="ideal_geodesic_pairing",
                source_program=PROGRAM,
                source_version=VERSION,
                construction_parameters={
                    "rank": rank, "gap": gap, "rotation_deg": rot, "sample_radius": args.sample_radius,
                    "endpoint_tol": args.endpoint_tol,
                },
                geometry_audit_pass=bool(ar0["pass_geometry_audit"]),
                finite_area=False,
                torsion_free=True,
                mainline_dataset_eligible=bool(ar0["pass_geometry_audit"]),
                exclusion_reason="" if ar0["pass_geometry_audit"] else "Schottky geometry audit failed",
            )
            # Restore Schottky-specific dataset flags after normalization.
            sj["surface_area_type"] = "noncompact_infinite_area_schottky_free_fuchsian"
            sj["mainline_finite_area_dataset_eligible"] = False
            sj["auxiliary_schottky_dataset_eligible"] = bool(ar0["pass_geometry_audit"])
            sj["dataset_role"] = "all_riemann_surfaces_zoo_noncompact_infinite_area_branch"
            write_json(run_root / "surfaces" / f"{surface}.json", sj)
            ar = dict(ar0)
            ar["mainline_dataset_eligible"] = bool(sj.get("mainline_dataset_eligible"))
            ar["exclusion_reason"] = str(sj.get("exclusion_reason", ""))
            audit_rows.append(ar)
            print(f"[audit] pass={ar['pass_geometry_audit']} rank={rank} gens={ar['generator_count']} disjoint={ar['intervals_disjoint']} tiles={ar['sampling_tile_count']} finite_area={ar['finite_area']}", flush=True)
            if args.ginn_smoke and ginn is not None:
                if not sj.get("mainline_dataset_eligible"):
                    smoke_rows.append({"surface": surface, "surface_id": sj.get("surface_id"), "pass_ginn_preflight": False, "error": "not mainline eligible"})
                    print("[ginn-smoke] skipped: not eligible", flush=True)
                else:
                    sr = run_smoke(ginn, sj, surface, args.ginn_pairs, args.ginn_depth, args.seed, args.ginn_max_word_ball)
                    smoke_rows.append(sr)
                    print(f"[ginn-smoke] PASS W={sr['word_ball_size']} shortcut={sr.get('shortcut_fraction')}", flush=True)
            if args.run_ginn and ginn is not None:
                tr = run_direct_ginn(ginn, sj, surface, run_root, args)
                train_rows.append(tr)
                print(f"[ginn-train] pass={tr['pass_ginn_training']} rc={tr['returncode']}", flush=True)
        except Exception as e:
            print(f"[FAIL] {surface}: {type(e).__name__}: {e}", flush=True)
            failures.append({"surface": surface, "surface_id": surface, "error_type": type(e).__name__, "error": str(e)})

    # Write tables with standard names for the master builder.
    write_csv(run_root / "tables" / "geometry_audit.csv", audit_rows, GEOMETRY_AUDIT_FIELDS)
    write_csv(run_root / "tables" / "schottky_geometry_audit.csv", audit_rows)
    write_csv(run_root / "tables" / "ginn_smoke_summary.csv", smoke_rows, GINN_SMOKE_FIELDS)
    write_csv(run_root / "tables" / "ginn_training_summary.csv", train_rows, GINN_TRAINING_FIELDS)
    write_csv(run_root / "tables" / "failures.csv", failures, FAILURE_FIELDS)
    write_json(run_root / "manifest.json", {
        "program": PROGRAM,
        "version": VERSION,
        "args": vars(args),
        "surfaces": [schottky_surface_id(r, g, rot) for r, g, rot in specs],
        "completed": len(audit_rows),
        "failures": len(failures),
        "family_status": "Schottky/free-Fuchsian Riemann-surface records with bounded sampling scaffold; noncompact/infinite-area branch of the zoo.",
    })
    print(f"[done] completed={len(audit_rows)} failures={len(failures)} run_root={run_root}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
