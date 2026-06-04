#!/usr/bin/env python3
"""FuchsianCompactPolygonTester_v1_2.py

Focused terminal tester for compact polygon Fuchsian/Riemann-surface feedstock.
Version 1.2 retrofits the family output contract used by the master dataset
builder: every emitted surface JSON is normalized with surface_id, family,
Riemann-surface/Kähler status, domain/sampling status, eligibility flags, and
source metadata.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from FuchsianSurfaceRecordTools_v1_0 import (
    FAILURE_FIELDS,
    GEOMETRY_AUDIT_FIELDS,
    GINN_SMOKE_FIELDS,
    GINN_TRAINING_FIELDS,
    audit_row_from_surface,
    finite_float,
    generator_su11_audit,
    normalize_surface_record,
    write_csv,
    write_json,
)

PROGRAM = "FuchsianCompactPolygonTester_v1_2.py"
VERSION = "1.2"


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_module(path: str, name: str):
    path = str(Path(path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compact_polygon_audit(surface_name: str, sj: Dict[str, Any], endpoint_tol: float, angle_tol: float) -> Dict[str, Any]:
    genus = sj.get("genus")
    area = finite_float(sj.get("area"))
    expected_area = 4.0 * math.pi * (int(genus) - 1) if isinstance(genus, int) and genus >= 2 else None
    area_err = abs(area - expected_area) if area is not None and expected_area is not None else None
    verts = sj.get("polygon_vertices") or []
    pairings = sj.get("side_pairings") or []
    ep_audit = sj.get("side_pairing_endpoint_audit") or []
    vaudit = sj.get("vertex_angle_audit") or []
    max_ep = max([abs(finite_float(x.get("endpoint_error"), 0.0) or 0.0) for x in ep_audit], default=None)
    max_ang = max([abs(finite_float(x.get("smooth_error_from_2pi"), 0.0) or 0.0) for x in vaudit], default=None)
    su = generator_su11_audit(sj)
    pass_geom = (
        sj.get("domain_type") == "compact_polygon"
        and isinstance(genus, int)
        and genus >= 2
        and len(verts) >= 4
        and len(pairings) >= 1
        and (area_err is not None and area_err < 1.0e-8)
        and (max_ep is None or max_ep < endpoint_tol)
        and (max_ang is None or max_ang < angle_tol)
        and su["bad_generator_count"] == 0
        and su["su11_max_det_error"] < 1.0e-8
    )
    return {
        "surface": surface_name,
        "surface_title": sj.get("name"),
        "family": "compact_polygon",
        "domain_type": sj.get("domain_type"),
        "genus": genus,
        "area": area,
        "expected_area": expected_area,
        "area_error": area_err,
        "polygon_vertex_count": len(verts),
        "side_pairing_count": len(pairings),
        "vertex_cycle_count": len(vaudit),
        "max_endpoint_error": max_ep,
        "max_vertex_angle_error": max_ang,
        "certification_level": (sj.get("certification") or {}).get("level") or sj.get("certification_level"),
        "pass_geometry_audit": bool(pass_geom),
        **su,
    }


def run_ginn_label_preflight(ginn, sj: Dict[str, Any], surface_name: str, pairs: int, depth: int, seed: int, max_word_ball: int) -> Dict[str, Any]:
    rows, X, D, word_ball, label_meta, feature_names = ginn.generate_ginn_dataset(sj, pairs, depth, seed, max_word_ball=max_word_ball)
    return {
        "surface": surface_name,
        "surface_id": sj.get("surface_id", surface_name),
        "pairs": pairs,
        "word_depth": depth,
        "word_ball_size": len(word_ball),
        "shortcut_fraction": label_meta.get("shortcut_fraction"),
        "mean_winner_depth": label_meta.get("mean_winner_depth") or label_meta.get("mean_shortest_lift_depth"),
        "max_word_ball": max_word_ball,
        "pass_ginn_preflight": True,
    }


def call_ginn_training(args, surface_name: str, run_root: Path) -> Dict[str, Any]:
    outroot = run_root / "ginn_runs"
    cmd = [
        sys.executable, args.ginn_script,
        "--surface", surface_name,
        "--maker", args.maker,
        "--outdir", str(outroot),
        "--profile", args.profile,
        "--pairs", str(args.ginn_pairs),
        "--depth", str(args.ginn_depth),
        "--epochs", str(args.ginn_epochs),
        "--batch-size", str(args.ginn_batch_size),
        "--device", args.ginn_device,
        "--max-word-ball", str(args.ginn_max_word_ball),
        "--max-generators", str(args.max_generators),
        "--max-cosets", str(args.max_cosets),
    ]
    if args.no_train:
        cmd.append("--no-train")
    t0 = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "surface": surface_name,
        "surface_id": surface_name,
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "wall_seconds": time.time() - t0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "pass_ginn_training": proc.returncode == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact polygon Fuchsian surface tester v1.2")
    ap.add_argument("--genera", nargs="*", type=int, default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--include-hurwitz", action="store_true", default=True)
    ap.add_argument("--no-hurwitz", dest="include_hurwitz", action="store_false")
    ap.add_argument("--surfaces", default="", help="Explicit comma-list overrides genera, e.g. regular_g2,hurwitz")
    ap.add_argument("--maker", default="FuchsianDomainMaker_v13.py")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--outroot", default="compact_polygon_tester_runs")
    ap.add_argument("--label", default="")
    ap.add_argument("--endpoint-tol", type=float, default=1.0e-8)
    ap.add_argument("--angle-tol", type=float, default=1.0e-8)
    ap.add_argument("--ginn-smoke", action="store_true")
    ap.add_argument("--run-ginn", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--ginn-pairs", type=int, default=400)
    ap.add_argument("--ginn-depth", type=int, default=2)
    ap.add_argument("--ginn-max-word-ball", type=int, default=50000)
    ap.add_argument("--ginn-epochs", type=int, default=60)
    ap.add_argument("--ginn-batch-size", type=int, default=128)
    ap.add_argument("--ginn-device", default="auto")
    ap.add_argument("--profile", default="balanced", choices=["balanced", "accurate", "fast", "manual"])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-generators", type=int, default=0)
    ap.add_argument("--max-cosets", type=int, default=20000)
    args = ap.parse_args()

    for attr in ["maker", "ginn_script"]:
        p = Path(getattr(args, attr))
        if not p.exists():
            p = Path(__file__).resolve().parent / p.name
        setattr(args, attr, str(p))

    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()] if args.surfaces else [f"regular_g{g}" for g in args.genera]
    if args.include_hurwitz and "hurwitz" not in surfaces:
        surfaces.append("hurwitz")

    run_root = Path(args.outroot) / f"run_{now_stamp()}{('_' + args.label) if args.label else ''}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"{PROGRAM}\nrun_root={run_root}\nsurfaces={surfaces}\n" + "-" * 78, flush=True)
    ginn = load_module(args.ginn_script, "ginn_v24_compact")

    geom_rows: List[Dict[str, Any]] = []
    smoke_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for s in surfaces:
        print("=" * 78)
        print(f"[surface] {s}", flush=True)
        try:
            raw = ginn.make_surface(s, args.maker, max_cosets=args.max_cosets, max_generators=args.max_generators)
            aud = compact_polygon_audit(s, raw, args.endpoint_tol, args.angle_tol)
            sj = normalize_surface_record(
                raw,
                surface_spec=s,
                surface_family="compact_polygon",
                surface_subfamily="hurwitz_klein" if s == "hurwitz" else "regular_genus",
                source_program=PROGRAM,
                source_version=VERSION,
                construction_parameters={"surface": s, "genera": args.genera},
                geometry_audit_pass=bool(aud["pass_geometry_audit"]),
                finite_area=True,
                torsion_free=True,
            )
            write_json(run_root / "surfaces" / f"{sj['surface_id']}.json", sj)
            row = {**audit_row_from_surface(sj, aud["pass_geometry_audit"]), **aud}
            geom_rows.append(row)
            print(f"[audit] pass={aud['pass_geometry_audit']} genus={aud['genus']} area_err={aud['area_error']} gens={aud['generator_count']} ep_err={aud['max_endpoint_error']}", flush=True)
            if args.ginn_smoke:
                gr = run_ginn_label_preflight(ginn, sj, s, args.ginn_pairs, args.ginn_depth, args.seed, args.ginn_max_word_ball)
                smoke_rows.append(gr)
                print(f"[ginn-smoke] PASS W={gr['word_ball_size']} shortcut={gr.get('shortcut_fraction')}", flush=True)
            if args.run_ginn:
                tr = call_ginn_training(args, s, run_root)
                train_rows.append(tr)
                print(f"[ginn-train] pass={tr['pass_ginn_training']} rc={tr['returncode']}", flush=True)
        except Exception as e:
            print(f"[FAIL] {s}: {type(e).__name__}: {e}", flush=True)
            failures.append({"surface": s, "surface_id": s, "error_type": type(e).__name__, "error": str(e)})

    write_csv(run_root / "tables" / "geometry_audit.csv", geom_rows, GEOMETRY_AUDIT_FIELDS)
    write_csv(run_root / "tables" / "compact_geometry_audit.csv", geom_rows, GEOMETRY_AUDIT_FIELDS)
    write_csv(run_root / "tables" / "ginn_smoke_summary.csv", smoke_rows, GINN_SMOKE_FIELDS)
    write_csv(run_root / "tables" / "ginn_training_summary.csv", train_rows, GINN_TRAINING_FIELDS)
    write_csv(run_root / "tables" / "failures.csv", failures, FAILURE_FIELDS)
    write_json(run_root / "manifest.json", {"program": PROGRAM, "version": VERSION, "surfaces": surfaces, "args": vars(args), "completed": len(geom_rows), "failures": len(failures), "contract_version": "fuchsian_surface_record_contract_v1"})
    print(f"[done] completed={len(geom_rows)} failures={len(failures)} run_root={run_root}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
