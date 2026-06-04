#!/usr/bin/env python3
"""FuchsianHeckeTester_v1_2.py

Focused terminal tester for torsion-free Hecke triangle-group cover feedstock.
Version 1.2 retrofits the common master-builder-ready surface-record contract.
Covers: abelian C2 x Cq kernels and nonabelian dihedral Dq kernels from
G_q = Delta(2,q,infinity).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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

PROGRAM = "FuchsianHeckeTester_v1_2.py"
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


def infer_compactified_genus(area: Any, cusps: Any):
    area = finite_float(area)
    if area is None or cusps is None:
        return None, None
    val = (area / (2.0 * math.pi) - int(cusps) + 2.0) / 2.0
    nearest = round(val)
    return nearest, abs(val - nearest)


def hecke_audit(surface: str, sj: Dict[str, Any]) -> Dict[str, Any]:
    q = int(surface.replace("hecke_ab", "").replace("hecke_d", "").replace("hecke_dihedral", ""))
    family = "hecke_abelian" if surface.startswith("hecke_ab") else "hecke_dihedral_nonabelian"
    area = finite_float(sj.get("area"))
    cusps = sj.get("cusp_count")
    ginf, gerr = infer_compactified_genus(area, cusps)
    su = generator_su11_audit(sj)
    torsion_free = bool(sj.get("torsion_free"))
    lam = 2.0 * math.cos(math.pi / q)
    pass_geom = (
        sj.get("domain_type") == "modular_ford_domain"
        and sj.get("compact") is False
        and torsion_free
        and cusps is not None
        and area is not None
        and gerr is not None
        and gerr < 1.0e-8
        and su["bad_generator_count"] == 0
        and su["su11_max_det_error"] < 1.0e-8
    )
    return {
        "surface": surface,
        "surface_title": sj.get("name"),
        "family": family,
        "q": q,
        "lambda_2cos_pi_over_q": lam,
        "domain_type": sj.get("domain_type"),
        "subdomain_type": sj.get("subdomain_type"),
        "subgroup": sj.get("subgroup"),
        "torsion_free": torsion_free,
        "compact": sj.get("compact"),
        "area": area,
        "cusp_count": cusps,
        "compactified_genus_inferred": ginf,
        "genus_integrality_error": gerr,
        "certification_level": sj.get("certification_level"),
        "pass_geometry_audit": bool(pass_geom),
        **su,
    }


def run_smoke(ginn, sj: Dict[str, Any], surface: str, pairs: int, depth: int, seed: int, max_word_ball: int) -> Dict[str, Any]:
    rows, X, D, word_ball, label_meta, feature_names = ginn.generate_ginn_dataset(sj, pairs, depth, seed, max_word_ball=max_word_ball)
    return {
        "surface": surface,
        "surface_id": sj.get("surface_id", surface),
        "pairs": pairs,
        "word_depth": depth,
        "word_ball_size": len(word_ball),
        "shortcut_fraction": label_meta.get("shortcut_fraction"),
        "mean_winner_depth": label_meta.get("mean_winner_depth") or label_meta.get("mean_shortest_lift_depth"),
        "max_word_ball": max_word_ball,
        "pass_ginn_preflight": True,
    }


def call_ginn(args, surface: str, run_root: Path) -> Dict[str, Any]:
    outroot = run_root / "ginn_runs"
    cmd = [
        sys.executable, args.ginn_script,
        "--surface", surface,
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
        "--candidate-chunk-size", str(args.candidate_chunk_size),
    ]
    if args.no_train:
        cmd.append("--no-train")
    t0 = time.time()
    p = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "surface": surface,
        "surface_id": surface,
        "returncode": p.returncode,
        "wall_seconds": time.time() - t0,
        "cmd": " ".join(cmd),
        "stdout_tail": p.stdout[-4000:],
        "stderr_tail": p.stderr[-4000:],
        "pass_ginn_training": p.returncode == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Hecke cover tester v1.2")
    ap.add_argument("--q", nargs="*", type=int, default=[3, 4, 5, 6, 7, 8, 9, 10, 12])
    ap.add_argument("--families", default="ab,d", help="comma-list: ab,d")
    ap.add_argument("--surfaces", default="", help="explicit comma-list overrides q/families")
    ap.add_argument("--maker", default="FuchsianDomainMaker_v13.py")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--outroot", default="hecke_tester_runs")
    ap.add_argument("--label", default="")
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
    ap.add_argument("--candidate-chunk-size", type=int, default=0)
    args = ap.parse_args()

    for attr in ["maker", "ginn_script"]:
        p = Path(getattr(args, attr))
        if not p.exists():
            p = Path(__file__).resolve().parent / p.name
        setattr(args, attr, str(p))

    if args.surfaces:
        surfaces = [x.strip() for x in args.surfaces.split(",") if x.strip()]
    else:
        fams = {x.strip() for x in args.families.split(",") if x.strip()}
        surfaces: List[str] = []
        for q in args.q:
            if "ab" in fams:
                surfaces.append(f"hecke_ab{q}")
            if "d" in fams or "dihedral" in fams:
                surfaces.append(f"hecke_d{q}")

    run_root = Path(args.outroot) / f"run_{now_stamp()}{('_' + args.label) if args.label else ''}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"{PROGRAM}\nrun_root={run_root}\nsurfaces={surfaces}\n" + "-" * 78, flush=True)
    ginn = load_module(args.ginn_script, "ginn_v24_hecke")

    audit_rows: List[Dict[str, Any]] = []
    smoke_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for s in surfaces:
        print("=" * 78)
        print(f"[surface] {s}", flush=True)
        try:
            raw = ginn.make_surface(s, args.maker, max_cosets=args.max_cosets, max_generators=args.max_generators)
            ar = hecke_audit(s, raw)
            q = ar["q"]
            sj = normalize_surface_record(
                raw,
                surface_spec=s,
                surface_family="hecke_triangle_cover",
                surface_subfamily="abelian_C2xCq" if s.startswith("hecke_ab") else "dihedral_Dq",
                source_program=PROGRAM,
                source_version=VERSION,
                construction_parameters={"q": q, "surface": s},
                geometry_audit_pass=bool(ar["pass_geometry_audit"]),
                finite_area=True,
                torsion_free=bool(raw.get("torsion_free")),
            )
            write_json(run_root / "surfaces" / f"{sj['surface_id']}.json", sj)
            audit_rows.append({**audit_row_from_surface(sj, ar["pass_geometry_audit"]), **ar})
            print(f"[audit] pass={ar['pass_geometry_audit']} q={ar['q']} fam={ar['family']} gbar={ar['compactified_genus_inferred']} cusps={ar['cusp_count']} gens={ar['generator_count']}", flush=True)
            if args.ginn_smoke:
                sr = run_smoke(ginn, sj, s, args.ginn_pairs, args.ginn_depth, args.seed, args.ginn_max_word_ball)
                smoke_rows.append(sr)
                print(f"[ginn-smoke] PASS W={sr['word_ball_size']} shortcut={sr.get('shortcut_fraction')}", flush=True)
            if args.run_ginn:
                tr = call_ginn(args, s, run_root)
                train_rows.append(tr)
                print(f"[ginn-train] pass={tr['pass_ginn_training']} rc={tr['returncode']}", flush=True)
        except Exception as e:
            print(f"[FAIL] {s}: {type(e).__name__}: {e}", flush=True)
            failures.append({"surface": s, "surface_id": s, "error_type": type(e).__name__, "error": str(e)})

    write_csv(run_root / "tables" / "geometry_audit.csv", audit_rows, GEOMETRY_AUDIT_FIELDS)
    write_csv(run_root / "tables" / "hecke_geometry_audit.csv", audit_rows, GEOMETRY_AUDIT_FIELDS)
    write_csv(run_root / "tables" / "ginn_smoke_summary.csv", smoke_rows, GINN_SMOKE_FIELDS)
    write_csv(run_root / "tables" / "ginn_training_summary.csv", train_rows, GINN_TRAINING_FIELDS)
    write_csv(run_root / "tables" / "failures.csv", failures, FAILURE_FIELDS)
    write_json(run_root / "manifest.json", {"program": PROGRAM, "version": VERSION, "surfaces": surfaces, "args": vars(args), "completed": len(audit_rows), "failures": len(failures), "contract_version": "fuchsian_surface_record_contract_v1"})
    print(f"[done] completed={len(audit_rows)} failures={len(failures)} run_root={run_root}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
