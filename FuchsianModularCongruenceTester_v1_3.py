#!/usr/bin/env python3
"""FuchsianModularCongruenceTester_v1_3.py

Focused terminal tester for modular congruence Fuchsian/Riemann-surface feedstock.
Version 1.3 keeps Gamma_0(N), retrofits the common master-builder-ready
surface-record contract.

Families:
  - principal Gamma(N), N>=3;
  - Gamma_1(N), N>=4;
  - Gamma_0(N), torsion-audited, with only torsion-free cases marked mainline.
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
from typing import Any, Dict, List, Tuple

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

PROGRAM = "FuchsianModularCongruenceTester_v1_3.py"
VERSION = "1.3"


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


def infer_compactified_genus(surface: str, sj: Dict[str, Any]) -> Tuple[Any, Any]:
    area = finite_float(sj.get("area"))
    cusps = sj.get("cusp_count")
    if area is None or cusps is None:
        return None, None
    tors = sj.get("torsion_free_audit") or {}
    e2 = len(tors.get("right_S_fixed_cosets_order2") or [])
    e3 = len(tors.get("right_ST_fixed_cosets_order3") or [])
    # Orbifold compactified genus formula.  In torsion-free cases e2=e3=0.
    val = 1.0 + area / (4.0 * math.pi) - e2 / 4.0 - e3 / 3.0 - int(cusps) / 2.0
    nearest = round(val)
    return nearest, abs(val - nearest)


def modular_family_and_level(surface: str) -> Tuple[str, int]:
    if surface.startswith("gamma1_"):
        return "Gamma_1", int(surface.split("_", 1)[1])
    if surface.startswith("gamma0_"):
        return "Gamma_0", int(surface.split("_", 1)[1])
    if surface.startswith("gamma"):
        return "Gamma", int(surface.replace("gamma", ""))
    raise ValueError(f"Not a modular surface spec: {surface}")


def modular_audit(surface: str, sj: Dict[str, Any], max_generators: int) -> Dict[str, Any]:
    fam, N = modular_family_and_level(surface)
    area = finite_float(sj.get("area"))
    cusps = sj.get("cusp_count")
    ginf, gerr = infer_compactified_genus(surface, sj)
    su = generator_su11_audit(sj)
    gea = sj.get("generator_export_audit") or {}
    truncated = bool(gea.get("generator_truncated_by_cli_max_generators", False))
    tokenized = bool(gea.get("v2_4_tokenized_generators", False) or gea.get("v2_3_tokenized_generators", False) or all(str(k).startswith("g") for k in (sj.get("generators") or {}).keys()))
    torsion_free = bool(sj.get("torsion_free"))
    minN = 4 if fam == "Gamma_1" else (3 if fam == "Gamma" else 2)
    tors = sj.get("torsion_free_audit") or {}
    e2 = len(tors.get("right_S_fixed_cosets_order2") or [])
    e3 = len(tors.get("right_ST_fixed_cosets_order3") or [])
    pass_geom = (
        sj.get("domain_type") == "modular_ford_domain"
        and N >= minN
        and sj.get("compact") is False
        and torsion_free
        and e2 == 0
        and e3 == 0
        and cusps is not None
        and area is not None
        and gerr is not None
        and gerr < 1.0e-8
        and not truncated
        and su["bad_generator_count"] == 0
        and su["su11_max_det_error"] < 1.0e-7
    )
    return {
        "surface": surface,
        "surface_title": sj.get("name"),
        "family": fam,
        "N": N,
        "domain_type": sj.get("domain_type"),
        "subdomain_type": sj.get("subdomain_type"),
        "subgroup": sj.get("subgroup"),
        "level_N": sj.get("level_N"),
        "index_in_psl2z": sj.get("index_in_psl2z"),
        "torsion_free": torsion_free,
        "elliptic_order2_fixed_count": e2,
        "elliptic_order3_fixed_count": e3,
        "compact": sj.get("compact"),
        "area": area,
        "cusp_count": cusps,
        "cusp_widths_json": json.dumps(sj.get("cusp_widths")),
        "compactified_genus_metadata": sj.get("compactified_genus"),
        "compactified_genus_inferred": ginf,
        "genus_integrality_error": gerr,
        "tokenized_generators": tokenized,
        "generator_truncated": truncated,
        "exported_generator_count": gea.get("exported_generator_count", su["generator_count"]),
        "underlying_schreier_generator_count": gea.get("underlying_schreier_generator_count"),
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
    ap = argparse.ArgumentParser(description="Modular congruence tester v1.2")
    ap.add_argument("--principal-N", nargs="*", type=int, default=[3, 4, 5])
    ap.add_argument("--gamma1-N", nargs="*", type=int, default=[4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--gamma0-N", nargs="*", type=int, default=[4, 6, 8, 9, 11, 12, 14, 15, 16, 18, 20])
    # v1.3 convenience aliases: these accept the natural command
    #   --families gamma,gamma1,gamma0 --N 3 4 5 ...
    # and translate it into the family-specific lists above.
    ap.add_argument("--families", default="", help="optional comma-list: gamma,gamma1,gamma0; used with --N as a convenience alias")
    ap.add_argument("--N", nargs="*", type=int, default=None, help="optional shared level list used with --families")
    ap.add_argument("--include-principal", action="store_true", default=True)
    ap.add_argument("--no-principal", dest="include_principal", action="store_false")
    ap.add_argument("--include-gamma1", action="store_true", default=True)
    ap.add_argument("--no-gamma1", dest="include_gamma1", action="store_false")
    ap.add_argument("--include-gamma0", action="store_true", default=True)
    ap.add_argument("--no-gamma0", dest="include_gamma0", action="store_false")
    ap.add_argument("--include-gamma0-orbifolds", action="store_true", help="also emit non-torsion-free Gamma0 audit records; they are excluded from mainline")
    ap.add_argument("--big", action="store_true", help="add principal Gamma(6), Gamma(7) and larger Gamma0 examples")
    ap.add_argument("--surfaces", default="", help="explicit comma-list overrides N lists")
    ap.add_argument("--maker", default="FuchsianDomainMaker_v13.py")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--outroot", default="modular_congruence_tester_runs")
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
    ap.add_argument("--max-generators", type=int, default=0, help="0 means no cap; refuse silent truncation")
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
    elif args.families.strip() or args.N is not None:
        # v1.3 convenience path.  This intentionally creates one explicit
        # surface spec per family/level rather than relying on any downstream
        # suite expansion.  Invalid low-level entries are skipped where the
        # family constructor is not defined as a smooth surface branch.
        raw_fams = [x.strip().lower() for x in args.families.split(",") if x.strip()]
        if not raw_fams:
            raw_fams = []
            if args.include_principal:
                raw_fams.append("gamma")
            if args.include_gamma1:
                raw_fams.append("gamma1")
            if args.include_gamma0:
                raw_fams.append("gamma0")
        fam_alias = {
            "principal": "gamma", "gamma(n)": "gamma", "gamma": "gamma",
            "g1": "gamma1", "gamma_1": "gamma1", "gamma1": "gamma1",
            "g0": "gamma0", "gamma_0": "gamma0", "gamma0": "gamma0",
        }
        fams = []
        for f in raw_fams:
            if f not in fam_alias:
                raise ValueError(f"Unknown --families entry {f!r}; use gamma,gamma1,gamma0")
            ff = fam_alias[f]
            if ff not in fams:
                fams.append(ff)
        levels = list(args.N or [])
        if not levels:
            # If --families is supplied without --N, fall back to the relevant
            # family-specific default level lists.
            if "gamma" in fams:
                levels.extend(args.principal_N)
            if "gamma1" in fams:
                levels.extend(args.gamma1_N)
            if "gamma0" in fams:
                levels.extend(args.gamma0_N)
            levels = sorted(set(levels))
        surfaces = []
        for N in levels:
            if "gamma" in fams and N >= 3:
                surfaces.append(f"gamma{N}")
            if "gamma1" in fams and N >= 4:
                surfaces.append(f"gamma1_{N}")
            if "gamma0" in fams and N >= 2:
                surfaces.append(f"gamma0_{N}")
    else:
        surfaces: List[str] = []
        if args.include_principal:
            Ns = list(args.principal_N)
            if args.big:
                for n in [6, 7]:
                    if n not in Ns:
                        Ns.append(n)
            surfaces += [f"gamma{N}" for N in Ns]
        if args.include_gamma1:
            surfaces += [f"gamma1_{N}" for N in args.gamma1_N]
        if args.include_gamma0:
            g0 = list(args.gamma0_N)
            if args.big:
                for n in [22, 24, 25, 26, 27, 28, 30]:
                    if n not in g0:
                        g0.append(n)
            surfaces += [f"gamma0_{N}" for N in g0]
        if args.include_gamma0_orbifolds:
            for n in [2, 3, 5, 7, 10, 13]:
                spec = f"gamma0_{n}"
                if spec not in surfaces:
                    surfaces.append(spec)

    # Preserve order but avoid duplicates, especially when shared --N lists overlap.
    seen_specs = set()
    deduped_specs: List[str] = []
    for spec in surfaces:
        if spec not in seen_specs:
            seen_specs.add(spec)
            deduped_specs.append(spec)
    surfaces = deduped_specs

    run_root = Path(args.outroot) / f"run_{now_stamp()}{('_' + args.label) if args.label else ''}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"{PROGRAM}\nrun_root={run_root}\nsurfaces={surfaces}\n" + "-" * 78, flush=True)
    ginn = load_module(args.ginn_script, "ginn_v24_modular")

    audit_rows: List[Dict[str, Any]] = []
    smoke_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for s in surfaces:
        print("=" * 78)
        print(f"[surface] {s}", flush=True)
        try:
            raw = ginn.make_surface(s, args.maker, max_cosets=args.max_cosets, max_generators=args.max_generators)
            ar = modular_audit(s, raw, args.max_generators)
            family, N = modular_family_and_level(s)
            eligible = bool(ar["pass_geometry_audit"])
            reason = "" if eligible else ("elliptic torsion/orbifold or failed modular audit")
            sj = normalize_surface_record(
                raw,
                surface_spec=s,
                surface_family="modular_congruence",
                surface_subfamily=family,
                source_program=PROGRAM,
                source_version=VERSION,
                construction_parameters={"N": N, "family": family, "surface": s},
                geometry_audit_pass=eligible,
                finite_area=True,
                torsion_free=bool(raw.get("torsion_free")),
                mainline_dataset_eligible=eligible,
                exclusion_reason=reason,
            )
            write_json(run_root / "surfaces" / f"{sj['surface_id']}.json", sj)
            audit_rows.append({**audit_row_from_surface(sj, ar["pass_geometry_audit"]), **ar})
            print(f"[audit] pass={ar['pass_geometry_audit']} fam={ar['family']} N={ar['N']} e2={ar['elliptic_order2_fixed_count']} e3={ar['elliptic_order3_fixed_count']} gbar={ar['compactified_genus_inferred']} cusps={ar['cusp_count']} gens={ar['generator_count']} truncated={ar['generator_truncated']}", flush=True)
            if args.ginn_smoke and eligible:
                sr = run_smoke(ginn, sj, s, args.ginn_pairs, args.ginn_depth, args.seed, args.ginn_max_word_ball)
                smoke_rows.append(sr)
                print(f"[ginn-smoke] PASS W={sr['word_ball_size']} shortcut={sr.get('shortcut_fraction')}", flush=True)
            elif args.ginn_smoke and not eligible:
                smoke_rows.append({"surface": s, "surface_id": sj["surface_id"], "pass_ginn_preflight": False, "error": "skipped because not mainline torsion-free Riemann surface"})
                print("[ginn-smoke] skipped: not mainline eligible", flush=True)
            if args.run_ginn and eligible:
                tr = call_ginn(args, s, run_root)
                train_rows.append(tr)
                print(f"[ginn-train] pass={tr['pass_ginn_training']} rc={tr['returncode']}", flush=True)
        except Exception as e:
            print(f"[FAIL] {s}: {type(e).__name__}: {e}", flush=True)
            failures.append({"surface": s, "surface_id": s, "error_type": type(e).__name__, "error": str(e)})

    write_csv(run_root / "tables" / "geometry_audit.csv", audit_rows, GEOMETRY_AUDIT_FIELDS)
    write_csv(run_root / "tables" / "modular_geometry_audit.csv", audit_rows, GEOMETRY_AUDIT_FIELDS)
    write_csv(run_root / "tables" / "ginn_smoke_summary.csv", smoke_rows, GINN_SMOKE_FIELDS)
    write_csv(run_root / "tables" / "ginn_training_summary.csv", train_rows, GINN_TRAINING_FIELDS)
    write_csv(run_root / "tables" / "failures.csv", failures, FAILURE_FIELDS)
    write_json(run_root / "manifest.json", {"program": PROGRAM, "version": VERSION, "surfaces": surfaces, "args": vars(args), "completed": len(audit_rows), "failures": len(failures), "contract_version": "fuchsian_surface_record_contract_v1"})
    print(f"[done] completed={len(audit_rows)} failures={len(failures)} run_root={run_root}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
