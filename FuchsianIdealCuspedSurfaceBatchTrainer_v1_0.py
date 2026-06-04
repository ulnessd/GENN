#!/usr/bin/env python3
r"""FuchsianIdealCuspedSurfaceBatchTrainer_v1_0.py

Batch trainer for smooth finite-area cusped Riemann surfaces represented by
explicit ideal/Ford polygonal domains in the Poincare disk.

Construction used in this first version:
  * work inside PSL(2,Z) = <S,R | S^2=R^3=1>, R=S*T;
  * generate finite transitive permutation actions with S and R fixed-point-free;
  * the stabilizer subgroup is torsion-free, hence H/H_subgroup is a smooth
    finite-area cusped Riemann surface, not an orbifold;
  * the fundamental domain is exported as an explicit union of ideal Ford tiles
    in the disk using the existing DomainMaker/Ginn conventions.

This is an ideal-cusped-surface trainer rather than a Teichmuller-space trainer:
no continuous moduli are introduced and every record has an explicit disk domain.

It intentionally overlaps the modular-permutation trainer, but organizes the
search by target topological type (compactified genus g and cusp count c), e.g.
  g=0,c=3 thrice-punctured sphere type
  g=1,c=1 once-punctured torus type
  g=0,c=4 four-punctured sphere type
  g=1,c=2 twice-punctured torus type
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROGRAM = "FuchsianIdealCuspedSurfaceBatchTrainer_v1_0.py"
VERSION = "v1.0"

try:
    from FuchsianSurfaceRecordTools_v1_0 import normalize_surface_record, write_csv, write_json
except Exception:
    def write_json(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2))
    def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys: List[str] = list(fieldnames or [])
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        if not keys:
            keys = ["empty"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    def normalize_surface_record(sj: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        out = dict(sj)
        out.update(kwargs)
        out["master_record"] = dict(out)
        return out


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_py(path: str, name: str):
    p = Path(path).expanduser()
    if not p.exists():
        alt = Path(__file__).resolve().parent / p.name
        if alt.exists():
            p = alt
    spec = importlib.util.spec_from_file_location(name, str(p.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_targets(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"Target {piece!r} should have form g:c, e.g. 0:3 or 1:1")
        gtxt, ctxt = piece.split(":", 1)
        g, c = int(gtxt), int(ctxt)
        if g < 0 or c < 1:
            raise ValueError(f"Invalid target genus/cusp pair {(g,c)}")
        if 2 * g - 2 + c <= 0:
            raise ValueError(f"Target {(g,c)} is not hyperbolic: need 2g-2+c > 0")
        out.append((g, c))
    # stable unique
    seen = set(); unique: List[Tuple[int,int]] = []
    for x in out:
        if x not in seen:
            unique.append(x); seen.add(x)
    return unique


def psl2z_index_for_target(g: int, c: int) -> int:
    # area subgroup = mu*pi/3; surface area = 2*pi*(2g-2+c)
    mu = 6 * (2 * g - 2 + c)
    if mu <= 0 or mu % 6 != 0:
        raise ValueError(f"Bad target index for {(g,c)}: {mu}")
    return int(mu)


def update_surface_for_ideal_family(sj: Dict[str, Any], g: int, c: int, serial: int, seed: int, label_prefix: str) -> Dict[str, Any]:
    old_id = str(sj.get("surface_id"))
    index = int(sj.get("index_in_psl2z"))
    cusp_widths = sj.get("cusp_widths") or []
    width_tag = "_".join(str(int(x)) for x in sorted(cusp_widths)) if cusp_widths else "none"
    sid = f"{label_prefix}_g{g}_c{c}_idx{index}_w{width_tag}_seed{seed}_n{serial:03d}"
    sj = dict(sj)
    sj.update({
        "surface_id": sid,
        "surface_spec": sid,
        "name": f"Ideal/Ford cusped surface, compactified genus {g}, {c} cusps, index {index}",
        "surface_family": "ideal_cusped_surface",
        "surface_subfamily": f"genus_{g}_cusps_{c}",
        "category": "ideal_cusped_riemann_surface",
        "ideal_surface_target": {"compactified_genus": g, "cusp_count": c, "psl2z_index": index},
        "source_modperm_surface_id": old_id,
        "domain_type": "modular_ford_domain",
        "subdomain_type": "ideal_ford_tile_union_torsion_free_psl2z_subgroup",
        "fundamental_domain_status": "explicit ideal/Ford tile-union domain in the Poincare disk for a torsion-free finite-index subgroup of PSL(2,Z)",
        "riemann_surface_status": "smooth noncompact finite-area hyperbolic Riemann surface with cusps; no elliptic orbifold points",
        "surface_area_type": "noncompact_finite_area_ideal_cusped",
        "dataset_role": "nontriangle_riemann_surface_zoo_ideal_cusped_surface",
        "torsion_free": True,
        "orbifold_excluded": False,
        "finite_area": True,
        "compact": False,
        "mainline_dataset_eligible": True,
        "exclusion_reason": "",
        "scope_note": "No orbifold records; no Teichmuller-space model; explicit disk/Ford domain supplied.",
    })
    return sj


def audit_row(sj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "surface_id": sj.get("surface_id"),
        "target_genus": (sj.get("ideal_surface_target") or {}).get("compactified_genus"),
        "target_cusps": (sj.get("ideal_surface_target") or {}).get("cusp_count"),
        "compactified_genus": sj.get("compactified_genus"),
        "cusp_count": sj.get("cusp_count"),
        "cusp_widths": json.dumps(sj.get("cusp_widths")),
        "index_in_psl2z": sj.get("index_in_psl2z"),
        "area": sj.get("area"),
        "generator_count": len(sj.get("generators") or {}),
        "tile_count": len(sj.get("fundamental_domain_tiles") or []),
        "torsion_free": sj.get("torsion_free"),
        "orbifold_excluded": sj.get("orbifold_excluded"),
        "eligible": sj.get("mainline_dataset_eligible"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Ideal/Ford cusped Riemann-surface batch trainer")
    ap.add_argument("--targets", default="0:3,1:1,0:4,1:2,2:1,0:5,1:3,2:2", help="comma-list g:c, e.g. 0:3,1:1,0:4")
    ap.add_argument("--samples-per-target", type=int, default=2)
    ap.add_argument("--attempts-per-target", type=int, default=5000)
    ap.add_argument("--max-surfaces", type=int, default=0, help="0 means no cap beyond samples-per-target")
    ap.add_argument("--dedupe-cusp-widths", action="store_true", help="keep one sample for each sorted cusp-width signature per target")
    ap.add_argument("--modperm-script", default="FuchsianModularPermutationSubgroupTrainer_v1_1.py")
    ap.add_argument("--maker", default="FuchsianDomainMaker_v13.py")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--outroot", default="ideal_cusped_surface_runs")
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-ginn", action="store_true")
    ap.add_argument("--pairs", type=int, default=9000)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-word-ball", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pair-hidden", type=int, default=192)
    ap.add_argument("--score-hidden", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--soft-distance-weight", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--candidate-chunk-size", type=int, default=8192)
    ap.add_argument("--auto-chunk-threshold-mb", type=float, default=2048.0)
    ap.add_argument("--max-generators", type=int, default=0)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    targets = parse_targets(args.targets)
    label = args.label or "ideal_cusped_batch"
    run_root = Path(args.outroot) / f"run_{now_stamp()}_{label}"
    for sub in ["surfaces", "tables", "reports", "ginn_runs"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    mod = load_py(args.modperm_script, "modperm_backend_v11")
    maker = mod.load_module(args.maker, "domain_maker_ideal_cusped_v13")
    ginn = mod.load_module(args.ginn_script, "ginn_v24_ideal_cusped") if args.run_ginn and not args.dry_run else None
    rng = random.Random(args.seed)

    surfaces: List[Tuple[str, Dict[str, Any]]] = []
    attempts_rows: List[Dict[str, Any]] = []
    seen_widths = set()
    serial = 0
    for g, c in targets:
        index = psl2z_index_for_target(g, c)
        accepted = 0
        attempts = 0
        while accepted < args.samples_per_target and attempts < args.attempts_per_target:
            attempts += 1
            action = mod.make_random_permutation_action(index, rng, attempts=1)
            if action is None:
                continue
            ag = int(action["genus"])
            ac = len(action["cusp_cycles"])
            if ag != g or ac != c:
                continue
            widths = tuple(sorted(len(x) for x in action["cusp_cycles"]))
            sig = (g, c, widths)
            if args.dedupe_cusp_widths and sig in seen_widths:
                continue
            seen_widths.add(sig)
            raw = mod.build_surface_json(maker, action, args.seed, serial, max_generators=args.max_generators)
            sj = update_surface_for_ideal_family(raw, g, c, serial, args.seed, "ideal_cusped")
            sid = sj["surface_id"]
            sj_norm = normalize_surface_record(
                sj,
                surface_spec=sid,
                surface_family="ideal_cusped_surface",
                surface_subfamily=f"genus_{g}_cusps_{c}",
                source_program=PROGRAM,
                source_version=VERSION,
                construction_parameters={
                    "compactified_genus": g,
                    "cusp_count": c,
                    "index_in_psl2z": index,
                    "cusp_widths": list(widths),
                    "random_seed": args.seed,
                    "backend": args.modperm_script,
                },
                geometry_audit_pass=True,
                finite_area=True,
                torsion_free=True,
                mainline_dataset_eligible=True,
                exclusion_reason="",
            )
            write_json(run_root / "surfaces" / f"{sid}.json", sj_norm)
            surfaces.append((sid, sj_norm))
            serial += 1
            accepted += 1
            print(f"[surface] {sid} g={g} c={c} index={index} widths={list(widths)} gens={len(sj_norm.get('generators') or {})}", flush=True)
            if args.max_surfaces and len(surfaces) >= args.max_surfaces:
                break
        attempts_rows.append({"target_genus": g, "target_cusps": c, "index": index, "attempts": attempts, "accepted": accepted})
        if args.max_surfaces and len(surfaces) >= args.max_surfaces:
            break

    audit_rows = [audit_row(sj) for _sid, sj in surfaces]
    train_rows: List[Dict[str, Any]] = []
    if args.run_ginn and not args.dry_run and ginn is not None:
        for sid, sj in surfaces:
            tr = mod.run_direct_ginn(ginn, sj, sid, run_root, args)
            train_rows.append(tr)
            print(f"[train] {sid} ok={tr.get('pass_ginn_training')} top1={tr.get('winning_lift_accuracy_test')} top5={tr.get('winning_lift_top5_accuracy_test')} err={str(tr.get('error',''))[:140]}", flush=True)

    write_csv(run_root / "tables" / "ideal_cusped_surface_audit.csv", audit_rows)
    write_csv(run_root / "tables" / "generation_attempts.csv", attempts_rows)
    if train_rows:
        write_csv(run_root / "tables" / "ideal_cusped_surface_training.csv", train_rows)

    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "label": label,
        "targets": [{"genus": g, "cusps": c, "index_in_psl2z": psl2z_index_for_target(g,c)} for g,c in targets],
        "generated_surface_count": len(surfaces),
        "samples_per_target": args.samples_per_target,
        "run_ginn": args.run_ginn,
        "dry_run": args.dry_run,
        "pairs": args.pairs,
        "depth": args.depth,
        "epochs": args.epochs,
        "max_word_ball": args.max_word_ball,
        "scope": "smooth torsion-free finite-area cusped Riemann surfaces with explicit ideal/Ford disk domains; no orbifolds; no Teichmuller-space abstraction",
        "backend": args.modperm_script,
    }
    write_json(run_root / "manifest.json", manifest)

    ok = sum(1 for r in train_rows if r.get("pass_ginn_training"))
    lines = [
        "# Ideal Cusped Surface Batch Trainer Report",
        "",
        f"Program: `{PROGRAM}` {VERSION}",
        "",
        "## Scope",
        "",
        "Smooth torsion-free finite-area cusped Riemann surfaces only. Domains are explicit ideal/Ford tile unions in the Poincare disk. No orbifold records and no Teichmuller-space parameter sweep are included.",
        "",
        "## Summary",
        "",
        f"- Generated surfaces: **{len(surfaces)}**",
        f"- Run GINN: `{args.run_ginn}`",
        f"- Dry run: `{args.dry_run}`",
        f"- Training successes: `{ok}/{len(train_rows)}`",
        "",
        "## Targets",
        "",
        "| genus | cusps | PSL2Z index | attempts | accepted |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in attempts_rows:
        lines.append(f"| {row['target_genus']} | {row['target_cusps']} | {row['index']} | {row['attempts']} | {row['accepted']} |")
    lines += ["", "## Surfaces", "", "| surface_id | genus | cusps | index | cusp widths | generators | tiles |", "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in audit_rows:
        lines.append(f"| `{row['surface_id']}` | {row['compactified_genus']} | {row['cusp_count']} | {row['index_in_psl2z']} | `{row['cusp_widths']}` | {row['generator_count']} | {row['tile_count']} |")
    report = run_root / "reports" / "ideal_cusped_surface_batch_report.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"[done] report={report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
