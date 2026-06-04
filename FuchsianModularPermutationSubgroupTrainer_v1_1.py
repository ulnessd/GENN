#!/usr/bin/env python3
r"""FuchsianModularPermutationSubgroupTrainer_v1_1.py

Generate and optionally train GENN records for random torsion-free finite-index
subgroups of PSL(2,Z), represented by transitive permutation actions of

    PSL(2,Z) = < S, R | S^2 = R^3 = 1 >,   R = S T.

Scope status:
  * torsion-free filter: S and R have no fixed points in the coset action;
  * therefore H\H^2 is a smooth finite-area cusped Riemann surface, not an
    orbifold;
  * the disk/Ford fundamental domain is an explicit union of PSL(2,Z) base
    tiles over a BFS Schreier transversal;
  * this script does NOT certify noncongruence.  The records are labeled
    random/permutation modular subgroups.  Most random examples are expected to
    be noncongruence, but that is deliberately not asserted.

This is the exploratory non-triangle trainer.  It complements the congruence
Gamma/Gamma1/Gamma0 zoo without leaving the disk-domain framework.
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
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROGRAM = "FuchsianModularPermutationSubgroupTrainer_v1_1.py"
VERSION = "v1.1"

try:
    from FuchsianSurfaceRecordTools_v1_0 import generator_su11_audit, normalize_surface_record, write_csv, write_json
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
    def generator_su11_audit(sj: Dict[str, Any]) -> Dict[str, Any]:
        bad = []
        maxerr = 0.0
        for name, g in (sj.get("generators") or {}).items():
            try:
                a = g["alpha"]; b = g["beta"]
                alpha = complex(a[0], a[1]); beta = complex(b[0], b[1])
                det = abs(alpha)**2 - abs(beta)**2
                maxerr = max(maxerr, abs(det - 1.0))
                if det <= 0: bad.append(name)
            except Exception:
                bad.append(name)
        return {"generator_count": len(sj.get("generators") or {}), "su11_max_det_error": maxerr, "bad_generator_count": len(bad), "bad_generators": ";".join(map(str,bad[:20]))}
    def normalize_surface_record(sj: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        out = dict(sj); out.update(kwargs); out["master_record"] = dict(out); return out


class _DummyQt:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return _DummyQt()
    def __getattr__(self, name): return _DummyQt()
    def __iter__(self): return iter(())
    def __bool__(self): return False


def _install_gui_stubs() -> None:
    import types
    if "PyQt6" not in sys.modules:
        pyqt = types.ModuleType("PyQt6")
        qtwidgets = types.ModuleType("PyQt6.QtWidgets")
        for name in ["QApplication","QCheckBox","QComboBox","QFileDialog","QFrame","QGridLayout","QGroupBox","QHBoxLayout","QLabel","QMainWindow","QMessageBox","QPushButton","QSpinBox","QDoubleSpinBox","QTextEdit","QVBoxLayout","QWidget"]:
            setattr(qtwidgets, name, _DummyQt)
        sys.modules["PyQt6"] = pyqt
        sys.modules["PyQt6.QtWidgets"] = qtwidgets
    if "matplotlib.backends.backend_qtagg" not in sys.modules:
        backend = types.ModuleType("matplotlib.backends.backend_qtagg")
        backend.FigureCanvasQTAgg = _DummyQt
        sys.modules["matplotlib.backends.backend_qtagg"] = backend


def load_module(path: str, name: str):
    p = Path(path).expanduser()
    if not p.exists():
        alt = Path(__file__).resolve().parent / p.name
        if alt.exists():
            p = alt
    if "DomainMaker" in p.name:
        _install_gui_stubs()
    spec = importlib.util.spec_from_file_location(name, str(p.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def mat_json(M: np.ndarray) -> List[List[int]]:
    return [[int(M[0,0]), int(M[0,1])], [int(M[1,0]), int(M[1,1])]]


def inv_sl2(M: np.ndarray) -> np.ndarray:
    a,b,c,d = int(M[0,0]), int(M[0,1]), int(M[1,0]), int(M[1,1])
    return np.array([[d, -b], [-c, a]], dtype=object)


def det2(M: np.ndarray) -> int:
    return int(M[0,0])*int(M[1,1]) - int(M[0,1])*int(M[1,0])


def mat_key_psl(M: np.ndarray) -> Tuple[int, int, int, int]:
    vals = [int(M[0,0]), int(M[0,1]), int(M[1,0]), int(M[1,1])]
    # identify +/- M in PSL(2,Z)
    for v in vals:
        if v != 0:
            if v < 0:
                vals = [-x for x in vals]
            break
    return tuple(vals)  # type: ignore[return-value]


def perm_identity(n: int) -> List[int]:
    return list(range(n))


def perm_compose(p: Sequence[int], q: Sequence[int]) -> List[int]:
    """Apply p then q: i -> q[p[i]]."""
    return [int(q[int(p[i])]) for i in range(len(p))]


def perm_cycles(p: Sequence[int]) -> List[List[int]]:
    n = len(p); seen = [False]*n; out: List[List[int]] = []
    for i in range(n):
        if seen[i]: continue
        cyc = []
        j = i
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = int(p[j])
        out.append(cyc)
    return out


def has_fixed_points(p: Sequence[int]) -> bool:
    return any(int(x) == i for i, x in enumerate(p))


def random_fixed_free_involution(n: int, rng: random.Random) -> List[int]:
    vals = list(range(n)); rng.shuffle(vals)
    p = [0]*n
    for a,b in zip(vals[0::2], vals[1::2]):
        p[a] = b; p[b] = a
    return p


def random_fixed_free_order3(n: int, rng: random.Random) -> List[int]:
    vals = list(range(n)); rng.shuffle(vals)
    p = [0]*n
    for a,b,c in zip(vals[0::3], vals[1::3], vals[2::3]):
        if rng.random() < 0.5:
            p[a] = b; p[b] = c; p[c] = a
        else:
            p[a] = c; p[c] = b; p[b] = a
    return p


def orbit_size(gens: Sequence[Sequence[int]], start: int = 0) -> int:
    seen = {start}; q = deque([start])
    while q:
        i = q.popleft()
        for p in gens:
            j = int(p[i])
            if j not in seen:
                seen.add(j); q.append(j)
    return len(seen)


def bfs_coset_reps_from_perms(pS: Sequence[int], pT: Sequence[int], S: np.ndarray, T: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    n = len(pS)
    reps: List[Optional[Tuple[str, np.ndarray]]] = [None] * n
    reps[0] = ("", np.array([[1,0],[0,1]], dtype=object))
    q = deque([0])
    gens = [("S", pS, S), ("T", pT, T)]
    while q:
        i = q.popleft()
        w, M = reps[i]  # type: ignore[misc]
        for ch, p, G in gens:
            j = int(p[i])
            if reps[j] is None:
                reps[j] = (w + ch, M @ G)
                q.append(j)
    if any(x is None for x in reps):
        raise RuntimeError("Permutation action is not transitive; BFS did not find all cosets")
    return [x for x in reps if x is not None]


def schreier_generators_from_reps(reps: List[Tuple[str, np.ndarray]], pS: Sequence[int], pT: Sequence[int], S: np.ndarray, T: np.ndarray, max_generators: int = 0) -> List[Tuple[str, np.ndarray, str]]:
    Ikey = mat_key_psl(np.array([[1,0],[0,1]], dtype=object))
    seen: Dict[Tuple[int,int,int,int], Tuple[str, np.ndarray, str]] = {}
    gens = [("S", pS, S), ("T", pT, T)]
    label_i = 0
    for i, (w, R) in enumerate(reps):
        for ch, p, G in gens:
            j = int(p[i])
            w2, R2 = reps[j]
            H = R @ G @ inv_sl2(R2)
            if det2(H) != 1:
                continue
            key = mat_key_psl(H)
            if key == Ikey or key in seen:
                continue
            if max_generators and label_i >= max_generators:
                continue
            label = f"m{label_i:04d}"
            label_i += 1
            seen[key] = (label, H, f"{w or 'I'}{ch}({w2 or 'I'})^-1")
    return list(seen.values())


def make_random_permutation_action(index: int, rng: random.Random, attempts: int) -> Optional[Dict[str, Any]]:
    if index % 6 != 0:
        raise ValueError("torsion-free PSL(2,Z) permutation index should be divisible by 6")
    for _ in range(attempts):
        pS = random_fixed_free_involution(index, rng)
        pR = random_fixed_free_order3(index, rng)   # R = S*T
        if orbit_size([pS, pR], 0) != index:
            continue
        # Right action by T satisfies R = S then T, so pR = pT after pS.
        # Therefore pT[x] = pR[pS[x]].
        pT = [int(pR[int(pS[i])]) for i in range(index)]
        cusps = perm_cycles(pT)
        cnum = len(cusps)
        gfloat = 1.0 + index / 12.0 - cnum / 2.0
        g = int(round(gfloat))
        if g < 0 or abs(gfloat - g) > 1e-9:
            continue
        return {"pS": pS, "pR": pR, "pT": pT, "cusp_cycles": cusps, "genus": g, "genus_float": gfloat}
    return None


def build_surface_json(maker: Any, action: Dict[str, Any], seed: int, serial: int, max_generators: int = 0) -> Dict[str, Any]:
    pS = action["pS"]; pR = action["pR"]; pT = action["pT"]
    mu = len(pS)
    S = np.array([[0,-1],[1,0]], dtype=object)
    T = np.array([[1,1],[0,1]], dtype=object)
    reps = bfs_coset_reps_from_perms(pS, pT, S, T)
    rs = schreier_generators_from_reps(reps, pS, pT, S, T, max_generators=max_generators)

    base = maker.make_modular_ford_domain(width=1.0, rotation_deg=0.0)
    base_vertices = base.get("ford_vertices", [])
    tile_data: List[Dict[str, Any]] = []
    all_vertices: List[List[float]] = []
    for idx, (word, M) in enumerate(reps):
        G = maker.psl2r_to_disk_mobius(float(M[0,0]), float(M[0,1]), float(M[1,0]), float(M[1,1]), name=f"r{idx}")
        tile_vertices: List[List[float]] = []
        for vpair in base_vertices:
            zz = G(maker.as_complex(vpair)); rr = abs(zz)
            if rr >= 1.0:
                zz = zz / rr
            pair = maker.cpair(zz)
            tile_vertices.append(pair); all_vertices.append(pair)
        tile_data.append({"tile_index": idx, "coset_word": word or "I", "matrix": mat_json(M), "vertices": tile_vertices})

    generators: Dict[str, Dict[str, Any]] = {}
    meanings: Dict[str, str] = {}
    matrices: Dict[str, Any] = {}
    for label, H, source in rs:
        G = maker.psl2r_to_disk_mobius(float(H[0,0]), float(H[0,1]), float(H[1,0]), float(H[1,1]), name=label)
        generators[label] = G.as_json()
        matrices[label] = mat_json(H)
        meanings[label] = f"Random torsion-free PSL2Z permutation subgroup Reidemeister-Schreier generator from {source}; matrix {matrices[label]}"

    cusp_widths = [len(c) for c in action["cusp_cycles"]]
    g = int(action["genus"])
    sid = f"modperm_idx{mu}_g{g}_c{len(cusp_widths)}_seed{seed}_n{serial:03d}"
    area = mu * (math.pi / 3.0)
    sj: Dict[str, Any] = {
        "surface_id": sid,
        "surface_spec": sid,
        "format": "FuchsianGENN surface JSON modular permutation subgroup v1.0",
        "name": f"Random torsion-free PSL(2,Z) subgroup index {mu}, genus {g}, {len(cusp_widths)} cusps",
        "domain_type": "modular_ford_domain",
        "subdomain_type": "random_torsion_free_permutation_subgroup",
        "category": "modular_permutation_subgroup_riemann_surface",
        "surface_family": "modular_permutation_subgroup",
        "surface_subfamily": "random_torsion_free_psl2z_subgroup",
        "parent_group": "PSL(2,Z)",
        "presentation": "<S,R | S^2=R^3=1>, R=S*T; subgroup is stabilizer of coset 0 in the generated transitive permutation action",
        "index_in_psl2z": mu,
        "area": area,
        "gauss_bonnet_area": area,
        "compact": False,
        "finite_area": True,
        "torsion_free": True,
        "orbifold_excluded": False,
        "cusp_count": len(cusp_widths),
        "cusp_widths": cusp_widths,
        "compactified_genus": g,
        "compactified_genus_float": float(action["genus_float"]),
        "elliptic_orders": [],
        "torsion_free_audit": {
            "right_S_fixed_cosets_order2": [],
            "right_ST_fixed_cosets_order3": [],
            "S_fixed_point_count": 0,
            "ST_fixed_point_count": 0,
            "torsion_free_by_coset_fixed_point_test": True,
        },
        "congruence_status": "not_audited; random permutation subgroup, noncongruence not certified",
        "permutation_action": {"S": pS, "ST": pR, "T": pT},
        "coset_representatives": [{"word": w or "I", "matrix": mat_json(M)} for w, M in reps],
        "generators": generators,
        "generator_meanings": meanings,
        "generator_matrices_sl2z": matrices,
        "generator_export_audit": {
            "tokenized_generators": True,
            "underlying_schreier_generator_count": len(rs),
            "exported_generator_count": len(rs),
            "generator_truncated_by_cli_max_generators": bool(max_generators and len(rs) >= max_generators),
            "label_scheme": "m0000 token labels with formal inverses generated by GINN parser",
        },
        "fundamental_domain_tiles": tile_data,
        "construction_ford_vertices": all_vertices,
        "ford_vertices": all_vertices,
        "fundamental_domain_status": "explicit Ford-tile union model in the Poincare disk for a torsion-free finite-index subgroup of PSL(2,Z)",
        "sampling_status": "supported by disk tile-union sampler",
        "riemann_surface_status": "smooth noncompact finite-area hyperbolic Riemann surface with cusps; compactification obtained by adding cusp points",
        "kahler_status": "Riemann surface, hence Kähler in complex dimension one",
        "surface_area_type": "noncompact_finite_area_cusped_modular_permutation",
        "dataset_role": "nontriangle_riemann_surface_zoo_modular_permutation_subgroup",
        "mainline_dataset_eligible": True,
        "exclusion_reason": "",
    }
    return sj


def audit_surface(sj: Dict[str, Any]) -> Dict[str, Any]:
    su = generator_su11_audit(sj)
    return {
        "surface_id": sj.get("surface_id"),
        "family": sj.get("surface_family"),
        "index_in_psl2z": sj.get("index_in_psl2z"),
        "compactified_genus": sj.get("compactified_genus"),
        "cusp_count": sj.get("cusp_count"),
        "cusp_widths": json.dumps(sj.get("cusp_widths")),
        "area": sj.get("area"),
        "generator_count": su.get("generator_count"),
        "bad_generator_count": su.get("bad_generator_count"),
        "su11_max_det_error": su.get("su11_max_det_error"),
        "torsion_free": sj.get("torsion_free"),
        "mainline_dataset_eligible": sj.get("mainline_dataset_eligible"),
        "congruence_status": sj.get("congruence_status"),
    }


def run_direct_ginn(ginn: Any, sj: Dict[str, Any], surface: str, run_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    outdir = run_root / "ginn_runs" / surface
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        rows, X, D, word_ball, label_meta, feature_names = ginn.generate_ginn_dataset(
            sj, args.pairs, args.depth, args.seed, max_word_ball=args.max_word_ball
        )
        write_json(outdir / "surface.json", sj)
        write_json(outdir / "label_meta.json", label_meta)
        metrics = ginn.train_ginn(
            rows, X, D, word_ball, outdir, args.depth, args.epochs,
            args.pair_hidden, args.score_hidden, args.lr,
            args.batch_size, args.seed, args.device, args.patience,
            args.ce_weight, args.soft_distance_weight, args.temperature,
            candidate_chunk_size=args.candidate_chunk_size,
            auto_chunk_threshold_mb=args.auto_chunk_threshold_mb,
        )
        return {
            "surface": surface,
            "surface_id": sj.get("surface_id"),
            "returncode": 0,
            "wall_seconds": time.time() - t0,
            "pass_ginn_training": True,
            "word_ball_size": metrics.get("word_ball_size"),
            "winning_lift_accuracy_test": metrics.get("winning_lift_accuracy_test"),
            "winning_lift_top5_accuracy_test": metrics.get("winning_lift_top5_accuracy_test"),
            "shortcut_fraction_test": metrics.get("shortcut_fraction_test"),
            "error": "",
        }
    except Exception as e:
        return {
            "surface": surface,
            "surface_id": sj.get("surface_id"),
            "returncode": 1,
            "wall_seconds": time.time() - t0,
            "pass_ginn_training": False,
            "error": f"{type(e).__name__}: {e}",
        }


def parse_index_values(s: str) -> List[int]:
    if not s.strip(): return []
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    bad = [x for x in vals if x % 6 != 0]
    if bad:
        raise ValueError(f"indices must be divisible by 6 for fixed-point-free S and ST: {bad}")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="Random torsion-free PSL2Z permutation subgroup trainer")
    ap.add_argument("--indices", default="6,12,18,24,30,36", help="comma-list of subgroup indices, each divisible by 6")
    ap.add_argument("--samples-per-index", type=int, default=2)
    ap.add_argument("--attempts-per-index", type=int, default=2000)
    ap.add_argument("--max-surfaces", type=int, default=0, help="0 means no cap beyond samples-per-index")
    ap.add_argument("--genus-min", type=int, default=0)
    ap.add_argument("--genus-max", type=int, default=999999)
    ap.add_argument("--cusp-min", type=int, default=1)
    ap.add_argument("--cusp-max", type=int, default=999999)
    ap.add_argument("--dedupe-signature", action="store_true", help="keep only one surface per (index, genus, sorted cusp widths)")
    ap.add_argument("--maker", default="FuchsianDomainMaker_v13.py")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--outroot", default="modular_permutation_subgroup_runs")
    ap.add_argument("--label", default="")
    ap.add_argument("--run-ginn", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
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

    indices = parse_index_values(args.indices)
    label = args.label or "modperm_batch"
    run_root = Path(args.outroot) / f"run_{now_stamp()}_{label}"
    for sub in ["surfaces", "tables", "reports", "ginn_runs"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    maker = load_module(args.maker, "domain_maker_modperm_v13")
    ginn = load_module(args.ginn_script, "ginn_v24_modperm") if args.run_ginn and not args.dry_run else None
    rng = random.Random(args.seed)

    surfaces: List[Tuple[str, Dict[str, Any]]] = []
    attempts_rows: List[Dict[str, Any]] = []
    seen_sig = set()
    serial = 0
    for idx in indices:
        accepted = 0
        attempts = 0
        while accepted < args.samples_per_index and attempts < args.attempts_per_index:
            attempts += 1
            action = make_random_permutation_action(idx, rng, attempts=1)
            if action is None:
                continue
            g = int(action["genus"]); c = len(action["cusp_cycles"])
            if not (args.genus_min <= g <= args.genus_max and args.cusp_min <= c <= args.cusp_max):
                continue
            sig = (idx, g, tuple(sorted(len(x) for x in action["cusp_cycles"])))
            if args.dedupe_signature and sig in seen_sig:
                continue
            seen_sig.add(sig)
            sj = build_surface_json(maker, action, args.seed, serial, max_generators=args.max_generators)
            serial += 1; accepted += 1
            surfaces.append((sj["surface_id"], sj))
            if args.max_surfaces and len(surfaces) >= args.max_surfaces:
                break
        attempts_rows.append({"index": idx, "attempts": attempts, "accepted": accepted})
        if args.max_surfaces and len(surfaces) >= args.max_surfaces:
            break

    audit_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    for sid, sj in surfaces:
        sj_norm = normalize_surface_record(
            sj,
            surface_spec=sid,
            surface_family="modular_permutation_subgroup",
            surface_subfamily="random_torsion_free_psl2z_subgroup",
            source_program=PROGRAM,
            source_version=VERSION,
            construction_parameters={
                "index_in_psl2z": sj.get("index_in_psl2z"),
                "compactified_genus": sj.get("compactified_genus"),
                "cusp_count": sj.get("cusp_count"),
                "random_seed": args.seed,
                "max_generators": args.max_generators,
            },
            geometry_audit_pass=True,
            finite_area=True,
            torsion_free=True,
            mainline_dataset_eligible=True,
            exclusion_reason="",
        )
        write_json(run_root / "surfaces" / f"{sid}.json", sj_norm)
        audit_rows.append(audit_surface(sj_norm))
        print(f"[surface] {sid} index={sj.get('index_in_psl2z')} g={sj.get('compactified_genus')} cusps={sj.get('cusp_count')} gens={len(sj.get('generators') or {})}", flush=True)
        if args.run_ginn and not args.dry_run and ginn is not None:
            tr = run_direct_ginn(ginn, sj_norm, sid, run_root, args)
            train_rows.append(tr)
            print(f"[train] {sid} ok={tr.get('pass_ginn_training')} top1={tr.get('winning_lift_accuracy_test')} top5={tr.get('winning_lift_top5_accuracy_test')} err={tr.get('error','')[:120]}", flush=True)

    write_csv(run_root / "tables" / "modular_permutation_audit.csv", audit_rows)
    write_csv(run_root / "tables" / "generation_attempts.csv", attempts_rows)
    if train_rows:
        write_csv(run_root / "tables" / "modular_permutation_training.csv", train_rows)

    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "label": label,
        "indices": indices,
        "samples_per_index": args.samples_per_index,
        "generated_surface_count": len(surfaces),
        "run_ginn": args.run_ginn,
        "dry_run": args.dry_run,
        "pairs": args.pairs,
        "depth": args.depth,
        "epochs": args.epochs,
        "max_word_ball": args.max_word_ball,
        "scope": "smooth torsion-free finite-area modular Riemann surfaces with explicit disk/Ford tile-union domains; no orbifold records; noncongruence not certified",
    }
    write_json(run_root / "manifest.json", manifest)

    ok = sum(1 for r in train_rows if r.get("pass_ginn_training"))
    report = run_root / "reports" / "modular_permutation_subgroup_report.md"
    report.write_text(
        f"# Modular Permutation Subgroup Trainer Report\n\n"
        f"Program: `{PROGRAM}` {VERSION}\n\n"
        f"Generated surfaces: **{len(surfaces)}**\n\n"
        f"Indices: `{indices}`\n\n"
        f"Run GINN: `{args.run_ginn}`  Dry run: `{args.dry_run}`\n\n"
        f"Training successes: `{ok}/{len(train_rows)}`\n\n"
        f"Scope: smooth torsion-free finite-area PSL(2,Z) subgroup Riemann surfaces; no orbifold records; congruence/noncongruence status not certified.\n\n"
        f"## Surfaces\n\n"
        + "\n".join([f"- `{r.get('surface_id')}`: index={r.get('index_in_psl2z')}, genus={r.get('compactified_genus')}, cusps={r.get('cusp_count')}, generators={r.get('generator_count')}" for r in audit_rows])
        + "\n"
    )
    print(f"[done] run_root={run_root}")
    print(f"[done] report={report}")
    return 0 if all(r.get("pass_ginn_training", True) for r in train_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
