#!/usr/bin/env python3
"""
FuchsianDownstairsGINN_v2_4.py

First true downstairs-learning experiment for the Fuchsian GENN project.

Purpose
-------
This script trains a neural network to learn quotient/downstairs geometry on a
fixed compact Fuchsian surface X = D/Gamma.  It does NOT merely relearn the
universal upstairs Poincare disk metric.  Instead it creates labeled point-pair
examples

    (p, q)  ->  d_X([p],[q]) = min_{gamma in B_R(Gamma)} d_D(p, gamma q)

using finite word searches in the deck group, then trains a small PyTorch model
to predict the downstairs quotient distance and shortest-lift word depth.

The original neural metric phi_theta remains useful as a calibration/sanity
checker, but this file is about the quotient/gluing behavior of the Riemann
surface.

Typical use
-----------
    python FuchsianDownstairsNet_v3.py --surface regular_g2 --pairs 6000 --depth 3
    python FuchsianDownstairsNet_v3.py --surface hurwitz --pairs 10000 --depth 3 --epochs 250
    python FuchsianDownstairsNet_v3.py --surface both --pairs 6000 --jobs 4

Outputs
-------
Creates a run directory:

    downstairs_runs/run_YYYYMMDD_HHMMSS_<surface>/
        surface.json
        word_ball.json
        pair_dataset.csv
        predictions_test.csv
        metrics.json
        train_log.csv
        run_manifest.json

Notes
-----
- v3 includes the main Poincare/Fuchsian families planned for the large dataset:
  compact regular polygons, Hurwitz/Klein PSL(2,7), Gamma(N), Gamma_1(N),
  torsion-free Hecke abelian covers, and torsion-free Hecke dihedral covers.
- The quotient distance labels are finite-search labels. They are excellent
  ML targets, but not global mathematical proofs unless the search is known to
  be exhaustive.
- v3 supports compact polygons and disk-tile/Ford-domain seeds. It adds early
  stopping, better run summaries, geodesic witness JSON files, and an optional
  two-expert committee protocol selected by validation RMSE.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import resource
import sys
import time
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:
    torch = None
    nn = None
    F = None

EPS = 1.0e-12


# -----------------------------------------------------------------------------
# Performance helpers
# -----------------------------------------------------------------------------

def rss_mb() -> float:
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 2)
        except Exception:
            pass
    try:
        # Linux reports ru_maxrss in KiB.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def cpu_seconds() -> float:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return float(ru.ru_utime + ru.ru_stime)
    except Exception:
        return float("nan")


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def print_perf(prefix: str, t0: float, c0: float) -> None:
    wall = time.time() - t0
    cpu = cpu_seconds() - c0
    eff = cpu / wall if wall > 0 else float("nan")
    print(f"[{prefix}] wall={wall:8.2f}s  cpu={cpu:8.2f}s  cpu/wall={eff:5.2f}  rss={rss_mb():8.1f} MB", flush=True)


# -----------------------------------------------------------------------------
# Headless DomainMaker import
# -----------------------------------------------------------------------------

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


def load_maker(path: str):
    _install_gui_stubs()
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"DomainMaker file not found: {p}")
    mod_name = f"_fuchsian_domain_maker_v13_{abs(hash(str(p))) & 0xffffffff:x}"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import DomainMaker from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # needed for dataclasses in imported file
    spec.loader.exec_module(mod)
    return mod


def default_maker_path() -> str:
    local = Path("FuchsianDomainMaker_v13.py")
    if local.exists():
        return str(local)
    return "/mnt/data/FuchsianDomainMaker_v13.py"


def main_surface_specs() -> List[str]:
    """Canonical mainline surfaces for the eventual large Poincare/Fuchsian dataset."""
    specs: List[str] = []
    specs += [f"regular_g{g}" for g in range(2, 9)]
    specs += ["hurwitz"]
    specs += [f"gamma{N}" for N in range(3, 8)]
    specs += [f"gamma1_{N}" for N in range(4, 11)]
    specs += [f"gamma0_{N}" for N in range(4, 31)]
    specs += [f"hecke_ab{q}" for q in [3, 4, 5, 6, 7, 8, 9, 10, 12]]
    specs += [f"hecke_d{q}" for q in [3, 4, 5, 6, 7, 8, 9, 10, 12]]
    return specs


def smoke_surface_specs() -> List[str]:
    return ["regular_g2", "regular_g3", "hurwitz", "gamma3", "gamma1_5", "gamma0_11", "hecke_ab5", "hecke_d5"]


def standard_surface_specs() -> List[str]:
    specs: List[str] = []
    specs += [f"regular_g{g}" for g in range(2, 6)]
    specs += ["hurwitz"]
    specs += [f"gamma{N}" for N in range(3, 7)]
    specs += [f"gamma1_{N}" for N in range(4, 8)]
    specs += [f"gamma0_{N}" for N in range(4, 16)]
    specs += [f"hecke_ab{q}" for q in range(3, 8)]
    specs += [f"hecke_d{q}" for q in range(3, 8)]
    return specs



def is_compact_surface_json(surface_json: Dict[str, Any]) -> bool:
    dt = str(surface_json.get('domain_type', '')).lower()
    if dt == 'compact_polygon':
        return True
    if surface_json.get('compact') is True:
        return True
    return False


def surface_family_from_name(surface_name: str) -> Tuple[str, Optional[int]]:
    if surface_name.startswith('regular_g'):
        try: return ('regular', int(surface_name.split('regular_g',1)[1]))
        except Exception: return ('regular', None)
    if surface_name == 'hurwitz':
        return ('hurwitz', 3)
    if surface_name.startswith('gamma1_'):
        try: return ('gamma1', int(surface_name.split('gamma1_',1)[1]))
        except Exception: return ('gamma1', None)
    if surface_name.startswith('gamma0_'):
        try: return ('gamma0', int(surface_name.split('gamma0_',1)[1]))
        except Exception: return ('gamma0', None)
    if surface_name.startswith('gamma'):
        try: return ('gamma', int(surface_name.split('gamma',1)[1]))
        except Exception: return ('gamma', None)
    if surface_name.startswith('hecke_ab'):
        try: return ('hecke_ab', int(surface_name.split('hecke_ab',1)[1]))
        except Exception: return ('hecke_ab', None)
    if surface_name.startswith('hecke_d'):
        try: return ('hecke_d', int(surface_name.split('hecke_d',1)[1]))
        except Exception: return ('hecke_d', None)
    return ('unknown', None)


def choose_auto_hyperparams(surface_name: str, surface_json: Dict[str, Any], profile: str) -> Dict[str, Any]:
    """Choose v2.2 production hyperparameters from the v1.2 sweep results.

    The choices are deliberately conservative.  Depth remains fixed by the CLI, but the
    recommended production depth is 2.  For Gamma(6)-like large noncompact word balls,
    the sweep showed that smaller networks and chunked scoring are safer.
    """
    fam, par = surface_family_from_name(surface_name)
    compact = is_compact_surface_json(surface_json)
    cfg: Dict[str, Any] = {}
    if profile == 'manual':
        return cfg
    if profile == 'fast':
        cfg.update(pairs=6000, pair_hidden=192, score_hidden=96, batch_size=256, lr=1.5e-3, epochs=160, patience=28, candidate_chunk_size=512)
        return cfg
    # balanced default
    if compact:
        if fam == 'regular':
            cfg.update(pairs=14000, pair_hidden=256, score_hidden=128, batch_size=256, lr=1.0e-3, epochs=220, patience=40, candidate_chunk_size=0)
        elif fam == 'hurwitz':
            cfg.update(pairs=14000, pair_hidden=256, score_hidden=128, batch_size=256, lr=1.0e-3, epochs=220, patience=40, candidate_chunk_size=0)
        else:
            cfg.update(pairs=14000, pair_hidden=256, score_hidden=128, batch_size=256, lr=1.0e-3, epochs=220, patience=40, candidate_chunk_size=0)
    else:
        large = (fam in {'gamma','gamma0'} and (par or 0) >= 6) or (fam in {'hecke_ab','hecke_d'} and (par or 0) >= 9)
        if large:
            cfg.update(pairs=9000, pair_hidden=192, score_hidden=96, batch_size=128, lr=1.5e-3, epochs=220, patience=40, candidate_chunk_size=512)
        else:
            cfg.update(pairs=9000, pair_hidden=384, score_hidden=192, batch_size=128, lr=7.0e-4, epochs=240, patience=45, candidate_chunk_size=512)
    if profile == 'accurate':
        if compact:
            if fam == 'regular':
                cfg.update(pairs=20000, pair_hidden=256, score_hidden=128, batch_size=256, lr=8.0e-4, epochs=260, patience=50)
            elif fam == 'hurwitz':
                cfg.update(pairs=14000, pair_hidden=256, score_hidden=192, batch_size=256, lr=1.0e-3, epochs=260, patience=50)
        else:
            large = (fam in {'gamma','gamma0'} and (par or 0) >= 6) or (fam in {'hecke_ab','hecke_d'} and (par or 0) >= 9)
            if large:
                cfg.update(pairs=12000, pair_hidden=256, score_hidden=128, batch_size=96, lr=1.0e-3, epochs=260, patience=50, candidate_chunk_size=384)
            else:
                cfg.update(pairs=14000, pair_hidden=384, score_hidden=192, batch_size=128, lr=7.0e-4, epochs=260, patience=50, candidate_chunk_size=512)
    return cfg



def _v24_label(i: int, prefix: str = "g") -> str:
    return f"{prefix}{i:04d}"


def _mat_json_any(maker, M: np.ndarray) -> List[List[int]]:
    try:
        return maker._matrix_json(M)
    except Exception:
        return [[int(M[0,0]), int(M[0,1])], [int(M[1,0]), int(M[1,1])]]


def _matrix_float_json_any(M: np.ndarray) -> List[List[float]]:
    return [[float(M[0,0]), float(M[0,1])], [float(M[1,0]), float(M[1,1])]]



def _v24_mat_mod(M: np.ndarray, N: int) -> np.ndarray:
    return (np.asarray(M, dtype=int) % int(N)).astype(int)


def _v24_in_gamma0_mod(M: np.ndarray, N: int) -> bool:
    A = _v24_mat_mod(M, N)
    return bool(int(A[1, 0]) % int(N) == 0)


def _v24_in_principal_gamma_mod(maker, M: np.ndarray, N: int) -> bool:
    return bool(maker._is_identity_psl_mod(M, N))


def _v24_in_gamma1_mod(maker, M: np.ndarray, N: int) -> bool:
    return bool(maker._in_gamma1_mod(M, N))


def _v24_subgroup_membership(maker, subgroup: str, N: int):
    if subgroup == "principal":
        return lambda M: _v24_in_principal_gamma_mod(maker, M, N)
    if subgroup == "gamma1":
        return lambda M: _v24_in_gamma1_mod(maker, M, N)
    if subgroup == "gamma0":
        return lambda M: _v24_in_gamma0_mod(M, N)
    raise ValueError(subgroup)


def _v24_coset_reps_modN(maker, N: int, subgroup: str, max_cosets: int = 20000) -> List[Tuple[str, np.ndarray]]:
    """Coset representatives for H\PSL(2,Z), generalized to Gamma0(N).

    DomainMaker v13 already has principal/Gamma1 support.  v2.4 keeps a local
    auditable copy so Gamma0(N) can be handled without changing the GUI maker.
    """
    from collections import deque
    S = maker._mat2(0, -1, 1, 0)
    T = maker._mat2(1, 1, 0, 1)
    gens = [("S", S), ("T", T)]
    I = maker._mat2(1, 0, 0, 1)
    in_H = _v24_subgroup_membership(maker, subgroup, N)
    reps: List[Tuple[str, np.ndarray]] = [("", I)]
    q = deque([0])

    def find_coset_index(M: np.ndarray) -> Optional[int]:
        for idx, (_w, R) in enumerate(reps):
            if in_H(M @ maker._mat_inv_sl2(R)):
                return idx
        return None

    while q:
        idx = q.popleft()
        w, R = reps[idx]
        for ch, G in gens:
            M = R @ G
            if find_coset_index(M) is None:
                if len(reps) >= max_cosets:
                    raise ValueError(f"Coset search exceeded max_cosets={max_cosets}; lower N or raise the explicit limit.")
                reps.append((w + ch, M))
                q.append(len(reps) - 1)
    return reps


def _v24_coset_action_permutation(maker, reps: List[Tuple[str, np.ndarray]], N: int, subgroup: str, G: np.ndarray) -> List[int]:
    in_H = _v24_subgroup_membership(maker, subgroup, N)
    perm: List[int] = []
    for _w, R in reps:
        M = R @ G
        found = None
        for j, (_wj, Rj) in enumerate(reps):
            if in_H(M @ maker._mat_inv_sl2(Rj)):
                found = j
                break
        if found is None:
            raise RuntimeError("Could not resolve coset action.")
        perm.append(found)
    return perm


def _full_modular_schreier_generators(maker, N: int, subgroup: str, reps: List[Tuple[str, np.ndarray]], max_generators: int = 0) -> List[Tuple[str, np.ndarray, str]]:
    """Token-labeled full Reidemeister-Schreier feedstock for Gamma(N)/Gamma_1(N).

    This intentionally avoids the old 26 one-letter cap.  A positive
    max_generators is an explicit CLI cap; 0 means no generator cap.
    """
    in_H = _v24_subgroup_membership(maker, subgroup, N)
    S = maker._mat2(0, -1, 1, 0)
    T = maker._mat2(1, 1, 0, 1)
    gens = [("S", S), ("T", T)]
    Ikey = maker._mat_key_psl_int(maker._mat2(1,0,0,1))
    seen: Dict[Tuple[int, ...], Tuple[str, np.ndarray, str]] = {}
    def find_rep(M: np.ndarray) -> Tuple[str, np.ndarray]:
        for wj, Rj in reps:
            if in_H(M @ maker._mat_inv_sl2(Rj)):
                return wj, Rj
        raise RuntimeError("Schreier representative not found")
    label_i = 0
    for w, R in reps:
        for ch, G in gens:
            M = R @ G
            w2, R2 = find_rep(M)
            H = R @ G @ maker._mat_inv_sl2(R2)
            if maker._det2(H) != 1:
                continue
            key = maker._mat_key_psl_int(H)
            if key == Ikey or key in seen:
                continue
            if max_generators and label_i >= max_generators:
                continue
            label = _v24_label(label_i, "g")
            label_i += 1
            seen[key] = (label, H, f"{w or 'I'}{ch}({w2 or 'I'})^-1")
    return list(seen.values())


def _make_modular_surface_v24(maker, N: int, subgroup: str, max_cosets: int, max_generators: int) -> Dict[str, Any]:
    """Build modular/Ford JSON without the old interactive N<=9 or 26-label cap.

    Explicit limits are now CLI limits: --max-cosets, --max-generators, and
    --max-word-ball.  This function keeps the JSON minimal but complete for
    GINN sampling and generator parsing.
    """
    if subgroup == "principal":
        if N < 3:
            raise ValueError("Gamma(N) torsion-free branch requires N>=3.")
        subgroup_name = f"Gamma({N})"
        subdomain_type = "torsion_free_principal_congruence_subgroup"
        category = "modular_principal_congruence_torsion_free"
    elif subgroup == "gamma1":
        if N < 4:
            raise ValueError("Gamma_1(N) torsion-free branch requires N>=4.")
        subgroup_name = f"Gamma_1({N})"
        subdomain_type = "torsion_free_gamma1_congruence_subgroup"
        category = "modular_gamma1_congruence_torsion_free"
    elif subgroup == "gamma0":
        if N < 2:
            raise ValueError("Gamma_0(N) requires N>=2.")
        subgroup_name = f"Gamma_0({N})"
        subdomain_type = "gamma0_congruence_subgroup_torsion_audited"
        category = "modular_gamma0_congruence_torsion_audited"
    else:
        raise ValueError(subgroup)

    reps = _v24_coset_reps_modN(maker, N, subgroup=subgroup, max_cosets=max_cosets)
    base = maker.make_modular_ford_domain(width=1.0, rotation_deg=0.0)
    base_vertices = base.get("ford_vertices", [])
    tile_data: List[dict] = []
    all_vertices: List[List[float]] = []
    for idx, (word, M) in enumerate(reps):
        G = maker.psl2r_to_disk_mobius(float(M[0,0]), float(M[0,1]), float(M[1,0]), float(M[1,1]), name=f"r{idx}")
        tile_vertices=[]
        for vpair in base_vertices:
            zz = G(maker.as_complex(vpair)); rr=abs(zz)
            if rr >= 1.0: zz = zz / rr
            pair = maker.cpair(zz)
            tile_vertices.append(pair); all_vertices.append(pair)
        tile_data.append({"tile_index": idx, "coset_word": word or "I", "matrix": _mat_json_any(maker, M), "vertices": tile_vertices})

    S = maker._mat2(0, -1, 1, 0); T = maker._mat2(1, 1, 0, 1); ST = S @ T
    perm_T = _v24_coset_action_permutation(maker, reps, N, subgroup, T)
    perm_S = _v24_coset_action_permutation(maker, reps, N, subgroup, S)
    perm_ST = _v24_coset_action_permutation(maker, reps, N, subgroup, ST)
    cusp_cycles = maker._perm_cycles(perm_T)
    cusp_widths = [len(c) for c in cusp_cycles]
    elliptic_order2_fixed = [i for i,j in enumerate(perm_S) if i==j]
    elliptic_order3_fixed = [i for i,j in enumerate(perm_ST) if i==j]
    mu = len(reps)
    area = mu * (math.pi/3.0)
    cnum = len(cusp_cycles)
    e2 = len(elliptic_order2_fixed)
    e3 = len(elliptic_order3_fixed)
    compactified_genus_float = 1.0 + mu/12.0 - e2/4.0 - e3/3.0 - cnum/2.0
    compactified_genus = int(round(compactified_genus_float))

    rs_full_uncapped = _full_modular_schreier_generators(maker, N, subgroup, reps, max_generators=0)
    rs = rs_full_uncapped if not max_generators else rs_full_uncapped[:max_generators]
    generators: Dict[str, dict] = {}
    meanings: Dict[str, str] = {}
    matrices: Dict[str, Any] = {}
    congruence_audit: List[dict] = []
    for label, H, source in rs:
        G = maker.psl2r_to_disk_mobius(float(H[0,0]), float(H[0,1]), float(H[1,0]), float(H[1,1]), name=label)
        generators[label] = G.as_json()
        matrices[label] = _mat_json_any(maker, H)
        mod_check = _v24_subgroup_membership(maker, subgroup, N)(H)
        meanings[label] = f"{subgroup_name} token generator from Reidemeister-Schreier word {source}; matrix {matrices[label]}"
        congruence_audit.append({"label":label,"source_word":source,"matrix":matrices[label],"matrix_mod_N":(maker._mat_mod(H,N)).tolist(),"determinant":maker._det2(H),"passes_subgroup_mod_N_check":bool(mod_check)})

    generator_export_audit = {
        "v2_4_tokenized_generators": True,
        "one_letter_label_cap_removed": True,
        "underlying_schreier_generator_count": len(rs_full_uncapped),
        "exported_generator_count": len(rs),
        "generator_truncated_by_cli_max_generators": bool(max_generators and len(rs_full_uncapped) > max_generators),
        "cli_max_generators": int(max_generators),
        "label_scheme": "g0000, g0001, ... with formal inverses token^-1",
    }
    return {
        "format":"FuchsianGENN surface JSON v24-tokenized",
        "domain_type":"modular_ford_domain",
        "subdomain_type":subdomain_type,
        "v12_polygon_compatible":False,
        "certification":{"status":f"certified_torsion_free_{subgroup_name.replace('(','_').replace(')','')}_v24_tokenized_seed","construction":f"{subgroup_name}; minimal Ford-tile union and full tokenized Reidemeister-Schreier generator export","audit":f"No hardcoded generator-label cap; {subgroup_name} torsion-free metadata/cusp/genus audit included; explicit CLI limits only.","warning":"Finite-area noncompact surface with cusps; GINN labels are finite word-ball labels, not global distance proofs.","explorer_loadable":False},
        "name":f"v2.4 tokenized {subgroup_name} modular/Ford Riemann-surface seed",
        "category":category,
        "parent_group":"PSL(2,Z)","subgroup":subgroup_name,"level_N":N,
        "index_in_psl2z":mu,"area":area,"gauss_bonnet_area":area,
        "cusp_count":cnum,"cusp_widths":cusp_widths,"elliptic_orders":[],"torsion_free":(len(elliptic_order2_fixed)==0 and len(elliptic_order3_fixed)==0),"compact":False,"finite_area":True,
        "mainline_dataset_eligible": (len(elliptic_order2_fixed)==0 and len(elliptic_order3_fixed)==0),
        "orbifold_excluded": not (len(elliptic_order2_fixed)==0 and len(elliptic_order3_fixed)==0),
        "exclusion_reason": "" if (len(elliptic_order2_fixed)==0 and len(elliptic_order3_fixed)==0) else "elliptic torsion fixed-cosets present; quotient is an orbifold, not a smooth Riemann surface",
        "compactification":{"name": (f"X({N})" if subgroup=='principal' else (f"X_1({N})" if subgroup=='gamma1' else f"X_0({N})")),"compactified_genus":compactified_genus,"compactified_genus_float":compactified_genus_float,"added_cusps":cnum},
        "compactified_genus": compactified_genus,
        "riemann_surface_status": (f"smooth noncompact finite-area Riemann surface H/{subgroup_name}; compactifies by adding cusp points" if (len(elliptic_order2_fixed)==0 and len(elliptic_order3_fixed)==0) else f"orbifold quotient H/{subgroup_name}; elliptic torsion fixed-cosets present, excluded from smooth Riemann-surface dataset"),
        "kahler_status":"complex dimension one; Kähler on the noncompact Riemann surface",
        "coset_representatives":[{"word":w or "I","matrix":_mat_json_any(maker,M)} for w,M in reps],
        "generators":generators,"generator_meanings":meanings,"generator_matrices_sl2z":matrices,"congruence_audit":congruence_audit,
        "torsion_free_audit":{"right_S_fixed_cosets_order2":elliptic_order2_fixed,"right_ST_fixed_cosets_order3":elliptic_order3_fixed,"torsion_free_by_coset_fixed_point_test":(len(elliptic_order2_fixed)==0 and len(elliptic_order3_fixed)==0)},
        "fundamental_domain_tiles":tile_data,
        "construction_ford_vertices": all_vertices,
        "ford_vertices": all_vertices,
        "ford_sides": [], "construction_ford_sides": [], "internal_ford_sides": [],
        "generator_export_audit": generator_export_audit,
        "mathematical_object":"torsion-free finite-area modular Riemann surface seed, noncompact with cusps",
        "certification_level":"torsion_free_modular_congruence_riemann_surface_seed",
        "explorer_mode_required":"modular_ford_domain","explorer_loadable":False,
        "batch_generation_status":"v2.4 tokenized feedstock for GINN; not constrained by one-letter GUI labels",
        "notes":"Generated by FuchsianDownstairsGINN v2.4 to avoid the previous 26-generator one-letter cap. Use --max-word-ball and --max-generators for explicit safety limits.",
    }


def _augment_hecke_full_generators_v24(surface_json: Dict[str, Any], maker, family: str, q: int, max_generators: int) -> Dict[str, Any]:
    """Replace one-letter-conservative Hecke generator export with token labels."""
    if family == 'hecke_ab':
        rs_full = maker._hecke_abelian_cover_generators(q, max_labels=100000)
    else:
        rs_full = maker._hecke_dihedral_cover_generators(q, max_labels=100000)
    rs = rs_full if not max_generators else rs_full[:max_generators]
    generators: Dict[str, dict] = {}
    meanings: Dict[str, str] = {}
    matrices: Dict[str, Any] = {}
    for i, (_old_label, H, source) in enumerate(rs):
        label = _v24_label(i, "h")
        G = maker.psl2r_to_disk_mobius(float(H[0,0]), float(H[0,1]), float(H[1,0]), float(H[1,1]), name=label)
        generators[label] = G.as_json()
        matrices[label] = _matrix_float_json_any(H)
        meanings[label] = f"Hecke {family} token generator from Reidemeister-Schreier word {source}; matrix {matrices[label]}"
    sj = dict(surface_json)
    sj['generators'] = generators
    sj['generator_meanings'] = meanings
    sj['generator_matrices_psl2r'] = matrices
    sj['generator_export_audit'] = {
        "v2_4_tokenized_generators": True,
        "one_letter_label_cap_removed": True,
        "underlying_schreier_generator_count": len(rs_full),
        "exported_generator_count": len(rs),
        "generator_truncated_by_cli_max_generators": bool(max_generators and len(rs_full) > max_generators),
        "cli_max_generators": int(max_generators),
        "label_scheme": "h0000, h0001, ... with formal inverses token^-1",
    }
    sj['format'] = str(sj.get('format','')) + ' + v24-tokenized-generators'
    sj['notes'] = str(sj.get('notes','')) + ' v2.4 replaced previous one-letter Hecke generator subset with tokenized full generator feedstock.'
    return sj

def make_surface(surface: str, maker_path: str, max_cosets: int = 20000, max_generators: int = 0) -> Dict[str, Any]:
    maker = load_maker(maker_path)
    surface = surface.lower().strip()
    if surface in {"hurwitz", "klein", "klein_quartic", "hurwitz_klein"}:
        return maker.make_hurwitz_klein_quartic_surface()
    if surface.startswith("regular_g"):
        return maker.make_regular_genus_surface(int(surface.replace("regular_g", "")))
    if surface.startswith("gamma1_"):
        N = int(surface.split("_", 1)[1])
        return _make_modular_surface_v24(maker, N, "gamma1", max_cosets=max_cosets, max_generators=max_generators)
    if surface.startswith("gamma0_"):
        N = int(surface.split("_", 1)[1])
        return _make_modular_surface_v24(maker, N, "gamma0", max_cosets=max_cosets, max_generators=max_generators)
    if surface.startswith("gamma"):
        N = int(surface.replace("gamma", ""))
        return _make_modular_surface_v24(maker, N, "principal", max_cosets=max_cosets, max_generators=max_generators)
    if surface.startswith("hecke_ab"):
        q = int(surface.replace("hecke_ab", ""))
        return _augment_hecke_full_generators_v24(maker.make_hecke_torsion_free_abelian_cover(q), maker, 'hecke_ab', q, max_generators)
    if surface.startswith("hecke_d") or surface.startswith("hecke_dihedral"):
        q = int(surface.replace("hecke_dihedral", "").replace("hecke_d", ""))
        return _augment_hecke_full_generators_v24(maker.make_hecke_torsion_free_dihedral_cover(q), maker, 'hecke_d', q, max_generators)
    if surface in {"g2", "genus2"}:
        return maker.make_regular_genus_surface(2)
    if surface in {"g3", "genus3"}:
        return maker.make_regular_genus_surface(3)
    raise ValueError(f"Unknown v2.4 surface {surface!r}. Use --list-surfaces to see supported specs.")


def surface_family(surface: str) -> str:
    s = surface.lower()
    if s.startswith("regular_g"):
        return "compact_regular_polygon"
    if s == "hurwitz" or "klein" in s:
        return "hurwitz_klein_psl27"
    if s.startswith("gamma1_"):
        return "modular_gamma1"
    if s.startswith("gamma0_"):
        return "modular_gamma0"
    if s.startswith("gamma"):
        return "modular_principal_gamma"
    if s.startswith("hecke_ab"):
        return "hecke_abelian_cover"
    if s.startswith("hecke_d"):
        return "hecke_dihedral_cover"
    return "unknown"


# -----------------------------------------------------------------------------
# SU(1,1), words, and disk geometry
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Mobius:
    alpha: complex
    beta: complex
    word: str = ""

    def __call__(self, z: complex) -> complex:
        denom = self.beta.conjugate() * z + self.alpha.conjugate()
        if abs(denom) < 1.0e-14:
            denom = 1.0e-14 + 0j
        return (self.alpha * z + self.beta) / denom

    def compose(self, other: "Mobius", word: Optional[str] = None) -> "Mobius":
        # self after other: self(other(z))
        a1, b1 = self.alpha, self.beta
        a2, b2 = other.alpha, other.beta
        a = a1 * a2 + b1 * b2.conjugate()
        b = a1 * b2 + b1 * a2.conjugate()
        return Mobius(a, b, self.word + other.word if word is None else word).normalized()

    def inverse(self, word: Optional[str] = None) -> "Mobius":
        return Mobius(self.alpha.conjugate(), -self.beta, word if word is not None else invert_word(self.word)).normalized()

    def normalized(self) -> "Mobius":
        det = abs(self.alpha) ** 2 - abs(self.beta) ** 2
        if det <= 0 or not np.isfinite(det):
            return self
        scale = 1.0 / math.sqrt(det)
        a = self.alpha * scale
        b = self.beta * scale
        # remove harmless global sign ambiguity by making Re(alpha) mostly positive
        if a.real < 0:
            a, b = -a, -b
        return Mobius(a, b, self.word)

    def trace_real(self) -> float:
        # SU(1,1) matrix trace alpha + alpha.conj = 2 Re alpha
        return float(2.0 * self.alpha.real)



def invert_letter(ch: str) -> str:
    """Legacy one-letter inverse helper retained for old compact-word output."""
    return ch.lower() if ch.isupper() else ch.upper()


def invert_word(w: str) -> str:
    """Legacy one-letter inverse helper.  Tokenized words use invert_token_word."""
    return "".join(invert_letter(c) for c in reversed(w))


def reduce_word(w: str) -> str:
    """Legacy one-letter reducer.  Tokenized words are reduced in build_word_ball."""
    st: List[str] = []
    for c in w:
        if st and invert_letter(c) == st[-1]:
            st.pop()
        else:
            st.append(c)
    return "".join(st)


def inverse_token(tok: str) -> str:
    """Return the formal inverse token for arbitrary generator labels.

    v2.4 removes the earlier one-character label assumption.  Positive
    generators may be A, B, g000, h042, etc.; their inverses are represented
    as A^-1, B^-1, g000^-1, h042^-1.  This keeps word parsing unambiguous.
    """
    tok = str(tok)
    return tok[:-3] if tok.endswith('^-1') else tok + '^-1'


def word_to_string(tokens: Tuple[str, ...] | List[str]) -> str:
    return " ".join(tokens)


def word_depth_string(w: str) -> int:
    if not w:
        return 0
    return len(str(w).split())


def parse_generators(surface_json: Dict[str, Any]) -> Dict[str, Mobius]:
    """Parse all SU(1,1) generators, including multi-token labels.

    Earlier versions silently ignored labels whose length was not one.  That
    created a hidden 26-generator cap for large modular examples.  v2.4 accepts
    arbitrary JSON generator labels and creates explicit formal inverse labels.
    """
    gens: Dict[str, Mobius] = {}
    raw = surface_json.get("generators", {})
    for label, g in raw.items():
        label = str(label)
        if g.get("type") != "su11":
            continue
        ar, ai = g["alpha"]
        br, bi = g["beta"]
        M = Mobius(complex(ar, ai), complex(br, bi), label).normalized()
        inv = inverse_token(label)
        if label in gens or inv in gens:
            raise ValueError(f"Generator label collision for {label!r}; labels and formal inverses must be unique.")
        gens[label] = M
        gens[inv] = M.inverse(inv)
    if not gens:
        raise ValueError("No SU(1,1) generators found in surface JSON.")
    return gens


def compose_token_word(tokens: Tuple[str, ...], gens: Dict[str, Mobius]) -> Mobius:
    current = Mobius(1.0 + 0j, 0.0 + 0j, "")
    for tok in tokens:
        current = gens[tok].compose(current, word="")
    return Mobius(current.alpha, current.beta, word_to_string(tokens)).normalized()


def build_word_ball(gens: Dict[str, Mobius], depth: int) -> List[Mobius]:
    """Build a reduced finite word ball using arbitrary token labels.

    The returned Mobius.word is a human-readable space-separated token string.
    The identity has word ''.
    """
    letters = sorted(gens.keys(), key=lambda c: (c.replace('^-1',''), c.endswith('^-1'), c))
    words: List[Tuple[str, ...]] = [tuple()]
    frontier: List[Tuple[str, ...]] = [tuple()]
    for _ in range(depth):
        new_frontier: List[Tuple[str, ...]] = []
        for w in frontier:
            last = w[-1] if w else None
            for tok in letters:
                if last is not None and inverse_token(tok) == last:
                    continue
                nw = w + (tok,)
                new_frontier.append(nw)
                words.append(nw)
        frontier = new_frontier
    seen = set()
    out: List[Mobius] = []
    for toks in words:
        if toks in seen:
            continue
        seen.add(toks)
        if not toks:
            out.append(Mobius(1 + 0j, 0 + 0j, ""))
        else:
            out.append(compose_token_word(toks, gens))
    return out

def disk_distance(z: complex, w: complex) -> float:
    az = abs(z)
    aw = abs(w)
    if az >= 1.0:
        z = z / (az + 1.0e-12) * (1.0 - 1.0e-12)
    if aw >= 1.0:
        w = w / (aw + 1.0e-12) * (1.0 - 1.0e-12)
    num = 2.0 * abs(z - w) ** 2
    den = max((1.0 - abs(z) ** 2) * (1.0 - abs(w) ** 2), 1.0e-300)
    arg = 1.0 + num / den
    return float(math.acosh(max(1.0, arg)))


def polygon_vertices_complex(surface_json: Dict[str, Any]) -> List[complex]:
    verts = []
    for xy in surface_json.get("polygon_vertices", []):
        verts.append(complex(float(xy[0]), float(xy[1])))
    if len(verts) < 3:
        raise ValueError("Surface has no compact polygon vertices.")
    return verts


def point_in_poly(x: float, y: float, poly: List[complex]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i].real, poly[i].imag
        xj, yj = poly[j].real, poly[j].imag
        if ((yi > y) != (yj > y)):
            x_inter = (xj - xi) * (y - yi) / ((yj - yi) + 1.0e-300) + xi
            if x < x_inter:
                inside = not inside
        j = i
    return inside


def triangle_area2(a: complex, b: complex, c: complex) -> float:
    return abs((b.real-a.real)*(c.imag-a.imag) - (b.imag-a.imag)*(c.real-a.real))


def sample_points_in_polygon(poly: List[complex], n: int, rng: random.Random) -> np.ndarray:
    xs = [z.real for z in poly]
    ys = [z.imag for z in poly]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pts: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = max(10000, n * 300)
    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if x * x + y * y >= 0.999999 ** 2:
            continue
        if point_in_poly(x, y, poly):
            pts.append((x, y))
    if len(pts) < n:
        raise RuntimeError(f"Only sampled {len(pts)} / {n} points inside polygon after {attempts} attempts.")
    return np.asarray(pts, dtype=np.float64)


def disk_tiles_from_surface(surface_json: Dict[str, Any]) -> List[List[complex]]:
    tiles: List[List[complex]] = []
    for tile in surface_json.get("fundamental_domain_tiles", []):
        verts = tile.get("vertices", [])
        if len(verts) >= 3:
            tri = [complex(float(xy[0]), float(xy[1])) for xy in verts[:3]]
            if triangle_area2(tri[0], tri[1], tri[2]) > 1.0e-12:
                tiles.append(tri)
    return tiles


def sample_points_in_triangle_union(tiles: List[List[complex]], n: int, rng: random.Random) -> np.ndarray:
    if not tiles:
        raise ValueError("No disk tiles available for sampling.")
    areas = np.asarray([triangle_area2(t[0], t[1], t[2]) for t in tiles], dtype=np.float64)
    probs = areas / areas.sum()
    cum = np.cumsum(probs)
    pts: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = max(10000, n * 100)
    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        u = rng.random()
        k = int(np.searchsorted(cum, u, side="right"))
        if k >= len(tiles):
            k = len(tiles) - 1
        a, b, c = tiles[k]
        r1 = math.sqrt(rng.random())
        r2 = rng.random()
        z = (1-r1)*a + r1*(1-r2)*b + r1*r2*c
        if abs(z) >= 0.999999:
            # Avoid ideal boundary points/cusp tips.
            z = z / (abs(z) + 1.0e-12) * (1.0 - 1.0e-6)
        pts.append((z.real, z.imag))
    if len(pts) < n:
        raise RuntimeError(f"Only sampled {len(pts)} / {n} points in tile union after {attempts} attempts.")
    return np.asarray(pts, dtype=np.float64)


def sample_points_on_surface(surface_json: Dict[str, Any], n: int, rng: random.Random) -> Tuple[np.ndarray, str]:
    if surface_json.get("domain_type") == "compact_polygon" and surface_json.get("polygon_vertices"):
        return sample_points_in_polygon(polygon_vertices_complex(surface_json), n, rng), "compact_polygon"
    tiles = disk_tiles_from_surface(surface_json)
    if tiles:
        return sample_points_in_triangle_union(tiles, n, rng), "disk_tile_union"
    raise ValueError("Surface cannot be sampled by v2: no compact polygon vertices or disk fundamental_domain_tiles.")


# -----------------------------------------------------------------------------
# Label generation
# -----------------------------------------------------------------------------

def label_pair(args: Tuple[int, float, float, float, float, List[Tuple[str, float, float]]]) -> Dict[str, Any]:
    idx, px, py, qx, qy, word_specs = args
    p = complex(px, py)
    q = complex(qx, qy)
    id_dist = disk_distance(p, q)
    best_word = ""
    best_dist = id_dist
    best_trace = 2.0
    best_gq = q
    equal_count = 1
    for w, ar, ai, br, bi in word_specs:  # type: ignore[misc]
        # This path supports legacy tuples if produced incorrectly, but main code uses 5-tuples.
        pass
    raise RuntimeError("Internal label_pair received bad packed word specs.")


def label_pair_packed(args: Tuple[int, float, float, float, float, List[Tuple[str, float, float, float, float]]]) -> Dict[str, Any]:
    idx, px, py, qx, qy, word_specs = args
    p = complex(px, py)
    q = complex(qx, qy)
    id_dist = disk_distance(p, q)
    best_word = ""
    best_dist = id_dist
    best_trace = 2.0
    best_gq = q
    equal_count = 1
    for w, ar, ai, br, bi in word_specs:
        alpha = complex(ar, ai)
        beta = complex(br, bi)
        denom = beta.conjugate() * q + alpha.conjugate()
        if abs(denom) < 1.0e-14:
            continue
        gq = (alpha * q + beta) / denom
        if abs(gq) >= 1.0:
            # Numerical drift only; SU(1,1) should preserve disk.
            gq = gq / (abs(gq) + 1.0e-12) * (1.0 - 1.0e-12)
        d = disk_distance(p, gq)
        if d + 1.0e-10 < best_dist:
            best_dist = d
            best_word = w
            best_trace = float(2.0 * alpha.real)
            best_gq = gq
            equal_count = 1
        elif abs(d - best_dist) <= 1.0e-8:
            equal_count += 1
    ratio = best_dist / id_dist if id_dist > 1.0e-12 else 1.0
    return {
        "pair_id": idx,
        "p_x": px,
        "p_y": py,
        "q_x": qx,
        "q_y": qy,
        "identity_distance": id_dist,
        "quotient_distance": best_dist,
        "quotient_ratio": ratio,
        "shortest_lift_word": best_word if best_word else "identity",
        "shortest_lift_depth": len(best_word),
        "shortest_lift_trace": best_trace,
        "lifted_q_x": float(best_gq.real),
        "lifted_q_y": float(best_gq.imag),
        "geodesic_length_check": best_dist,
        "nontrivial_shortcut": int(best_word != ""),
        "equal_shortest_lift_count": equal_count,
        # v2 proxy: the selected deck word is the first downstairs gluing label.
        # Full side-crossing itinerary will be added later.
        "crossing_word_proxy": best_word,
        "crossing_count_proxy": len(best_word),
    }


def generate_pair_dataset(
    surface_json: Dict[str, Any],
    n_pairs: int,
    depth: int,
    seed: int,
    jobs: int,
    progress_every: int = 1000,
) -> Tuple[List[Dict[str, Any]], List[Mobius], Dict[str, Any]]:
    print(f"[labels] parsing generators and building word ball depth={depth} ...", flush=True)
    gens = parse_generators(surface_json)
    word_ball = build_word_ball(gens, depth)
    print(f"[labels] generators={len(gens)//2}  word_ball_size={len(word_ball)}", flush=True)
    rng = random.Random(seed)
    pts, sampler_kind = sample_points_on_surface(surface_json, 2 * n_pairs, rng)
    print(f"[labels] sampler={sampler_kind}  sampled_points={len(pts)}", flush=True)
    word_specs = [(m.word, m.alpha.real, m.alpha.imag, m.beta.real, m.beta.imag) for m in word_ball]
    tasks = []
    for i in range(n_pairs):
        p = pts[2 * i]
        q = pts[2 * i + 1]
        tasks.append((i, float(p[0]), float(p[1]), float(q[0]), float(q[1]), word_specs))

    t0, c0 = time.time(), cpu_seconds()
    rows: List[Dict[str, Any]] = []
    if jobs <= 1:
        for k, task in enumerate(tasks, 1):
            rows.append(label_pair_packed(task))
            if k % progress_every == 0 or k == len(tasks):
                print(f"[labels] {k}/{len(tasks)} pairs labeled", flush=True)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(label_pair_packed, task) for task in tasks]
            for fut in as_completed(futs):
                rows.append(fut.result())
                done += 1
                if done % progress_every == 0 or done == len(tasks):
                    print(f"[labels] {done}/{len(tasks)} pairs labeled", flush=True)
    rows.sort(key=lambda r: int(r["pair_id"]))
    print_perf("labels", t0, c0)

    shortcut_frac = float(np.mean([r["nontrivial_shortcut"] for r in rows])) if rows else 0.0
    depth_vals = np.asarray([r["shortest_lift_depth"] for r in rows], dtype=float)
    meta = {
        "word_ball_depth": depth,
        "word_ball_size": len(word_ball),
        "sampler_kind": sampler_kind,
        "shortcut_fraction": shortcut_frac,
        "mean_shortest_lift_depth": float(depth_vals.mean()) if len(depth_vals) else None,
        "max_shortest_lift_depth": int(depth_vals.max()) if len(depth_vals) else None,
    }
    print(f"[labels] shortcut_fraction={shortcut_frac:.3f}  mean_depth={meta['mean_shortest_lift_depth']:.3f}", flush=True)
    return rows, word_ball, meta



# -----------------------------------------------------------------------------
# Geometry-informed downstairs neural network (GINN)
# -----------------------------------------------------------------------------

"""
GINN v1 conceptual shift
------------------------
The earlier DownstairsNet MLP learned

    (p,q, simple features) -> d_X([p],[q])

directly.  That proved the quotient-distance target is learnable, but the
architecture did not strongly respect the known Fuchsian geometry.

This GINN version embeds the geometry of the quotient-distance formula into the
architecture.  For a fixed surface and finite word ball B_R(Gamma), it computes
candidate lifted distances

    d_gamma(p,q) = d_D(p, gamma q)

for every gamma in B_R(Gamma).  The neural network then learns the hard part:
which candidate sheet/lift is the relevant downstairs branch.

The model is therefore a lift-selector/soft-min geometry-informed learner:

    neural scores over gamma  ->  predicted winning lift gamma_hat
    gamma_hat + exact d_D     ->  predicted downstairs distance

This is not a pure mathematical proof and not a global exact solver.  It is a
finite-word, learned branch-selection surrogate for downstairs quotient geometry.
"""


def disk_distance_array(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Vectorized Poincare disk distance for complex arrays p,w."""
    p = np.asarray(p, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    ap = np.abs(p)
    aw = np.abs(w)
    # Clip rare numerical drift.
    p = np.where(ap >= 1.0, p / (ap + 1.0e-12) * (1.0 - 1.0e-12), p)
    w = np.where(aw >= 1.0, w / (aw + 1.0e-12) * (1.0 - 1.0e-12), w)
    num = 2.0 * np.abs(p - w) ** 2
    den = np.maximum((1.0 - np.abs(p) ** 2) * (1.0 - np.abs(w) ** 2), 1.0e-300)
    arg = 1.0 + num / den
    return np.arccosh(np.maximum(1.0, arg)).astype(np.float64)


def apply_mobius_to_array(alpha: complex, beta: complex, q: np.ndarray) -> np.ndarray:
    denom = beta.conjugate() * q + alpha.conjugate()
    z = (alpha * q + beta) / denom
    az = np.abs(z)
    z = np.where(az >= 1.0, z / (az + 1.0e-12) * (1.0 - 1.0e-12), z)
    return z


def apply_mobius_to_point(alpha: complex, beta: complex, q: complex) -> complex:
    denom = beta.conjugate() * q + alpha.conjugate()
    z = (alpha * q + beta) / denom
    if abs(z) >= 1.0:
        z = z / (abs(z) + 1.0e-12) * (1.0 - 1.0e-12)
    return z


def pair_features_from_arrays(p: np.ndarray, q: np.ndarray, identity_dist: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    px = p.real; py = p.imag; qx = q.real; qy = q.imag
    pr = np.abs(p); qr = np.abs(q)
    pa = np.angle(p); qa = np.angle(q)
    X = np.column_stack([
        px, py, qx, qy,
        identity_dist,
        np.abs(p-q),
        pr, qr,
        np.cos(pa), np.sin(pa), np.cos(qa), np.sin(qa),
    ]).astype(np.float32)
    feature_names = [
        "p_x", "p_y", "q_x", "q_y",
        "identity_distance", "euclidean_pair_distance",
        "p_radius", "q_radius",
        "p_angle_cos", "p_angle_sin", "q_angle_cos", "q_angle_sin",
    ]
    return X, feature_names


def generate_ginn_dataset(
    surface_json: Dict[str, Any],
    n_pairs: int,
    depth: int,
    seed: int,
    max_word_ball: int = 50000,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[Mobius], Dict[str, Any], List[str]]:
    """Generate point pairs plus all candidate lift distances.

    Returns rows, X pair features, D candidate distance matrix, word_ball, meta, feature_names.
    D[i,j] = d_D(p_i, gamma_j q_i), with word_ball[0] the identity.
    """
    print(f"[ginn-labels] building word ball depth={depth} ...", flush=True)
    gens = parse_generators(surface_json)
    word_ball = build_word_ball(gens, depth)
    print(f"[ginn-labels] generators={len(gens)//2}  word_ball_size={len(word_ball)}", flush=True)
    if max_word_ball > 0 and len(word_ball) > max_word_ball:
        raise RuntimeError(f"word_ball_size={len(word_ball)} exceeds --max-word-ball={max_word_ball}. Use smaller depth, larger max, or future staged-pruning mode.")

    rng = random.Random(seed)
    pts, sampler_kind = sample_points_on_surface(surface_json, 2*n_pairs, rng)
    p = pts[0::2,0] + 1j*pts[0::2,1]
    q = pts[1::2,0] + 1j*pts[1::2,1]
    if len(p) != n_pairs:
        raise RuntimeError("Internal point sampling mismatch.")

    t0, c0 = time.time(), cpu_seconds()
    W = len(word_ball)
    D = np.empty((n_pairs, W), dtype=np.float32)
    lifted_best_x = np.empty(n_pairs, dtype=np.float32)
    lifted_best_y = np.empty(n_pairs, dtype=np.float32)
    print(f"[ginn-labels] computing candidate distances matrix {n_pairs} x {W} ...", flush=True)
    for j, m in enumerate(word_ball):
        gq = apply_mobius_to_array(m.alpha, m.beta, q)
        D[:, j] = disk_distance_array(p, gq).astype(np.float32)
        if j == 0 or (j+1) % max(50, W//10) == 0 or j == W-1:
            print(f"[ginn-labels] candidate {j+1}/{W}", flush=True)
    best_idx = np.argmin(D, axis=1).astype(np.int64)
    identity_dist = D[:,0].copy()
    quotient_dist = D[np.arange(n_pairs), best_idx]

    # Lifted q for exact winning word, used for geodesic witnesses.
    for i, bi in enumerate(best_idx):
        m = word_ball[int(bi)]
        z = apply_mobius_to_point(m.alpha, m.beta, q[i])
        lifted_best_x[i] = float(z.real)
        lifted_best_y[i] = float(z.imag)

    X, feature_names = pair_features_from_arrays(p, q, identity_dist)
    rows: List[Dict[str, Any]] = []
    for i in range(n_pairs):
        word = word_ball[int(best_idx[i])].word
        rows.append({
            "pair_id": i,
            "p_x": float(p[i].real), "p_y": float(p[i].imag),
            "q_x": float(q[i].real), "q_y": float(q[i].imag),
            "identity_distance": float(identity_dist[i]),
            "quotient_distance": float(quotient_dist[i]),
            "quotient_ratio": float(quotient_dist[i] / identity_dist[i]) if identity_dist[i] > 1e-12 else 1.0,
            "shortest_lift_word": word if word else "identity",
            "shortest_lift_index": int(best_idx[i]),
            "shortest_lift_depth": len(word),
            "shortest_lift_trace": float(word_ball[int(best_idx[i])].trace_real()),
            "lifted_q_x": float(lifted_best_x[i]),
            "lifted_q_y": float(lifted_best_y[i]),
            "geodesic_length_check": float(quotient_dist[i]),
            "nontrivial_shortcut": int(best_idx[i] != 0),
            "crossing_word_proxy": word,
            "crossing_count_proxy": len(word),
        })
    shortcut_frac = float(np.mean(best_idx != 0))
    depth_vals = np.asarray([word_depth_string(word_ball[int(k)].word) for k in best_idx], dtype=float)
    meta = {
        "word_ball_depth": depth,
        "word_ball_size": W,
        "sampler_kind": sampler_kind,
        "shortcut_fraction": shortcut_frac,
        "mean_shortest_lift_depth": float(depth_vals.mean()),
        "max_shortest_lift_depth": int(depth_vals.max()) if len(depth_vals) else 0,
        "label_type": "finite-word quotient-distance plus exact winning-lift index",
    }
    print_perf("ginn-labels", t0, c0)
    print(f"[ginn-labels] shortcut_fraction={shortcut_frac:.3f} mean_depth={meta['mean_shortest_lift_depth']:.3f}", flush=True)
    return rows, X, D, word_ball, meta, feature_names



class CandidateRankGINN(nn.Module):  # type: ignore[misc]
    """Geometry-informed branch/ranking network that is not handed exact d_gamma.

    For each pair (p,q) and candidate deck word gamma, the model sees:
      - pair context features for (p,q)
      - candidate lifted endpoint coordinates gamma(q)
      - cheap geometric descriptors such as Euclidean displacement and angles
      - word metadata such as depth and trace proxy

    It does NOT receive the exact Poincare candidate distances d_D(p,gamma q)
    as input.  Those distances are used only by the labeling oracle and by the
    final top-k pruning audit.  The neural task is therefore to learn the
    downstairs branch structure: which candidate sheet should be examined.
    """
    def __init__(self, pair_dim: int, cand_dim: int, hidden: int = 256, context_dim: int = 128):
        super().__init__()
        self.pair_net = nn.Sequential(
            nn.Linear(pair_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, context_dim), nn.SiLU(),
        )
        self.cand_net = nn.Sequential(
            nn.Linear(cand_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, context_dim), nn.SiLU(),
        )
        self.score_net = nn.Sequential(
            nn.Linear(3 * context_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.shortcut_head = nn.Linear(context_dim, 1)

    def forward(self, x_pair, x_cand):
        # x_pair: [B,P], x_cand: [B,W,C]
        B, W, _ = x_cand.shape
        pc = self.pair_net(x_pair)                  # [B,H]
        ce = self.cand_net(x_cand.reshape(B * W, -1)).reshape(B, W, -1)
        pc_exp = pc[:, None, :].expand(-1, W, -1)
        joint = torch.cat([pc_exp, ce, pc_exp * ce], dim=-1)
        scores = self.score_net(joint).squeeze(-1)  # [B,W]
        shortcut_logit = self.shortcut_head(pc).squeeze(-1)
        return scores, shortcut_logit, pc


def build_candidate_feature_cube(rows: List[Dict[str, Any]], word_ball: List[Mobius], max_depth: int) -> Tuple[np.ndarray, List[str]]:
    """Candidate features for GINN v2.2, deliberately excluding exact Poincare distances."""
    n = len(rows)
    W = len(word_ball)
    p = np.asarray([complex(float(r['p_x']), float(r['p_y'])) for r in rows], dtype=np.complex128)
    q = np.asarray([complex(float(r['q_x']), float(r['q_y'])) for r in rows], dtype=np.complex128)
    pa = np.angle(p)
    pr = np.abs(p)
    C = np.empty((n, W, 16), dtype=np.float32)
    for j, m in enumerate(word_ball):
        gq = apply_mobius_to_array(m.alpha, m.beta, q)
        dx = gq.real - p.real
        dy = gq.imag - p.imag
        euc = np.sqrt(dx*dx + dy*dy)
        gr = np.abs(gq)
        ga = np.angle(gq)
        dang = ga - pa
        tr = np.clip(float(m.trace_real()), -4.0, 4.0) / 4.0
        dep = word_depth_string(m.word) / max(1, max_depth)
        C[:, j, :] = np.column_stack([
            gq.real, gq.imag,
            dx, dy,
            euc,
            pr, gr,
            np.cos(pa), np.sin(pa), np.cos(ga), np.sin(ga),
            np.cos(dang), np.sin(dang),
            np.full(n, dep),
            np.full(n, 1.0 if word_depth_string(m.word) == 0 else 0.0),
            np.full(n, tr),
        ]).astype(np.float32)
    names = [
        'lifted_q_x', 'lifted_q_y', 'dx_lifted_minus_p', 'dy_lifted_minus_p',
        'euclidean_lifted_distance', 'p_radius', 'lifted_q_radius',
        'p_angle_cos', 'p_angle_sin', 'lifted_angle_cos', 'lifted_angle_sin',
        'angle_delta_cos', 'angle_delta_sin', 'word_depth_norm', 'is_identity', 'trace_clipped_over_4'
    ]
    return C, names


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - y
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / denom if denom > 0 else float('nan')
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def ordered_topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return candidate indices sorted by decreasing learned score.

    v2 used argpartition for some top-k metrics and argsort for displayed words.
    That is mathematically fine for unordered top-k membership, but confusing in
    the CSV because the displayed top-k list could appear inconsistent with
    the top-1 prediction in tie/near-tie cases.  v2.2 uses this one ordered
    ranking everywhere for prediction, top-k recall, and reporting.
    """
    kk = min(k, scores.shape[1])
    return np.argsort(-scores, axis=1)[:, :kk]


def classification_topk_accuracy(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    top = ordered_topk_indices(scores, k)
    return float(np.mean([int(y[i]) in set(top[i].tolist()) for i in range(len(y))]))


def topk_pruned_distance(D: np.ndarray, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # Learned pruning: take the model's ordered top-k candidates, then do the
    # exact Poincare distance minimization only inside that small candidate set.
    top = ordered_topk_indices(scores, k)
    vals = np.take_along_axis(D, top, axis=1)
    arg = np.argmin(vals, axis=1)
    chosen = top[np.arange(D.shape[0]), arg]
    dist = vals[np.arange(D.shape[0]), arg]
    return chosen.astype(np.int64), dist.astype(np.float32)


def entropy_from_scores(scores: np.ndarray) -> np.ndarray:
    s = scores - np.max(scores, axis=1, keepdims=True)
    p = np.exp(s)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
    return -np.sum(p * np.log(np.maximum(p, 1e-30)), axis=1)



def _estimate_candidate_tensor_mb(batch_size: int, word_count: int, cand_dim: int, hidden: int, context_dim: int) -> float:
    """Very rough float32 activation-size estimate for one unchunked candidate-scoring pass."""
    # This intentionally overestimates only the large B*W activations, not model params.
    elems = batch_size * word_count * (cand_dim + hidden + context_dim * 4 + hidden)
    return float(elems * 4 / (1024 ** 2))


def _score_batch_chunked(
    model: CandidateRankGINN,
    X_t: Any,
    Cn_cpu: np.ndarray,
    b: np.ndarray,
    dev: Any,
    candidate_chunk_size: int,
) -> Tuple[Any, Any, Any]:
    """Score all candidates for a batch, optionally in candidate chunks.

    This is the central v2.2 memory fix.  The exact candidate distances are still
    not model inputs.  We only chunk the candidate-feature tensor Cn so large word
    balls such as Gamma(6) depth 2 do not require a huge B x W x F activation on
    the GPU at once.
    """
    xb = X_t[b]
    W = Cn_cpu.shape[1]
    if candidate_chunk_size <= 0 or candidate_chunk_size >= W:
        cb = torch.tensor(Cn_cpu[b], dtype=torch.float32, device=dev)
        return model(xb, cb)
    scores_parts = []
    shortcut_logit = None
    context = None
    for j0 in range(0, W, candidate_chunk_size):
        j1 = min(W, j0 + candidate_chunk_size)
        cb = torch.tensor(Cn_cpu[np.ix_(b, np.arange(j0, j1))], dtype=torch.float32, device=dev)
        sj, slj, ctxj = model(xb, cb)
        scores_parts.append(sj)
        if shortcut_logit is None:
            shortcut_logit = slj
            context = ctxj
    return torch.cat(scores_parts, dim=1), shortcut_logit, context

def train_ginn(
    rows: List[Dict[str, Any]],
    X: np.ndarray,
    D: np.ndarray,
    word_ball: List[Mobius],
    outdir: Path,
    depth: int,
    epochs: int,
    pair_hidden: int,
    score_hidden: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str,
    patience: int,
    ce_weight: float,
    soft_distance_weight: float,
    temperature: float,
    candidate_chunk_size: int = 0,
    auto_chunk_threshold_mb: float = 2048.0,
) -> Dict[str, Any]:
    if torch is None or nn is None or F is None:
        raise RuntimeError('PyTorch is required for GINN training but is not available.')

    rng = np.random.default_rng(seed)
    n, W = D.shape
    y_idx = np.asarray([int(r['shortest_lift_index']) for r in rows], dtype=np.int64)
    y_depth = np.asarray([int(r['shortest_lift_depth']) for r in rows], dtype=np.int64)
    y_shortcut = (y_idx != 0).astype(np.float32)
    y_dist = D[np.arange(n), y_idx].astype(np.float32)
    id_dist = D[:, 0].astype(np.float32)

    idx = np.arange(n); rng.shuffle(idx)
    n_train = int(0.70*n); n_val = int(0.15*n)
    train_idx = idx[:n_train]; val_idx = idx[n_train:n_train+n_val]; test_idx = idx[n_train+n_val:]

    # Pair features are allowed; candidate features exclude exact Poincare d_gamma.
    C, cand_feature_names = build_candidate_feature_cube(rows, word_ball, depth)
    x_mean = X[train_idx].mean(axis=0)
    x_std = X[train_idx].std(axis=0); x_std[x_std < 1e-6] = 1.0
    Xn = ((X - x_mean) / x_std).astype(np.float32)
    c_mean = C[train_idx].reshape(-1, C.shape[-1]).mean(axis=0)
    c_std = C[train_idx].reshape(-1, C.shape[-1]).std(axis=0); c_std[c_std < 1e-6] = 1.0
    Cn = ((C - c_mean[None,None,:]) / c_std[None,None,:]).astype(np.float32)

    dev = torch.device(device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[ginn-v2.2-train] device={dev} n={n} W={W} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}', flush=True)
    print('[ginn-v2.2-train] exact candidate distances are NOT model inputs; they are label/audit geometry only', flush=True)
    model = CandidateRankGINN(X.shape[1], C.shape[-1], hidden=pair_hidden, context_dim=max(32, score_hidden)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)

    X_t = torch.tensor(Xn, dtype=torch.float32, device=dev)
    # v2.2: keep candidate features on CPU by default and transfer candidate chunks to GPU.
    # This avoids CUDA OOM for large word balls such as Gamma(6) at depth 2.
    est_mb = _estimate_candidate_tensor_mb(batch_size, W, C.shape[-1], pair_hidden, max(32, score_hidden))
    if candidate_chunk_size < 0:
        candidate_chunk_size = 0
    if candidate_chunk_size == 0 and str(dev).startswith('cuda') and est_mb > auto_chunk_threshold_mb:
        candidate_chunk_size = 512 if W > 512 else 0
    if candidate_chunk_size > 0:
        print(f'[ginn-v2.2-train] chunked candidate scoring enabled: chunk={candidate_chunk_size} W={W} estimated_unchunked_activation_mb={est_mb:.1f}', flush=True)
    else:
        print(f'[ginn-v2.2-train] unchunked candidate scoring: W={W} estimated_activation_mb={est_mb:.1f}', flush=True)
    Cn_cpu = Cn.astype(np.float32, copy=False)
    yidx_t = torch.tensor(y_idx, dtype=torch.long, device=dev)
    yshortcut_t = torch.tensor(y_shortcut, dtype=torch.float32, device=dev)

    train_log: List[Dict[str, Any]] = []
    best_val = float('inf'); best_state = None; best_epoch = 0; no_imp = 0
    t0, c0 = time.time(), cpu_seconds()
    for ep in range(1, epochs+1):
        model.train()
        perm = train_idx.copy(); rng.shuffle(perm)
        total=ce_tot=sc_tot=0.0; nb=0
        for start in range(0, len(perm), batch_size):
            b = perm[start:start+batch_size]
            scores, shortcut_logit, _ = _score_batch_chunked(model, X_t, Cn_cpu, b, dev, candidate_chunk_size)
            ce = F.cross_entropy(scores, yidx_t[b])
            shortcut_loss = F.binary_cross_entropy_with_logits(shortcut_logit, yshortcut_t[b])
            loss = ce_weight*ce + 0.25*shortcut_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item()); ce_tot += float(ce.item()); sc_tot += float(shortcut_loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            sv, slogv, _ = _score_batch_chunked(model, X_t, Cn_cpu, val_idx, dev, candidate_chunk_size)
            ce_v = F.cross_entropy(sv, yidx_t[val_idx]).item()
            sc_v = F.binary_cross_entropy_with_logits(slogv, yshortcut_t[val_idx]).item()
            val_loss = ce_weight*ce_v + 0.25*sc_v
        if val_loss < best_val - 1e-7:
            best_val = val_loss; best_epoch = ep; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_imp += 1
        train_log.append({
            'epoch': ep,
            'train_loss': total/max(nb,1),
            'train_ce': ce_tot/max(nb,1),
            'train_shortcut_bce': sc_tot/max(nb,1),
            'val_loss': val_loss,
            'val_ce': ce_v,
            'val_shortcut_bce': sc_v,
            'best_epoch_so_far': best_epoch,
            'epochs_without_improve': no_imp,
        })
        if ep == 1 or ep % max(1, epochs//10) == 0 or ep == epochs:
            print(f'[ginn-v2.2-train] epoch {ep:4d}/{epochs} train_loss={total/max(nb,1):.6f} val_loss={val_loss:.6f} val_ce={ce_v:.4f}', flush=True)
            print_perf('ginn-v2.2-train', t0, c0)
        if patience > 0 and no_imp >= patience:
            print(f'[ginn-v2.2-train] early stopping at epoch {ep}; best_epoch={best_epoch} best_val={best_val:.6f}', flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    scores_all_list=[]; shortcut_all=[]; context_all=[]
    with torch.no_grad():
        for start in range(0, n, max(256, batch_size)):
            b = np.arange(start, min(n, start+max(256,batch_size)))
            s, sl, ctx = _score_batch_chunked(model, X_t, Cn_cpu, b, dev, candidate_chunk_size)
            scores_all_list.append(s.detach().cpu().numpy())
            shortcut_all.append(torch.sigmoid(sl).detach().cpu().numpy())
            context_all.append(ctx.detach().cpu().numpy())
    scores_all = np.vstack(scores_all_list)
    shortcut_prob = np.concatenate(shortcut_all)
    context_emb = np.vstack(context_all)

    # A single ordered ranking is used everywhere in v2.2.
    ordered_score_idx_all = ordered_topk_indices(scores_all, W)
    pred_idx = ordered_score_idx_all[:, 0].astype(np.int64)
    pred_hard_dist = D[np.arange(n), pred_idx]
    pred_depth = np.asarray([word_depth_string(word_ball[int(k)].word) for k in pred_idx], dtype=np.int64)
    ent = entropy_from_scores(scores_all)
    sorted_scores = np.take_along_axis(scores_all, ordered_score_idx_all, axis=1)
    if W > 1:
        score_margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    else:
        score_margin = np.zeros(n, dtype=np.float32)

    # Exact distance gap to the second-best candidate; a small gap marks quotient seams/near ties.
    exact_ordered_idx_all = np.argsort(D, axis=1)
    sortedD = np.take_along_axis(D, exact_ordered_idx_all, axis=1)
    exact_distance_gap = sortedD[:,1] - sortedD[:,0] if W > 1 else np.zeros(n, dtype=np.float32)
    exact_equiv_tol = 1.0e-5
    pred_exact_equiv = pred_hard_dist <= (y_dist + exact_equiv_tol)

    def reg_metrics_for(split_idx, pred):
        return regression_metrics(y_dist[split_idx], pred[split_idx])

    topk_metrics: Dict[str, Any] = {}
    for k in [1, 3, 5, 10, 20]:
        if k > W: continue
        chosen, dist = topk_pruned_distance(D, scores_all, k)
        topk_metrics[f'top{k}'] = {
            'recall_true_winner_test': classification_topk_accuracy(scores_all[test_idx], y_idx[test_idx], k),
            'pruned_exact_distance_test': reg_metrics_for(test_idx, dist),
            'candidate_fraction_examined': float(k / W),
            'speedup_factor_vs_full_word_ball': float(W / k),
            'selected_word_accuracy_after_pruned_min_test': float(np.mean(chosen[test_idx] == y_idx[test_idx])),
        }

    metrics = {
        'model_type': 'geometry_informed_branch_ranker_ginn_v2_2_chunked_auto_hyperparams_no_exact_distances_as_inputs',
        'interpretation': 'Learns downstairs quotient branch structure by ranking deck-word candidates from lifted coordinates and word metadata, without receiving exact Poincare candidate distances as inputs.',
        'train': {'hard_selected_distance': reg_metrics_for(train_idx, pred_hard_dist)},
        'val': {'hard_selected_distance': reg_metrics_for(val_idx, pred_hard_dist)},
        'test': {'hard_selected_distance': reg_metrics_for(test_idx, pred_hard_dist)},
        'baseline_identity_test': regression_metrics(y_dist[test_idx], id_dist[test_idx]),
        'baseline_mean_test': regression_metrics(y_dist[test_idx], np.full_like(y_dist[test_idx], y_dist[train_idx].mean())),
        'winning_lift_accuracy_test': float(np.mean(pred_idx[test_idx] == y_idx[test_idx])),
        'winning_lift_exact_equivalent_accuracy_test_tol_1e_5': float(np.mean(pred_exact_equiv[test_idx])),
        'winning_lift_top3_accuracy_test': classification_topk_accuracy(scores_all[test_idx], y_idx[test_idx], 3),
        'winning_lift_top5_accuracy_test': classification_topk_accuracy(scores_all[test_idx], y_idx[test_idx], 5),
        'depth_accuracy_test': float(np.mean(pred_depth[test_idx] == y_depth[test_idx])),
        'shortcut_fraction_test': float(np.mean(y_idx[test_idx] != 0)),
        'predicted_shortcut_fraction_test': float(np.mean(pred_idx[test_idx] != 0)),
        'shortcut_auc_proxy_accuracy_test': float(np.mean((shortcut_prob[test_idx] >= 0.5) == (y_shortcut[test_idx] > 0.5))),
        'topk_pruned_search': topk_metrics,
        'branch_atlas_test': {
            'learned_branch_entropy_mean': float(np.mean(ent[test_idx])),
            'learned_branch_entropy_median': float(np.median(ent[test_idx])),
            'learned_top1_score_margin_mean': float(np.mean(score_margin[test_idx])),
            'learned_top1_score_margin_median': float(np.median(score_margin[test_idx])),
            'exact_distance_gap_mean': float(np.mean(exact_distance_gap[test_idx])),
            'exact_distance_gap_median': float(np.median(exact_distance_gap[test_idx])),
            'near_seam_fraction_gap_lt_0p02': float(np.mean(exact_distance_gap[test_idx] < 0.02)),
            'near_seam_fraction_gap_lt_0p05': float(np.mean(exact_distance_gap[test_idx] < 0.05)),
            'unique_true_branches_test': int(len(set(y_idx[test_idx].tolist()))),
            'unique_predicted_branches_test': int(len(set(pred_idx[test_idx].tolist()))),
        },
        'word_ball_size': int(W),
        'best_val_loss': float(best_val),
        'best_epoch': int(best_epoch),
        'epochs_ran': int(train_log[-1]['epoch'] if train_log else 0),
        'normalization': {
            'x_mean': x_mean.tolist(), 'x_std': x_std.tolist(),
            'candidate_mean': c_mean.tolist(), 'candidate_std': c_std.tolist(),
        },
        'candidate_feature_names': cand_feature_names,
        'candidate_chunk_size': int(candidate_chunk_size),
        'estimated_unchunked_activation_mb': float(est_mb),
        'loss_weights': {'ce_weight': ce_weight, 'shortcut_bce_weight': 0.25},
        'not_model_inputs': ['exact candidate Poincare distances d_gamma(p,q)', 'argmin distances except as supervised labels'],
        'topk_reporting': 'v2.2 uses one ordered score ranking for top-1 prediction, top-k recall, top-k word lists, and pruned-search audits',
    }
    h_rmse = metrics['test']['hard_selected_distance']['rmse']
    b_rmse = metrics['baseline_identity_test']['rmse']
    metrics['identity_baseline_hard_rmse_improvement_fraction'] = float((b_rmse-h_rmse)/b_rmse) if b_rmse > 0 else None

    # Save a learned branch embedding for downstream visualization; this is a learned object,
    # not a direct classical invariant.
    np.savez_compressed(outdir / 'learned_branch_context_embeddings.npz', context=context_emb.astype(np.float32), test_idx=test_idx.astype(np.int64))

    pred_path = outdir / 'predictions_test.csv'
    fieldnames = list(rows[0].keys()) + [
        'pred_winning_lift_index', 'pred_winning_lift_word', 'pred_winning_lift_depth',
        'pred_hard_quotient_distance', 'winner_correct', 'top3_contains_true', 'top5_contains_true',
        'shortcut_probability', 'branch_entropy', 'score_margin_top1_top2', 'exact_distance_gap_top1_top2',
        'top3_pruned_distance', 'top5_pruned_distance', 'top10_pruned_distance',
        'top5_words', 'exact_top5_words', 'true_word_model_rank',
        'pred_exact_equivalent_tol_1e_5', 'split',
    ]
    top3 = ordered_topk_indices(scores_all, min(3,W))
    top5 = ordered_topk_indices(scores_all, min(5,W))
    _, top3_dist = topk_pruned_distance(D, scores_all, min(3,W))
    _, top5_dist = topk_pruned_distance(D, scores_all, min(5,W))
    _, top10_dist = topk_pruned_distance(D, scores_all, min(10,W))
    with pred_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in test_idx:
            ii = int(i)
            rr = dict(rows[ii])
            pw = word_ball[int(pred_idx[ii])].word
            ordered = ordered_score_idx_all[ii, :min(5,W)]
            exact_ordered = exact_ordered_idx_all[ii, :min(5,W)]
            words = [word_ball[int(j)].word if word_ball[int(j)].word else 'identity' for j in ordered]
            exact_words = [word_ball[int(j)].word if word_ball[int(j)].word else 'identity' for j in exact_ordered]
            # Rank of the exact winning index in the learned score order.
            true_rank_arr = np.where(ordered_score_idx_all[ii] == y_idx[ii])[0]
            true_rank = int(true_rank_arr[0] + 1) if len(true_rank_arr) else -1
            rr.update({
                'pred_winning_lift_index': int(pred_idx[ii]),
                'pred_winning_lift_word': pw if pw else 'identity',
                'pred_winning_lift_depth': int(pred_depth[ii]),
                'pred_hard_quotient_distance': float(pred_hard_dist[ii]),
                'winner_correct': int(pred_idx[ii] == y_idx[ii]),
                'top3_contains_true': int(y_idx[ii] in set(top3[ii].tolist())),
                'top5_contains_true': int(y_idx[ii] in set(top5[ii].tolist())),
                'shortcut_probability': float(shortcut_prob[ii]),
                'branch_entropy': float(ent[ii]),
                'score_margin_top1_top2': float(score_margin[ii]),
                'exact_distance_gap_top1_top2': float(exact_distance_gap[ii]),
                'top3_pruned_distance': float(top3_dist[ii]),
                'top5_pruned_distance': float(top5_dist[ii]),
                'top10_pruned_distance': float(top10_dist[ii]),
                'top5_words': '|'.join(words),
                'exact_top5_words': '|'.join(exact_words),
                'true_word_model_rank': int(true_rank),
                'pred_exact_equivalent_tol_1e_5': int(pred_exact_equiv[ii]),
                'split': 'test',
            })
            writer.writerow(rr)

    save_ginn_geodesic_witnesses(outdir, pred_path, word_ball, max_witnesses=24)

    with (outdir / 'train_log.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
        writer.writeheader(); writer.writerows(train_log)

    with (outdir / 'branch_atlas_summary.json').open('w') as f:
        json.dump(metrics['branch_atlas_test'], f, indent=2)

    ckpt = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'pair_hidden': pair_hidden,
        'context_dim': max(32, score_hidden),
        'feature_names': ['see run_manifest; pair features from fixed generator'],
        'candidate_feature_names': cand_feature_names,
        'candidate_chunk_size': int(candidate_chunk_size),
        'estimated_unchunked_activation_mb': float(est_mb),
        'word_ball_words': [m.word if m.word else 'identity' for m in word_ball],
        'note': 'GINN v2.2 ranker does not consume exact candidate distances as model inputs.',
    }
    torch.save(ckpt, outdir / 'downstairs_ginn_v2_4.pt')

    print('[ginn-v2.2-result] TEST branch-ranking GINN', flush=True)
    print(f"                 hard distance RMSE = {metrics['test']['hard_selected_distance']['rmse']:.6f}", flush=True)
    print(f"                 identity RMSE      = {metrics['baseline_identity_test']['rmse']:.6f}", flush=True)
    print(f"                 hard R^2           = {metrics['test']['hard_selected_distance']['r2']:.4f}", flush=True)
    print(f"                 winner acc         = {metrics['winning_lift_accuracy_test']:.3f}", flush=True)
    print(f"                 top3 winner acc    = {metrics['winning_lift_top3_accuracy_test']:.3f}", flush=True)
    print(f"                 top5 winner acc    = {metrics['winning_lift_top5_accuracy_test']:.3f}", flush=True)
    if 'top5' in topk_metrics:
        print(f"                 top5 pruned RMSE   = {topk_metrics['top5']['pruned_exact_distance_test']['rmse']:.6f}", flush=True)
        print(f"                 top5 search speedup= {topk_metrics['top5']['speedup_factor_vs_full_word_ball']:.1f}x", flush=True)
    print_perf('ginn-v2.2-train-final', t0, c0)
    return metrics

def save_ginn_geodesic_witnesses(outdir: Path, prediction_csv: Path, word_ball: List[Mobius], max_witnesses: int = 24) -> None:
    if not prediction_csv.exists():
        return
    rows=[]
    with prediction_csv.open(newline="") as f:
        reader=csv.DictReader(f)
        for r in reader:
            try:
                r["hard_abs_error"] = abs(float(r["pred_hard_quotient_distance"])-float(r["quotient_distance"]))
                rows.append(r)
            except Exception:
                pass
    rows.sort(key=lambda r: float(r.get("hard_abs_error",0.0)), reverse=True)
    chosen = rows[:max_witnesses//2]
    if len(rows) > len(chosen):
        step=max(1, len(rows)//max(1, max_witnesses-len(chosen)))
        chosen += rows[::step][:max_witnesses-len(chosen)]
    wdir=outdir/"geodesic_witnesses"; wdir.mkdir(exist_ok=True)
    for k,r in enumerate(chosen):
        px,py=float(r["p_x"]),float(r["p_y"])
        qx,qy=float(r["q_x"]),float(r["q_y"])
        pred_i=int(float(r["pred_winning_lift_index"]))
        pred_m=word_ball[pred_i]
        pred_z=apply_mobius_to_point(pred_m.alpha, pred_m.beta, complex(qx,qy))
        witness={
            "pair_id": int(float(r["pair_id"])),
            "p": [px,py],
            "q_base": [qx,qy],
            "exact_winning_word": r.get("shortest_lift_word",""),
            "exact_lifted_q": [float(r.get("lifted_q_x", qx)), float(r.get("lifted_q_y", qy))],
            "predicted_winning_word": r.get("pred_winning_lift_word",""),
            "predicted_lifted_q": [float(pred_z.real), float(pred_z.imag)],
            "identity_distance": float(r["identity_distance"]),
            "exact_quotient_distance": float(r["quotient_distance"]),
            "predicted_hard_distance_from_predicted_lift": float(r["pred_hard_quotient_distance"]),
            "predicted_soft_expected_distance": float(r.get("pred_soft_quotient_distance", r.get("top5_pruned_distance", r["pred_hard_quotient_distance"]))),
            "hard_absolute_error": float(r.get("hard_abs_error",0.0)),
            "winner_correct": int(float(r.get("winner_correct",0))),
            "verification_statement": "Exact and predicted distances are witnessed by Poincare geodesics from p to the exact/predicted lifted q endpoints upstairs.",
        }
        with (wdir/f"witness_{k:04d}_pair_{witness['pair_id']:06d}.json").open("w") as f:
            json.dump(witness, f, indent=2)


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def compact_word_ball_summary(word_ball: List[Mobius]) -> List[Dict[str, Any]]:
    out=[]
    for i,m in enumerate(word_ball):
        tr=m.trace_real()
        typ="identity" if m.word=="" else ("hyperbolic" if abs(tr)>2+1e-10 else "elliptic_or_relation")
        out.append({"index": i, "word": m.word if m.word else "identity", "depth": word_depth_string(m.word), "trace": tr, "type_proxy": typ})
    return out


def run_one_surface_ginn(args: argparse.Namespace, surface_name: str) -> Path:
    run_id=f"run_{now_stamp()}_{surface_name}_ginn_v2_4"
    outdir=Path(args.outdir)/run_id; outdir.mkdir(parents=True, exist_ok=True)
    print("="*78, flush=True)
    print(f"[start] FuchsianDownstairsGINN v2.4 surface={surface_name} outdir={outdir}", flush=True)
    print("="*78, flush=True)
    overall_t0, overall_c0 = time.time(), cpu_seconds()

    surface_json = make_surface(surface_name, args.maker, max_cosets=args.max_cosets, max_generators=args.max_generators)
    with (outdir/"surface.json").open("w") as f: json.dump(surface_json, f, indent=2)
    print(f"[surface] {surface_json.get('name', surface_name)}", flush=True)
    print(f"[surface] domain_type={surface_json.get('domain_type')} genus={surface_json.get('genus')} area={surface_json.get('area')}", flush=True)

    effective = {
        'pairs': args.pairs, 'epochs': args.epochs, 'patience': args.patience,
        'pair_hidden': args.pair_hidden, 'score_hidden': args.score_hidden,
        'batch_size': args.batch_size, 'lr': args.lr,
        'candidate_chunk_size': args.candidate_chunk_size,
    }
    auto_cfg = choose_auto_hyperparams(surface_name, surface_json, args.profile)
    if auto_cfg:
        effective.update(auto_cfg)
        print(f"[auto-profile] profile={args.profile} applied hyperparameters: {auto_cfg}", flush=True)
    else:
        print("[auto-profile] manual profile: using CLI hyperparameters", flush=True)
    if args.depth != 2:
        print(f"[warning] depth={args.depth}; v2.2 production recommendation is depth=2. Larger depths may be infeasible for noncompact groups.", flush=True)

    rows, X, D, word_ball, label_meta, feature_names = generate_ginn_dataset(surface_json, effective['pairs'], args.depth, args.seed, max_word_ball=args.max_word_ball)
    write_csv_rows(outdir/"pair_dataset.csv", rows)
    np.savez_compressed(outdir/"candidate_distances.npz", D=D.astype(np.float32), X=X.astype(np.float32))
    with (outdir/"word_ball.json").open("w") as f: json.dump(compact_word_ball_summary(word_ball), f, indent=2)

    metrics={}
    if args.no_train:
        print("[ginn-train] skipped because --no-train was supplied", flush=True)
    else:
        metrics=train_ginn(
            rows=rows, X=X, D=D, word_ball=word_ball, outdir=outdir,
            depth=args.depth, epochs=effective['epochs'], pair_hidden=effective['pair_hidden'],
            score_hidden=effective['score_hidden'], lr=effective['lr'], batch_size=effective['batch_size'],
            seed=args.seed, device=args.device, patience=effective['patience'],
            ce_weight=args.ce_weight, soft_distance_weight=args.soft_distance_weight,
            temperature=args.temperature, candidate_chunk_size=effective['candidate_chunk_size'],
            auto_chunk_threshold_mb=args.auto_chunk_threshold_mb,
        )
    manifest={
        "program": "FuchsianDownstairsGINN_v2_4.py",
        "surface_requested": surface_name,
        "surface_name": surface_json.get("name"),
        "domain_type": surface_json.get("domain_type"),
        "genus": surface_json.get("genus"),
        "area": surface_json.get("area"),
        "maker": args.maker,
        "max_cosets": args.max_cosets,
        "max_generators": args.max_generators,
        "profile": args.profile,
        "effective_hyperparameters": effective,
        "pairs": effective['pairs'],
        "word_depth": args.depth,
        "word_ball_size": len(word_ball),
        "pair_hidden": effective['pair_hidden'],
        "score_hidden": effective['score_hidden'],
        "batch_size": effective['batch_size'],
        "lr": effective['lr'],
        "epochs": effective['epochs'],
        "patience": effective['patience'],
        "seed": args.seed,
        "label_meta": label_meta,
        "metrics": metrics,
        "wall_seconds_total": time.time()-overall_t0,
        "cpu_seconds_total": cpu_seconds()-overall_c0,
        "rss_mb_final": rss_mb(),
        "feature_names": feature_names,
        "interpretation": "Geometry-informed branch-ranking network. It sees lifted candidate endpoints and word metadata but not exact candidate Poincare distances. The trainable part learns the downstairs branch atlas, top-k lift set, seam uncertainty, and pruned-search candidates.",
        "caveat": "The labels and candidates are finite word-ball approximations. This is not a proof of global distance unless the word ball is known to be exhaustive for the pair.",
    }
    with (outdir/"metrics.json").open("w") as f: json.dump(metrics, f, indent=2)
    with (outdir/"run_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
    print_perf("overall", overall_t0, overall_c0)
    print(f"[done] outputs written to {outdir}", flush=True)
    return outdir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser=argparse.ArgumentParser(description="Train/test a geometry-informed branch-ranking neural network for downstairs quotient geometry on Fuchsian Riemann surfaces.")
    parser.add_argument("--surface", default="regular_g2", help="Surface spec, comma-list, smoke, standard, all, regular_g2, hurwitz, gamma3, gamma1_5, hecke_ab5, hecke_d5, etc.")
    parser.add_argument("--list-surfaces", action="store_true")
    parser.add_argument("--maker", default=default_maker_path(), help="Path to FuchsianDomainMaker_v13.py")
    parser.add_argument("--outdir", default="downstairs_ginn_v2_4_runs", help="Output directory root")
    parser.add_argument("--profile", choices=["balanced", "accurate", "fast", "manual"], default="balanced", help="Smart hyperparameter profile. Use manual to respect CLI values exactly.")
    parser.add_argument("--max-cosets", type=int, default=20000, help="Explicit coset-search safety limit for tokenized modular surfaces; replaces older hardcoded interactive limits.")
    parser.add_argument("--max-generators", type=int, default=0, help="Explicit cap on exported positive generators; 0 means no cap. Use with --max-word-ball for safety.")
    parser.add_argument("--max-word-ball", type=int, default=50000, help="Explicit safety limit for candidate word ball size; 0 disables.")
    parser.add_argument("--candidate-chunk-size", type=int, default=0, help="Candidate chunk size for GPU memory control; 0 means auto/unchunked depending on profile and estimate.")
    parser.add_argument("--auto-chunk-threshold-mb", type=float, default=2048.0, help="Estimated activation MB threshold above which chunking is enabled automatically on CUDA.")
    parser.add_argument("--pairs", type=int, default=9000, help="Number of random point pairs per surface")
    parser.add_argument("--depth", type=int, default=2, help="Reduced-word search depth / candidate lift ball")
    parser.add_argument("--epochs", type=int, default=220, help="Maximum GINN training epochs")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience; 0 disables")
    parser.add_argument("--pair-hidden", type=int, default=256, help="Pair-context hidden width")
    parser.add_argument("--score-hidden", type=int, default=128, help="Candidate scoring hidden width")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--ce-weight", type=float, default=1.0, help="Winning-lift cross-entropy loss weight")
    parser.add_argument("--soft-distance-weight", type=float, default=0.2, help="Soft expected distance MSE loss weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for soft expected distance")
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--no-train", action="store_true")
    args=parser.parse_args(argv)

    if args.depth < 0 or args.depth > 5:
        print("[error] depth should be 0..5; depth 2 is recommended initially.", file=sys.stderr); return 2
    if args.pairs < 100:
        print("[error] use at least 100 pairs.", file=sys.stderr); return 2
    if not args.no_train and torch is None:
        print("[error] PyTorch is not available.", file=sys.stderr); return 2
    random.seed(args.seed); np.random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)
        # Small/medium tabular GINN batches often run faster and more predictably
        # with limited intra-op threading than with PyTorch's default oversubscription.
        try:
            torch.set_num_threads(max(1, int(os.environ.get("GINN_TORCH_THREADS", "2"))))
            torch.set_num_interop_threads(max(1, int(os.environ.get("GINN_TORCH_INTEROP_THREADS", "1"))))
        except Exception:
            pass

    if args.list_surfaces:
        print("Smoke suite:", ", ".join(smoke_surface_specs()))
        print("Standard suite:", ", ".join(standard_surface_specs()))
        print("All mainline suite:", ", ".join(main_surface_specs()))
        return 0

    surf_arg=args.surface.lower().strip()
    if surf_arg in {"both", "smoke"}: surfaces=smoke_surface_specs()
    elif surf_arg == "standard": surfaces=standard_surface_specs()
    elif surf_arg in {"all", "main", "mainline"}: surfaces=main_surface_specs()
    elif "," in args.surface: surfaces=[x.strip() for x in args.surface.split(",") if x.strip()]
    else: surfaces=[args.surface]

    print("FuchsianDownstairsGINN v2.4", flush=True)
    print(f"maker={args.maker}", flush=True)
    print(f"surfaces={surfaces}", flush=True)
    print(f"profile={args.profile} pairs={args.pairs} depth={args.depth} epochs={args.epochs} pair_hidden={args.pair_hidden} score_hidden={args.score_hidden}", flush=True)
    print("architecture=geometry-informed branch ranker; exact candidate distances are labels/audits, not model inputs", flush=True)
    print("v2.4 tokenized generators: no one-letter/26-generator cap; use CLI limits for cosets/generators/word ball", flush=True)
    print(f"initial rss={rss_mb():.1f} MB", flush=True)
    try:
        completed=[]
        for s in surfaces:
            out=run_one_surface_ginn(args, s)
            completed.append(str(out))
        summary_root=Path(args.outdir)/f"summary_{now_stamp()}"; summary_root.mkdir(parents=True, exist_ok=True)
        with (summary_root/"completed_runs.json").open("w") as f: json.dump({"completed_runs": completed, "surfaces": surfaces}, f, indent=2)
        print(f"[summary] completed {len(completed)} surface runs; summary={summary_root}", flush=True)
    except Exception as exc:
        print("[fatal] GINN run failed:", exc, file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
