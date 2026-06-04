#!/usr/bin/env python3
"""FuchsianElementaryTester_v1_1.py

Classical small/elementary Fuchsian Riemann-surface tester for the GENN zoo.

This tester adds normalized records for:
  * Gamma(2): the thrice-punctured sphere, a torsion-free finite-area modular surface.
  * The modular commutator subgroup [PSL(2,Z),PSL(2,Z)]: the once-punctured torus.
  * Cyclic hyperbolic quotients: smooth noncompact infinite-area annulus/cylinder-type surfaces.
  * Cyclic parabolic quotients: smooth noncompact infinite-area cyclic cusp/funnel-type surfaces.
  * Optional cyclic elliptic quotients: explicitly excluded orbifold reference records.

The cyclic records use bounded sampling scaffolds for GINN preflight; those scaffolds
are not claimed to be canonical finite-area fundamental domains.  Gamma(2) and the
commutator subgroup use Ford-tile union domain models.
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
import types
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

PROGRAM = "FuchsianElementaryTester_v1_1.py"
VERSION = "v1.1"

try:
    from FuchsianSurfaceRecordTools_v1_0 import (
        GEOMETRY_AUDIT_FIELDS, GINN_SMOKE_FIELDS, GINN_TRAINING_FIELDS, FAILURE_FIELDS,
        generator_su11_audit, normalize_surface_record, write_csv, write_json,
    )
except Exception:  # pragma: no cover
    GEOMETRY_AUDIT_FIELDS = None
    GINN_SMOKE_FIELDS = None
    GINN_TRAINING_FIELDS = None
    FAILURE_FIELDS = None

    def write_json(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2))

    def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
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
                if det <= 0 or not math.isfinite(det):
                    bad.append(str(name))
            except Exception:
                bad.append(str(name))
        return {"generator_count": len(gens), "su11_max_det_error": maxerr, "bad_generator_count": len(bad), "bad_generators": ";".join(bad[:20])}

    def normalize_surface_record(sj: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        out = dict(sj)
        out.update(kwargs)
        out["surface_id"] = str(out.get("surface_id") or kwargs.get("surface_spec", "elementary"))
        out["mainline_dataset_eligible"] = bool(kwargs.get("mainline_dataset_eligible", False))
        out["exclusion_reason"] = str(kwargs.get("exclusion_reason", ""))
        out["master_record"] = {k: out.get(k) for k in out.keys()}
        return out


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


class _DummyQt:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return _DummyQt()
    def __getattr__(self, name): return _DummyQt()
    def __iter__(self): return iter(())
    def __bool__(self): return False


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


def safe_float_label(x: float) -> str:
    return (f"{x:.4g}").replace("-", "m").replace(".", "p")


def cpair(z: complex) -> List[float]:
    return [float(z.real), float(z.imag)]


def mat_json(M: np.ndarray) -> List[List[int]]:
    return [[int(M[0, 0]), int(M[0, 1])], [int(M[1, 0]), int(M[1, 1])]]


def bounded_disk_scaffold(radius: float = 0.72, sides: int = 12, rotation_deg: float = 0.0, kind: str = "bounded_sampling_scaffold") -> List[Dict[str, Any]]:
    theta0 = math.radians(rotation_deg)
    pts = [radius * complex(math.cos(theta0 + 2.0 * math.pi * k / sides), math.sin(theta0 + 2.0 * math.pi * k / sides)) for k in range(sides)]
    tiles: List[Dict[str, Any]] = []
    center = 0.0 + 0.0j
    for i in range(sides):
        tiles.append({
            "tile_index": i,
            "tile_kind": kind,
            "vertices": [cpair(center), cpair(pts[i]), cpair(pts[(i + 1) % sides])],
        })
    return tiles


# -----------------------------------------------------------------------------
# Elementary cyclic records
# -----------------------------------------------------------------------------

def make_cyclic_hyperbolic(maker: Any, strength: float, rotation: float, sample_radius: float) -> Dict[str, Any]:
    sj = maker.make_cyclic_quotient(float(strength), float(rotation))
    sid = f"cyclic_hyperbolic_a{safe_float_label(strength)}_rot{safe_float_label(rotation)}"
    sj.update({
        "surface_id": sid,
        "surface_spec": sid,
        "surface_family": "elementary_cyclic",
        "surface_subfamily": "cyclic_hyperbolic",
        "domain_type": "cyclic_hyperbolic_bounded_scaffold",
        "category": "cyclic_hyperbolic_riemann_surface",
        "compact": False,
        "finite_area": False,
        "torsion_free": True,
        "orbifold_excluded": False,
        "cusp_count": 0,
        "surface_area_type": "noncompact_infinite_area_cyclic_hyperbolic",
        "riemann_surface_status": "smooth noncompact hyperbolic Riemann surface D/<A>; cyclic hyperbolic annulus/cylinder-type quotient with bounded sampling scaffold",
        "kahler_status": "Riemann surface, hence Kähler in complex dimension one",
        "fundamental_domain_status": "cyclic hyperbolic quotient; bounded disk sampling scaffold supplied for GINN, not a full finite-area domain",
        "sampling_status": "supported by bounded cyclic sampling scaffold; not a canonical finite-area measure",
        "fundamental_domain_tiles": bounded_disk_scaffold(sample_radius, sides=12, rotation_deg=rotation, kind="cyclic_hyperbolic_bounded_sampling_triangle"),
        "sampling_scaffold": {"type": "bounded_disk_window_for_cyclic_hyperbolic", "sample_radius": sample_radius, "warning": "sampling window only; full quotient has infinite area"},
        "word_ball_recommended_depth": 4,
        "dataset_role": "all_riemann_surfaces_zoo_elementary_infinite_area_branch",
    })
    return sj


def make_cyclic_parabolic(maker: Any, width: float, sample_radius: float) -> Dict[str, Any]:
    G = maker.psl2r_to_disk_mobius(1.0, float(width), 0.0, 1.0, name="P")
    sid = f"cyclic_parabolic_w{safe_float_label(width)}"
    sj = {
        "format": "FuchsianGENN surface JSON elementary v1.1",
        "surface_id": sid,
        "surface_spec": sid,
        "name": f"Cyclic parabolic quotient <T_{width:g}>",
        "domain_type": "cyclic_parabolic_bounded_scaffold",
        "category": "cyclic_parabolic_riemann_surface",
        "surface_family": "elementary_cyclic",
        "surface_subfamily": "cyclic_parabolic",
        "generators": {"P": G.as_json()},
        "generator_parameters": {"P": {"type": "parabolic", "upper_half_plane_action": f"tau -> tau + {width:g}", "width": float(width)}},
        "compact": False,
        "finite_area": False,
        "torsion_free": True,
        "orbifold_excluded": False,
        "cusp_count": 1,
        "surface_area_type": "noncompact_infinite_area_cyclic_parabolic",
        "riemann_surface_status": "smooth noncompact hyperbolic Riemann surface H/<tau->tau+w>; cyclic parabolic cusp/funnel-type quotient with bounded sampling scaffold",
        "kahler_status": "Riemann surface, hence Kähler in complex dimension one",
        "fundamental_domain_status": "cyclic parabolic quotient; bounded disk sampling scaffold supplied for GINN, not a full finite-area domain",
        "sampling_status": "supported by bounded cyclic sampling scaffold; not a canonical finite-area measure",
        "fundamental_domain_tiles": bounded_disk_scaffold(sample_radius, sides=12, rotation_deg=0.0, kind="cyclic_parabolic_bounded_sampling_triangle"),
        "sampling_scaffold": {"type": "bounded_disk_window_for_cyclic_parabolic", "sample_radius": sample_radius, "warning": "sampling window only; full cyclic quotient has infinite area"},
        "word_ball_recommended_depth": 4,
        "dataset_role": "all_riemann_surfaces_zoo_elementary_infinite_area_branch",
        "notes": "Parabolic generator is represented by the Cayley transform of tau -> tau + width.",
    }
    return sj


def make_cyclic_elliptic_reference(maker: Any, order: int, sample_radius: float) -> Dict[str, Any]:
    if order < 2:
        raise ValueError("elliptic order must be >=2")
    G = maker.disk_rotation(2.0 * math.pi / float(order), name="E")
    sid = f"cyclic_elliptic_order{order}_orbifold_reference"
    sj = {
        "format": "FuchsianGENN surface JSON elementary v1.1",
        "surface_id": sid,
        "surface_spec": sid,
        "name": f"Excluded cyclic elliptic order-{order} orbifold reference",
        "domain_type": "cyclic_elliptic_orbifold_reference",
        "category": "cyclic_elliptic_orbifold_reference",
        "surface_family": "elementary_cyclic",
        "surface_subfamily": "cyclic_elliptic_orbifold_reference",
        "generators": {"E": G.as_json()},
        "generator_parameters": {"E": {"type": "elliptic_rotation_about_0", "order": int(order)}},
        "compact": False,
        "finite_area": False,
        "torsion_free": False,
        "orbifold_excluded": True,
        "cusp_count": 0,
        "elliptic_orders": [int(order)],
        "surface_area_type": "orbifold_reference_not_riemann_surface",
        "riemann_surface_status": "excluded: quotient has elliptic torsion/cone point and is an orbifold, not a smooth Riemann surface",
        "kahler_status": "not asserted because smooth Riemann-surface status is not cleared",
        "fundamental_domain_status": "orbifold reference only",
        "sampling_status": "not used for main GINN; bounded scaffold included only for visual/reference consistency",
        "fundamental_domain_tiles": bounded_disk_scaffold(sample_radius, sides=order if order >= 3 else 6, rotation_deg=0.0, kind="cyclic_elliptic_reference_triangle"),
        "word_ball_recommended_depth": 1,
        "dataset_role": "excluded_orbifold_reference",
        "exclusion_reason": "elliptic torsion: cyclic elliptic quotient is an orbifold/cone-point quotient, not a smooth Riemann surface",
    }
    return sj


# -----------------------------------------------------------------------------
# Modular small surfaces: Gamma(2) and PSL2Z commutator subgroup
# -----------------------------------------------------------------------------

def membership_principal(maker: Any, N: int) -> Callable[[np.ndarray], bool]:
    return lambda M: bool(maker._is_identity_psl_mod(M, N))


def abel_class_word(ch: str) -> int:
    # PSL(2,Z) abelianization C6 using presentation <S,T | S^2=(ST)^3=1>.
    # T maps to 1 and S maps to 3 in C6.
    return 3 if ch == "S" else 1


def abel_class_matrix_word(word: str) -> int:
    return sum(abel_class_word(ch) for ch in word) % 6


def membership_commutator(maker: Any) -> Callable[[np.ndarray], bool]:
    # For matrices reached in the PSL(2,Z) BFS, compare against representatives T^k.
    # M lies in the commutator subgroup iff it is in the coset class 0 of the C6 abelianization.
    # This function is only used through the custom commutator coset routines below.
    raise NotImplementedError


def mat_key_psl(maker: Any, M: np.ndarray) -> Tuple[int, ...]:
    return maker._mat_key_psl_int(M)


def coset_reps_by_finite_classes(maker: Any, class_count: int, class_of_word: Callable[[str], int]) -> List[Tuple[str, np.ndarray]]:
    """Build one PSL2Z matrix representative for each finite coset class."""
    S = maker._mat2(0, -1, 1, 0)
    T = maker._mat2(1, 1, 0, 1)
    gens = [("S", S), ("T", T)]
    reps_by_class: Dict[int, Tuple[str, np.ndarray]] = {0: ("", maker._mat2(1, 0, 0, 1))}
    q = deque([""])
    word_to_matrix: Dict[str, np.ndarray] = {"": maker._mat2(1, 0, 0, 1)}
    while q and len(reps_by_class) < class_count:
        w = q.popleft()
        R = word_to_matrix[w]
        for ch, G in gens:
            nw = w + ch
            if nw in word_to_matrix:
                continue
            M = R @ G
            word_to_matrix[nw] = M
            q.append(nw)
            c = class_of_word(nw) % class_count
            if c not in reps_by_class:
                reps_by_class[c] = (nw, M)
    if len(reps_by_class) != class_count:
        raise RuntimeError(f"Could not find all {class_count} coset classes; found {sorted(reps_by_class)}")
    return [reps_by_class[i] for i in range(class_count)]


def commutator_coset_reps(maker: Any) -> List[Tuple[str, np.ndarray]]:
    return coset_reps_by_finite_classes(maker, 6, abel_class_matrix_word)


def commutator_find_class(word: str) -> int:
    return abel_class_matrix_word(word)


def modular_generic_coset_reps(maker: Any, subgroup_kind: str, N: int = 2, max_cosets: int = 20000) -> List[Tuple[str, np.ndarray]]:
    if subgroup_kind == "gamma2":
        return maker._coset_reps_modN(N, subgroup="principal", max_cosets=max_cosets)
    if subgroup_kind == "commutator":
        return commutator_coset_reps(maker)
    raise ValueError(subgroup_kind)


def modular_in_H(maker: Any, subgroup_kind: str, N: int) -> Callable[[np.ndarray], bool]:
    if subgroup_kind == "gamma2":
        return membership_principal(maker, N)
    if subgroup_kind == "commutator":
        reps = commutator_coset_reps(maker)
        invs = [maker._mat_inv_sl2(R) for _w, R in reps]
        # The commutator subgroup is the class-0 coset.  M is in H iff M lies in H*I;
        # resolve class by comparing to the six C6 representatives using a short BFS table.
        # Because we only test Schreier generators produced from reps, a matrix is in H iff
        # it belongs to the same left coset as the identity.
        def same_as_identity(M: np.ndarray) -> bool:
            # H is normal; use the induced C6 homomorphism by matching M against T^k reps.
            # We can find k such that M * R_k^{-1} is in H by testing action on cosets below;
            # here use a conservative fallback: M is identity-coset if it is generated by
            # words of abelian class 0 in the Schreier routine.  This predicate is only used
            # for resolving cosets; the commutator-specific resolver below does not call it.
            raise NotImplementedError("commutator membership should use class arithmetic")
        return same_as_identity
    raise ValueError(subgroup_kind)


def coset_action_permutation_gamma2(maker: Any, reps: List[Tuple[str, np.ndarray]], N: int, G: np.ndarray) -> List[int]:
    in_H = membership_principal(maker, N)
    perm: List[int] = []
    for _w, R in reps:
        M = R @ G
        found = None
        for j, (_wj, Rj) in enumerate(reps):
            if in_H(M @ maker._mat_inv_sl2(Rj)):
                found = j
                break
        if found is None:
            raise RuntimeError("Could not resolve Gamma(2) coset action")
        perm.append(found)
    return perm


def coset_action_permutation_commutator(reps: List[Tuple[str, np.ndarray]], G_letter: str) -> List[int]:
    delta = abel_class_word(G_letter)
    class_to_index = {abel_class_matrix_word(w): i for i, (w, _R) in enumerate(reps)}
    out: List[int] = []
    for w, _R in reps:
        c2 = (abel_class_matrix_word(w) + delta) % 6
        out.append(class_to_index[c2])
    return out


def schreier_generators_gamma2(maker: Any, reps: List[Tuple[str, np.ndarray]], N: int, max_generators: int = 0) -> List[Tuple[str, np.ndarray, str]]:
    in_H = membership_principal(maker, N)
    S = maker._mat2(0, -1, 1, 0)
    T = maker._mat2(1, 1, 0, 1)
    gens = [("S", S), ("T", T)]
    Ikey = mat_key_psl(maker, maker._mat2(1, 0, 0, 1))
    seen: Dict[Tuple[int, ...], Tuple[str, np.ndarray, str]] = {}
    label_i = 0
    def find_rep(M: np.ndarray) -> Tuple[str, np.ndarray]:
        for wj, Rj in reps:
            if in_H(M @ maker._mat_inv_sl2(Rj)):
                return wj, Rj
        raise RuntimeError("Schreier representative not found")
    for w, R in reps:
        for ch, G in gens:
            M = R @ G
            w2, R2 = find_rep(M)
            H = R @ G @ maker._mat_inv_sl2(R2)
            if maker._det2(H) != 1:
                continue
            key = mat_key_psl(maker, H)
            if key == Ikey or key in seen:
                continue
            if max_generators and label_i >= max_generators:
                continue
            label = f"g{label_i:04d}"
            label_i += 1
            seen[key] = (label, H, f"{w or 'I'}{ch}({w2 or 'I'})^-1")
    return list(seen.values())


def schreier_generators_commutator(maker: Any, reps: List[Tuple[str, np.ndarray]], max_generators: int = 0) -> List[Tuple[str, np.ndarray, str]]:
    S = maker._mat2(0, -1, 1, 0)
    T = maker._mat2(1, 1, 0, 1)
    gens = [("S", S), ("T", T)]
    class_to_rep = {abel_class_matrix_word(w): (w, R) for w, R in reps}
    Ikey = mat_key_psl(maker, maker._mat2(1, 0, 0, 1))
    seen: Dict[Tuple[int, ...], Tuple[str, np.ndarray, str]] = {}
    label_i = 0
    for w, R in reps:
        c = abel_class_matrix_word(w)
        for ch, G in gens:
            c2 = (c + abel_class_word(ch)) % 6
            w2, R2 = class_to_rep[c2]
            H = R @ G @ maker._mat_inv_sl2(R2)
            if maker._det2(H) != 1:
                continue
            key = mat_key_psl(maker, H)
            if key == Ikey or key in seen:
                continue
            if max_generators and label_i >= max_generators:
                continue
            label = f"c{label_i:04d}"
            label_i += 1
            seen[key] = (label, H, f"{w or 'I'}{ch}({w2 or 'I'})^-1")
    return list(seen.values())


def build_modular_tiles_and_generators(maker: Any, subgroup_kind: str, max_generators: int = 0) -> Dict[str, Any]:
    S = maker._mat2(0, -1, 1, 0)
    T = maker._mat2(1, 1, 0, 1)
    ST = S @ T
    if subgroup_kind == "gamma2":
        N = 2
        reps = modular_generic_coset_reps(maker, "gamma2", N=N)
        perm_T = coset_action_permutation_gamma2(maker, reps, N, T)
        perm_S = coset_action_permutation_gamma2(maker, reps, N, S)
        perm_ST = coset_action_permutation_gamma2(maker, reps, N, ST)
        rs = schreier_generators_gamma2(maker, reps, N, max_generators=max_generators)
        subgroup_name = "Gamma(2)"
        surface_family = "elementary_modular"
        surface_subfamily = "gamma2_thrice_punctured_sphere"
        sid = "gamma2_thrice_punctured_sphere"
        compactified_genus_expected = 0
        cusp_count_expected = 3
    elif subgroup_kind == "commutator":
        reps = modular_generic_coset_reps(maker, "commutator")
        perm_T = coset_action_permutation_commutator(reps, "T")
        perm_S = coset_action_permutation_commutator(reps, "S")
        # ST increments by S then T in the word convention used by right multiplication.
        # For the C6 abelianization this is +4.
        class_to_index = {abel_class_matrix_word(w): i for i, (w, _R) in enumerate(reps)}
        perm_ST = []
        for w, _R in reps:
            perm_ST.append(class_to_index[(abel_class_matrix_word(w) + 4) % 6])
        # The commutator subgroup is free of rank two.  The full Reidemeister-
        # Schreier feedstock has redundant generators; for the zoo record use a
        # standard two-generator once-punctured-torus set whose commutator is
        # parabolic (up to sign): A=[[2,1],[1,1]], B=[[0,-1],[1,3]].
        A_min = maker._mat2(2, 1, 1, 1)
        B_min = maker._mat2(0, -1, 1, 3)
        rs = [("A", A_min, "minimal commutator generator A = [[2,1],[1,1]]"),
              ("B", B_min, "minimal commutator generator B = [[0,-1],[1,3]]")]
        subgroup_name = "[PSL(2,Z),PSL(2,Z)]"
        surface_family = "elementary_modular"
        surface_subfamily = "modular_commutator_once_punctured_torus"
        sid = "modular_commutator_once_punctured_torus"
        compactified_genus_expected = 1
        cusp_count_expected = 1
        N = 0
    else:
        raise ValueError(subgroup_kind)

    base = maker.make_modular_ford_domain(width=1.0, rotation_deg=0.0)
    base_vertices = base.get("ford_vertices", [])
    tile_data: List[Dict[str, Any]] = []
    all_vertices: List[List[float]] = []
    for idx, (word, M) in enumerate(reps):
        G = maker.psl2r_to_disk_mobius(float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1]), name=f"r{idx}")
        tile_vertices: List[List[float]] = []
        for vpair in base_vertices:
            zz = G(maker.as_complex(vpair)); rr = abs(zz)
            if rr >= 1.0:
                zz = zz / rr
            pair = maker.cpair(zz)
            tile_vertices.append(pair); all_vertices.append(pair)
        tile_data.append({"tile_index": idx, "coset_word": word or "I", "matrix": mat_json(M), "vertices": tile_vertices})

    cusp_cycles = maker._perm_cycles(perm_T)
    elliptic2 = [i for i, j in enumerate(perm_S) if i == j]
    elliptic3 = [i for i, j in enumerate(perm_ST) if i == j]
    mu = len(reps)
    cnum = len(cusp_cycles)
    e2 = len(elliptic2); e3 = len(elliptic3)
    area = mu * (math.pi / 3.0)
    gbar_float = 1.0 + mu / 12.0 - e2 / 4.0 - e3 / 3.0 - cnum / 2.0
    gbar = int(round(gbar_float))

    generators: Dict[str, Dict[str, Any]] = {}
    meanings: Dict[str, str] = {}
    matrices: Dict[str, Any] = {}
    for label, H, source in rs:
        G = maker.psl2r_to_disk_mobius(float(H[0, 0]), float(H[0, 1]), float(H[1, 0]), float(H[1, 1]), name=label)
        generators[label] = G.as_json()
        matrices[label] = mat_json(H)
        meanings[label] = f"{subgroup_name} Reidemeister-Schreier generator from {source}; matrix {matrices[label]}"

    return {
        "surface_id": sid,
        "surface_spec": sid,
        "format": "FuchsianGENN surface JSON elementary v1.1",
        "name": "Gamma(2) thrice-punctured sphere" if subgroup_kind == "gamma2" else "Modular commutator subgroup once-punctured torus",
        "domain_type": "modular_ford_domain",
        "subdomain_type": surface_subfamily,
        "category": surface_subfamily,
        "surface_family": surface_family,
        "surface_subfamily": surface_subfamily,
        "parent_group": "PSL(2,Z)",
        "subgroup": subgroup_name,
        "index_in_psl2z": mu,
        "area": area,
        "gauss_bonnet_area": area,
        "compact": False,
        "finite_area": True,
        "torsion_free": (e2 == 0 and e3 == 0),
        "orbifold_excluded": not (e2 == 0 and e3 == 0),
        "cusp_count": cnum,
        "cusp_widths": [len(c) for c in cusp_cycles],
        "compactified_genus": gbar,
        "compactified_genus_float": gbar_float,
        "expected_compactified_genus": compactified_genus_expected,
        "expected_cusp_count": cusp_count_expected,
        "elliptic_orders": [],
        "torsion_free_audit": {"right_S_fixed_cosets_order2": elliptic2, "right_ST_fixed_cosets_order3": elliptic3, "torsion_free_by_coset_fixed_point_test": (e2 == 0 and e3 == 0)},
        "coset_representatives": [{"word": w or "I", "matrix": mat_json(M)} for w, M in reps],
        "generators": generators,
        "generator_meanings": meanings,
        "generator_matrices_sl2z": matrices,
        "generator_export_audit": {"tokenized_generators": True, "underlying_schreier_generator_count": len(rs), "exported_generator_count": len(rs), "generator_truncated_by_cli_max_generators": False, "label_scheme": "g0000/c0000 token labels with formal inverses"},
        "fundamental_domain_tiles": tile_data,
        "construction_ford_vertices": all_vertices,
        "ford_vertices": all_vertices,
        "fundamental_domain_status": "explicit Ford-tile union model in the Poincare disk for a torsion-free finite-index subgroup of PSL(2,Z)",
        "sampling_status": "supported by disk tile-union sampler",
        "riemann_surface_status": "smooth noncompact finite-area hyperbolic Riemann surface with cusps; compactification obtained by adding cusp points",
        "kahler_status": "Riemann surface, hence Kähler in complex dimension one",
        "surface_area_type": "noncompact_finite_area_cusped_modular",
        "word_ball_recommended_depth": 2,
        "dataset_role": "all_riemann_surfaces_zoo_classical_small_modular_branch",
    }


def make_gamma2_surface(maker: Any) -> Dict[str, Any]:
    return build_modular_tiles_and_generators(maker, "gamma2")


def make_commutator_surface(maker: Any) -> Dict[str, Any]:
    return build_modular_tiles_and_generators(maker, "commutator")


# -----------------------------------------------------------------------------
# Audits and GINN calls
# -----------------------------------------------------------------------------

def elementary_audit(surface: str, sj: Dict[str, Any]) -> Dict[str, Any]:
    su = generator_su11_audit(sj)
    dt = sj.get("domain_type")
    family = sj.get("surface_family")
    subfam = sj.get("surface_subfamily")
    tile_count = len(sj.get("fundamental_domain_tiles") or [])
    torsion_free = bool(sj.get("torsion_free"))
    finite_area = bool(sj.get("finite_area"))
    compact = bool(sj.get("compact"))
    orbifold = bool(sj.get("orbifold_excluded"))
    # Family-specific consistency checks.
    pass_family = True
    reason = ""
    if subfam == "gamma2_thrice_punctured_sphere":
        pass_family = (sj.get("compactified_genus") == 0 and sj.get("cusp_count") == 3 and finite_area and torsion_free)
        if not pass_family: reason = "Gamma(2) expected genus 0 with 3 cusps and torsion-free finite area"
    elif subfam == "modular_commutator_once_punctured_torus":
        pass_family = (sj.get("compactified_genus") == 1 and sj.get("cusp_count") == 1 and finite_area and torsion_free)
        if not pass_family: reason = "commutator subgroup expected genus 1 with 1 cusp and torsion-free finite area"
    elif subfam in {"cyclic_hyperbolic", "cyclic_parabolic"}:
        pass_family = (not compact and not finite_area and torsion_free and not orbifold and tile_count >= 3)
        if not pass_family: reason = "cyclic hyperbolic/parabolic expected torsion-free noncompact infinite-area record with sampling scaffold"
    elif subfam == "cyclic_elliptic_orbifold_reference":
        pass_family = (not torsion_free and orbifold)
        if not pass_family: reason = "cyclic elliptic reference expected to be excluded as orbifold"
    pass_geom = bool(
        su["generator_count"] > 0
        and su["bad_generator_count"] == 0
        and float(su["su11_max_det_error"]) < 1.0e-8
        and tile_count >= 3
        and pass_family
    )
    eligible = bool(pass_geom and torsion_free and not orbifold)
    exclusion = "" if eligible else (str(sj.get("exclusion_reason") or reason or "not a smooth eligible Riemann-surface record"))
    return {
        "surface": surface,
        "surface_id": sj.get("surface_id"),
        "surface_family": family,
        "surface_subfamily": subfam,
        "mainline_dataset_eligible": eligible,
        "exclusion_reason": exclusion,
        "riemann_surface_status": sj.get("riemann_surface_status"),
        "domain_type": dt,
        "subdomain_type": sj.get("subdomain_type"),
        "torsion_free": torsion_free,
        "orbifold_excluded": orbifold,
        "compact": compact,
        "finite_area": finite_area,
        "genus": sj.get("genus"),
        "compactified_genus": sj.get("compactified_genus"),
        "area": sj.get("area"),
        "cusp_count": sj.get("cusp_count"),
        "generator_count": su["generator_count"],
        "generator_truncated": bool(sj.get("generator_truncated")),
        "pass_geometry_audit": pass_geom,
        "source_program": PROGRAM,
        "source_version": VERSION,
        "su11_max_det_error": su["su11_max_det_error"],
        "bad_generator_count": su["bad_generator_count"],
        "bad_generators": su["bad_generators"],
        "sampling_tile_count": tile_count,
        "surface_area_type": sj.get("surface_area_type"),
    }


def run_smoke(ginn: Any, sj: Dict[str, Any], surface: str, pairs: int, depth: int, seed: int, max_word_ball: int) -> Dict[str, Any]:
    rows, X, D, word_ball, label_meta, feature_names = ginn.generate_ginn_dataset(sj, pairs, depth, seed, max_word_ball=max_word_ball)
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
    t0 = time.time()
    outdir = run_root / "ginn_runs" / surface
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        rows, X, D, word_ball, label_meta, feature_names = ginn.generate_ginn_dataset(
            sj, args.ginn_pairs, args.ginn_depth, args.seed, max_word_ball=args.ginn_max_word_ball
        )
        write_json(outdir / "surface.json", sj)
        write_json(outdir / "label_meta.json", label_meta)
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


def build_requested_surfaces(maker: Any, args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    families = {x.strip().lower() for x in args.families.split(",") if x.strip()}
    if "all" in families:
        families = {"gamma2", "commutator", "cyclic_hyperbolic", "cyclic_parabolic", "cyclic_elliptic"}
    if "gamma2" in families or "punctured_sphere" in families:
        sj = make_gamma2_surface(maker)
        out.append((sj["surface_id"], sj))
    if "commutator" in families or "punctured_torus" in families:
        sj = make_commutator_surface(maker)
        out.append((sj["surface_id"], sj))
    if "cyclic" in families:
        families.update({"cyclic_hyperbolic", "cyclic_parabolic"})
        if args.include_elliptic:
            families.add("cyclic_elliptic")
    if "cyclic_hyperbolic" in families:
        for a in args.hyperbolic_strengths:
            for rot in args.rotations:
                sj = make_cyclic_hyperbolic(maker, a, rot, args.sample_radius)
                out.append((sj["surface_id"], sj))
    if "cyclic_parabolic" in families:
        for w in args.parabolic_widths:
            sj = make_cyclic_parabolic(maker, w, args.sample_radius)
            out.append((sj["surface_id"], sj))
    if "cyclic_elliptic" in families or args.include_elliptic:
        for n in args.elliptic_orders:
            sj = make_cyclic_elliptic_reference(maker, n, args.sample_radius)
            out.append((sj["surface_id"], sj))
    if args.surfaces:
        # append explicitly named surfaces if not already covered
        for spec in [s.strip() for s in args.surfaces.split(",") if s.strip()]:
            if spec in {sid for sid, _ in out}:
                continue
            if spec in {"gamma2", "thrice_punctured_sphere"}:
                sj = make_gamma2_surface(maker); out.append((sj["surface_id"], sj))
            elif spec in {"commutator", "once_punctured_torus"}:
                sj = make_commutator_surface(maker); out.append((sj["surface_id"], sj))
            elif spec.startswith("cyclic_hyperbolic"):
                a = args.hyperbolic_strengths[0]
                sj = make_cyclic_hyperbolic(maker, a, args.rotations[0], args.sample_radius); out.append((sj["surface_id"], sj))
            elif spec.startswith("cyclic_parabolic"):
                sj = make_cyclic_parabolic(maker, args.parabolic_widths[0], args.sample_radius); out.append((sj["surface_id"], sj))
            else:
                raise ValueError(f"Unknown explicit elementary surface spec {spec!r}")
    # stable unique by surface id
    seen: set[str] = set()
    unique: List[Tuple[str, Dict[str, Any]]] = []
    for sid, sj in out:
        if sid not in seen:
            unique.append((sid, sj)); seen.add(sid)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description="Classical elementary Fuchsian surface tester v1.1")
    ap.add_argument("--families", default="gamma2,commutator,cyclic", help="comma-list: gamma2,commutator,cyclic,cyclic_hyperbolic,cyclic_parabolic,cyclic_elliptic,all")
    ap.add_argument("--surfaces", default="", help="optional comma-list of explicit specs: gamma2,commutator,once_punctured_torus,...")
    ap.add_argument("--hyperbolic-strengths", nargs="*", type=float, default=[0.28, 0.45, 0.62])
    ap.add_argument("--parabolic-widths", nargs="*", type=float, default=[1.0, 2.0])
    ap.add_argument("--elliptic-orders", nargs="*", type=int, default=[2, 3, 4, 6])
    ap.add_argument("--rotations", nargs="*", type=float, default=[0.0])
    ap.add_argument("--include-elliptic", action="store_true", help="include excluded cyclic elliptic orbifold reference records")
    ap.add_argument("--sample-radius", type=float, default=0.72)
    ap.add_argument("--maker", default="FuchsianDomainMaker_v13.py")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--outroot", default="elementary_tester_runs")
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

    maker = load_module(args.maker, "domain_maker_elementary_v13")
    ginn = load_module(args.ginn_script, "ginn_v24_elementary") if (args.ginn_smoke or args.run_ginn) else None
    surfaces = build_requested_surfaces(maker, args)
    run_root = Path(args.outroot) / f"run_{now_stamp()}{('_' + args.label) if args.label else ''}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"{PROGRAM}\nrun_root={run_root}\nsurfaces={[sid for sid, _ in surfaces]}\n" + "-"*78, flush=True)

    audit_rows: List[Dict[str, Any]] = []
    smoke_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sid, sj0 in surfaces:
        print("="*78, flush=True)
        print(f"[surface] {sid}", flush=True)
        try:
            ar0 = elementary_audit(sid, sj0)
            eligible = bool(ar0["mainline_dataset_eligible"])
            sj = normalize_surface_record(
                sj0,
                surface_spec=sid,
                surface_family=str(sj0.get("surface_family", "elementary")),
                surface_subfamily=str(sj0.get("surface_subfamily", "")),
                source_program=PROGRAM,
                source_version=VERSION,
                construction_parameters={"families": args.families, "sample_radius": args.sample_radius},
                geometry_audit_pass=bool(ar0["pass_geometry_audit"]),
                finite_area=bool(sj0.get("finite_area")),
                torsion_free=bool(sj0.get("torsion_free")),
                mainline_dataset_eligible=eligible,
                exclusion_reason=str(ar0.get("exclusion_reason", "")),
            )
            # Preserve auxiliary labels normalized tools do not know about.
            for key in ["surface_area_type", "dataset_role", "mainline_finite_area_dataset_eligible", "auxiliary_schottky_dataset_eligible"]:
                if key in sj0:
                    sj[key] = sj0[key]
            write_json(run_root / "surfaces" / f"{sid}.json", sj)
            ar = dict(ar0)
            ar["mainline_dataset_eligible"] = bool(sj.get("mainline_dataset_eligible"))
            ar["exclusion_reason"] = str(sj.get("exclusion_reason", ""))
            audit_rows.append(ar)
            print(f"[audit] pass={ar['pass_geometry_audit']} family={ar['surface_subfamily']} gens={ar['generator_count']} compact={ar['compact']} finite_area={ar['finite_area']} cusps={ar['cusp_count']} eligible={ar['mainline_dataset_eligible']}", flush=True)
            if args.ginn_smoke and ginn is not None:
                if not sj.get("mainline_dataset_eligible"):
                    smoke_rows.append({"surface": sid, "surface_id": sid, "pass_ginn_preflight": False, "error": "not eligible"})
                    print("[ginn-smoke] skipped: not eligible", flush=True)
                else:
                    sr = run_smoke(ginn, sj, sid, args.ginn_pairs, args.ginn_depth, args.seed, args.ginn_max_word_ball)
                    smoke_rows.append(sr)
                    print(f"[ginn-smoke] PASS W={sr['word_ball_size']} shortcut={sr.get('shortcut_fraction')}", flush=True)
            if args.run_ginn and ginn is not None:
                tr = run_direct_ginn(ginn, sj, sid, run_root, args)
                train_rows.append(tr)
                print(f"[ginn-train] pass={tr['pass_ginn_training']} rc={tr['returncode']}", flush=True)
        except Exception as e:
            print(f"[FAIL] {sid}: {type(e).__name__}: {e}", flush=True)
            failures.append({"surface": sid, "surface_id": sid, "error_type": type(e).__name__, "error": str(e)})

    write_csv(run_root / "tables" / "geometry_audit.csv", audit_rows, GEOMETRY_AUDIT_FIELDS)
    write_csv(run_root / "tables" / "elementary_geometry_audit.csv", audit_rows)
    write_csv(run_root / "tables" / "ginn_smoke_summary.csv", smoke_rows, GINN_SMOKE_FIELDS)
    write_csv(run_root / "tables" / "ginn_training_summary.csv", train_rows, GINN_TRAINING_FIELDS)
    write_csv(run_root / "tables" / "failures.csv", failures, FAILURE_FIELDS)
    write_json(run_root / "manifest.json", {
        "program": PROGRAM,
        "version": VERSION,
        "args": vars(args),
        "surfaces": [sid for sid, _sj in surfaces],
        "completed": len(audit_rows),
        "failures": len(failures),
        "family_status": "Classical elementary Riemann-surface records plus excluded elliptic orbifold references.",
    })
    print(f"[done] completed={len(audit_rows)} failures={len(failures)} run_root={run_root}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
