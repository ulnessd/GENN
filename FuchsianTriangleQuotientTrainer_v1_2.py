#!/usr/bin/env python3
"""
FuchsianTriangleQuotientTrainer_v1_2.py

Generic finite quotient of a hyperbolic triangle group
    Delta^+(p,q,r) = < X,Y | X^p = Y^q = (XY)^r = 1 >
-> Fuchsian kernel surface -> exact finite word-ball atlas -> candidate-pool
GINN reranker.

This is a small, zoo-oriented generalization of the PSL(2,q) Hurwitz trainer.
It is intended first for the two classical non-Hurwitz anchor animals:

  --signature 2,3,8 --quotient GL2_3    Bolza surface, genus 2, |G|=48
  --signature 2,4,5 --quotient S5       Bring curve, genus 4, |G|=120

It keeps the same selected-kernel-generator convention:

  --kernel-generator-mode all       export every nonidentity Schreier generator
  --kernel-generator-mode first     export the first N nonidentity generators
  --kernel-generator-mode shortest  scan all kernel edges and export the N
                                    shortest by SU(1,1) identity displacement

Interpretation warning
----------------------
When --kernel-generator-mode shortest/first is used, the word ball is the exact
reduced length-<=depth word ball in the selected generator pool.  It is not the
full word ball in the complete Reidemeister-Schreier generating set.
"""

from __future__ import annotations

import argparse
import csv
import cmath
import heapq
import importlib.util
import json
import math
import platform
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROGRAM = "FuchsianTriangleQuotientTrainer_v1_2.py"
VERSION = "1.0"

# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_slug(label: str) -> str:
    s = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(label))
    return s.strip('_') or 'run'


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k); keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def load_module(path: str | Path, module_prefix: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Required module not found: {p}")
    if str(p.parent) not in sys.path:
        sys.path.insert(0, str(p.parent))
    mod_name = f"_{module_prefix}_{abs(hash(str(p))) & 0xffffffff:x}"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def default_local(name: str) -> str:
    candidates = [Path.cwd() / name, Path(__file__).resolve().parent / name]
    for p in candidates:
        if p.exists():
            return str(p)
    return name


def default_zoo_script() -> str:
    return default_local("FuchsianBigHurwitzZoo_v1_8.py")


def default_big_trainer_script() -> str:
    return default_local("FuchsianBigHurwitzTrainer_v1_7.py")


def default_ginn_script() -> str:
    return default_local("FuchsianDownstairsGINN_v2_4.py")


def parse_int_list(s: str | Sequence[int]) -> List[int]:
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    out: List[int] = []
    if not s:
        return out
    for part in str(s).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_signature(s: str | Sequence[int]) -> Tuple[int, int, int]:
    if isinstance(s, (list, tuple)):
        vals = [int(x) for x in s]
    else:
        vals = parse_int_list(str(s).replace("x", ",").replace("/", ","))
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("signature must have three integers, e.g. 2,3,8")
    p, q, r = vals
    if p < 2 or q < 2 or r < 2:
        raise argparse.ArgumentTypeError("triangle orders must be >= 2")
    if 1.0 / p + 1.0 / q + 1.0 / r >= 1.0:
        raise argparse.ArgumentTypeError(f"signature ({p},{q},{r}) is not hyperbolic")
    return p, q, r


def signature_slug(sig: Tuple[int, int, int]) -> str:
    return f"{sig[0]}_{sig[1]}_{sig[2]}"


def triangle_defect(sig: Tuple[int, int, int]) -> float:
    p, q, r = sig
    return 1.0 - 1.0 / p - 1.0 / q - 1.0 / r


def triangle_genus(order: int, sig: Tuple[int, int, int]) -> Optional[int]:
    # |G| * area(orientation-preserving orbifold fundamental kite)
    # = |G| * 2*pi*defect = 4*pi(g-1)
    val = 1.0 + float(order) * triangle_defect(sig) / 2.0
    nearest = round(val)
    if abs(val - nearest) < 1.0e-9:
        return int(nearest)
    return None

# -----------------------------------------------------------------------------
# SU(1,1) representation of Delta^+(p,q,r)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SU11Mobius:
    alpha: complex
    beta: complex
    label: str = ""

    def normalized(self, label: Optional[str] = None) -> "SU11Mobius":
        det = abs(self.alpha) ** 2 - abs(self.beta) ** 2
        if det <= 0 or not math.isfinite(det):
            return self
        scale = 1.0 / math.sqrt(det)
        a = self.alpha * scale
        b = self.beta * scale
        if a.real < -1.0e-14 or (abs(a.real) <= 1.0e-14 and a.imag < 0):
            a, b = -a, -b
        return SU11Mobius(a, b, self.label if label is None else label)

    def compose(self, other: "SU11Mobius", label: Optional[str] = None) -> "SU11Mobius":
        a = self.alpha * other.alpha + self.beta * other.beta.conjugate()
        b = self.alpha * other.beta + self.beta * other.alpha.conjugate()
        return SU11Mobius(a, b, self.label + other.label if label is None else label).normalized()

    def inverse(self, label: Optional[str] = None) -> "SU11Mobius":
        return SU11Mobius(self.alpha.conjugate(), -self.beta, self.label + "^-1" if label is None else label).normalized(label)

    def apply(self, z: complex) -> complex:
        return (self.alpha * z + self.beta) / (self.beta.conjugate() * z + self.alpha.conjugate())

    def trace_real(self) -> float:
        return float(2.0 * self.alpha.real)

    def displacement_from_identity(self) -> float:
        return math.hypot(self.alpha.real - 1.0, self.alpha.imag) + abs(self.beta)

    def as_json(self) -> Dict[str, Any]:
        return {
            "type": "su11",
            "alpha": [float(self.alpha.real), float(self.alpha.imag)],
            "beta": [float(self.beta.real), float(self.beta.imag)],
        }


def su11_identity(label: str = "I") -> SU11Mobius:
    return SU11Mobius(1.0 + 0j, 0.0 + 0j, label)


def su11_rotation_about_zero(theta: float, label: str) -> SU11Mobius:
    return SU11Mobius(cmath.exp(0.5j * theta), 0.0j, label).normalized(label)


def su11_real_translation_to(r: float, label: str = "T") -> SU11Mobius:
    if not (abs(r) < 1.0):
        raise ValueError("Disk translation parameter must satisfy |r| < 1")
    a = 1.0 / math.sqrt(1.0 - r * r)
    return SU11Mobius(a + 0j, r * a + 0j, label).normalized(label)


def su11_rotation_about_real_point(r: float, theta: float, label: str) -> SU11Mobius:
    T = su11_real_translation_to(r, "T")
    R = su11_rotation_about_zero(theta, "R")
    return T.compose(R).compose(T.inverse(), label=label).normalized(label)


def fixed_point_inside_disk(M: SU11Mobius) -> complex:
    A = M.beta.conjugate()
    B = M.alpha.conjugate() - M.alpha
    C = -M.beta
    if abs(A) < 1.0e-14:
        return 0.0j
    disc = B * B - 4 * A * C
    roots = [(-B + cmath.sqrt(disc)) / (2 * A), (-B - cmath.sqrt(disc)) / (2 * A)]
    return sorted(roots, key=lambda z: abs(z))[0]


def mobius_pow(M: SU11Mobius, n: int) -> SU11Mobius:
    out = su11_identity()
    for _ in range(int(n)):
        out = M.compose(out)
    return out.normalized()


def build_delta_pqr_su11(sig: Tuple[int, int, int]) -> Tuple[Dict[str, SU11Mobius], Dict[str, Any]]:
    """Concrete PSU(1,1) representation of Delta^+(p,q,r).

    X is a rotation of angle 2*pi/p about 0.  Y is a rotation of angle
    2*pi/q about a real point t.  We solve t so |trace(XY)| = 2 cos(pi/r).
    """
    p, q, r = sig
    target = abs(2.0 * math.cos(math.pi / r))
    X = su11_rotation_about_zero(2.0 * math.pi / p, "X")

    def val(t: float) -> float:
        Y = su11_rotation_about_real_point(t, 2.0 * math.pi / q, "Y")
        return abs(X.compose(Y, label="XY").trace_real())

    # Bracket a root.  For hyperbolic signatures a solution lies in (0,1).
    lo = 0.0
    prev_t = lo
    prev = val(prev_t) - target
    found = False
    bracket = (0.0, 0.95)
    for i in range(1, 9990):
        t = i / 10000.0
        cur = val(t) - target
        if prev == 0 or prev * cur <= 0:
            bracket = (prev_t, t)
            found = True
            break
        prev_t, prev = t, cur
    if not found:
        raise RuntimeError(f"Could not bracket Delta^+({p},{q},{r}) SU(1,1) parameter")
    lo, hi = bracket
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (val(lo) - target) * (val(mid) - target) <= 0:
            hi = mid
        else:
            lo = mid
    t = 0.5 * (lo + hi)
    Y = su11_rotation_about_real_point(t, 2.0 * math.pi / q, "Y")
    XY = X.compose(Y, label="XY")
    Z = XY.inverse(label="Z")
    gens = {"X": X, "Y": Y, "Y^-1": Y.inverse("Y^-1")}
    if p != 2:
        gens["X^-1"] = X.inverse("X^-1")
    fpz = fixed_point_inside_disk(Z)
    rel = {
        "X_order_identity_error": mobius_pow(X, p).displacement_from_identity(),
        "Y_order_identity_error": mobius_pow(Y, q).displacement_from_identity(),
        "XY_order_identity_error": mobius_pow(XY, r).displacement_from_identity(),
    }
    audit = {
        "representation": f"PSU(1,1) concrete Delta^+({p},{q},{r}) representation",
        "triangle_signature": [p, q, r],
        "defect": triangle_defect(sig),
        "X_rotation_angle": 2.0 * math.pi / p,
        "Y_rotation_angle": 2.0 * math.pi / q,
        "XY_order_expected": r,
        "Y_fixed_point_real_r": t,
        "hyperbolic_distance_between_XY_centers": 2.0 * math.atanh(t),
        "target_abs_trace_order_r": target,
        "abs_trace_XY": abs(XY.trace_real()),
        "trace_error": abs(abs(XY.trace_real()) - target),
        "relations_numerical": rel,
        "X_fixed_point": [0.0, 0.0],
        "Y_fixed_point": [float(t), 0.0],
        "Z_fixed_point": [float(fpz.real), float(fpz.imag)],
        "base_orbifold_triangle_vertices": [[0.0, 0.0], [float(t), 0.0], [float(fpz.real), float(fpz.imag)]],
    }
    return gens, audit


def normalize_su11_pair(alpha: complex, beta: complex) -> Tuple[complex, complex]:
    det = abs(alpha) ** 2 - abs(beta) ** 2
    if det > 0 and math.isfinite(det):
        scale = 1.0 / math.sqrt(det)
        alpha *= scale; beta *= scale
    if alpha.real < -1.0e-14 or (abs(alpha.real) <= 1.0e-14 and alpha.imag < 0):
        alpha, beta = -alpha, -beta
    return alpha, beta


def su11_pair_inverse(alpha: complex, beta: complex) -> Tuple[complex, complex]:
    return normalize_su11_pair(alpha.conjugate(), -beta)


def su11_pair_compose(a: complex, b: complex, c: complex, d: complex) -> Tuple[complex, complex]:
    return normalize_su11_pair(a * c + b * d.conjugate(), a * d + b * c.conjugate())


def su11_pair_apply(alpha: complex, beta: complex, z: complex) -> complex:
    return (alpha * z + beta) / (beta.conjugate() * z + alpha.conjugate())


def su11_pair_displacement_from_identity(alpha: complex, beta: complex) -> float:
    return math.hypot(alpha.real - 1.0, alpha.imag) + abs(beta)


def su11_pair_as_json(alpha: complex, beta: complex) -> Dict[str, Any]:
    return {"type": "su11", "alpha": [float(alpha.real), float(alpha.imag)], "beta": [float(beta.real), float(beta.imag)]}

# -----------------------------------------------------------------------------
# Finite quotient groups: GL(2,3) and S5 anchor animals
# -----------------------------------------------------------------------------

Element = Tuple[int, ...]

@dataclass
class FiniteTriangleQuotient:
    name: str
    family: str
    signature: Tuple[int, int, int]
    elements: List[Element]
    identity: Element
    x: Element
    y: Element
    order: int
    metadata: Dict[str, Any]

    def mul(self, a: Element, b: Element) -> Element:
        if self.family in {"perm", "Sn", "An"}:
            # a after b
            return tuple(a[i] for i in b)
        if self.family == "psl2p":
            pp = int(self.metadata["field_p"])
            return canonical_psl2p(((a[0]*b[0] + a[1]*b[2]) % pp,
                                    (a[0]*b[1] + a[1]*b[3]) % pp,
                                    (a[2]*b[0] + a[3]*b[2]) % pp,
                                    (a[2]*b[1] + a[3]*b[3]) % pp), pp)
        if self.family == "gl2p":
            p = int(self.metadata["field_p"])
            return ((a[0]*b[0] + a[1]*b[2]) % p,
                    (a[0]*b[1] + a[1]*b[3]) % p,
                    (a[2]*b[0] + a[3]*b[2]) % p,
                    (a[2]*b[1] + a[3]*b[3]) % p)
        raise ValueError(self.family)

    def inv(self, a: Element) -> Element:
        if self.family in {"perm", "Sn", "An"}:
            out = [0] * len(a)
            for i, j in enumerate(a):
                out[j] = i
            return tuple(out)
        if self.family in {"gl2p", "psl2p"}:
            p = int(self.metadata["field_p"])
            det = (a[0]*a[3] - a[1]*a[2]) % p
            dinv = pow(det, -1, p)
            raw = ((a[3]*dinv) % p, (-a[1]*dinv) % p, (-a[2]*dinv) % p, (a[0]*dinv) % p)
            if self.family == "psl2p":
                return canonical_psl2p(raw, p)
            return raw
        raise ValueError(self.family)

    def pow(self, a: Element, n: int) -> Element:
        out = self.identity
        base = a
        k = int(n)
        while k > 0:
            if k & 1:
                out = self.mul(out, base)
            base = self.mul(base, base)
            k >>= 1
        return out

    def order_of(self, a: Element, max_n: Optional[int] = None) -> int:
        out = self.identity
        lim = max_n or max(2, self.order * 2)
        for n in range(1, lim + 1):
            out = self.mul(out, a)
            if out == self.identity:
                return n
        raise RuntimeError("order computation failed")

    def generated_subgroup_order(self, gens: Sequence[Element], stop_at_full: bool = True) -> int:
        gl: List[Element] = []
        for g in gens:
            if g not in gl:
                gl.append(g)
            ig = self.inv(g)
            if ig not in gl:
                gl.append(ig)
        seen = {self.identity}
        qd: deque[Element] = deque([self.identity])
        while qd:
            h = qd.popleft()
            for g in gl:
                hg = self.mul(h, g)
                if hg not in seen:
                    seen.add(hg)
                    if stop_at_full and len(seen) == self.order:
                        return len(seen)
                    qd.append(hg)
        return len(seen)

    def to_json_element(self, a: Element) -> Any:
        if self.family in {"perm", "Sn", "An"}:
            return [int(x) + 1 for x in a]
        if self.family in {"gl2p", "psl2p"}:
            return [[int(a[0]), int(a[1])], [int(a[2]), int(a[3])]]
        return list(a)

    def validate_triangle(self) -> Dict[str, Any]:
        p, q, r = self.signature
        xy = self.mul(self.x, self.y)
        xo = self.order_of(self.x)
        yo = self.order_of(self.y)
        xyo = self.order_of(xy)
        gen_order = self.generated_subgroup_order([self.x, self.y], stop_at_full=True)
        return {
            "x_order": xo,
            "y_order": yo,
            "xy_order": xyo,
            "expected_signature": [p, q, r],
            "exact_signature": bool(xo == p and yo == q and xyo == r),
            "generated_order": gen_order,
            "surjective": bool(gen_order == self.order),
        }


def build_gl2_3_bolza() -> FiniteTriangleQuotient:
    p = 3
    elems: List[Element] = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    if (a*d - b*c) % p != 0:
                        elems.append((a, b, c, d))
    # Found by brute force: x^2=y^3=(xy)^8=1 and <x,y>=GL(2,3)
    x = (0, 1, 1, 0)
    y = (1, 0, 1, 1)
    return FiniteTriangleQuotient(
        name="GL2_3_Bolza",
        family="gl2p",
        signature=(2, 3, 8),
        elements=sorted(elems),
        identity=(1, 0, 0, 1),
        x=x,
        y=y,
        order=len(elems),
        metadata={"field_p": 3, "description": "GL(2,3), order 48, Bolza surface automorphism group quotient of Delta^+(2,3,8)"},
    )


def build_s5_bring() -> FiniteTriangleQuotient:
    from itertools import permutations
    n = 5
    elems = [tuple(p) for p in permutations(range(n))]
    # x=(4 5), y=(1 2 3 4) in one-line 0-based notation. xy is a 5-cycle.
    x = (0, 1, 2, 4, 3)
    y = (1, 2, 3, 0, 4)
    return FiniteTriangleQuotient(
        name="S5_Bring",
        family="perm",
        signature=(2, 4, 5),
        elements=elems,
        identity=tuple(range(n)),
        x=x,
        y=y,
        order=len(elems),
        metadata={"degree": 5, "description": "S5, order 120, Bring curve automorphism group quotient of Delta^+(2,4,5)"},
    )



# -----------------------------------------------------------------------------
# Generic finite quotient builders for search-manifest candidates
# -----------------------------------------------------------------------------

def perm_parity(p: Element) -> int:
    invs = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                invs += 1
    return invs % 2


def build_symmetric_elements(n: int) -> List[Element]:
    from itertools import permutations
    return [tuple(pp) for pp in permutations(range(n))]


def build_alternating_elements(n: int) -> List[Element]:
    from itertools import permutations
    return [tuple(pp) for pp in permutations(range(n)) if perm_parity(tuple(pp)) == 0]


def canonical_psl2p(m: Element, p: int) -> Element:
    a = tuple(int(x) % p for x in m)
    neg = tuple((-x) % p for x in a)
    return min(a, neg)


def psl2_order(p: int) -> int:
    return p * (p * p - 1) // math.gcd(2, p - 1)


def build_psl2p_elements(p: int) -> List[Element]:
    elems_set = set()
    for a in range(p):
        for b in range(p):
            for c in range(p):
                if a % p != 0:
                    d = ((1 + b * c) * pow(a, -1, p)) % p
                    elems_set.add(canonical_psl2p((a, b, c, d), p))
                else:
                    if b % p != 0 and c % p != 0 and (-b * c) % p == 1:
                        for d in range(p):
                            elems_set.add(canonical_psl2p((a, b, c, d), p))
    elems = sorted(elems_set)
    exp = psl2_order(p)
    if len(elems) != exp:
        raise RuntimeError(f"PSL2({p}) enumeration produced {len(elems)}, expected {exp}")
    return elems


def parse_perm_element(obj: Any) -> Element:
    vals = [int(v) for v in obj]
    # Searcher writes permutations 1-based for JSON.
    if vals and min(vals) == 1:
        vals = [v - 1 for v in vals]
    return tuple(vals)


def parse_matrix_element(obj: Any, p: int, projective: bool = False) -> Element:
    if isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(obj[0], (list, tuple)):
        raw = (int(obj[0][0]), int(obj[0][1]), int(obj[1][0]), int(obj[1][1]))
    else:
        raw = tuple(int(x) for x in obj)
    raw = tuple(x % p for x in raw)
    return canonical_psl2p(raw, p) if projective else raw


def quotient_from_descriptor(desc: Dict[str, Any], requested_sig: Tuple[int, int, int]) -> FiniteTriangleQuotient:
    sig = tuple(int(x) for x in desc.get("signature_tuple", requested_sig))
    if sig != tuple(requested_sig):
        raise RuntimeError(f"quotient descriptor signature {sig} != requested {requested_sig}")
    group = str(desc.get("group") or desc.get("group_name") or desc.get("quotient") or "")
    family = str(desc.get("family") or "")
    meta = dict(desc.get("metadata") or {})
    xj = desc.get("x", meta.get("x"))
    yj = desc.get("y", meta.get("y"))
    if xj is None or yj is None:
        raise RuntimeError("quotient descriptor must include x and y generator data")

    if group in {"GL2_3_Bolza", "GL2_3", "Bolza"} or (family == "gl2p" and int(meta.get("field_p", 0)) == 3):
        p = int(meta.get("field_p", 3))
        elems = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    for d in range(p):
                        if (a*d - b*c) % p != 0:
                            elems.append((a,b,c,d))
        x = parse_matrix_element(xj, p, projective=False)
        y = parse_matrix_element(yj, p, projective=False)
        return FiniteTriangleQuotient(group or "GL2_3_Bolza", "gl2p", sig, sorted(elems), (1,0,0,1), x, y, len(elems), {"field_p": p, **meta})

    if group.startswith("PSL2_") or family == "psl2p":
        pp = int(meta.get("field_p") or group.split("_")[-1])
        elems = build_psl2p_elements(pp)
        x = parse_matrix_element(xj, pp, projective=True)
        y = parse_matrix_element(yj, pp, projective=True)
        ident = canonical_psl2p((1,0,0,1), pp)
        return FiniteTriangleQuotient(group or f"PSL2_{pp}", "psl2p", sig, elems, ident, x, y, len(elems), {"field_p": pp, **meta})

    if group.startswith("A") and group[1:].isdigit() or family == "An":
        n = int(meta.get("degree") or group[1:])
        elems = build_alternating_elements(n)
        x = parse_perm_element(xj)
        y = parse_perm_element(yj)
        return FiniteTriangleQuotient(group or f"A{n}", "An", sig, elems, tuple(range(n)), x, y, len(elems), {"degree": n, **meta})

    if group.startswith("S") and group[1:].isdigit() or family in {"Sn", "perm"}:
        n = int(meta.get("degree") or group[1:])
        elems = build_symmetric_elements(n)
        x = parse_perm_element(xj)
        y = parse_perm_element(yj)
        return FiniteTriangleQuotient(group or f"S{n}", "Sn", sig, elems, tuple(range(n)), x, y, len(elems), {"degree": n, **meta})

    raise RuntimeError(f"Unsupported quotient descriptor group={group!r} family={family!r}")

def build_quotient(args: argparse.Namespace) -> FiniteTriangleQuotient:
    sig = tuple(args.signature)
    if getattr(args, "quotient_json", ""):
        desc = json.loads(Path(args.quotient_json).read_text(encoding="utf-8"))
        G = quotient_from_descriptor(desc, sig)
        cert = G.validate_triangle()
        if not cert["exact_signature"] or not cert["surjective"]:
            raise RuntimeError(f"Descriptor quotient failed triangle certificate: {cert}")
        return G
    qname = str(args.quotient)
    if qname == "auto":
        if sig == (2, 3, 8):
            qname = "GL2_3"
        elif sig == (2, 4, 5):
            qname = "S5"
        else:
            raise RuntimeError(f"No built-in auto quotient for signature {sig}; choose --quotient GL2_3 or S5 where appropriate")
    if qname in {"GL2_3", "Bolza", "bolza"}:
        G = build_gl2_3_bolza()
    elif qname in {"S5", "Bring", "bring"}:
        G = build_s5_bring()
    else:
        raise RuntimeError(f"Unknown quotient {args.quotient}. Supported: auto, GL2_3, S5, or --quotient-json from searcher")
    if tuple(G.signature) != sig:
        raise RuntimeError(f"Quotient {G.name} has signature {G.signature}, but CLI requested {sig}")
    cert = G.validate_triangle()
    if not cert["exact_signature"] or not cert["surjective"]:
        raise RuntimeError(f"Built-in quotient failed triangle certificate: {cert}")
    return G

# -----------------------------------------------------------------------------
# GINN word-ball convention hook
# -----------------------------------------------------------------------------

def install_leq_depth_word_ball_builder(ginn: Any) -> None:
    if not hasattr(ginn, "Mobius"):
        raise AttributeError("GINN module does not expose Mobius; cannot install word-ball builder")

    def _inverse_token(tok: str) -> str:
        if hasattr(ginn, "inverse_token"):
            return ginn.inverse_token(tok)
        return tok[:-3] if tok.endswith("^-1") else tok + "^-1"

    def _word_to_string(tokens: Tuple[str, ...]) -> str:
        if hasattr(ginn, "word_to_string"):
            return ginn.word_to_string(tokens)
        return " ".join(tokens)

    def _compose_token_word(tokens: Tuple[str, ...], gens: Dict[str, Any]) -> Any:
        current = ginn.Mobius(1.0 + 0j, 0.0 + 0j, "")
        for tok in tokens:
            current = gens[tok].compose(current, word="")
        return ginn.Mobius(current.alpha, current.beta, _word_to_string(tokens)).normalized()

    def build_word_ball_leq_depth(gens: Dict[str, Any], depth: int) -> List[Any]:
        depth = int(depth)
        letters = sorted(gens.keys(), key=lambda c: (c.replace('^-1', ''), c.endswith('^-1'), c))
        words: List[Tuple[str, ...]] = [tuple()]
        frontier: List[Tuple[str, ...]] = [tuple()]
        for _ in range(max(0, depth)):
            new_frontier: List[Tuple[str, ...]] = []
            for w in frontier:
                last = w[-1] if w else None
                for tok in letters:
                    if last is not None and _inverse_token(tok) == last:
                        continue
                    nw = w + (tok,)
                    new_frontier.append(nw)
                    words.append(nw)
            frontier = new_frontier
        out: List[Any] = []
        seen: set[Tuple[str, ...]] = set()
        for toks in words:
            if toks in seen:
                continue
            seen.add(toks)
            if not toks:
                out.append(ginn.Mobius(1.0 + 0j, 0.0 + 0j, ""))
            else:
                out.append(_compose_token_word(toks, gens))
        return out

    ginn.build_word_ball = build_word_ball_leq_depth

# -----------------------------------------------------------------------------
# Surface construction
# -----------------------------------------------------------------------------

def build_triangle_kernel_surface(G: FiniteTriangleQuotient, args: argparse.Namespace, run_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.perf_counter()
    sig = tuple(args.signature)
    p, q, r = sig
    delta_gens, delta_audit = build_delta_pqr_su11(sig)

    finite_gens: Dict[str, Element] = {"X": G.x, "Y": G.y, "Y^-1": G.inv(G.y)}
    token_order = ["X", "Y", "Y^-1"]
    if p != 2:
        finite_gens["X^-1"] = G.inv(G.x)
        token_order = ["X", "X^-1", "Y", "Y^-1"]

    # Schreier transversal BFS in finite quotient.
    rep_id: Dict[Element, int] = {G.identity: 0}
    rep_elems: List[Element] = [G.identity]
    parent_id: List[int] = [-1]
    parent_tok: List[str] = [""]
    qd: deque[int] = deque([0])
    while qd:
        hid = qd.popleft()
        h = rep_elems[hid]
        for tok in token_order:
            nh = G.mul(h, finite_gens[tok])
            if nh not in rep_id:
                nid = len(rep_elems)
                rep_id[nh] = nid
                rep_elems.append(nh)
                parent_id.append(hid)
                parent_tok.append(tok)
                qd.append(nid)
    if len(rep_elems) != G.order:
        raise RuntimeError(f"Schreier transversal incomplete: {len(rep_elems)} of {G.order}")

    coset_alpha = np.empty(G.order, dtype=np.complex128)
    coset_beta = np.empty(G.order, dtype=np.complex128)
    coset_alpha[0] = 1.0 + 0j
    coset_beta[0] = 0.0 + 0j
    delta_pair = {tok: (complex(delta_gens[tok].alpha), complex(delta_gens[tok].beta)) for tok in token_order}
    for i in range(1, G.order):
        pid = parent_id[i]
        tok = parent_tok[i]
        a, b = su11_pair_compose(complex(coset_alpha[pid]), complex(coset_beta[pid]), delta_pair[tok][0], delta_pair[tok][1])
        coset_alpha[i] = a; coset_beta[i] = b

    mode = str(args.kernel_generator_mode)
    limit = int(args.kernel_generator_limit)
    if mode == "all":
        limit = 0
    if mode not in {"all", "first", "shortest"}:
        raise ValueError(f"Unknown kernel_generator_mode={mode}")

    raw_nonidentity = 0
    identity_like = 0
    scanned_slots = 0
    selected_all: List[Dict[str, Any]] = []
    selected_heap: List[Tuple[float, int, Dict[str, Any]]] = []

    def keep_record(rec: Dict[str, Any]) -> None:
        if mode in ("all", "first"):
            if mode == "all" or limit <= 0 or len(selected_all) < limit:
                selected_all.append(rec)
            return
        disp = float(rec["identity_displacement"])
        item = (-disp, int(rec["scan_index"]), rec)
        if limit <= 0:
            selected_all.append(rec)
        elif len(selected_heap) < limit:
            heapq.heappush(selected_heap, item)
        elif disp < -selected_heap[0][0]:
            heapq.heapreplace(selected_heap, item)

    for hid, h in enumerate(rep_elems):
        ah = complex(coset_alpha[hid]); bh = complex(coset_beta[hid])
        for tok in token_order:
            scanned_slots += 1
            hs = G.mul(h, finite_gens[tok])
            hs_id = rep_id[hs]
            asg, bsg = delta_pair[tok]
            a1, b1 = su11_pair_compose(ah, bh, asg, bsg)
            ai, bi = su11_pair_inverse(complex(coset_alpha[hs_id]), complex(coset_beta[hs_id]))
            ak, bk = su11_pair_compose(a1, b1, ai, bi)
            disp = su11_pair_displacement_from_identity(ak, bk)
            if disp < float(args.identity_tol):
                identity_like += 1
                continue
            raw_nonidentity += 1
            trace_real = float(2.0 * ak.real)
            rec = {
                "scan_index": scanned_slots - 1,
                "source_coset_id": hid,
                "edge_token": tok,
                "target_coset_id": hs_id,
                "identity_displacement": float(disp),
                "trace_real": trace_real,
                "is_hyperbolic_by_trace": bool(abs(trace_real) > 2.0 + 1.0e-8),
                "finite_image_identity": True,
                "su11": su11_pair_as_json(ak, bk),
            }
            keep_record(rec)
        if mode == "first" and limit > 0 and len(selected_all) >= limit:
            break
        if getattr(args, "verbose", False) and hid > 0 and hid % max(1, int(args.kernel_progress_every)) == 0:
            kept = len(selected_all) if mode != "shortest" else len(selected_heap)
            print(f"[kernel {G.name}] scanned cosets={hid:,}/{G.order:,} raw_nonidentity={raw_nonidentity:,} kept={kept:,}", flush=True)

    if mode == "shortest" and limit > 0:
        selected = [item[2] for item in selected_heap]
        selected.sort(key=lambda rr: (float(rr["identity_displacement"]), int(rr["scan_index"])))
    else:
        selected = selected_all
        if mode == "shortest":
            selected.sort(key=lambda rr: (float(rr["identity_displacement"]), int(rr["scan_index"])))

    gen_json: Dict[str, Dict[str, Any]] = {}
    meanings: Dict[str, str] = {}
    audit_sample: List[Dict[str, Any]] = []
    for idx, rec in enumerate(selected):
        label = f"h{idx:06d}"
        gen_json[label] = rec["su11"]
        meanings[label] = (
            f"Delta^+({p},{q},{r})/{G.name} selected Schreier kernel edge generator {label}: "
            f"coset {rec['source_coset_id']} --{rec['edge_token']}--> {rec['target_coset_id']}; "
            f"selection={mode}; displacement={float(rec['identity_displacement']):.8g}; trace={float(rec['trace_real']):.8g}"
        )
        if len(audit_sample) < int(args.kernel_audit_sample_rows):
            audit_sample.append({k: v for k, v in rec.items() if k != "su11"})

    kernel_generator_scan_complete = not (mode == "first" and limit > 0 and len(selected_all) >= limit and scanned_slots < G.order * len(token_order))
    generator_export_complete = (mode == "all" and kernel_generator_scan_complete)

    base_verts = [complex(x, y) for x, y in delta_audit["base_orbifold_triangle_vertices"]]
    max_tiles = int(args.max_tiles) if int(args.max_tiles) > 0 else G.order
    tiles = []
    for tile_idx in range(min(G.order, max_tiles)):
        a = complex(coset_alpha[tile_idx]); b = complex(coset_beta[tile_idx])
        verts = [su11_pair_apply(a, b, z) for z in base_verts]
        tiles.append({
            "tile_index": tile_idx,
            "coset_id": tile_idx,
            "coset_element": G.to_json_element(rep_elems[tile_idx]),
            "vertices": [[float(z.real), float(z.imag)] for z in verts],
        })
    tile_scaffold_complete = len(tiles) == G.order

    cert = G.validate_triangle()
    relation_max_error = max(delta_audit["relations_numerical"].values())
    finite_certificate_complete = bool(cert["exact_signature"] and cert["surjective"])
    atlas_training_ready = bool(gen_json) and bool(tiles) and finite_certificate_complete and relation_max_error < 1.0e-8
    pass_geometry_audit = atlas_training_ready and generator_export_complete and tile_scaffold_complete

    partial_reasons: List[str] = []
    if not generator_export_complete:
        partial_reasons.append(f"kernel generator export is selected/truncated: mode={mode}, selected={len(gen_json)}, raw_nonidentity={raw_nonidentity}")
    if not kernel_generator_scan_complete:
        partial_reasons.append("kernel edge scan stopped early")
    if not tile_scaffold_complete:
        partial_reasons.append("triangle-tile scaffold truncated by --max-tiles")
    if relation_max_error >= 1.0e-8:
        partial_reasons.append("PSU(1,1) triangle relation numerical error above tolerance")
    if not finite_certificate_complete:
        partial_reasons.append("finite quotient exact triangle/surjectivity certificate failed")
    if not gen_json:
        partial_reasons.append("no nonidentity kernel generators exported")
    exclusion_reason = "; ".join(partial_reasons)

    genus = triangle_genus(G.order, sig)
    sid = f"triangle_{p}_{q}_{r}_{stable_slug(G.name)}_kernel_{mode}{len(gen_json)}"
    surface = {
        "format": "FuchsianGENN surface JSON v1.2 triangle-quotient-tokenized-kernel",
        "surface_id": sid,
        "name": f"Triangle quotient kernel surface Delta^+({p},{q},{r}) -> {G.name}",
        "surface_type": "triangle_quotient_kernel_surface",
        "domain_type": "triangle_kernel_tile_union",
        "compact": True,
        "finite_area": True,
        "torsion_free": True,
        "orbifold_excluded": False,
        "mainline_dataset_eligible": pass_geometry_audit,
        "atlas_training_ready": atlas_training_ready,
        "riemann_surface_status": f"smooth compact Riemann surface D/Gamma, with Gamma the torsion-free kernel of Delta^+({p},{q},{r}) -> {G.name}",
        "kahler_status": "complex dimension one; automatically Kähler",
        "genus": genus,
        "area": 4.0 * math.pi * (genus - 1) if genus is not None else None,
        "gauss_bonnet_area": 4.0 * math.pi * (genus - 1) if genus is not None else None,
        "triangle_group": f"Delta^+({p},{q},{r})",
        "triangle_signature": [p, q, r],
        "triangle_defect": triangle_defect(sig),
        "finite_quotient": G.name,
        "finite_quotient_order": G.order,
        "quotient_order": G.order,
        "ginn_ready": atlas_training_ready,
        "explorer_loadable": False,
        "v1_0_triangle_tokenized_generators": True,
        "generator_count": len(gen_json),
        "generator_selection_mode": mode,
        "generator_selection_limit": int(args.kernel_generator_limit),
        "generator_truncated": not generator_export_complete,
        "generator_export_complete": generator_export_complete,
        "kernel_generator_scan_complete": kernel_generator_scan_complete,
        "raw_schreier_slots": G.order * len(token_order),
        "raw_nonidentity_schreier_generators": raw_nonidentity,
        "identity_like_generators_filtered": identity_like,
        "tile_scaffold_complete": tile_scaffold_complete,
        "tiles_truncated_by_cli_max_tiles": not tile_scaffold_complete,
        "tile_count": len(tiles),
        "expected_tile_count": G.order,
        "exclusion_reason": exclusion_reason,
        "generators": gen_json,
        "generator_meanings": meanings,
        "kernel_generator_audit_sample": audit_sample,
        "fundamental_domain_tiles": tiles,
        "tile_scaffold_warning": "Tile union is built from triangle-orbifold coset representatives. Full scaffold means all quotient cosets are present. It is a computational sampling scaffold, not a polished side-paired compact polygon.",
        "finite_group_triple": {
            "triple_index": 0,
            "quotient": G.name,
            "quotient_order": G.order,
            "signature": [p, q, r],
            "x": G.to_json_element(G.x),
            "y": G.to_json_element(G.y),
            "xy": G.to_json_element(G.mul(G.x, G.y)),
            **cert,
            "genus": genus,
        },
        "psu11_triangle_audit": delta_audit,
        "schreier_audit": {
            "transversal_size": len(rep_elems),
            "expected_transversal_size": G.order,
            "raw_schreier_slots": G.order * len(token_order),
            "raw_nonidentity_schreier_generators": raw_nonidentity,
            "selected_kernel_generators": len(gen_json),
            "identity_like_generators_filtered": identity_like,
            "kernel_generator_export_complete": generator_export_complete,
            "kernel_generator_scan_complete": kernel_generator_scan_complete,
            "generator_selection_mode": mode,
            "generator_selection_limit": int(args.kernel_generator_limit),
            "tile_scaffold_complete": tile_scaffold_complete,
            "tile_count": len(tiles),
            "expected_tile_count": G.order,
        },
        "certification": {
            "status": "complete_ginn_ready_triangle_kernel_surface" if pass_geometry_audit else "selected_generator_triangle_kernel_surface",
            "finite_quotient_certificate": f"exact ({p},{q},{r}) relation and BFS generation order check",
            "psu11_triangle_certificate": f"numerical SU(1,1) Delta^+({p},{q},{r}) relation checks",
            "kernel_certificate": "selected Reidemeister-Schreier generators k=t*s*rep(ts)^-1 map to identity in the finite quotient",
            "torsion_free_reason": "For a triangle-group quotient with exact generator/product orders p,q,r, the kernel intersects elliptic vertex stabilizers trivially; hence the kernel is torsion-free.",
            "remaining_caveat": "Unless --kernel-generator-mode all is used, the exported generator set is a selected computational pool, not a complete kernel generator set or minimal side-paired polygon presentation.",
        },
        "maker_run_id": run_id,
    }
    audit = {
        "surface_id": sid,
        "signature": f"({p},{q},{r})",
        "quotient": G.name,
        "genus": genus,
        "quotient_order": G.order,
        "transversal_size": len(rep_elems),
        "raw_schreier_slots": G.order * len(token_order),
        "raw_nonidentity_schreier_generators": raw_nonidentity,
        "kernel_generators_exported": len(gen_json),
        "kernel_generator_export_complete": generator_export_complete,
        "kernel_generator_scan_complete": kernel_generator_scan_complete,
        "generator_truncated": not generator_export_complete,
        "identity_like_filtered": identity_like,
        "tile_count": len(tiles),
        "expected_tile_count": G.order,
        "tile_scaffold_complete": tile_scaffold_complete,
        "tiles_truncated": not tile_scaffold_complete,
        "psu11_relation_max_error": relation_max_error,
        "pass_geometry_audit": pass_geometry_audit,
        "atlas_training_ready": atlas_training_ready,
        "mainline_dataset_eligible": pass_geometry_audit,
        "exclusion_reason": exclusion_reason,
        "ginn_ready": atlas_training_ready,
        "build_seconds": time.perf_counter() - t0,
    }
    return surface, audit

# -----------------------------------------------------------------------------
# Report and CLI
# -----------------------------------------------------------------------------

def write_triangle_report(run_root: Path, args: argparse.Namespace, surface_rows: List[Dict[str, Any]], atlas_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Fuchsian Triangle Quotient Trainer v1.2 Report")
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This run builds finite quotient kernel surfaces for hyperbolic triangle groups Delta^+(p,q,r), generates exact finite-word top-k atlases, and trains candidate-pool GINN rerankers.")
    lines.append("")
    lines.append("## Practical caveat")
    lines.append("")
    lines.append("When `--kernel-generator-mode all` is used, the exported Schreier generator set is complete. When `shortest` or `first` is used, the atlas is exact for the selected generator pool and selected depth, not for the complete Reidemeister-Schreier generator set or full infinite kernel.")
    lines.append("")
    lines.append("## Word-ball convention")
    lines.append("")
    lines.append("Version 1.0 explicitly installs a local reduced word-ball builder using length `<= depth`. For `m` oriented letters and `depth = 2`, raw size is `1 + m + m(m-1)`, before geometric/projective deduplication.")
    lines.append("")
    lines.append("## Run parameters")
    lines.append("")
    for k in ["signature", "quotient", "quotient_json", "kernel_generator_mode", "kernel_generator_limit", "depth", "pairs", "top_k_max", "train_pool_size", "epochs", "engine", "candidate_chunk_size", "pair_batch_size", "target_vram_mb", "stream_huge_word_ball"]:
        lines.append(f"- `{k}`: `{getattr(args, k, '')}`")
    lines.append("")
    lines.append("## Surface summary")
    lines.append("")
    if surface_rows:
        cols = ["surface_id", "signature", "quotient", "quotient_order", "genus", "kernel_generators_exported", "raw_nonidentity_schreier_generators", "tile_count", "tile_scaffold_complete", "atlas_training_ready"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for rr in surface_rows:
            lines.append("| " + " | ".join(str(round(rr.get(c), 5)) if isinstance(rr.get(c), float) else str(rr.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Atlas summary")
    lines.append("")
    if atlas_rows:
        cols = ["surface_id", "word_ball_size_raw", "word_ball_size_unique", "n_pairs", "engine", "wall_seconds", "evals_per_second", "shortcut_fraction", "median_gap12"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for rr in atlas_rows:
            lines.append("| " + " | ".join(str(round(rr.get(c), 5)) if isinstance(rr.get(c), float) else str(rr.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Training summary")
    lines.append("")
    if train_rows:
        cols = ["surface_id", "unique_word_ball_size", "pool_size", "train_pairs", "val_pairs", "test_pairs", "device", "epochs_ran", "test_recall_at_1", "test_recall_at_5", "test_recall_at_20", "test_top5_pruned_rmse", "train_seconds"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for rr in train_rows:
            lines.append("| " + " | ".join(str(round(rr.get(c), 5)) if isinstance(rr.get(c), float) else str(rr.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Recall@1 is the correct-lift win rate relative to the exact atlas for this run's finite word ball. If the generator mode is selected/truncated, this is a selected-atlas result, not a complete-generator word-ball result.")
    (run_root / "report").mkdir(parents=True, exist_ok=True)
    (run_root / "report" / "triangle_quotient_training_report.md").write_text("\n".join(lines), encoding="utf-8")
    (run_root / "report" / "big_hurwitz_training_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build triangle-group quotient kernel atlases and train candidate-pool GINN rerankers.")
    ap.add_argument("--signature", type=parse_signature, default=(2, 3, 8), help="Triangle signature, e.g. 2,3,8 or 2,4,5")
    ap.add_argument("--quotient", type=str, default="auto", help="Finite quotient: auto, GL2_3, S5, or descriptor name")
    ap.add_argument("--quotient-json", type=str, default="", help="Path to one candidate JSON descriptor from TriangleQuotientSearcher")
    ap.add_argument("--mode", choices=["smoke", "train"], default="train")
    ap.add_argument("--pairs", type=int, default=9000)
    ap.add_argument("--smoke-pairs", type=int, default=60)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--top-k-max", type=int, default=100)
    ap.add_argument("--top-k-list", type=parse_int_list, default=[1, 3, 5, 10, 20, 50, 100])
    ap.add_argument("--csv-top-k", type=int, default=20)
    ap.add_argument("--candidate-chunk-size", type=int, default=8192)
    ap.add_argument("--pair-batch-size", type=int, default=0)
    ap.add_argument("--engine", choices=["auto", "gpu_torch", "cpu_vec", "cpu_loop"], default="auto")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--target-ram-mb", type=float, default=8192.0)
    ap.add_argument("--target-vram-mb", type=float, default=8192.0)
    ap.add_argument("--max-word-ball", type=int, default=1000000)
    ap.add_argument("--max-unique-word-ball", type=int, default=1000000)
    ap.add_argument("--allow-huge-word-ball", action="store_true")
    ap.add_argument("--stream-huge-word-ball", action="store_true")
    ap.add_argument("--virtual-topk-buffer", type=int, default=5000)
    ap.add_argument("--virtual-topk-dedupe-tol", type=float, default=0.0)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--dedupe-tol", type=float, default=1.0e-10)
    ap.add_argument("--alias-sample-limit", type=int, default=5)
    ap.add_argument("--alias-summary-rows", type=int, default=100)
    ap.add_argument("--frequency-rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outroot", type=str, default="triangle_quotient_training_runs")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-atlas", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--write-word-ball-summary", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-perf-log", action="store_true")

    ap.add_argument("--kernel-generator-mode", choices=["all", "first", "shortest"], default="shortest")
    ap.add_argument("--kernel-generator-limit", type=int, default=256)
    ap.add_argument("--kernel-progress-every", type=int, default=10000)
    ap.add_argument("--kernel-audit-sample-rows", type=int, default=50)
    ap.add_argument("--identity-tol", type=float, default=1.0e-9)
    ap.add_argument("--max-tiles", type=int, default=0)

    # Reranker training args expected by FuchsianBigHurwitzTrainer_v1_7.py
    ap.add_argument("--train-pool-size", type=int, default=256)
    ap.add_argument("--pool-sizes", type=str, default="128,256")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min-delta", type=float, default=1.0e-5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--context-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--soft-distance-weight", type=float, default=0.10)
    ap.add_argument("--soft-distance-tau", type=float, default=0.50)
    ap.add_argument("--train-device", type=str, default="auto")
    ap.add_argument("--no-train-gpu", action="store_true")
    ap.add_argument("--cache-tensors-gpu", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--feature-batch-size", type=int, default=2048)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--no-shuffle-pool", action="store_true")
    ap.add_argument("--print-every", type=int, default=5)

    ap.add_argument("--zoo-script", type=str, default=default_zoo_script())
    ap.add_argument("--big-trainer-script", type=str, default=default_big_trainer_script())
    ap.add_argument("--ginn-script", type=str, default=default_ginn_script())
    # Compatibility with zoo functions: q is only used as a seed component/result field.
    ap.add_argument("--q", type=int, default=0, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    sig = tuple(args.signature)

    if args.mode == "smoke":
        args.pairs = int(args.smoke_pairs)
        args.top_k_max = min(int(args.top_k_max), 20)
        args.csv_top_k = min(int(args.csv_top_k), int(args.top_k_max))
        args.epochs = min(int(args.epochs), 4)
        args.min_epochs = min(int(args.min_epochs), 1)
        args.patience = min(int(args.patience), 3)
        args.train_pool_size = min(int(args.train_pool_size), 64)
        if args.kernel_generator_mode == "shortest":
            args.kernel_generator_limit = min(int(args.kernel_generator_limit), 64)
        if not args.label:
            args.label = "smoke_train"

    pool_sizes = sorted(set([x for x in parse_int_list(args.pool_sizes) if x > 0] + [int(args.train_pool_size)]))
    args.pool_sizes = ",".join(str(x) for x in pool_sizes)
    if not args.label:
        args.label = f"triangle_{signature_slug(sig)}_{args.quotient}_kg{args.kernel_generator_mode}{args.kernel_generator_limit}_depth{args.depth}_pairs{args.pairs}"

    G = build_quotient(args)
    cert = G.validate_triangle()
    genus = triangle_genus(G.order, sig)
    print(f"[triangle-preflight] signature={sig} quotient={G.name} order={G.order} genus={genus} cert={cert}", flush=True)

    # Stable pseudo-q so zoo sample seed differs by signature/quotient but remains integer.
    if int(args.q) == 0:
        args.q = int(100000 * sig[0] + 1000 * sig[1] + 10 * sig[2] + (G.order % 10))

    stamp = now_stamp()
    run_name = f"run_{stamp}_{stable_slug(args.label)}"
    run_root = Path(args.outroot) / run_name
    for sub in ["group", "surfaces", "kernel_audits", "atlas", "training", "tables", "report"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    zoo = load_module(args.zoo_script, "big_hurwitz_zoo_v18")
    bt = load_module(args.big_trainer_script, "big_hurwitz_trainer_v17")
    perf = zoo.PerfTracker(run_root / "tables" / "performance_log.csv", enabled=(not args.no_perf_log))
    t_all = time.perf_counter()
    print(f"{PROGRAM} v{VERSION}")
    print(f"run_root={run_root}")
    print(f"signature={sig} quotient={G.name} depth={args.depth} pairs={args.pairs} kernel_mode={args.kernel_generator_mode} kernel_limit={args.kernel_generator_limit} train_pool={args.train_pool_size}")
    print("-" * 78)

    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "args": vars(args).copy(),
        "python": sys.version,
        "platform": platform.platform(),
        "purpose": "Triangle-group quotient selected/complete Schreier-kernel atlas generation plus candidate-pool GINN reranker training.",
    }
    write_json(run_root / "manifest.json", manifest)
    write_json(run_root / "group" / f"{stable_slug(G.name)}_certificate.json", {"quotient": G.name, "order": G.order, "signature": list(sig), "certificate": cert, "metadata": G.metadata})

    try:
        bt.require_torch()
        perf.log("module_load_start", ginn_script=args.ginn_script, zoo_script=args.zoo_script, big_trainer_script=args.big_trainer_script)
        ginn = zoo.load_module(args.ginn_script, "ginn_v24")
        install_leq_depth_word_ball_builder(ginn)
        perf.log("module_load_done")

        run_id = run_root.name.replace("run_", "")
        print(f"[surface] building selected kernel for {G.name}", flush=True)
        perf.log("surface_build_start", quotient=G.name, signature=str(sig))
        surface, audit = build_triangle_kernel_surface(G, args, run_id)
        surface["triangle_quotient_trainer_record"] = True
        surface["triangle_quotient_program"] = PROGRAM
        surface["triangle_quotient_version"] = VERSION
        sid = str(surface.get("surface_id"))
        write_json(run_root / "surfaces" / f"{sid}.json", surface)
        write_json(run_root / "kernel_audits" / f"{sid}_audit.json", audit)
        perf.log("surface_build_done", surface_id=sid, genus=surface.get("genus"), generator_count=surface.get("generator_count"), tile_count=surface.get("tile_count"), raw_nonidentity=surface.get("raw_nonidentity_schreier_generators"))

        surface_rows = [{
            "surface_id": sid,
            "signature": f"({sig[0]},{sig[1]},{sig[2]})",
            "quotient": G.name,
            "genus": surface.get("genus"),
            "quotient_order": G.order,
            "kernel_generators_exported": audit.get("kernel_generators_exported"),
            "raw_nonidentity_schreier_generators": audit.get("raw_nonidentity_schreier_generators"),
            "generator_count": surface.get("generator_count"),
            "tile_count": surface.get("tile_count"),
            "tile_scaffold_complete": audit.get("tile_scaffold_complete"),
            "mainline_dataset_eligible": surface.get("mainline_dataset_eligible"),
            "ginn_ready": surface.get("ginn_ready"),
            "atlas_training_ready": surface.get("atlas_training_ready"),
            "kernel_generator_export_complete": audit.get("kernel_generator_export_complete"),
            "kernel_generator_scan_complete": audit.get("kernel_generator_scan_complete"),
        }]
        write_csv(run_root / "tables" / "triangle_surface_summary.csv", surface_rows)
        write_csv(run_root / "tables" / "big_hurwitz_surface_summary.csv", surface_rows)

        atlas_results: List[Any] = []
        failure_rows: List[Dict[str, Any]] = []
        if not args.skip_atlas:
            old_mode = args.mode
            args.mode = "atlas" if old_mode == "train" else "smoke"
            try:
                atlas_results.append(zoo.atlas_for_surface(args, ginn, surface, run_root, 0, perf=perf))
            except Exception as e:
                print(f"[atlas fail] {sid}: {type(e).__name__}: {e}", flush=True)
                failure_rows.append({"stage": "atlas", "surface_id": sid, "error_type": type(e).__name__, "error": str(e)})
            args.mode = old_mode
        atlas_rows = [r.__dict__ for r in atlas_results]
        write_csv(run_root / "tables" / "triangle_atlas_summary.csv", atlas_rows)
        write_csv(run_root / "tables" / "big_hurwitz_atlas_summary.csv", atlas_rows)

        train_rows: List[Dict[str, Any]] = []
        if not args.no_train and not any(fr.get("stage") == "atlas" for fr in failure_rows):
            try:
                tr = bt.train_surface_reranker(args, zoo, ginn, surface, run_root, perf=perf)
                train_rows = [tr.__dict__]
            except Exception as e:
                print(f"[train fail] {sid}: {type(e).__name__}: {e}", flush=True)
                failure_rows.append({"stage": "train", "surface_id": sid, "error_type": type(e).__name__, "error": str(e)})
        write_csv(run_root / "tables" / "triangle_training_summary.csv", train_rows)
        write_csv(run_root / "tables" / "big_hurwitz_training_summary.csv", train_rows)
        write_csv(run_root / "tables" / "failures.csv", failure_rows, ["stage", "surface_id", "error_type", "error"])
        write_triangle_report(run_root, args, surface_rows, atlas_rows, train_rows)
        summary = {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": time.perf_counter() - t_all,
            "surfaces_built": 1,
            "atlases_completed": len(atlas_results),
            "trained_surfaces": len(train_rows),
            "failures": len(failure_rows),
            "run_root": str(run_root),
            "process_peak_rss_mb": perf.peak_rss_mb,
        }
        write_json(run_root / "run_summary.json", summary)
        perf.log("run_done", **summary)
        perf.write()
        print("=" * 78)
        print(f"[done] surfaces=1 atlases={len(atlas_results)} trained={len(train_rows)} failures={len(failure_rows)}")
        print(f"[done] run_root={run_root}")
        return 0 if not failure_rows else 1
    except Exception as e:
        err = {"error_type": type(e).__name__, "error": str(e), "wall_seconds": time.perf_counter() - t_all}
        write_json(run_root / "run_error.json", err)
        try:
            perf.log("run_fatal", error_type=type(e).__name__, error=str(e))
            perf.write()
        except Exception:
            pass
        print(f"[fatal] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
