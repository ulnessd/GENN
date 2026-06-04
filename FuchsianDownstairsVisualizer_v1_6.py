#!/usr/bin/env python3
"""
FuchsianDownstairsVisualizer_v1_6.py

Interactive two-view visualizer for explicit Fuchsian quotient examples.

Left: exact upstairs Poincare-disk candidate-lift geometry.
Right: rotatable ordinary R^3 topological / illustrative embedding sketch.

Important caveat
----------------
The Poincare disk panel is the mathematically faithful view for lift candidates:
it computes d_D(p, gamma q) for a finite word ball and displays exact top-k
candidate lifts.  The 3D panel is intentionally a visual/topological companion,
not a globally isometric embedding of the hyperbolic metric.  In v1.6,
3D paths are drawn as sampled topological traces of the exact upstairs
geodesic representatives rather than as ambient straight lines.

This PyQt6 version supports built-in elementary animals and explicit top-k path-bundle visualization:
  - cyclic hyperbolic quotient
  - cyclic parabolic quotient
  - Gamma(2), thrice-punctured sphere
  - modular commutator subgroup, once-punctured torus
  - regular compact genus-2 octagon surface
  - Schottky rank-2 sketch

It can also load a GENN surface JSON with SU(1,1) generators and optional polygon
vertices.  Loaded surfaces use exact disk generators in the left panel and a
generic 3D sketch unless a built-in embedding type is selected.

New in v1.6:
  - The 3D companion now enforces the quotient endpoint rule: every displayed candidate trace
    starts at the displayed [p] and ends at the displayed [q].
  - The 3D companion now enforces the quotient endpoint rule: every displayed candidate trace
    starts at the displayed [p] and ends at the displayed [q].
  - Cyclic cylinder/cusp and punctured-torus companions now use explicit surface-parameter
    traces for every selected candidate, including the emphasized winner.  This prevents
    the winner from being drawn as an ambient Euclidean chord through the surface.
  - The disk view remains the authoritative finite-word hyperbolic geometry; the 3D view is a
    quotient-aware visual companion, not a metric embedding.

Dependencies: numpy, matplotlib, PyQt6.  Headless demo/self-test do not require PyQt6.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Disk Mobius utilities
# -----------------------------------------------------------------------------

EPS = 1.0e-12


def cpair(z: complex) -> List[float]:
    return [float(z.real), float(z.imag)]


def cunpair(x) -> complex:
    if isinstance(x, complex):
        return x
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return complex(float(x[0]), float(x[1]))
    raise ValueError(f"Cannot decode complex pair: {x!r}")


@dataclass(frozen=True)
class MobiusDisk:
    """General disk-coordinate Mobius transformation z -> (a z + b)/(c z + d)."""

    a: complex
    b: complex
    c: complex
    d: complex
    name: str = ""

    def __call__(self, z: complex) -> complex:
        den = self.c * z + self.d
        if abs(den) < EPS:
            den = EPS + 0j
        return (self.a * z + self.b) / den

    def compose(self, other: "MobiusDisk", name: str = "") -> "MobiusDisk":
        """Return self o other."""
        A = np.array([[self.a, self.b], [self.c, self.d]], dtype=complex)
        B = np.array([[other.a, other.b], [other.c, other.d]], dtype=complex)
        C = A @ B
        return MobiusDisk(C[0, 0], C[0, 1], C[1, 0], C[1, 1], name or (self.name + other.name)).normalized()

    def inverse(self, name: str = "") -> "MobiusDisk":
        return MobiusDisk(self.d, -self.b, -self.c, self.a, name or inv_name(self.name)).normalized()

    def normalized(self) -> "MobiusDisk":
        # Normalize by determinant magnitude when possible.  Mobius maps are projective;
        # this improves numerical conditioning without changing the transformation.
        det = self.a * self.d - self.b * self.c
        if abs(det) < EPS:
            return self
        s = complex(math.sqrt(abs(det)), 0.0)
        return MobiusDisk(self.a / s, self.b / s, self.c / s, self.d / s, self.name)

    @staticmethod
    def identity() -> "MobiusDisk":
        return MobiusDisk(1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, "I")

    @staticmethod
    def from_su11(alpha: complex, beta: complex, name: str = "") -> "MobiusDisk":
        return MobiusDisk(alpha, beta, beta.conjugate(), alpha.conjugate(), name).normalized()

    def as_tuple_rounded(self, ndigits: int = 10) -> Tuple[float, ...]:
        return tuple(round(v, ndigits) for z in (self.a, self.b, self.c, self.d) for v in (z.real, z.imag))


def inv_name(name: str) -> str:
    if name.endswith("^-1"):
        return name[:-3]
    if name == "I":
        return "I"
    return name + "^-1"


def su11_real_translation(strength: float, name: str = "A") -> MobiusDisk:
    if abs(strength) >= 1:
        raise ValueError("strength must be |a| < 1")
    scale = 1.0 / math.sqrt(1.0 - strength * strength)
    return MobiusDisk.from_su11(scale + 0j, strength * scale + 0j, name=name)


def disk_rotation(theta: float, name: str = "R") -> MobiusDisk:
    # z -> exp(i theta) z in ordinary Mobius matrix form.
    return MobiusDisk(complex(math.cos(theta), math.sin(theta)), 0j, 0j, 1 + 0j, name).normalized()


def disk_move_to_zero(z0: complex) -> MobiusDisk:
    r2 = abs(z0) ** 2
    if r2 >= 1.0:
        raise ValueError("Point must lie in disk")
    scale = 1.0 / math.sqrt(1.0 - r2)
    return MobiusDisk.from_su11(scale + 0j, -z0 * scale, name="M0")


def disk_move_from_zero(w0: complex) -> MobiusDisk:
    r2 = abs(w0) ** 2
    if r2 >= 1.0:
        raise ValueError("Point must lie in disk")
    scale = 1.0 / math.sqrt(1.0 - r2)
    return MobiusDisk.from_su11(scale + 0j, w0 * scale, name="M1")


def disk_isometry_from_two_point_pairs(z1: complex, z2: complex, w1: complex, w2: complex, name: str = "G") -> MobiusDisk:
    """Unique orientation-preserving disk isometry sending z1->w1, z2->w2."""
    Mz = disk_move_to_zero(z1)
    u = Mz(z2)
    Mw = disk_move_to_zero(w1)
    v = Mw(w2)
    if abs(u) < 1.0e-14 or abs(v) < 1.0e-14:
        raise ValueError("Degenerate endpoint data")
    lam = v / u
    lam = lam / abs(lam)
    R = disk_rotation(math.atan2(lam.imag, lam.real))
    return disk_move_from_zero(w1).compose(R.compose(Mz), name=name).normalized()


def regular_hyperbolic_polygon_radius(p: int, q: int) -> float:
    cosh_R = math.cos(math.pi / q) / math.sin(math.pi / p)
    if cosh_R <= 1.0:
        raise ValueError("Not hyperbolic")
    return math.tanh(0.5 * math.acosh(cosh_R))


def psl_to_disk_matrix(M: np.ndarray) -> MobiusDisk:
    """Conjugate an upper-half-plane PSL(2,R) map to disk coordinates.

    Cayley C: disk -> H, w = i(1+z)/(1-z).
    Disk map = C^{-1} M C.
    """
    C = np.array([[1j, 1j], [-1.0, 1.0]], dtype=complex)
    Cinv = np.array([[1.0, -1j], [1.0, 1j]], dtype=complex)  # z=(w-i)/(w+i)
    D = Cinv @ M.astype(complex) @ C
    return MobiusDisk(D[0, 0], D[0, 1], D[1, 0], D[1, 1]).normalized()


def disk_distance(z: complex, w: complex) -> float:
    rz = min(abs(z), 1.0 - 1e-12)
    rw = min(abs(w), 1.0 - 1e-12)
    if rz != abs(z):
        z = z / abs(z) * rz
    if rw != abs(w):
        w = w / abs(w) * rw
    arg = 1.0 + 2.0 * abs(z - w) ** 2 / max((1.0 - abs(z) ** 2) * (1.0 - abs(w) ** 2), EPS)
    return math.acosh(max(1.0, arg))


def safe_inside(z: complex, radius: float = 0.985) -> complex:
    r = abs(z)
    if r >= radius:
        return z / r * radius
    return z


# -----------------------------------------------------------------------------
# Geodesics in the disk
# -----------------------------------------------------------------------------


def disk_geodesic_points(z1: complex, z2: complex, n: int = 128) -> np.ndarray:
    """Sample the Poincare geodesic segment between two disk points."""
    z1, z2 = safe_inside(z1), safe_inside(z2)
    cross = abs((z1.conjugate() * z2).imag)
    if abs(z1) < 1e-8 or abs(z2) < 1e-8 or cross < 1e-7:
        return np.linspace([z1.real, z1.imag], [z2.real, z2.imag], n)

    # Solve for center C=x+iy of circle orthogonal to unit circle:
    # 2 Re(z_i conj(C)) = |z_i|^2 + 1.
    A = np.array([[z1.real, z1.imag], [z2.real, z2.imag]], dtype=float)
    b = np.array([(abs(z1) ** 2 + 1.0) / 2.0, (abs(z2) ** 2 + 1.0) / 2.0], dtype=float)
    try:
        x, y = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linspace([z1.real, z1.imag], [z2.real, z2.imag], n)
    C = complex(x, y)
    R = math.sqrt(max(abs(C) ** 2 - 1.0, 0.0))
    if R <= 1e-8 or not np.isfinite(R):
        return np.linspace([z1.real, z1.imag], [z2.real, z2.imag], n)
    a1 = math.atan2((z1 - C).imag, (z1 - C).real)
    a2 = math.atan2((z2 - C).imag, (z2 - C).real)
    # Choose shorter arc inside disk.
    delta = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    angles = a1 + np.linspace(0, delta, n)
    pts = C + R * np.exp(1j * angles)
    # If numerical arc wandered outside too much, use complementary arc.
    if np.nanmax(np.abs(pts)) > 1.01:
        delta2 = delta - math.copysign(2 * math.pi, delta)
        angles = a1 + np.linspace(0, delta2, n)
        pts = C + R * np.exp(1j * angles)
    return np.column_stack([pts.real, pts.imag])


# -----------------------------------------------------------------------------
# Surface examples and word balls
# -----------------------------------------------------------------------------


@dataclass
class SurfaceExample:
    key: str
    name: str
    regime: str
    description: str
    generators: Dict[str, MobiusDisk]
    embedding: str
    polygon_vertices: Optional[np.ndarray] = None
    notes: str = ""


def add_inverses(gens: Dict[str, MobiusDisk]) -> Dict[str, MobiusDisk]:
    out = dict(gens)
    for name, g in list(gens.items()):
        out[inv_name(name)] = g.inverse(inv_name(name))
    return out


def word_ball(gens_pos: Dict[str, MobiusDisk], depth: int = 2) -> List[Tuple[str, MobiusDisk]]:
    gens = add_inverses(gens_pos)
    items = [("I", MobiusDisk.identity())]
    seen = {MobiusDisk.identity().as_tuple_rounded()}
    frontier = [("", MobiusDisk.identity(), "")]
    for _ in range(depth):
        new_frontier = []
        for word, M, last in frontier:
            for name, g in gens.items():
                if last and name == inv_name(last):
                    continue
                new_word = name if not word else word + " " + name
                new_M = g.compose(M, name=new_word).normalized()
                key = new_M.as_tuple_rounded(9)
                if key not in seen:
                    seen.add(key)
                    items.append((new_word, new_M))
                    new_frontier.append((new_word, new_M, name))
        frontier = new_frontier
    return items


def make_cyclic_hyperbolic() -> SurfaceExample:
    A = su11_real_translation(0.43, "A")
    return SurfaceExample(
        key="cyclic_hyperbolic",
        name="Cyclic hyperbolic quotient",
        regime="noncompact infinite-area / hyperbolic cylinder",
        description="Deck group <A> = Z generated by one hyperbolic disk translation.",
        generators={"A": A},
        embedding="cylinder",
        notes="3D view is a cylinder/trumpet sketch; disk view is exact for this generator.",
    )


def make_cyclic_parabolic() -> SurfaceExample:
    T = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    A = psl_to_disk_matrix(T)
    A = MobiusDisk(A.a, A.b, A.c, A.d, "A")
    return SurfaceExample(
        key="cyclic_parabolic",
        name="Cyclic parabolic quotient",
        regime="noncompact cusp / parabolic cylinder",
        description="Deck group <T> = Z generated by one parabolic, conjugated from H to disk.",
        generators={"A": A},
        embedding="cusp",
        notes="Qualitative 3D cusp sketch; disk view is exact for the parabolic generator.",
    )


def make_gamma2() -> SurfaceExample:
    # Gamma(2) generated by T^2 and ST^{-2}S; convenient matrices [[1,2],[0,1]], [[1,0],[2,1]].
    A_h = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    B_h = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=float)
    A = psl_to_disk_matrix(A_h); B = psl_to_disk_matrix(B_h)
    return SurfaceExample(
        key="gamma2_trinion",
        name="Gamma(2): thrice-punctured sphere",
        regime="finite-area cusped / pair of pants",
        description="Gamma(2), a free rank-2 torsion-free modular subgroup. Quotient is a sphere with three cusps.",
        generators={"A": MobiusDisk(A.a, A.b, A.c, A.d, "A"), "B": MobiusDisk(B.a, B.b, B.c, B.d, "B")},
        embedding="pants",
        notes="The 3D pair-of-pants is topological; disk candidate lifts are exact for the chosen generators.",
    )


def make_once_punctured_torus() -> SurfaceExample:
    A_h = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=float)
    B_h = np.array([[0.0, -1.0], [1.0, 3.0]], dtype=float)
    A = psl_to_disk_matrix(A_h); B = psl_to_disk_matrix(B_h)
    return SurfaceExample(
        key="once_punctured_torus",
        name="Modular commutator: once-punctured torus",
        regime="finite-area cusped / punctured torus",
        description="Two-generator presentation of the modular commutator subgroup; commutator is parabolic.",
        generators={"A": MobiusDisk(A.a, A.b, A.c, A.d, "A"), "B": MobiusDisk(B.a, B.b, B.c, B.d, "B")},
        embedding="punctured_torus",
        notes="3D view is a punctured torus sketch; disk view uses exact matrix generators.",
    )


def make_regular_genus2() -> SurfaceExample:
    genus = 2
    sides = 4 * genus
    pairs = 2 * genus
    q = 4 * genus
    rho = regular_hyperbolic_polygon_radius(sides, q)
    rotation = math.radians(22.5)
    angles = np.array([rotation + 2.0 * math.pi * k / sides for k in range(sides)])
    vertices_c = [rho * complex(math.cos(a), math.sin(a)) for a in angles]
    gens: Dict[str, MobiusDisk] = {}
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i in range(pairs):
        j = i + pairs
        z1 = vertices_c[i]
        z2 = vertices_c[(i + 1) % sides]
        w1 = vertices_c[(j + 1) % sides]
        w2 = vertices_c[j % sides]
        gens[letters[i]] = disk_isometry_from_two_point_pairs(z1, z2, w1, w2, name=letters[i])
    vertices = np.array([[z.real, z.imag] for z in vertices_c])
    return SurfaceExample(
        key="regular_genus2",
        name="Regular genus-2 compact surface",
        regime="compact closed / double torus",
        description="Regular hyperbolic octagon with opposite-side pairings.",
        generators=gens,
        embedding="genus2",
        polygon_vertices=vertices,
        notes="The 3D double-torus is topological; disk octagon and side-pairing generators are exact.",
    )


def make_schottky_rank2() -> SurfaceExample:
    A = su11_real_translation(0.55, "A")
    R = disk_rotation(math.radians(72), "R")
    B = R.compose(su11_real_translation(0.45, "B0")).compose(R.inverse(), name="B")
    return SurfaceExample(
        key="schottky_rank2",
        name="Schottky/free Fuchsian rank 2 sketch",
        regime="noncompact infinite-area / free Fuchsian",
        description="Simple rank-2 free Fuchsian-style example with two hyperbolic generators.",
        generators={"A": A, "B": B},
        embedding="pants",
        notes="Illustrative Schottky-style rank-2 example; not a finite-area canonical sample.",
    )


def built_in_examples() -> List[SurfaceExample]:
    return [
        make_cyclic_hyperbolic(),
        make_cyclic_parabolic(),
        make_gamma2(),
        make_once_punctured_torus(),
        make_regular_genus2(),
        make_schottky_rank2(),
    ]


def load_surface_json(path: str) -> SurfaceExample:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gens: Dict[str, MobiusDisk] = {}
    for name, gj in data.get("generators", {}).items():
        if gj.get("type") == "su11":
            alpha = cunpair(gj["alpha"])
            beta = cunpair(gj["beta"])
            gens[name] = MobiusDisk.from_su11(alpha, beta, name=name)
        elif all(k in gj for k in ["a", "b", "c", "d"]):
            gens[name] = MobiusDisk(cunpair(gj["a"]), cunpair(gj["b"]), cunpair(gj["c"]), cunpair(gj["d"]), name=name)
    if not gens:
        raise ValueError("No recognizable SU(1,1)/Mobius generators found in JSON")
    vertices = None
    if "polygon_vertices" in data:
        vertices = np.array(data["polygon_vertices"], dtype=float)
    name = data.get("surface_id") or data.get("name") or Path(path).stem
    compact = data.get("compact")
    finite_area = data.get("finite_area")
    regime_bits = []
    if compact is not None:
        regime_bits.append("compact" if compact else "noncompact")
    if finite_area is not None:
        regime_bits.append("finite-area" if finite_area else "infinite-area")
    return SurfaceExample(
        key="loaded_" + Path(path).stem,
        name="Loaded: " + str(name),
        regime=" / ".join(regime_bits) or str(data.get("domain_type", "loaded surface")),
        description="Loaded GENN surface JSON. Disk generators are used exactly when recognizable.",
        generators=gens,
        embedding="generic",
        polygon_vertices=vertices,
        notes=f"Loaded from {path}",
    )


# -----------------------------------------------------------------------------
# Candidate computation
# -----------------------------------------------------------------------------


@dataclass
class CandidateResult:
    word: str
    point: complex
    distance: float
    rank: int


def compute_candidates(surface: SurfaceExample, p: complex, q: complex, depth: int) -> List[CandidateResult]:
    wb = word_ball(surface.generators, depth=depth)
    rows = []
    for word, M in wb:
        z = safe_inside(M(q))
        rows.append((word, z, disk_distance(p, z)))
    rows.sort(key=lambda x: x[2])
    return [CandidateResult(word=w, point=z, distance=d, rank=i + 1) for i, (w, z, d) in enumerate(rows)]


# -----------------------------------------------------------------------------
# 3D visual sketches
# -----------------------------------------------------------------------------


def plot_cylinder(ax):
    theta = np.linspace(0, 2 * np.pi, 80)
    z = np.linspace(-2.2, 2.2, 40)
    T, Z = np.meshgrid(theta, z)
    R = 0.85
    X = R * np.cos(T); Y = R * np.sin(T)
    ax.plot_surface(X, Y, Z, color="#d0d0d0", alpha=0.55, linewidth=0, antialiased=True)


def plot_cusp(ax):
    theta = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, 4.5, 70)
    T, V = np.meshgrid(theta, v)
    R = 1.05 * np.exp(-0.42 * V) + 0.08
    X = R * np.cos(T); Y = R * np.sin(T); Z = V - 2.0
    ax.plot_surface(X, Y, Z, color="#d0d0d0", alpha=0.60, linewidth=0, antialiased=True)


def torus_mesh(R=1.3, r=0.34, center=(0, 0, 0), u0=0.0, u1=2 * np.pi, n=80, m=30):
    u = np.linspace(u0, u1, n)
    v = np.linspace(0, 2 * np.pi, m)
    U, V = np.meshgrid(u, v)
    X = center[0] + (R + r * np.cos(V)) * np.cos(U)
    Y = center[1] + (R + r * np.cos(V)) * np.sin(U)
    Z = center[2] + r * np.sin(V)
    return X, Y, Z


def plot_punctured_torus(ax):
    """Draw a smooth torus-core companion with a marked puncture/cusp direction.

    Earlier versions drew a torus with a large omitted sector and a separate tube,
    which made endpoint-anchored traces appear to leave the surface.  v1.6 keeps
    a smooth torus core so parametric traces can lie on the displayed surface,
    and marks the puncture/cusp direction with a small translucent funnel marker.
    This is topological/schematic, not a hyperbolic isometric embedding.
    """
    X, Y, Z = torus_mesh(R=1.25, r=0.35, u0=0.0, u1=2 * np.pi, n=96, m=36)
    ax.plot_surface(X, Y, Z, color="#d0d0d0", alpha=0.60, linewidth=0, antialiased=True)
    # Mark a puncture/cusp direction without cutting the visual torus core.
    plot_tube(ax, np.array([[1.58, 0.0, 0.0], [2.10, 0.0, 0.18], [2.55, 0.0, 0.36]]), radius=0.09, alpha=0.30)


def plot_genus2(ax):
    X1, Y1, Z1 = torus_mesh(R=0.75, r=0.24, center=(-0.9, 0, 0))
    X2, Y2, Z2 = torus_mesh(R=0.75, r=0.24, center=(0.9, 0, 0))
    ax.plot_surface(X1, Y1, Z1, color="#d0d0d0", alpha=0.62, linewidth=0, antialiased=True)
    ax.plot_surface(X2, Y2, Z2, color="#d0d0d0", alpha=0.62, linewidth=0, antialiased=True)
    plot_tube(ax, np.array([[-0.45, 0, 0], [0.0, 0, 0], [0.45, 0, 0]]), radius=0.22, alpha=0.55)


def plot_pants(ax):
    """Smooth qualitative pair-of-pants / thrice-punctured sphere companion.

    This is still not an isometric hyperbolic embedding, but it avoids the
    misleading central-ball-with-surgically-attached-tubes look by drawing three
    smoothly tapered cusp/funnel legs that meet through an implicit trinion-like
    body.
    """
    # Three smooth tapered legs meeting near the origin.
    directions = [
        np.array([1.65, 0.0, -0.70]),
        np.array([-0.95, 1.35, -0.70]),
        np.array([-0.95, -1.35, -0.70]),
    ]
    for d in directions:
        d = d / np.linalg.norm(d)
        pts = np.vstack([
            0.15 * d + np.array([0.0, 0.0, 0.25]),
            0.65 * d + np.array([0.0, 0.0, 0.05]),
            1.45 * d + np.array([0.0, 0.0, -0.15]),
            2.15 * d + np.array([0.0, 0.0, -0.35]),
        ])
        plot_tube(ax, pts, radius=0.28, alpha=0.46, flare=True)

    # A smooth central neck/body.  It visually blends the three ends without
    # implying a metric claim.
    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 36)
    U, V = np.meshgrid(u, v)
    Rxy = 0.62 * (1.0 + 0.10 * np.cos(3 * U) * np.sin(V) ** 2)
    X = Rxy * np.cos(U) * np.sin(V)
    Y = Rxy * np.sin(U) * np.sin(V)
    Z = 0.45 * np.cos(V) + 0.08 * np.cos(3 * U) * np.sin(V) ** 2
    ax.plot_surface(X, Y, Z, color="#d0d0d0", alpha=0.36, linewidth=0, antialiased=True)


def plot_generic(ax):
    # Generic saddle-ish sheet for loaded surfaces without embedding metadata.
    u = np.linspace(-1.8, 1.8, 60)
    v = np.linspace(-1.2, 1.2, 45)
    U, V = np.meshgrid(u, v)
    Z = 0.25 * np.sin(1.6 * U) * np.cos(2.2 * V)
    ax.plot_surface(U, V, Z, color="#d0d0d0", alpha=0.55, linewidth=0)


def plot_tube(ax, pts: np.ndarray, radius=0.1, alpha=0.6, flare: bool = False, color: str = "#d0d0d0") :
    # Sweep a tube along a polyline using local frames.
    pts = np.asarray(pts, dtype=float)
    samples = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        for t in np.linspace(0, 1, 18, endpoint=False):
            samples.append((1 - t) * a + t * b)
    samples.append(pts[-1])
    C = np.array(samples)
    tang = np.gradient(C, axis=0)
    theta = np.linspace(0, 2 * np.pi, 18)
    X = np.zeros((len(C), len(theta))); Y = X.copy(); Z = X.copy()
    for i, (c, tvec) in enumerate(zip(C, tang)):
        tnorm = np.linalg.norm(tvec)
        if tnorm < 1e-9:
            tvec = np.array([1.0, 0, 0])
        else:
            tvec = tvec / tnorm
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, tvec)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        n1 = np.cross(tvec, ref); n1 /= max(np.linalg.norm(n1), 1e-9)
        n2 = np.cross(tvec, n1); n2 /= max(np.linalg.norm(n2), 1e-9)
        rr = radius * (1.0 + (0.8 * i / max(1, len(C) - 1) if flare else 0.0))
        ring = c + rr * np.cos(theta)[:, None] * n1 + rr * np.sin(theta)[:, None] * n2
        X[i, :] = ring[:, 0]; Y[i, :] = ring[:, 1]; Z[i, :] = ring[:, 2]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


# -----------------------------------------------------------------------------
# Plotting core
# -----------------------------------------------------------------------------


def draw_disk(
    ax,
    surface: SurfaceExample,
    p: complex,
    q: complex,
    candidates: List[CandidateResult],
    path_k: int,
    preview_k: int = 20,
):
    """Draw the exact finite-word upstairs lift geometry in the Poincare disk.

    The selected path bundle consists of the exact top path_k candidates by
    hyperbolic distance among the finite word ball.  The winning downstairs
    geodesic [p]->[q] is represented upstairs by p -> gamma*q for the rank-1
    candidate.
    """
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color="black", lw=1.0)
    ax.fill(np.cos(th), np.sin(th), alpha=0.025)

    if surface.polygon_vertices is not None and len(surface.polygon_vertices) > 0:
        V = surface.polygon_vertices
        VV = np.vstack([V, V[0]])
        ax.plot(VV[:, 0], VV[:, 1], lw=1.3, color="tab:gray", alpha=0.9, label="fundamental polygon")
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i), fontsize=7, color="gray")

    path_k = max(1, min(int(path_k), len(candidates)))
    preview_k = max(preview_k, path_k)

    # Plot a faint cloud of low-rank candidates so the selected top-k pool has context.
    show = candidates[: min(len(candidates), preview_k)]
    xs = [c.point.real for c in show]
    ys = [c.point.imag for c in show]
    ax.scatter(xs, ys, s=14, alpha=0.25, color="tab:blue", label="candidate lifts")

    # Selected exact top-k candidate paths.
    for c in candidates[:path_k]:
        if c.rank == 1:
            color = "tab:red"
            lw = 2.8
            alpha = 0.78
            size = 80
            label = "winner path"
        elif c.rank <= 3:
            color = "tab:green"
            lw = 1.8
            alpha = 0.48
            size = 50
            label = "top-3 paths" if c.rank == 2 else None
        elif c.rank <= 5:
            color = "tab:purple"
            lw = 1.35
            alpha = 0.36
            size = 42
            label = "top-5 paths" if c.rank == 4 else None
        elif c.rank <= 10:
            color = "tab:olive"
            lw = 1.0
            alpha = 0.25
            size = 34
            label = "top-10 paths" if c.rank == 6 else None
        else:
            color = "tab:gray"
            lw = 0.85
            alpha = 0.18
            size = 26
            label = "top-20 paths" if c.rank == 11 else None

        ax.scatter([c.point.real], [c.point.imag], s=size, color=color, edgecolor="black", zorder=5)
        if c.rank <= 10 or c.rank == path_k:
            ax.text(c.point.real + 0.015, c.point.imag + 0.015, f"{c.rank}:{short_word(c.word)}", fontsize=8)
        pts = disk_geodesic_points(p, c.point, 120)
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, lw=lw, label=label)

    # Original upstairs points p and q.  Downstairs these represent [p] and [q].
    ax.scatter([p.real], [p.imag], s=90, color="black", marker="o", label="p upstairs", zorder=6)
    ax.scatter([q.real], [q.imag], s=90, color="tab:orange", marker="s", label="q upstairs", zorder=6)
    ax.text(p.real + 0.02, p.imag + 0.02, "p  ([p])", fontsize=10, weight="bold")
    ax.text(q.real + 0.02, q.imag + 0.02, "q  ([q])", fontsize=10, weight="bold")

    if candidates:
        w = candidates[0]
        ax.text(
            0.02,
            -1.02,
            f"Downstairs geodesic [p]→[q] is represented upstairs by p→γ*q; winner γ={short_word(w.word)}, d={w.distance:.4f}",
            fontsize=8,
            ha="left",
            va="top",
        )

    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Exact Poincare-disk finite-word paths: selected top-{path_k}")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.15)


def short_word(w: str, maxlen: int = 18) -> str:
    if w == "I": return "I"
    s = w.replace(" ", "")
    return s if len(s) <= maxlen else s[:maxlen-1] + "…"



def companion_point(surface: SurfaceExample, z: complex) -> np.ndarray:
    """Map a disk point to a point on/near the schematic R^3 companion surface.

    This is deliberately topological and visual, not metric.  It gives the 3D
    panel a consistent place to mark [p] and [q] and to draw a schematic path
    bundle corresponding to the exact disk-panel lift candidates.
    """
    z = safe_inside(z, 0.96)
    x, y = z.real, z.imag
    r = min(abs(z), 0.96)
    theta = math.atan2(y, x)
    kind = surface.embedding

    if kind == "cylinder":
        R = 0.87
        return np.array([R * math.cos(theta), R * math.sin(theta), 2.0 * x])

    if kind == "cusp":
        V = 0.8 + 3.2 * (1.0 - r)
        R = 1.05 * math.exp(-0.42 * V) + 0.08
        return np.array([R * math.cos(theta), R * math.sin(theta), V - 2.0])

    if kind == "pants":
        dirs = [
            np.array([1.65, 0.0, -0.70]),
            np.array([-0.95, 1.35, -0.70]),
            np.array([-0.95, -1.35, -0.70]),
        ]
        # Use angular sectors to choose a leg and radial coordinate to move outward.
        sector = int(((theta + math.pi) / (2 * math.pi)) * 3) % 3
        d = dirs[sector] / np.linalg.norm(dirs[sector])
        t = max(0.0, min(1.0, (r - 0.15) / 0.80))
        central = np.array([0.45 * x, 0.45 * y, 0.20 * (1 - r)])
        outer = 1.65 * d + np.array([0.0, 0.0, -0.20])
        return (1 - t) * central + t * outer

    if kind == "punctured_torus":
        R0, rr = 1.25, 0.35
        u = theta
        v = 2 * math.pi * r
        return np.array([(R0 + rr * math.cos(v)) * math.cos(u), (R0 + rr * math.cos(v)) * math.sin(u), rr * math.sin(v)])

    if kind == "genus2":
        R0, rr = 0.75, 0.24
        center = np.array([-0.9, 0.0, 0.0]) if x < 0 else np.array([0.9, 0.0, 0.0])
        u = math.pi * y
        v = 2 * math.pi * min(abs(x), 0.95)
        return center + np.array([(R0 + rr * math.cos(v)) * math.cos(u), (R0 + rr * math.cos(v)) * math.sin(u), rr * math.sin(v)])

    # Generic companion sheet.
    return np.array([1.5 * x, 1.2 * y, 0.25 * math.sin(2.0 * x) * math.cos(2.0 * y)])


def bezier3(a: np.ndarray, b: np.ndarray, control: np.ndarray, n: int = 80) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * control + t ** 2 * b


def color_for_rank(rank: int) -> Tuple[str, float, float]:
    """Return color, linewidth, alpha for a candidate rank."""
    if rank == 1:
        return "tab:red", 3.0, 0.86
    if rank <= 3:
        return "tab:green", 1.9, 0.58
    if rank <= 5:
        return "tab:purple", 1.45, 0.45
    if rank <= 10:
        return "tab:olive", 1.05, 0.32
    return "tab:gray", 0.9, 0.22




def torus_point_from_uv(u: np.ndarray, v: np.ndarray, R0: float = 1.25, rr: float = 0.35) -> np.ndarray:
    """Map torus parameters to R^3.  Works for scalar or vector u,v."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    X = (R0 + rr * np.cos(v)) * np.cos(u)
    Y = (R0 + rr * np.cos(v)) * np.sin(u)
    Z = rr * np.sin(v)
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def punctured_torus_params_from_z(z: complex) -> Tuple[float, float]:
    """Schematic quotient-coordinate parameters for the punctured-torus companion."""
    z = safe_inside(z, 0.96)
    u = math.atan2(z.imag, z.real)
    v = 2.0 * math.pi * min(abs(z), 0.96)
    return u, v


def punctured_torus_surface_trace(surface: SurfaceExample, p: complex, q: complex, candidate: CandidateResult, n: int = 420) -> Optional[np.ndarray]:
    """Surface-parameter trace for the punctured torus companion.

    This is not the hyperbolic geodesic in an isometric embedding.  It is a
    quotient-aware topological trace on the displayed torus core.  Candidate word
    exponents are used as schematic winding numbers, so different lift classes
    become visibly different surface paths while every path begins at [p] and
    ends at [q].
    """
    if surface.embedding != "punctured_torus":
        return None
    u0, v0 = punctured_torus_params_from_z(p)
    u1, v1 = punctured_torus_params_from_z(q)
    wind_u = signed_power_in_word(candidate.word, "A")
    wind_v = signed_power_in_word(candidate.word, "B")
    du = angle_delta_with_winding(u0, u1, wind_u)
    dv = angle_delta_with_winding(v0, v1, wind_v)
    t = np.linspace(0.0, 1.0, n)
    # Smooth endpoint-zero sideband separates equal-winding candidates visually
    # without moving endpoints off [p] or [q].
    sideband = 0.10 * math.sin(candidate.rank) * np.sin(np.pi * t)
    u = u0 + t * du
    v = v0 + t * dv + sideband
    C = torus_point_from_uv(u, v)
    C[0] = companion_point(surface, p)
    C[-1] = companion_point(surface, q)
    return C


def signed_power_in_word(word: str, letter: str = "A") -> int:
    """Return the signed exponent sum of a generator letter in a word string.

    The visualizer word format is a space-separated reduced word such as
    "A", "A^-1", "A A", or "A^-1 B".  For cyclic examples this exponent is
    used as a schematic winding number on the R^3 cylinder/cusp companion.
    """
    if not word or word == "I":
        return 0
    total = 0
    for tok in word.split():
        if tok == letter:
            total += 1
        elif tok == f"{letter}^-1":
            total -= 1
    return total


def angle_delta_with_winding(theta0: float, theta1: float, winding: int) -> float:
    """Angle change from theta0 to theta1 plus an integer winding.

    The base part is the shortest angular difference, then 2π*winding is added.
    This makes distinct lift candidates visibly wrap differently while all traces
    still begin at [p] and end at [q].
    """
    base = (theta1 - theta0 + math.pi) % (2.0 * math.pi) - math.pi
    return base + 2.0 * math.pi * int(winding)


def cylinder_or_cusp_surface_trace(surface: SurfaceExample, p: complex, q: complex, candidate: CandidateResult, n: int = 420) -> Optional[np.ndarray]:
    """Surface-constrained traces for cyclic cylinder/cusp companion models.

    These are still visual companions, not isometric hyperbolic embeddings.  But
    unlike the generic endpoint correction, the returned points lie on the plotted
    cylinder/cusp model.  This fixes the misleading red winner chord seen in v1.4.
    """
    kind = surface.embedding
    if kind not in ("cylinder", "cusp"):
        return None
    P = companion_point(surface, p)
    Q = companion_point(surface, q)
    theta0 = math.atan2(P[1], P[0])
    theta1 = math.atan2(Q[1], Q[0])
    winding = signed_power_in_word(candidate.word, "A")
    dtheta = angle_delta_with_winding(theta0, theta1, winding)
    t = np.linspace(0.0, 1.0, n)
    theta = theta0 + t * dtheta

    if kind == "cylinder":
        R = 0.85
        z = (1.0 - t) * P[2] + t * Q[2]
        C = np.column_stack([R * np.cos(theta), R * np.sin(theta), z])
    else:
        # In the cusp model, z is the cusp height parameter.  Radius follows
        # the plotted exponential cusp profile at that height.
        z = (1.0 - t) * P[2] + t * Q[2]
        V = z + 2.0
        R = 1.05 * np.exp(-0.42 * V) + 0.08
        C = np.column_stack([R * np.cos(theta), R * np.sin(theta), z])

    C[0] = P
    C[-1] = Q
    return C


def anti_chord_safeguard(C: np.ndarray, rank: int) -> np.ndarray:
    """Add a small endpoint-zero bulge if a trace is visually almost a chord.

    This is only for generic topological companion traces.  It prevents the
    emphasized winner from looking like a straight ambient-space segment when the
    raw quotient-sketch projection is visually degenerate.  The endpoints remain
    exactly fixed at [p] and [q].
    """
    if C.shape[0] < 5:
        return C
    P = C[0].copy(); Q = C[-1].copy()
    chord = Q - P
    L = float(np.linalg.norm(chord))
    if L < 1e-9:
        return C
    t = np.linspace(0.0, 1.0, C.shape[0])[:, None]
    line = (1.0 - t) * P + t * Q
    deviation = np.linalg.norm(C - line, axis=1).max()
    # Only nudge traces that are too chord-like; make rank-1 slightly more visible.
    threshold = 0.035 * L
    if deviation > threshold:
        return C
    # Choose a stable normal direction perpendicular to the chord.
    ref = np.array([0.0, 0.0, 1.0])
    normal = np.cross(chord, ref)
    if np.linalg.norm(normal) < 1e-9:
        ref = np.array([0.0, 1.0, 0.0])
        normal = np.cross(chord, ref)
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    # Add a secondary vertical component to keep paths visible on flat companion regions.
    side = np.cross(normal, chord)
    if np.linalg.norm(side) > 1e-9:
        side = side / np.linalg.norm(side)
    else:
        side = np.zeros(3)
    amp = (0.13 if rank == 1 else 0.08) * L
    bulge = amp * np.sin(np.pi * t) * (0.75 * normal + 0.25 * side)
    out = C + bulge
    out[0] = P
    out[-1] = Q
    return out


def companion_trace_for_candidate(surface: SurfaceExample, p: complex, q: complex, candidate: CandidateResult, n: int = 420) -> np.ndarray:
    """Return an endpoint-anchored 3D topological trace for a candidate path.

    Mathematical source:
        The left/disk panel computes the exact finite-word upstairs geodesic
        representative p -> gamma*q.  Its quotient projection is a path on
        X = D/Gamma from [p] to [q].

    Visualization problem:
        The companion map E_sketch(z) used for the R^3 panel is not a genuine
        quotient map: generally E_sketch(gamma*q) != E_sketch(q).  If we mapped
        the sampled geodesic directly, different lift candidates would appear
        to have different endpoints in R^3, which is wrong downstairs.

    v1.6 fix:
        Sample p -> gamma*q in the disk, map those points to the companion model,
        and then apply a smooth endpoint correction so the displayed trace starts
        exactly at E([p]) and ends exactly at E([q]).  The interior remains a
        schematic/topological trace of the upstairs candidate, not a metric
        geodesic of the R^3 embedding.
    """
    surface_trace = cylinder_or_cusp_surface_trace(surface, p, q, candidate, n=n)
    if surface_trace is not None:
        return surface_trace
    surface_trace = punctured_torus_surface_trace(surface, p, q, candidate, n=n)
    if surface_trace is not None:
        return surface_trace

    P = companion_point(surface, p)
    Q = companion_point(surface, q)
    pts2 = disk_geodesic_points(p, candidate.point, n=n)
    raw = np.asarray([companion_point(surface, complex(float(x), float(y))) for x, y in pts2], dtype=float)
    if len(raw) < 2:
        return np.vstack([P, Q])

    raw0 = raw[0].copy()
    raw1 = raw[-1].copy()
    t = np.linspace(0.0, 1.0, len(raw))[:, None]

    # Smoothly translate the raw trace so endpoints obey the quotient relation:
    # p maps to [p], while every gamma*q maps to [q].
    correction = (1.0 - t) * (P - raw0) + t * (Q - raw1)
    anchored = raw + correction
    anchored[0] = P
    anchored[-1] = Q
    return anti_chord_safeguard(anchored, candidate.rank)


def draw_companion_path_bundle(ax, surface: SurfaceExample, p: complex, q: complex, candidates: List[CandidateResult], path_k: int):
    """Draw endpoint-anchored topological traces for selected candidate geodesics.

    Every candidate lift gamma*q represents a downstairs path from [p] to [q].
    Therefore every displayed 3D trace is anchored at the same two displayed
    points P=E([p]) and Q=E([q]).  The interior shows a schematic trace derived
    from the exact upstairs geodesic, but is not a metric geodesic in R^3.
    """
    if not candidates:
        return
    P = companion_point(surface, p)
    Q = companion_point(surface, q)
    path_k = max(1, min(path_k, len(candidates)))

    # Mark downstairs equivalence classes on the companion model.
    ax.scatter([P[0]], [P[1]], [P[2]], s=95, color="black", depthshade=True)
    ax.scatter([Q[0]], [Q[1]], [Q[2]], s=95, color="tab:orange", marker="s", depthshade=True)
    ax.text(P[0], P[1], P[2] + 0.12, "[p]", fontsize=10, color="black")
    ax.text(Q[0], Q[1], Q[2] + 0.12, "[q]", fontsize=10, color="tab:orange")

    for c in candidates[:path_k]:
        color, lw, alpha = color_for_rank(c.rank)
        C = companion_trace_for_candidate(surface, p, q, c, n=420)
        ax.plot(C[:, 0], C[:, 1], C[:, 2], color=color, lw=lw, alpha=alpha)
        if c.rank == 1:
            mid = C[len(C)//2]
            ax.text(mid[0], mid[1], mid[2] + 0.10, "winning trace", fontsize=8, color=color)

def draw_embedding(
    ax,
    surface: SurfaceExample,
    p: Optional[complex] = None,
    q: Optional[complex] = None,
    candidates: Optional[List[CandidateResult]] = None,
    path_k: int = 1,
):
    ax.clear()
    kind = surface.embedding
    if kind == "cylinder":
        plot_cylinder(ax)
    elif kind == "cusp":
        plot_cusp(ax)
    elif kind == "pants":
        plot_pants(ax)
    elif kind == "punctured_torus":
        plot_punctured_torus(ax)
    elif kind == "genus2":
        plot_genus2(ax)
    else:
        plot_generic(ax)

    if p is not None and q is not None and candidates is not None:
        draw_companion_path_bundle(ax, surface, p, q, candidates, path_k)

    ax.set_title("Rotatable R^3 companion: endpoint-anchored [p]→[q] candidate traces")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.text2D(
        0.02,
        0.02,
        "All 3D traces are anchored at [p] and [q]; cyclic and punctured-torus examples use surface-constrained traces; metric truth remains the disk view.",
        transform=ax.transAxes,
        fontsize=8,
    )
    set_axes_equal(ax)



# -----------------------------------------------------------------------------
# PyQt6 GUI
# -----------------------------------------------------------------------------


class VisualizerApp:
    """PyQt6/Matplotlib interactive two-view visualizer.

    The GUI import is intentionally delayed so --self-test and --save-demo remain
    usable on headless machines without PyQt6 installed.
    """

    def __init__(self):
        try:
            from PyQt6 import QtCore, QtWidgets
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure
        except Exception as e:
            raise RuntimeError(
                "PyQt6 GUI dependencies are missing. Install with:\n"
                "    pip install PyQt6\n"
                "Headless modes still work:\n"
                "    python FuchsianDownstairsVisualizer_v1_6.py --self-test\n"
                "    python FuchsianDownstairsVisualizer_v1_6.py --save-demo demo.png"
            ) from e

        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.FigureCanvas = FigureCanvas
        self.NavigationToolbar = NavigationToolbar
        self.Figure = Figure

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Fuchsian Downstairs Visualizer v1.6 (PyQt6)")
        self.window.resize(1400, 820)

        self.examples: List[SurfaceExample] = built_in_examples()
        self.p = 0.15 + 0.23j
        self.q = -0.22 + 0.18j
        self.random_seed = 7

        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Surface:"))
        self.surface_combo = QtWidgets.QComboBox()
        self.surface_combo.addItems([s.name for s in self.examples])
        self.surface_combo.setMinimumWidth(440)
        self.surface_combo.currentIndexChanged.connect(self.update_plot)
        controls.addWidget(self.surface_combo)

        controls.addSpacing(10)
        controls.addWidget(QtWidgets.QLabel("Depth:"))
        self.depth_spin = QtWidgets.QSpinBox()
        self.depth_spin.setRange(0, 5)
        self.depth_spin.setValue(2)
        self.depth_spin.valueChanged.connect(self.update_plot)
        controls.addWidget(self.depth_spin)

        controls.addWidget(QtWidgets.QLabel("Paths:"))
        self.path_combo = QtWidgets.QComboBox()
        self.path_combo.addItems(["winner only", "top-3", "top-5", "top-10", "top-20", "custom"])
        self.path_combo.setCurrentText("top-5")
        self.path_combo.currentIndexChanged.connect(self.update_plot)
        controls.addWidget(self.path_combo)

        controls.addWidget(QtWidgets.QLabel("Custom k:"))
        self.topk_spin = QtWidgets.QSpinBox()
        self.topk_spin.setRange(1, 20)
        self.topk_spin.setValue(5)
        self.topk_spin.valueChanged.connect(self.update_plot)
        controls.addWidget(self.topk_spin)

        self.update_button = QtWidgets.QPushButton("Update")
        self.update_button.clicked.connect(self.update_plot)
        controls.addWidget(self.update_button)

        self.random_button = QtWidgets.QPushButton("Random pair")
        self.random_button.clicked.connect(self.random_pair)
        controls.addWidget(self.random_button)

        self.load_button = QtWidgets.QPushButton("Load JSON…")
        self.load_button.clicked.connect(self.load_json_dialog)
        controls.addWidget(self.load_button)

        self.save_button = QtWidgets.QPushButton("Save PNG…")
        self.save_button.clicked.connect(self.save_png_dialog)
        controls.addWidget(self.save_button)

        controls.addStretch(1)
        outer.addLayout(controls)

        self.fig = Figure(figsize=(12.5, 6.2), dpi=100)
        self.ax_disk = self.fig.add_subplot(1, 2, 1)
        self.ax_3d = self.fig.add_subplot(1, 2, 2, projection="3d")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()
        self.toolbar = NavigationToolbar(self.canvas, self.window)
        outer.addWidget(self.toolbar)
        outer.addWidget(self.canvas, stretch=1)

        self.status = QtWidgets.QStatusBar()
        self.window.setStatusBar(self.status)
        self.window.setCentralWidget(central)

        help_text = (
            "Left: exact Poincare-disk finite-word lift candidates.  "
            "Right: rotatable R^3 topological companion; not an isometric embedding; all candidate traces are anchored at [p] and [q]."
        )
        self.status.showMessage(help_text)
        self.update_plot()

    def current_surface(self) -> SurfaceExample:
        return self.examples[self.surface_combo.currentIndex()]

    def selected_path_k(self) -> int:
        text = self.path_combo.currentText() if hasattr(self, "path_combo") else "top-5"
        mapping = {
            "winner only": 1,
            "top-3": 3,
            "top-5": 5,
            "top-10": 10,
            "top-20": 20,
        }
        if text in mapping:
            return mapping[text]
        return int(self.topk_spin.value())

    def random_pair(self):
        self.random_seed += random.randint(1, 100000)
        random.seed(self.random_seed)

        def rand_disk(rmax=0.68):
            r = rmax * math.sqrt(random.random())
            th = 2 * math.pi * random.random()
            return r * complex(math.cos(th), math.sin(th))

        self.p = rand_disk()
        self.q = rand_disk()
        self.update_plot()

    def load_json_dialog(self):
        QtWidgets = self.QtWidgets
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Load GENN surface JSON",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            surf = load_surface_json(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.window, "Load failed", str(e))
            return
        self.examples.append(surf)
        self.surface_combo.addItem(surf.name)
        self.surface_combo.setCurrentIndex(len(self.examples) - 1)
        self.update_plot()

    def save_png_dialog(self):
        QtWidgets = self.QtWidgets
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save current view as PNG",
            "fuchsian_downstairs_visualizer.png",
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.fig.savefig(path, dpi=180, bbox_inches="tight")
        self.status.showMessage(f"Saved {path}")

    def update_plot(self):
        try:
            surf = self.current_surface()
            depth = int(self.depth_spin.value())
            path_k = self.selected_path_k()
            candidates = compute_candidates(surf, self.p, self.q, depth)
            draw_disk(self.ax_disk, surf, self.p, self.q, candidates, path_k)
            draw_embedding(self.ax_3d, surf, self.p, self.q, candidates, path_k)
            self.fig.suptitle(f"{surf.name}\n{surf.regime}", fontsize=12)
            self.fig.tight_layout(rect=[0, 0.02, 1, 0.92])
            self.canvas.draw_idle()
            best = candidates[0]
            self.status.showMessage(
                f"{surf.name} | generators={len(surf.generators)} | W(depth {depth})={len(candidates)} | "
                f"showing top-{path_k} exact path candidates | winner={short_word(best.word)} d={best.distance:.4f} | "
                f"Disk geometry is exact for finite word ball; 3D traces are endpoint-anchored topological projections; cyclic/torus traces are surface-parameterized, not metric geodesics."
            )
        except Exception as e:
            self.status.showMessage(f"Update failed: {e}")
            raise

    def run(self):
        self.window.show()
        return self.app.exec()


# -----------------------------------------------------------------------------
# Headless demo / self-test
# -----------------------------------------------------------------------------


def render_demo(path: str, surface_key: str = "gamma2_trinion", depth: int = 2, top_k: int = 5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    examples = {s.key: s for s in built_in_examples()}
    surf = examples.get(surface_key, make_gamma2())
    p = 0.15 + 0.23j
    q = -0.22 + 0.18j
    candidates = compute_candidates(surf, p, q, depth)
    fig = plt.figure(figsize=(12, 5.5), dpi=120)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_disk(ax1, surf, p, q, candidates, top_k)
    draw_embedding(ax2, surf, p, q, candidates, top_k)
    fig.suptitle(f"{surf.name} demo")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path, len(candidates), candidates[0].word, candidates[0].distance


def self_test() -> int:
    print("FuchsianDownstairsVisualizer_v1_6 self-test")
    ok = True
    for surf in built_in_examples():
        try:
            rows = compute_candidates(surf, 0.12 + 0.2j, -0.18 + 0.1j, depth=2)
            print(f"  {surf.key:28s} gens={len(surf.generators):2d} W={len(rows):4d} winner={short_word(rows[0].word):8s} d={rows[0].distance:.4f}")
            if not rows or not np.isfinite(rows[0].distance):
                ok = False
        except Exception as e:
            ok = False
            print(f"  FAIL {surf.key}: {e}")
    out = "/tmp/fuchsian_downstairs_visualizer_v1_6_demo.png"
    try:
        render_demo(out)
        print(f"  demo render: {out} ({os.path.getsize(out)} bytes)")
    except Exception as e:
        ok = False
        print(f"  FAIL render_demo: {e}")
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Interactive PyQt6 Fuchsian downstairs candidate visualizer.")
    ap.add_argument("--self-test", action="store_true", help="Run headless internal tests and render a demo PNG.")
    ap.add_argument("--save-demo", default="", help="Render a headless demo PNG to this path instead of launching GUI.")
    ap.add_argument("--demo-surface", default="gamma2_trinion", help="Built-in surface key for --save-demo.")
    ap.add_argument("--demo-top-k", type=int, default=5, help="Number of exact top-k paths to show in --save-demo.")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.save_demo:
        path, W, word, dist = render_demo(args.save_demo, surface_key=args.demo_surface, top_k=args.demo_top_k)
        print(f"saved {path}; W={W}; winner={short_word(word)}; distance={dist:.6f}")
        return 0
    app = VisualizerApp()
    return int(app.run())


if __name__ == "__main__":
    raise SystemExit(main())
