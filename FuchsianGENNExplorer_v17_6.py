#!/usr/bin/env python3
"""
FuchsianGENNExplorer_v17_3.py

A mathematically careful student-facing explorer for the Poincare disk,
certified Fuchsian quotient examples, neural metric learning, and lifted
geodesic computation.

Version 17.4 tests explicit, user-triggered tessellation: modular and polygonal domains load by showing only the principal/base domain, and the finite word-patch tessellation is drawn only when the user requests it. It keeps the
geometry-feedstock tools (classical differential geometry, finite word-length
candidates, sampled injectivity radius, and base-polygon side-crossing
intersections) and adds a quotient-aware point-cloud graph-Laplacian spectral
prototype. It keeps the rigorous modes
already established:

1. Exact Poincare disk / no quotient.
2. Certified cyclic hyperbolic quotient <A>.
3. Built-in certified regular genus-2 octagon surface.
4. Advanced user-supplied polygon surfaces, accepted by fiat as externally
   certified Fuchsian data.

Earlier versions added cached/fast finite word-patch rendering, a Geometry Audit tab, workflow
buttons for student-safe examples, generator-ready JSON import/export, and certification metadata display.
v17.2 made torsion-free modular congruence metadata visible in the Audit and Invariants tabs: subgroup, level N, index, cusps, cusp widths, compactified genus, boundary cleanup, and torsion-free audit. Compact-polygon JSON still supports arbitrary one-letter generator labels A,B,C,... .

The 3D window shows the universal-cover scalar field

    z = phi(x,y) = log(2) - log(1 - x^2 - y^2),

not an isometric embedding of the quotient surface.  Gamma changes quotient
identifications, not the local universal-cover Poincare conformal factor.

Dependencies:
    pip install pyqt6 matplotlib numpy torch

Torch is needed for the neural-relaxation and neural-metric modes.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - depends on user environment
    torch = None
    TORCH_AVAILABLE = False

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QFileDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


EPS = 1.0e-12


# -----------------------------------------------------------------------------
# Poincare disk geometry
# -----------------------------------------------------------------------------

def inside_unit_disk(p: np.ndarray, margin: float = 1.0e-9) -> bool:
    p = np.asarray(p, dtype=float)
    return float(np.dot(p, p)) < 1.0 - margin


def complex_from_point(p: np.ndarray) -> complex:
    return complex(float(p[0]), float(p[1]))


def point_from_complex(z: complex) -> np.ndarray:
    return np.array([float(z.real), float(z.imag)], dtype=float)


def circle_points(cx: float, cy: float, r: float, n: int = 720) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])


def poincare_phi_np(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    r2 = np.sum(pts * pts, axis=-1)
    return math.log(2.0) - np.log(np.maximum(1.0 - r2, 1.0e-12))


def poincare_scale_np(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    r2 = np.sum(pts * pts, axis=-1)
    return 2.0 / np.maximum(1.0 - r2, 1.0e-12)


def poincare_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p2 = float(np.dot(p, p))
    q2 = float(np.dot(q, q))
    diff2 = float(np.dot(p - q, p - q))
    denom = max((1.0 - p2) * (1.0 - q2), 1.0e-15)
    arg = 1.0 + 2.0 * diff2 / denom
    return float(np.arccosh(max(arg, 1.0)))


def exact_poincare_geodesic(p: np.ndarray, q: np.ndarray, n: int = 420) -> np.ndarray:
    """Exact geodesic segment between two points in the Poincare disk."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if np.linalg.norm(p - q) < 1.0e-13:
        return np.repeat(p[None, :], n, axis=0)

    cross = p[0] * q[1] - p[1] * q[0]
    if abs(cross) < 1.0e-11:
        return np.linspace(p, q, n)

    # The Euclidean circle for a disk geodesic has center c satisfying
    #     2 p.c = |p|^2 + 1,     2 q.c = |q|^2 + 1.
    A = 2.0 * np.vstack([p, q])
    b = np.array([np.dot(p, p) + 1.0, np.dot(q, q) + 1.0], dtype=float)
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linspace(p, q, n)

    R2 = float(np.dot(c, c) - 1.0)
    if R2 <= 0.0:
        return np.linspace(p, q, n)
    R = math.sqrt(R2)

    a1 = math.atan2(p[1] - c[1], p[0] - c[0])
    a2 = math.atan2(q[1] - c[1], q[0] - c[0])
    delta = (a2 - a1) % (2.0 * math.pi)

    theta_a = a1 + np.linspace(0.0, delta, n)
    theta_b = a1 + np.linspace(0.0, delta - 2.0 * math.pi, n)
    arc_a = np.column_stack([c[0] + R * np.cos(theta_a), c[1] + R * np.sin(theta_a)])
    arc_b = np.column_stack([c[0] + R * np.cos(theta_b), c[1] + R * np.sin(theta_b)])

    max_a = float(np.max(np.sum(arc_a * arc_a, axis=1)))
    max_b = float(np.max(np.sum(arc_b * arc_b, axis=1)))
    return arc_a if max_a <= max_b else arc_b



def ideal_geodesic_arc(u: np.ndarray, v: np.ndarray, n: int = 180, shrink: float = 0.999999) -> np.ndarray:
    """Sample the Poincare geodesic with ideal endpoints u and v on the unit circle.

    This is used for Schottky-style ideal geodesic domains.  The endpoints are
    allowed to lie on the unit circle, unlike ordinary finite polygon vertices.
    The returned plotted arc is nudged slightly inside the disk at the endpoints.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
    if nu < EPS or nv < EPS:
        return np.empty((0, 2), dtype=float)
    u = u / nu
    v = v / nv
    # Antipodal ideal points define a diameter.
    if abs(float(np.cross(u, v))) < 1.0e-10:
        return np.linspace(shrink * u, shrink * v, n)
    A = np.vstack([u, v])
    b = np.array([1.0, 1.0], dtype=float)
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linspace(shrink * u, shrink * v, n)
    R2 = float(np.dot(c, c) - 1.0)
    if R2 <= 0.0:
        return np.linspace(shrink * u, shrink * v, n)
    R = math.sqrt(R2)
    a1 = math.atan2(u[1] - c[1], u[0] - c[0])
    a2 = math.atan2(v[1] - c[1], v[0] - c[0])
    delta = (a2 - a1) % (2.0 * math.pi)
    theta_a = a1 + np.linspace(0.0, delta, n)
    theta_b = a1 + np.linspace(0.0, delta - 2.0 * math.pi, n)
    arc_a = np.column_stack([c[0] + R * np.cos(theta_a), c[1] + R * np.sin(theta_a)])
    arc_b = np.column_stack([c[0] + R * np.cos(theta_b), c[1] + R * np.sin(theta_b)])
    # Choose the arc lying inside the disk.
    arc = arc_a if float(np.nanmax(np.sum(arc_a * arc_a, axis=1))) <= float(np.nanmax(np.sum(arc_b * arc_b, axis=1))) else arc_b
    rr = np.linalg.norm(arc, axis=1)
    mask = rr >= shrink
    if np.any(mask):
        arc[mask] *= (shrink / rr[mask])[:, None]
    return arc

def discrete_hyperbolic_length(curve: np.ndarray) -> float:
    pts = np.asarray(curve, dtype=float)
    if len(pts) < 2:
        return 0.0
    mid = 0.5 * (pts[:-1] + pts[1:])
    ds_euclid = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return float(np.sum(poincare_scale_np(mid) * ds_euclid))


def discrete_hyperbolic_energy(curve: np.ndarray) -> float:
    pts = np.asarray(curve, dtype=float)
    if len(pts) < 2:
        return 0.0
    n = len(pts)
    dt = 1.0 / (n - 1)
    dz_dt = (pts[1:] - pts[:-1]) / dt
    mid = 0.5 * (pts[:-1] + pts[1:])
    e2phi = poincare_scale_np(mid) ** 2
    return float(0.5 * np.sum(e2phi * np.sum(dz_dt * dz_dt, axis=1)) * dt)


def hyperbolic_segment_speeds(curve: np.ndarray) -> np.ndarray:
    pts = np.asarray(curve, dtype=float)
    if len(pts) < 2:
        return np.array([], dtype=float)
    dt = 1.0 / (len(pts) - 1)
    mid = 0.5 * (pts[:-1] + pts[1:])
    ds_euclid = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return poincare_scale_np(mid) * ds_euclid / dt


def gauss_hyperbolic_length_energy(
    curve: np.ndarray,
    order: int = 24,
) -> Tuple[float, float, np.ndarray]:
    """High-accuracy diagnostics for a sampled curve.

    The optimizer records a midpoint-quadrature training energy because that is
    fast and differentiable.  For final reporting, however, we integrate the
    displayed piecewise-linear curve with Gauss-Legendre quadrature on each
    segment.  This matters near the unit circle, where the Poincare conformal
    factor changes rapidly and midpoint quadrature can slightly underestimate
    the true energy.

    Returns
    -------
    length, energy, segment_speeds
        The energy is the constant-parameter energy on t in [0,1].
    """
    pts = np.asarray(curve, dtype=float)
    if len(pts) < 2:
        return 0.0, 0.0, np.array([], dtype=float)

    order = int(max(4, min(order, 64)))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    s_nodes = 0.5 * (nodes + 1.0)
    s_weights = 0.5 * weights

    nseg = len(pts) - 1
    dt = 1.0 / nseg
    seg_lengths = np.zeros(nseg, dtype=float)
    energy = 0.0

    for i in range(nseg):
        a = pts[i]
        b = pts[i + 1]
        d = b - a
        chord_len = float(np.linalg.norm(d))
        if chord_len < EPS:
            continue
        samples = a[None, :] + s_nodes[:, None] * d[None, :]
        lam = poincare_scale_np(samples)
        int_lam = float(np.sum(s_weights * lam))
        int_lam2 = float(np.sum(s_weights * lam * lam))
        seg_lengths[i] = chord_len * int_lam
        energy += 0.5 * (chord_len * chord_len / dt) * int_lam2

    speeds = seg_lengths / dt
    return float(np.sum(seg_lengths)), float(energy), speeds


def diagnostic_length_energy_speeds(curve: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Default high-accuracy diagnostic integral for GUI reporting."""
    return gauss_hyperbolic_length_energy(curve, order=24)


def make_initial_curve(p: np.ndarray, q: np.ndarray, n: int, perturb: float) -> np.ndarray:
    """Smooth non-geodesic starter curve with fixed endpoints."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    t = np.linspace(0.0, 1.0, n)
    base = (1.0 - t)[:, None] * p[None, :] + t[:, None] * q[None, :]
    chord = q - p
    norm = float(np.linalg.norm(chord))
    normal = np.array([0.0, 1.0]) if norm < EPS else np.array([-chord[1], chord[0]]) / norm
    bump = perturb * np.sin(math.pi * t)[:, None] * normal[None, :]
    curve = base + bump

    for _ in range(14):
        if float(np.max(np.sum(curve * curve, axis=1))) < 0.965:
            return curve
        bump *= 0.5
        curve = base + bump
    return curve


# -----------------------------------------------------------------------------
# Disk Mobius transformations and the certified cyclic group
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DiskMobius:
    """Orientation-preserving disk isometry in SU(1,1) form.

    f(z) = (alpha z + beta)/(conj(beta) z + conj(alpha)).
    """

    alpha: complex
    beta: complex
    name: str = ""

    def __call__(self, z: complex) -> complex:
        denom = self.beta.conjugate() * z + self.alpha.conjugate()
        if abs(denom) < 1.0e-14:
            denom = complex(1.0e-14, 0.0)
        return (self.alpha * z + self.beta) / denom

    def apply_point(self, p: np.ndarray) -> np.ndarray:
        return point_from_complex(self(complex_from_point(p)))

    def apply_curve(self, curve: np.ndarray) -> np.ndarray:
        return np.array([self.apply_point(p) for p in np.asarray(curve, dtype=float)], dtype=float)

    def compose(self, other: "DiskMobius", name: str = "") -> "DiskMobius":
        """Return self o other."""
        a1, b1 = self.alpha, self.beta
        a2, b2 = other.alpha, other.beta
        alpha = a1 * a2 + b1 * b2.conjugate()
        beta = a1 * b2 + b1 * a2.conjugate()
        return DiskMobius(alpha, beta, name=name).normalized()

    def inverse(self, name: str = "") -> "DiskMobius":
        return DiskMobius(self.alpha.conjugate(), -self.beta, name=name or self.name + "^-1").normalized()

    def normalized(self) -> "DiskMobius":
        det = abs(self.alpha) ** 2 - abs(self.beta) ** 2
        if det <= 0.0:
            return self
        scale = math.sqrt(det)
        return DiskMobius(self.alpha / scale, self.beta / scale, self.name)

    @staticmethod
    def identity(name: str = "I") -> "DiskMobius":
        return DiskMobius(1.0 + 0.0j, 0.0 + 0.0j, name=name)

    @staticmethod
    def real_translation(a: float, name: str = "A") -> "DiskMobius":
        """Hyperbolic translation along the real diameter, sending 0 to a."""
        if abs(a) >= 1.0:
            raise ValueError("real_translation requires |a| < 1.")
        scale = 1.0 / math.sqrt(1.0 - a * a)
        return DiskMobius(scale + 0.0j, a * scale + 0.0j, name=name).normalized()




def disk_rotation(theta: float, name: str = "R") -> DiskMobius:
    """Rotation z -> exp(i theta) z as an SU(1,1) disk isometry."""
    return DiskMobius(complex(math.cos(theta / 2.0), math.sin(theta / 2.0)), 0.0 + 0.0j, name=name).normalized()


def axial_translation(a: float, theta: float, name: str = "G") -> DiskMobius:
    """Hyperbolic translation with length 2 atanh(a) along the diameter at angle theta.

    This is R_theta o T_a o R_-theta.  It sends 0 to a*exp(i theta).
    """
    a = float(a)
    theta = float(theta)
    T = DiskMobius.real_translation(a, name=name)
    R = disk_rotation(theta, name="R")
    Rinv = disk_rotation(-theta, name="R^-1")
    return R.compose(T.compose(Rinv), name=name).normalized()




def disk_move_to_zero(a: complex, name: str = "M") -> DiskMobius:
    """Disk isometry z -> (z-a)/(1-conj(a)z), sending a to 0."""
    if abs(a) >= 1.0:
        raise ValueError("disk_move_to_zero requires |a|<1")
    scale = 1.0 / math.sqrt(max(1.0 - abs(a) ** 2, 1.0e-15))
    return DiskMobius(scale + 0.0j, -a * scale, name=name).normalized()


def disk_move_from_zero(a: complex, name: str = "M^-1") -> DiskMobius:
    """Inverse of disk_move_to_zero(a)."""
    if abs(a) >= 1.0:
        raise ValueError("disk_move_from_zero requires |a|<1")
    scale = 1.0 / math.sqrt(max(1.0 - abs(a) ** 2, 1.0e-15))
    return DiskMobius(scale + 0.0j, a * scale, name=name).normalized()


def disk_isometry_from_two_point_pairs(
    z1: complex,
    z2: complex,
    w1: complex,
    w2: complex,
    name: str = "G",
) -> DiskMobius:
    """Unique orientation-preserving disk isometry sending z1->w1 and z2->w2.

    The two pairs must have the same hyperbolic separation.  This routine is
    used to construct side-pairing maps for the regular octagon.  It does not
    prove a polygon theorem; it constructs the unique PSU(1,1) map determined
    by the two endpoint correspondences.
    """
    Mz = disk_move_to_zero(z1, name="Mz")
    Mw_inv = disk_move_from_zero(w1, name="Mw^-1")
    u = Mz(z2)
    Mw = disk_move_to_zero(w1, name="Mw")
    v = Mw(w2)
    if abs(u) < 1.0e-14 or abs(v) < 1.0e-14:
        raise ValueError("Degenerate side-pairing endpoint data.")
    lam = v / u
    lam = lam / abs(lam)
    R = disk_rotation(math.atan2(lam.imag, lam.real), name="R")
    return Mw_inv.compose(R.compose(Mz), name=name).normalized()


def regular_hyperbolic_polygon_radius(p: int, q: int) -> float:
    """Euclidean circumradius in the disk for the regular {p,q} polygon.

    The hyperbolic circumradius R satisfies cosh(R)=cos(pi/q)/sin(pi/p), and
    the Poincare disk Euclidean radius is tanh(R/2).
    """
    cosh_R = math.cos(math.pi / q) / math.sin(math.pi / p)
    R = math.acosh(cosh_R)
    return math.tanh(0.5 * R)

def inverse_letter(ch: str) -> str:
    """Return the formal inverse letter for one-letter generator names.

    v12.2 allows compact-polygon JSON files to define arbitrary one-letter
    uppercase generators A, B, C, ... .  Lowercase letters denote inverses.
    """
    if not isinstance(ch, str) or len(ch) != 1 or not ch.isalpha():
        return ""
    return ch.lower() if ch.isupper() else ch.upper()


def clean_fuchsian_word(text: str) -> str:
    """Keep one-letter generator symbols.

    Surface-specific validity is checked later against the generators supplied
    by the active surface.  This generic cleaner intentionally allows letters
    beyond A-D so externally generated genus-g polygon files can use E, F, ... .
    """
    return "".join(ch for ch in text.strip() if ch.isalpha() and len(ch) == 1)


def reduce_fuchsian_word(word: str) -> str:
    stack = []
    for ch in clean_fuchsian_word(word):
        if stack and inverse_letter(ch) == stack[-1]:
            stack.pop()
        else:
            stack.append(ch)
    return "".join(stack)


def generate_reduced_words(depth: int, alphabet: str = "ABab") -> List[str]:
    """Generate reduced words up to a given alphabet of generators/inverses."""
    depth = int(depth)
    words = [""]
    frontier = [""]
    for _ in range(depth):
        new_frontier = []
        for w in frontier:
            for ch in alphabet:
                if w and inverse_letter(ch) == w[-1]:
                    continue
                nw = w + ch
                words.append(nw)
                new_frontier.append(nw)
        frontier = new_frontier
    return words


def cyclic_power_parameter(a: float, n: int) -> float:
    """Parameter a_n such that A_a^n = A_{a_n}."""
    if n == 0:
        return 0.0
    tau = math.atanh(float(a))
    return math.tanh(n * tau)


def cyclic_power(a: float, n: int) -> DiskMobius:
    return DiskMobius.real_translation(cyclic_power_parameter(a, n), name=f"A^{n}")


def parse_cyclic_word(text: str) -> int:
    """Return the exponent represented by a word in A and a=A^{-1}.

    Examples: '' -> 0, 'A' -> 1, 'AAa' -> 1, '-3' -> -3.
    """
    raw = text.strip()
    if raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        pass
    total = 0
    cleaned = []
    for ch in raw:
        if ch == "A":
            total += 1
            cleaned.append(ch)
        elif ch == "a":
            total -= 1
            cleaned.append(ch)
    return total


def exponent_to_word(n: int) -> str:
    if n > 0:
        return "A" * n
    if n < 0:
        return "a" * (-n)
    return ""


def cyclic_right_boundary_arc(a: float, samples: int = 260) -> np.ndarray:
    """Right boundary of the Dirichlet fundamental strip for <A> at 0.

    A(z)=(z+a)/(az+1).  The right boundary is the perpendicular bisector of
    0 and A(0)=a.  It is the disk geodesic whose ideal endpoints are

        a +/- i sqrt(1-a^2).

    It is represented by the Euclidean circle centered at 1/a with radius
    sqrt(1/a^2 - 1), and we sample the arc lying inside the unit disk.
    """
    a = float(a)
    if not (0.0 < a < 1.0):
        return np.empty((0, 2), dtype=float)
    c = 1.0 / a
    R = math.sqrt(1.0 / (a * a) - 1.0)
    s = math.sqrt(max(0.0, 1.0 - a * a))
    top_angle = math.atan2(s, a - c)
    bottom_angle = math.atan2(-s, a - c)
    delta = (bottom_angle - top_angle) % (2.0 * math.pi)
    theta = top_angle + np.linspace(0.0, delta, samples)
    arc = np.column_stack([c + R * np.cos(theta), R * np.sin(theta)])
    # Nudge exact ideal endpoints slightly inside for clean plotting.
    r = np.linalg.norm(arc, axis=1)
    mask = r >= 0.999999
    if np.any(mask):
        arc[mask] *= (0.999999 / r[mask])[:, None]
    return arc


def cyclic_strip_boundaries(a: float, depth: int, samples: int = 260) -> List[Tuple[int, np.ndarray]]:
    """Return boundaries A^k(B_0) of the cyclic Dirichlet strip tessellation."""
    base = cyclic_right_boundary_arc(a, samples=samples)
    if base.size == 0:
        return []
    out: List[Tuple[int, np.ndarray]] = []
    for k in range(-depth - 1, depth + 1):
        g = cyclic_power(a, k)
        out.append((k, g.apply_curve(base)))
    return out


# -----------------------------------------------------------------------------
# Certified surface interface
# -----------------------------------------------------------------------------


def surface_signature(surface: "FuchsianSurface") -> str:
    """Stable-ish signature used for tessellation cache invalidation."""
    parts = [surface.__class__.__name__, getattr(surface, "key", "")]
    for attr in ("a", "theta_a_deg", "b", "theta_b_deg", "name", "rho"):
        if hasattr(surface, attr):
            parts.append(f"{attr}={getattr(surface, attr)}")
    if hasattr(surface, "polygon_vertices"):
        try:
            verts = np.asarray(surface.polygon_vertices, dtype=float)
            parts.append("verts=" + np.array2string(verts, precision=7, separator=","))
        except Exception:
            pass
    if hasattr(surface, "side_pairings"):
        parts.append("pairs=" + repr(getattr(surface, "side_pairings")))
    if hasattr(surface, "geodesic_sides"):
        parts.append("ideal_sides=" + repr(getattr(surface, "geodesic_sides")))
    if hasattr(surface, "generator_specs"):
        parts.append("genspec=" + repr(getattr(surface, "generator_specs")))
    return "|".join(parts)

class FuchsianSurface:
    """Minimal surface interface for the moderate-arbitrary direction.

    The GUI assumes supplied surface data are already Fuchsian/certified.  This
    class is not a discreteness prover and does not attempt to discover a
    fundamental domain from arbitrary generators.  Concrete subclasses provide
    the group action, lifted endpoint convention, and any verified fundamental
    boundaries they can display.
    """

    key = "abstract"
    display_name = "Abstract Fuchsian surface"
    supports_exponent = False
    supports_orbit_search = False
    supports_strip_boundaries = False

    def gamma(self, n: int = 0) -> DiskMobius:
        return DiskMobius.identity()

    def lifted_endpoint(self, q: np.ndarray, n: int = 0) -> np.ndarray:
        return self.gamma(n).apply_point(q)

    def lift_label(self, n: int = 0) -> str:
        return "q"

    def orbit_points(self, q: Optional[np.ndarray], depth: int) -> List[Tuple[int, np.ndarray]]:
        return []

    def strip_edges(self, depth: int, samples: int = 260) -> List[Tuple[int, np.ndarray]]:
        return []

    def description(self) -> str:
        return "Abstract Fuchsian surface interface."

    def diagnostics(self) -> List[Tuple[str, str]]:
        return [("surface interface", self.__class__.__name__)]


class TrivialDiskSurface(FuchsianSurface):
    key = "disk"
    display_name = "Exact Poincare disk / trivial Gamma"

    def description(self) -> str:
        return (
            "Exact Poincare disk mode. Gamma is trivial, so q_lift=q. "
            "All distances, geodesics, and metric benchmarks are exact disk quantities."
        )

    def diagnostics(self) -> List[Tuple[str, str]]:
        return [
            ("surface interface", self.__class__.__name__),
            ("Gamma", "trivial"),
            ("fundamental domain", "whole disk"),
        ]


class CyclicQuotientSurface(FuchsianSurface):
    key = "cyclic"
    display_name = "Certified cyclic quotient Gamma = <A>"
    supports_exponent = True
    supports_orbit_search = True
    supports_strip_boundaries = True

    def __init__(self, a: float):
        self.a = float(a)

    @property
    def translation_length(self) -> float:
        return 2.0 * math.atanh(self.a)

    def gamma(self, n: int = 0) -> DiskMobius:
        return cyclic_power(self.a, int(n))

    def lift_label(self, n: int = 0) -> str:
        return f"A^{int(n)}(q)"

    def orbit_points(self, q: Optional[np.ndarray], depth: int) -> List[Tuple[int, np.ndarray]]:
        if q is None:
            return []
        out: List[Tuple[int, np.ndarray]] = []
        for n in range(-int(depth), int(depth) + 1):
            pt = self.gamma(n).apply_point(q)
            if inside_unit_disk(pt, margin=1.0e-8):
                out.append((n, pt))
        return out

    def strip_edges(self, depth: int, samples: int = 260) -> List[Tuple[int, np.ndarray]]:
        return cyclic_strip_boundaries(self.a, depth=int(depth), samples=samples)

    def description(self) -> str:
        return (
            "Certified cyclic Fuchsian quotient.\n\n"
            f"A(z)=(z+a)/(a z+1), 0<a<1, with a={self.a:.3f}. "
            "The generator translates along the horizontal diameter and has "
            f"translation length 2 atanh(a) = {self.translation_length:.6f}.\n\n"
            "The strip boundaries shown in the disk are Dirichlet-strip boundaries "
            "generated from this same Gamma. A selected path from [p] to [q] is "
            "lifted as p -> A^n(q)."
        )

    def diagnostics(self) -> List[Tuple[str, str]]:
        return [
            ("surface interface", self.__class__.__name__),
            ("generator", f"A(z) = (z+a)/(a z+1),  a = {self.a:.6f}"),
            ("translation length", f"{self.translation_length:.9f}"),
            ("fundamental domain", "Dirichlet strip for basepoint 0"),
        ]




class TwoGeneratorByFiatSurface(FuchsianSurface):
    """Moderate-arbitrary Fuchsian mode with two supplied disk isometries.

    The app does not prove discreteness or compute a fundamental polygon.  The
    user supplies A and B as honest orientation-preserving disk isometries, and
    the surface is treated by fiat as a Fuchsian quotient model.  This supports
    rigorous lifted-endpoint/orbit/geodesic calculations for the supplied group
    action, but it does not yet draw a certified tessellation unless future
    versions also supply a fundamental polygon and side-pairing data.
    """

    key = "twogen"
    display_name = "By-fiat two-generator Fuchsian group <A,B>"
    supports_word = True
    supports_orbit_search = True
    supports_strip_boundaries = False

    def __init__(self, a: float, theta_a_deg: float, b: float, theta_b_deg: float):
        self.a = float(a)
        self.theta_a_deg = float(theta_a_deg)
        self.b = float(b)
        self.theta_b_deg = float(theta_b_deg)
        self.theta_a = math.radians(self.theta_a_deg)
        self.theta_b = math.radians(self.theta_b_deg)
        self.A = axial_translation(self.a, self.theta_a, name="A")
        self.B = axial_translation(self.b, self.theta_b, name="B")
        self.generators = {
            "A": self.A,
            "a": self.A.inverse(name="a"),
            "B": self.B,
            "b": self.B.inverse(name="b"),
        }

    @property
    def length_A(self) -> float:
        return 2.0 * math.atanh(self.a)

    @property
    def length_B(self) -> float:
        return 2.0 * math.atanh(self.b)

    def gamma(self, word: str = "") -> DiskMobius:
        word = reduce_fuchsian_word(str(word))
        g = DiskMobius.identity()
        # GUI convention: read left-to-right as successive actions on q.
        # Thus word AB sends q -> B(A(q)).
        for ch in word:
            g = self.generators[ch].compose(g, name=word)
        return g.normalized()

    def lift_label(self, word: str = "") -> str:
        word = reduce_fuchsian_word(str(word))
        return "q" if word == "" else f"{word}(q)"

    def orbit_points(self, q: Optional[np.ndarray], depth: int) -> List[Tuple[str, np.ndarray]]:
        if q is None:
            return []
        out: List[Tuple[str, np.ndarray]] = []
        for word in generate_reduced_words(int(depth)):
            pt = self.gamma(word).apply_point(q)
            if inside_unit_disk(pt, margin=1.0e-8):
                out.append((word, pt))
        return out

    def description(self) -> str:
        return (
            "By-fiat two-generator Fuchsian mode.\n\n"
            "The generators A and B are constructed as hyperbolic disk isometries, each a "
            "translation along a chosen diameter. The program assumes the generated group "
            "is the Fuchsian group you intend to study. It does not prove discreteness, "
            "compute a fundamental polygon, or certify side pairings.\n\n"
            "Words use A,B for generators and a,b for inverses. The GUI convention is "
            "left-to-right action: word AB means q -> A(q) -> B(A(q)). This mode is "
            "rigorous for lifted orbit points and geodesics of the supplied disk isometries, "
            "but tessellation/fundamental-domain display is intentionally absent until "
            "polygon data are supplied."
        )

    def diagnostics(self) -> List[Tuple[str, str]]:
        return [
            ("surface interface", self.__class__.__name__),
            ("generator A", f"translation a={self.a:.6f}, axis angle={self.theta_a_deg:.3f} deg"),
            ("A translation length", f"{self.length_A:.9f}"),
            ("generator B", f"translation b={self.b:.6f}, axis angle={self.theta_b_deg:.3f} deg"),
            ("B translation length", f"{self.length_B:.9f}"),
            ("fundamental domain", "not supplied in this mode; no certified tessellation"),
            ("certification status", "group taken by fiat to be Fuchsian"),
        ]



def _complex_from_json_pair(value) -> complex:
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    if isinstance(value, str):
        return complex(value.replace("i", "j"))
    if isinstance(value, Sequence) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    raise ValueError(f"Cannot parse complex value {value!r}")


def disk_mobius_from_generator_spec(name: str, spec: dict) -> DiskMobius:
    """Build a disk isometry from a JSON generator spec.

    Supported forms:
        {"type":"axial", "strength":0.45, "angle_deg":65}
        {"alpha":[re,im], "beta":[re,im]}

    The second form is normalized to |alpha|^2-|beta|^2=1.  The app assumes
    the user has supplied a genuine disk isometry; normalization is only a
    numerical cleanup, not a certification of a Fuchsian group.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Generator {name} must be an object.")
    kind = str(spec.get("type", "")).lower().strip()
    if kind in ("axial", "translation", "hyperbolic") or "strength" in spec:
        strength = float(spec.get("strength", spec.get("a", 0.4)))
        angle_deg = float(spec.get("angle_deg", spec.get("angle", 0.0)))
        if not (0.0 < abs(strength) < 1.0):
            raise ValueError(f"Generator {name}: axial strength must satisfy 0<|strength|<1.")
        return axial_translation(strength, math.radians(angle_deg), name=name)
    if "alpha" in spec and "beta" in spec:
        alpha = _complex_from_json_pair(spec["alpha"])
        beta = _complex_from_json_pair(spec["beta"])
        g = DiskMobius(alpha, beta, name=name).normalized()
        det = abs(g.alpha) ** 2 - abs(g.beta) ** 2
        if det <= 0.0:
            raise ValueError(f"Generator {name}: |alpha|^2-|beta|^2 must be positive.")
        return g
    raise ValueError(f"Generator {name} must have either axial strength/angle or alpha/beta.")


class RegularOctagonGenus2Surface(FuchsianSurface):
    """Certified compact genus-2 surface from a regular hyperbolic octagon.

    The fundamental polygon is the regular {8,8} octagon in the Poincare disk:
    its interior angles are pi/4, so eight polygon corners around the single
    quotient vertex sum to 2*pi.  Opposite sides are paired by the unique
    orientation-preserving disk isometries sending each side to its opposite
    with reversed boundary orientation.  This is the standard closed genus-2
    octagon construction.
    """

    key = "octagon_g2"
    display_name = "Certified regular octagon genus-2 surface"
    supports_word = True
    supports_orbit_search = True
    supports_strip_boundaries = True

    def __init__(self):
        self.name = "Regular {8,8} octagon, opposite-side pairings, genus 2"
        self.rho = regular_hyperbolic_polygon_radius(8, 8)
        # Rotate by pi/8 so the sides are not exactly vertical/horizontal in the
        # initial view; this also makes the side labels easier to see.
        self.angles = np.array([math.pi / 8.0 + k * math.pi / 4.0 for k in range(8)], dtype=float)
        self.polygon_vertices = np.column_stack([self.rho * np.cos(self.angles), self.rho * np.sin(self.angles)])
        self.generators = {}
        names = ["A", "B", "C", "D"]
        self.side_pairings = []
        verts = self.polygon_vertices
        for i, name in enumerate(names):
            j = i + 4
            # Side i is v_i -> v_{i+1}.  The paired opposite side is j, but the
            # orientation must reverse: v_i -> v_{j+1}, v_{i+1} -> v_j.
            z1 = complex_from_point(verts[i])
            z2 = complex_from_point(verts[(i + 1) % 8])
            w1 = complex_from_point(verts[(j + 1) % 8])
            w2 = complex_from_point(verts[j % 8])
            g = disk_isometry_from_two_point_pairs(z1, z2, w1, w2, name=name)
            self.generators[name] = g
            self.generators[name.lower()] = g.inverse(name=name.lower())
            self.side_pairings.append({"side": i, "paired_with": j, "word": name})

    def available_alphabet(self) -> str:
        return "ABCDabcd"

    def gamma(self, word: str = "") -> DiskMobius:
        word = reduce_fuchsian_word(str(word))
        g = DiskMobius.identity()
        for ch in word:
            if ch not in self.generators:
                raise ValueError(f"Word contains generator {ch!r}, which is not available for the octagon surface.")
            g = self.generators[ch].compose(g, name=word)
        return g.normalized()

    def lift_label(self, word: str = "") -> str:
        word = reduce_fuchsian_word(str(word))
        return "q" if word == "" else f"{word}(q)"

    def orbit_points(self, q: Optional[np.ndarray], depth: int) -> List[Tuple[str, np.ndarray]]:
        if q is None:
            return []
        # The 8-letter alphabet grows fast.  Keep the GUI responsive.
        depth = min(int(depth), 4)
        out: List[Tuple[str, np.ndarray]] = []
        for word in generate_reduced_words(depth, alphabet=self.available_alphabet()):
            pt = self.gamma(word).apply_point(q)
            if inside_unit_disk(pt, margin=1.0e-8):
                out.append((word, pt))
        return out

    def polygon_edge_curves(self, samples_per_edge: int = 72) -> List[np.ndarray]:
        edges = []
        verts = self.polygon_vertices
        for i in range(8):
            edges.append(exact_poincare_geodesic(verts[i], verts[(i + 1) % 8], n=samples_per_edge))
        return edges

    def strip_edges(self, depth: int, samples: int = 260) -> List[Tuple[str, np.ndarray]]:
        depth = min(int(depth), 4)
        edges = self.polygon_edge_curves(samples_per_edge=max(36, samples // 7))
        out: List[Tuple[str, np.ndarray]] = []
        for word in generate_reduced_words(depth, alphabet=self.available_alphabet()):
            g = self.gamma(word)
            for edge in edges:
                out.append((word, g.apply_curve(edge)))
        return out

    def description(self) -> str:
        return (
            "Certified compact genus-2 octagon surface.\n\n"
            "The fundamental domain F is a regular hyperbolic octagon with interior "
            "angle pi/4, realized in the Poincare disk as a regular {8,8} polygon. "
            "Opposite sides are paired by the unique orientation-preserving disk "
            "isometries that map each side to its opposite with reversed side orientation.\n\n"
            "The displayed finite word patch is gamma(F), generated from these "
            "side-pairing maps.  This is the first built-in compact Fuchsian surface "
            "in the app.  Words may use A,B,C,D and a,b,c,d for inverses."
        )

    def diagnostics(self) -> List[Tuple[str, str]]:
        pair_summary = [f"{sp['side']}<->{sp['paired_with']} by {sp['word']}" for sp in self.side_pairings]
        area = 4.0 * math.pi
        return [
            ("surface interface", self.__class__.__name__),
            ("surface name", self.name),
            ("fundamental polygon", "regular {8,8} octagon; interior angle pi/4"),
            ("Euclidean vertex radius", f"{self.rho:.9f}"),
            ("side pairings", "; ".join(pair_summary)),
            ("quotient genus", "2"),
            ("Gauss-Bonnet area", f"4*pi = {area:.9f}"),
            ("certification status", "built-in regular octagon model"),
        ]


class UserPolygonFuchsianSurface(FuchsianSurface):
    """User-supplied moderate-arbitrary Fuchsian surface.

    This is an advanced by-fiat model.  The app does not discover or certify the
    group.  Instead, the user supplies the required quotient data by fiat:

        generators A,B as disk isometries,
        a fundamental polygon F as vertices in the disk,
        side-pairing metadata.

    Once F is supplied, the displayed finite word patch is not decorative: it is the
    orbit gamma(F) over the finite word set chosen by the depth control.  The
    finite patch is of course only a displayed patch, not the whole infinite
    tessellation.
    """

    key = "polygon"
    display_name = "User-supplied polygon Fuchsian surface"
    supports_word = True
    supports_orbit_search = True
    supports_strip_boundaries = True

    def __init__(self, data: Optional[dict] = None):
        self.data = data or {}
        self.name = str(self.data.get("name", "User-supplied Fuchsian surface"))
        self.generators = {}
        self.generator_specs = self.data.get("generators", {}) if isinstance(self.data.get("generators", {}), dict) else {}
        # v12.2 generator-ready JSON: accept arbitrary one-letter uppercase
        # generator labels A, B, C, ... . Lowercase inverses are created
        # automatically.  This lets the regular genus-g generator use A through
        # the required 2g-th letter without changing the Explorer.
        for name in sorted(self.generator_specs.keys()):
            if isinstance(name, str) and len(name) == 1 and name.isalpha() and name.isupper():
                self.generators[name] = disk_mobius_from_generator_spec(name, self.generator_specs[name])
                self.generators[name.lower()] = self.generators[name].inverse(name=name.lower())
        self.domain_type = str(self.data.get("domain_type", "compact_polygon"))
        self.v12_polygon_compatible = bool(self.data.get("v12_polygon_compatible", self.domain_type in ("compact_polygon", "polygon")))
        cert = self.data.get("certification", {})
        self.certification = cert if isinstance(cert, dict) else {"status": str(cert)}
        verts = self.data.get("polygon_vertices", [])
        self.polygon_vertices = np.array(verts, dtype=float) if len(verts) else np.empty((0, 2), dtype=float)
        if self.polygon_vertices.size and (self.polygon_vertices.ndim != 2 or self.polygon_vertices.shape[1] != 2):
            raise ValueError("polygon_vertices must be a list of [x,y] pairs.")
        for v in self.polygon_vertices:
            if not inside_unit_disk(v, margin=1.0e-7):
                raise ValueError(f"Polygon vertex {v.tolist()} is not strictly inside the unit disk.")
        self.side_pairings = self.data.get("side_pairings", []) if isinstance(self.data.get("side_pairings", []), list) else []
        self.notes = str(self.data.get("notes", ""))

    @classmethod
    def empty(cls) -> "UserPolygonFuchsianSurface":
        return cls({"name": "No user polygon loaded", "generators": {}, "polygon_vertices": [], "side_pairings": []})

    @classmethod
    def from_json_text(cls, text: str) -> "UserPolygonFuchsianSurface":
        raw = text.strip()
        if not raw:
            raise ValueError("No JSON surface data supplied.")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Surface JSON must be an object.")
        surf = cls(data)
        if not surf.v12_polygon_compatible or surf.domain_type not in ("compact_polygon", "polygon"):
            raise ValueError(
                "This JSON is valid-looking FuchsianGENN data, but it is not a v12 compact-polygon surface. "
                f"domain_type={surf.domain_type!r}, v12_polygon_compatible={surf.v12_polygon_compatible!r}. "
                "It needs a matching Explorer mode such as Schottky, triangle-orbifold, or Ford-domain mode."
            )
        upper_generators = [ch for ch in surf.generators if len(ch) == 1 and ch.isupper()]
        if len(upper_generators) == 0:
            raise ValueError("advanced user-polygon mode requires at least one one-letter uppercase generator, e.g. A, B, C, ...")
        if len(surf.polygon_vertices) < 3:
            raise ValueError("advanced user-polygon mode requires polygon_vertices with at least three vertices.")
        return surf

    def available_alphabet(self) -> str:
        uppers = sorted(ch for ch in self.generators if len(ch) == 1 and ch.isupper())
        return "".join(uppers + [ch.lower() for ch in uppers if ch.lower() in self.generators])

    def gamma(self, word: str = "") -> DiskMobius:
        word = reduce_fuchsian_word(str(word))
        g = DiskMobius.identity()
        for ch in word:
            if ch not in self.generators:
                raise ValueError(f"Word contains generator {ch!r}, which is not supplied.")
            g = self.generators[ch].compose(g, name=word)
        return g.normalized()

    def lift_label(self, word: str = "") -> str:
        word = reduce_fuchsian_word(str(word))
        return "q" if word == "" else f"{word}(q)"

    def orbit_points(self, q: Optional[np.ndarray], depth: int) -> List[Tuple[str, np.ndarray]]:
        if q is None or not self.generators:
            return []
        alphabet = self.available_alphabet()
        if not alphabet:
            return []
        out: List[Tuple[str, np.ndarray]] = []
        for word in generate_reduced_words(int(depth), alphabet=alphabet):
            pt = self.gamma(word).apply_point(q)
            if inside_unit_disk(pt, margin=1.0e-8):
                out.append((word, pt))
        return out

    def polygon_edge_curves(self, samples_per_edge: int = 60) -> List[np.ndarray]:
        if len(self.polygon_vertices) < 3:
            return []
        edges = []
        verts = self.polygon_vertices
        for i in range(len(verts)):
            p = verts[i]
            q = verts[(i + 1) % len(verts)]
            edges.append(exact_poincare_geodesic(p, q, n=samples_per_edge))
        return edges

    def strip_edges(self, depth: int, samples: int = 260) -> List[Tuple[str, np.ndarray]]:
        if len(self.polygon_vertices) < 3 or not self.generators:
            return []
        alphabet = self.available_alphabet()
        edges = self.polygon_edge_curves(samples_per_edge=max(24, samples // 8))
        out: List[Tuple[str, np.ndarray]] = []
        for word in generate_reduced_words(int(depth), alphabet=alphabet):
            g = self.gamma(word)
            for edge in edges:
                out.append((word, g.apply_curve(edge)))
        return out

    def description(self) -> str:
        return (
            "User-supplied polygon Fuchsian mode.\n\n"
            "This is the moderate-arbitrary mode: the program assumes, by fiat, that "
            "the supplied generators, fundamental polygon F, and side-pairing data "
            "are already valid Fuchsian quotient data. The app does not prove "
            "discreteness or verify that the polygon is truly fundamental.\n\n"
            "Because F is supplied, the app can draw a finite word patch "
            "gamma(F) using the selected word-depth control. Geodesic calculations "
            "are performed upstairs in the Poincare disk from p to word(q)."
        )

    def diagnostics(self) -> List[Tuple[str, str]]:
        pair_summary = []
        for sp in self.side_pairings:
            if isinstance(sp, dict):
                pair_summary.append(f"{sp.get('side')}<->{sp.get('paired_with')} by {sp.get('word')}")
        return [
            ("surface interface", self.__class__.__name__),
            ("surface name", self.name),
            ("generators supplied", ", ".join(sorted(k for k in self.generators if len(k) == 1 and k.isupper())) or "none"),
            ("fundamental polygon", f"{len(self.polygon_vertices)} vertices supplied" if len(self.polygon_vertices) else "not supplied"),
            ("side pairings", "; ".join(pair_summary[:4]) + ("; ..." if len(pair_summary) > 4 else "") if pair_summary else "not supplied"),
            ("domain type", self.domain_type),
            ("JSON certification", str(self.certification.get("status", "not supplied"))),
            ("certification status", "externally certified by metadata; Explorer consumes, does not prove" if self.certification.get("status") else "user-supplied by fiat; not verified by program"),
        ]




class SchottkyIdealGeodesicSurface(FuchsianSurface):
    """Schottky-style domain bounded by paired ideal geodesics.

    This mode consumes Domain Maker JSON with domain_type
    "schottky_ideal_geodesic_domain".  The data are not a compact finite-vertex
    polygon.  Instead, the displayed domain is bounded by ideal geodesic arcs
    whose endpoints lie on the unit circle.  The Explorer assumes the supplied
    data are certified by construction; it does not prove discreteness or
    disjointness from scratch.
    """

    key = "schottky"
    display_name = "Schottky ideal-geodesic-domain surface"
    supports_word = True
    supports_orbit_search = True
    supports_strip_boundaries = True

    def __init__(self, data: Optional[dict] = None):
        self.data = data or {}
        self.name = str(self.data.get("name", "Schottky ideal-geodesic domain"))
        self.domain_type = str(self.data.get("domain_type", "schottky_ideal_geodesic_domain"))
        cert = self.data.get("certification", {})
        self.certification = cert if isinstance(cert, dict) else {"status": str(cert)}
        self.generator_specs = self.data.get("generators", {}) if isinstance(self.data.get("generators", {}), dict) else {}
        self.generators = {}
        for name in sorted(self.generator_specs.keys()):
            if isinstance(name, str) and len(name) == 1 and name.isalpha() and name.isupper():
                self.generators[name] = disk_mobius_from_generator_spec(name, self.generator_specs[name])
                self.generators[name.lower()] = self.generators[name].inverse(name=name.lower())
        raw_sides = self.data.get("geodesic_sides", [])
        self.geodesic_sides = raw_sides if isinstance(raw_sides, list) else []
        self.side_pairings = self.data.get("side_pairings", []) if isinstance(self.data.get("side_pairings", []), list) else []
        if not self.generators:
            raise ValueError("Schottky JSON requires at least one one-letter uppercase generator.")
        if not self.geodesic_sides:
            raise ValueError("Schottky JSON requires geodesic_sides with ideal endpoints.")

    @classmethod
    def from_json_text(cls, text: str) -> "SchottkyIdealGeodesicSurface":
        raw = text.strip()
        if not raw:
            raise ValueError("No JSON surface data supplied.")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Surface JSON must be an object.")
        if str(data.get("domain_type", "")) != "schottky_ideal_geodesic_domain":
            raise ValueError("This is not a Schottky ideal-geodesic-domain JSON file.")
        return cls(data)

    def available_alphabet(self) -> str:
        uppers = sorted(ch for ch in self.generators if len(ch) == 1 and ch.isupper())
        return "".join(uppers + [ch.lower() for ch in uppers if ch.lower() in self.generators])

    def gamma(self, word: str = "") -> DiskMobius:
        word = reduce_fuchsian_word(str(word))
        g = DiskMobius.identity()
        for ch in word:
            if ch not in self.generators:
                raise ValueError(f"Word contains generator {ch!r}, which is not supplied.")
            g = self.generators[ch].compose(g, name=word)
        return g.normalized()

    def lift_label(self, word: str = "") -> str:
        word = reduce_fuchsian_word(str(word))
        return "q" if word == "" else f"{word}(q)"

    def orbit_points(self, q: Optional[np.ndarray], depth: int) -> List[Tuple[str, np.ndarray]]:
        if q is None or not self.generators:
            return []
        alphabet = self.available_alphabet()
        out: List[Tuple[str, np.ndarray]] = []
        for word in generate_reduced_words(int(depth), alphabet=alphabet):
            pt = self.gamma(word).apply_point(q)
            if inside_unit_disk(pt, margin=1.0e-8):
                out.append((word, pt))
        return out

    def base_geodesic_edges(self, samples_per_edge: int = 100) -> List[np.ndarray]:
        edges = []
        for side in self.geodesic_sides:
            try:
                endpoints = side.get("ideal_endpoints", [])
                if len(endpoints) != 2:
                    continue
                u = np.asarray(endpoints[0], dtype=float)
                v = np.asarray(endpoints[1], dtype=float)
                edges.append(ideal_geodesic_arc(u, v, n=samples_per_edge))
            except Exception:
                continue
        return [e for e in edges if len(e) >= 2]

    def strip_edges(self, depth: int, samples: int = 260) -> List[Tuple[str, np.ndarray]]:
        if not self.generators:
            return []
        alphabet = self.available_alphabet()
        edges = self.base_geodesic_edges(samples_per_edge=max(36, samples // 3))
        out: List[Tuple[str, np.ndarray]] = []
        for word in generate_reduced_words(int(depth), alphabet=alphabet):
            g = self.gamma(word)
            for edge in edges:
                out.append((word, g.apply_curve(edge)))
        return out

    def description(self) -> str:
        return (
            "Schottky ideal-geodesic-domain mode.\n\n"
            "This mode displays a domain bounded by paired ideal geodesics, with endpoints "
            "on the unit circle.  It is not a compact finite-vertex polygon, so it needs its "
            "own Explorer interpretation rather than the compact-polygon mode.  The supplied "
            "JSON is taken by fiat/certification metadata as valid Schottky-type Fuchsian data."
        )

    def diagnostics(self) -> List[Tuple[str, str]]:
        pair_summary = []
        for sp in self.side_pairings:
            if isinstance(sp, dict):
                pair_summary.append(f"{sp.get('side')}<->{sp.get('paired_with')} by {sp.get('word')}")
        return [
            ("surface interface", self.__class__.__name__),
            ("surface name", self.name),
            ("domain type", self.domain_type),
            ("generators supplied", ", ".join(sorted(k for k in self.generators if len(k) == 1 and k.isupper())) or "none"),
            ("ideal geodesic sides", f"{len(self.geodesic_sides)} supplied"),
            ("side pairings", "; ".join(pair_summary[:4]) + ("; ..." if len(pair_summary) > 4 else "") if pair_summary else "not supplied"),
            ("JSON certification", str(self.certification.get("status", "not supplied"))),
            ("certification status", "externally certified/by construction metadata; Explorer consumes, does not prove" if self.certification.get("status") else "by fiat; not verified by program"),
        ]



class ModularFordSurface(FuchsianSurface):
    """Classical PSL(2,Z) Ford-domain orbifold mode.

    This mode consumes Domain Maker v6 JSON with domain_type
    "modular_ford_domain".  The domain is the classical finite-area Ford
    fundamental region for PSL(2,Z), transported from the upper half-plane to
    the disk by the Cayley map.  It is an orbifold/domain seed with cusp and
    elliptic points, not a smooth compact Riemann surface.
    """

    key = "modular_ford"
    display_name = "Modular/Ford PSL(2,Z) orbifold domain"
    supports_word = True
    supports_orbit_search = True
    supports_strip_boundaries = True

    def __init__(self, data: Optional[dict] = None):
        self.data = data or {}
        self.name = str(self.data.get("name", "Modular/Ford domain seed"))
        # User-facing label is dynamic: PSL(2,Z) is an orbifold, but torsion-free
        # congruence subgroups are smooth noncompact Riemann-surface domains.
        subgroup_label = str(self.data.get("subgroup", "") or "")
        torsion_free_label = self.data.get("torsion_free", None)
        is_hecke = bool(self.data.get("hecke_q") is not None or "hecke" in str(self.data.get("category", "")).lower() or "Hecke" in str(self.data.get("parent_group", "")))
        if torsion_free_label is True and subgroup_label:
            prefix = "Hecke/Ford" if is_hecke else "Modular/Ford"
            self.display_name = f"{prefix} {subgroup_label} torsion-free Riemann-surface domain"
        elif self.data.get("elliptic_orders"):
            self.display_name = "Hecke/Ford orbifold domain" if is_hecke else "Modular/Ford PSL(2,Z) orbifold domain"
        else:
            self.display_name = "Hecke/Ford finite-area domain" if is_hecke else "Modular/Ford finite-area domain"
        self.domain_type = str(self.data.get("domain_type", "modular_ford_domain"))
        cert = self.data.get("certification", {})
        self.certification = cert if isinstance(cert, dict) else {"status": str(cert)}
        self.generator_specs = self.data.get("generators", {}) if isinstance(self.data.get("generators", {}), dict) else {}
        self.generators = {}
        for name in sorted(self.generator_specs.keys()):
            if isinstance(name, str) and len(name) == 1 and name.isalpha() and name.isupper():
                self.generators[name] = disk_mobius_from_generator_spec(name, self.generator_specs[name])
                self.generators[name.lower()] = self.generators[name].inverse(name=name.lower())
        self.generator_meanings = self.data.get("generator_meanings", {}) if isinstance(self.data.get("generator_meanings", {}), dict) else {}
        raw_sides = self.data.get("ford_sides", [])
        self.ford_sides = raw_sides if isinstance(raw_sides, list) else []
        self.side_pairings = self.data.get("side_pairings", []) if isinstance(self.data.get("side_pairings", []), list) else []
        self.ford_vertices = np.asarray(self.data.get("ford_vertices", []), dtype=float) if isinstance(self.data.get("ford_vertices", []), list) else np.empty((0,2))
        if not self.generators:
            raise ValueError("Modular/Ford JSON requires one-letter uppercase generator specs, usually A=T and B=S.")
        if not self.ford_sides:
            raise ValueError("Modular/Ford JSON requires ford_sides with segment endpoints.")

    @classmethod
    def from_json_text(cls, text: str) -> "ModularFordSurface":
        raw = text.strip()
        if not raw:
            raise ValueError("No JSON surface data supplied.")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Surface JSON must be an object.")
        if str(data.get("domain_type", "")) != "modular_ford_domain":
            raise ValueError("This is not a modular_ford_domain JSON file.")
        return cls(data)

    def available_alphabet(self) -> str:
        uppers = sorted(ch for ch in self.generators if len(ch) == 1 and ch.isupper())
        return "".join(uppers + [ch.lower() for ch in uppers if ch.lower() in self.generators])

    def gamma(self, word: str = "") -> DiskMobius:
        word = reduce_fuchsian_word(str(word))
        g = DiskMobius.identity()
        for ch in word:
            if ch not in self.generators:
                raise ValueError(f"Word contains generator {ch!r}, which is not supplied.")
            g = self.generators[ch].compose(g, name=word)
        return g.normalized()

    def lift_label(self, word: str = "") -> str:
        word = reduce_fuchsian_word(str(word))
        return "q" if word == "" else f"{word}(q)"

    def orbit_points(self, q: Optional[np.ndarray], depth: int) -> List[Tuple[str, np.ndarray]]:
        if q is None or not self.generators:
            return []
        alphabet = self.available_alphabet()
        out: List[Tuple[str, np.ndarray]] = []
        for word in generate_reduced_words(int(depth), alphabet=alphabet):
            try:
                pt = self.gamma(word).apply_point(q)
                if inside_unit_disk(pt, margin=1.0e-8):
                    out.append((word, pt))
            except Exception:
                continue
        return out

    def _segment_curve(self, endpoints: list, n: int = 140) -> np.ndarray:
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            return np.empty((0, 2), dtype=float)
        a = np.asarray(endpoints[0], dtype=float).copy()
        b = np.asarray(endpoints[1], dtype=float).copy()
        for pt in (a, b):
            rr = float(np.linalg.norm(pt))
            if rr >= 0.999999:
                pt *= 0.999999 / rr
        return exact_poincare_geodesic(a, b, n=max(16, int(n)))

    def base_geodesic_edges(self, samples_per_edge: int = 100) -> List[np.ndarray]:
        edges = []
        for side in self.ford_sides:
            try:
                endpoints = side.get("segment_endpoints", [])
                curve = self._segment_curve(endpoints, n=samples_per_edge)
                if len(curve) >= 2:
                    edges.append(curve)
            except Exception:
                continue
        return edges

    def strip_edges(self, depth: int, samples: int = 260) -> List[Tuple[str, np.ndarray]]:
        alphabet = self.available_alphabet()
        base_edges = self.base_geodesic_edges(samples_per_edge=max(36, samples // 3))
        out: List[Tuple[str, np.ndarray]] = []
        for word in generate_reduced_words(int(depth), alphabet=alphabet):
            try:
                g = self.gamma(word)
                for edge in base_edges:
                    out.append((word, g.apply_curve(edge)))
            except Exception:
                continue
        return out

    def modular_metadata(self) -> dict:
        """Return a compact modular-surface metadata dictionary for audit/fingerprint output."""
        data = self.data if isinstance(self.data, dict) else {}
        comp = data.get("compactification", {}) if isinstance(data.get("compactification", {}), dict) else {}
        audit = data.get("torsion_free_audit", {}) if isinstance(data.get("torsion_free_audit", {}), dict) else {}
        cleanup = data.get("boundary_cleanup_audit", {}) if isinstance(data.get("boundary_cleanup_audit", {}), dict) else {}
        return {
            "subdomain_type": data.get("subdomain_type"),
            "category": data.get("category"),
            "parent_group": data.get("parent_group"),
            "subgroup": data.get("subgroup"),
            "level_N": data.get("level_N"),
            "hecke_q": data.get("hecke_q"),
            "hecke_lambda": data.get("lambda"),
            "index_in_hecke_group": data.get("index_in_hecke_group"),
            "torsion_free": data.get("torsion_free"),
            "compact": data.get("compact"),
            "index_in_psl2z": data.get("index_in_psl2z"),
            "area": data.get("area"),
            "cusp_count": data.get("cusp_count"),
            "cusp_widths": data.get("cusp_widths"),
            "elliptic_orders": data.get("elliptic_orders", data.get("elliptic_points", None)),
            "compactification": comp,
            "compactified_genus": comp.get("compactified_genus"),
            "riemann_surface_status": data.get("riemann_surface_status"),
            "kahler_status": data.get("kahler_status"),
            "mathematical_object": data.get("mathematical_object"),
            "upper_half_plane_domain": data.get("upper_half_plane_domain"),
            "torsion_free_audit": audit,
            "boundary_cleanup_audit": cleanup,
            "construction_tile_count": len(data.get("fundamental_domain_tiles", [])) if isinstance(data.get("fundamental_domain_tiles", []), list) else None,
            "construction_edge_count": len(data.get("construction_ford_sides", [])) if isinstance(data.get("construction_ford_sides", []), list) else None,
            "internal_edge_count": len(data.get("internal_ford_sides", [])) if isinstance(data.get("internal_ford_sides", []), list) else None,
            "exterior_edge_count": len(data.get("ford_sides", [])) if isinstance(data.get("ford_sides", []), list) else None,
        }

    def description(self) -> str:
        meanings = ", ".join(f"{k}={v}" for k, v in sorted(self.generator_meanings.items()))
        md = self.modular_metadata()
        subgroup = md.get("subgroup") or (f"Hecke G_{md.get('hecke_q')}" if md.get("hecke_q") else "PSL(2,Z)")
        torsion_free = md.get("torsion_free")
        is_hecke = md.get("hecke_q") is not None or "hecke" in str(md.get("category", "")).lower()
        if torsion_free is True:
            family = "Hecke" if is_hecke else "modular"
            compact_curve = "compact Hecke-cover curve" if is_hecke else "compact modular curve"
            surface_line = (
                f"The JSON represents a torsion-free finite-index {family} subgroup {subgroup}. "
                "The quotient H/Gamma is a smooth noncompact finite-area Riemann surface; "
                f"adding cusps gives the {compact_curve}."
            )
        else:
            family = "Hecke/Ford" if is_hecke else "modular/Ford"
            surface_line = (
                f"The JSON represents a {family} finite-area domain. "
                "If elliptic points are present, the quotient is an orbifold with a cusp, not a smooth surface."
            )
        return (
            "Ford-domain mode.\n\n"
            + surface_line + " "
            "The displayed boundary is a Ford-domain exterior-boundary representation in the disk. "
            f"Generator meanings: {meanings or 'not supplied'}."
        )

    def diagnostics(self) -> List[Tuple[str, str]]:
        pair_summary = []
        for sp in self.side_pairings:
            if isinstance(sp, dict):
                pair_summary.append(f"{sp.get('side')}<->{sp.get('paired_with')} by {sp.get('word')}")
        md = self.modular_metadata()
        rows = [
            ("surface interface", self.__class__.__name__),
            ("surface name", self.name),
            ("domain type", self.domain_type),
            ("subdomain type", str(md.get("subdomain_type") or "not supplied")),
            ("modular subgroup", str(md.get("subgroup") or "PSL(2,Z) / not supplied")),
            ("modular level N", str(md.get("level_N") if md.get("level_N") is not None else "not supplied")),
            ("Hecke q", str(md.get("hecke_q") if md.get("hecke_q") is not None else "not supplied")),
            ("Hecke lambda", str(md.get("hecke_lambda") if md.get("hecke_lambda") is not None else "not supplied")),
            ("index in Hecke group", str(md.get("index_in_hecke_group") if md.get("index_in_hecke_group") is not None else "not supplied")),
            ("torsion free", str(md.get("torsion_free") if md.get("torsion_free") is not None else "not supplied")),
            ("compact", str(md.get("compact") if md.get("compact") is not None else "not supplied")),
            ("compactified genus", str(md.get("compactified_genus") if md.get("compactified_genus") is not None else "not supplied")),
            ("index in PSL(2,Z)", str(md.get("index_in_psl2z") if md.get("index_in_psl2z") is not None else "not supplied")),
            ("cusp count", str(md.get("cusp_count") if md.get("cusp_count") is not None else "not supplied")),
            ("cusp widths", str(md.get("cusp_widths") if md.get("cusp_widths") is not None else "not supplied")),
            ("generators supplied", ", ".join(sorted(k for k in self.generators if len(k) == 1 and k.isupper())) or "none"),
            ("generator meanings", "; ".join(f"{k}: {v}" for k, v in sorted(self.generator_meanings.items())) or "not supplied"),
            ("exterior Ford sides", f"{len(self.ford_sides)} supplied"),
            ("construction tiles", str(md.get("construction_tile_count") if md.get("construction_tile_count") is not None else "not supplied")),
            ("internal scaffold edges", str(md.get("internal_edge_count") if md.get("internal_edge_count") is not None else "not supplied")),
            ("side pairings", "; ".join(pair_summary[:4]) + ("; ..." if len(pair_summary) > 4 else "") if pair_summary else "not supplied"),
            ("orbifold signature", str(self.data.get("orbifold_signature", "not supplied"))),
            ("orbifold area", str(self.data.get("orbifold_area", "not supplied"))),
            ("JSON certification", str(self.certification.get("status", "not supplied"))),
            ("certification status", "externally certified/by construction metadata; Explorer consumes, does not prove" if self.certification.get("status") else "by fiat; not verified by program"),
        ]
        return rows

def fuchsian_surface_from_json_text(text: str) -> Tuple[str, FuchsianSurface]:
    """Parse generator output JSON and return (model_key, surface)."""
    data = json.loads(text.strip())
    if not isinstance(data, dict):
        raise ValueError("Surface JSON must be an object.")
    domain_type = str(data.get("domain_type", "compact_polygon"))
    if domain_type in ("compact_polygon", "polygon") and bool(data.get("v12_polygon_compatible", True)):
        return ("polygon", UserPolygonFuchsianSurface.from_json_text(text))
    if domain_type == "schottky_ideal_geodesic_domain":
        return ("schottky", SchottkyIdealGeodesicSurface.from_json_text(text))
    if domain_type == "modular_ford_domain":
        return ("modular_ford", ModularFordSurface.from_json_text(text))
    raise ValueError(
        "This FuchsianGENN JSON type is recognized only as data, not as an implemented Explorer mode yet. "
        f"domain_type={domain_type!r}, v12_polygon_compatible={data.get('v12_polygon_compatible')!r}. "
        "Implemented JSON modes in v17.6: compact_polygon, schottky_ideal_geodesic_domain, and modular_ford_domain/Ford-domain JSON including modular and Hecke variants. Compact-polygon files may also carry triangle_source metadata, such as the (2,3,7)-compatible 14-gon compact surface from Domain Maker v4."
    )

# -----------------------------------------------------------------------------
# Neural curve relaxation
# -----------------------------------------------------------------------------

@dataclass
class RelaxationResult:
    curve: np.ndarray
    energy_history: np.ndarray
    final_energy: float
    final_length: float


if TORCH_AVAILABLE:
    class NeuralCurveModel(torch.nn.Module):
        def __init__(self, p: np.ndarray, q: np.ndarray, perturb: float, width: int = 32):
            super().__init__()
            self.register_buffer("p", torch.tensor(p, dtype=torch.float64))
            self.register_buffer("q", torch.tensor(q, dtype=torch.float64))

            chord = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
            norm = float(np.linalg.norm(chord))
            normal = np.array([0.0, 1.0], dtype=float) if norm < EPS else np.array([-chord[1], chord[0]]) / norm

            self.net = torch.nn.Sequential(
                torch.nn.Linear(1, width, dtype=torch.float64),
                torch.nn.Tanh(),
                torch.nn.Linear(width, width, dtype=torch.float64),
                torch.nn.Tanh(),
                torch.nn.Linear(width, 2, dtype=torch.float64),
            )
            for module in self.net:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.045)
                    torch.nn.init.zeros_(module.bias)
            with torch.no_grad():
                self.net[-1].bias[:] = torch.tensor(perturb * normal, dtype=torch.float64)

        def forward(self, t: "torch.Tensor") -> "torch.Tensor":
            base = (1.0 - t) * self.p[None, :] + t * self.q[None, :]
            envelope = torch.sin(math.pi * t)
            return base + envelope * self.net(t)


if TORCH_AVAILABLE:
    class MetricPhiNetwork(torch.nn.Module):
        """Neural conformal factor for disk validation.

        The ansatz is

            phi_theta(x,y) = -log(1-r^2+eps) + u_theta(x,y),

        so the network only has to learn the smoother correction u.  For the
        exact Poincare metric, u is the constant log(2).  This is deliberately
        a stable validation case for the Liouville residual.
        """

        def __init__(self, width: int = 48, eps: float = 1.0e-6):
            super().__init__()
            self.eps = float(eps)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, width, dtype=torch.float64),
                torch.nn.Tanh(),
                torch.nn.Linear(width, width, dtype=torch.float64),
                torch.nn.Tanh(),
                torch.nn.Linear(width, 1, dtype=torch.float64),
            )
            for module in self.net:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.035)
                    torch.nn.init.zeros_(module.bias)
            with torch.no_grad():
                # Start close to phi=-log(1-r^2); training learns the log(2)
                # correction from the PDE/gauge/boundary losses.
                self.net[-1].bias[:] = torch.tensor([0.0], dtype=torch.float64)

        def phi(self, xy: "torch.Tensor") -> "torch.Tensor":
            r2 = torch.sum(xy * xy, dim=1, keepdim=True)
            base = -torch.log(torch.clamp(1.0 - r2 + self.eps, min=self.eps))
            return base + self.net(xy)


def sample_disk_torch(n: int, r_max: float, device=None) -> "torch.Tensor":
    theta = 2.0 * math.pi * torch.rand(n, dtype=torch.float64, device=device)
    r = float(r_max) * torch.sqrt(torch.rand(n, dtype=torch.float64, device=device))
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)


def metric_laplacian(model: "MetricPhiNetwork", xy: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    xy = xy.clone().detach().requires_grad_(True)
    phi = model.phi(xy)
    grad_phi = torch.autograd.grad(phi.sum(), xy, create_graph=True)[0]
    lap = torch.zeros_like(phi[:, 0])
    for j in range(2):
        g_j = grad_phi[:, j]
        second_j = torch.autograd.grad(g_j.sum(), xy, create_graph=True)[0][:, j]
        lap = lap + second_j
    return phi[:, 0], lap


def evaluate_metric_network_np(model: "MetricPhiNetwork", points: np.ndarray) -> np.ndarray:
    if model is None:
        return poincare_phi_np(points)
    pts = np.asarray(points, dtype=float)
    shape = pts.shape[:-1]
    flat = pts.reshape(-1, 2)
    with torch.no_grad():
        xy = torch.tensor(flat, dtype=torch.float64)
        phi = model.phi(xy).detach().cpu().numpy().reshape(-1)
    return phi.reshape(shape)


def train_metric_network_pytorch(
    model: "MetricPhiNetwork",
    steps: int = 900,
    samples: int = 256,
    lr: float = 0.004,
    r_max: float = 0.82,
    boundary_r: float = 0.93,
    callback: Optional[Callable[[int, dict], None]] = None,
    callback_every: int = 25,
) -> Tuple[np.ndarray, dict]:
    """Train phi_theta by Liouville residual on the disk.

    Uses the sign convention for the Poincare disk metric

        K = - exp(-2 phi) Delta phi.

    Thus K=-1 corresponds to Delta phi = exp(2 phi).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    hist: List[Tuple[float, float, float, float, float]] = []

    val_pts = sample_disk_torch(384, min(0.86, float(r_max) + 0.04))
    val_true = torch.log(torch.tensor(2.0, dtype=torch.float64)) - torch.log(
        torch.clamp(1.0 - torch.sum(val_pts * val_pts, dim=1, keepdim=True), min=1.0e-10)
    )

    for step in range(1, int(steps) + 1):
        opt.zero_grad(set_to_none=True)
        xy = sample_disk_torch(int(samples), float(r_max))
        phi, lap = metric_laplacian(model, xy)
        e2phi = torch.exp(torch.clamp(2.0 * phi, max=18.0))
        # A relative residual is more stable near the boundary where e^(2phi)
        # becomes large.
        residual = (lap - e2phi) / torch.clamp(e2phi.detach(), min=1.0)
        pde_loss = torch.mean(residual * residual)

        zero = torch.zeros((1, 2), dtype=torch.float64)
        center_loss = torch.mean((model.phi(zero) - math.log(2.0)) ** 2)

        theta = 2.0 * math.pi * torch.rand(max(32, int(samples) // 4), dtype=torch.float64)
        br = torch.full_like(theta, float(boundary_r))
        bxy = torch.stack([br * torch.cos(theta), br * torch.sin(theta)], dim=1)
        bphi = model.phi(bxy)[:, 0]
        boundary_target = math.log(2.0) - torch.log(torch.clamp(1.0 - br * br, min=1.0e-10))
        boundary_loss = torch.mean((bphi - boundary_target) ** 2)

        loss = pde_loss + 2.0 * center_loss + 0.15 * boundary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()

        with torch.no_grad():
            pred = model.phi(val_pts)
            abs_err = torch.abs(pred - val_true)
            mean_abs = float(torch.mean(abs_err).cpu())
            max_abs = float(torch.max(abs_err).cpu())

        row = (
            float(loss.detach().cpu()),
            float(pde_loss.detach().cpu()),
            float(center_loss.detach().cpu()),
            float(boundary_loss.detach().cpu()),
            mean_abs,
        )
        hist.append(row)
        if callback is not None and (step == 1 or step % callback_every == 0 or step == int(steps)):
            callback(step, {
                "loss": row[0],
                "pde": row[1],
                "center": row[2],
                "boundary": row[3],
                "mean_abs_error": mean_abs,
                "max_abs_error": max_abs,
            })

    return np.array(hist, dtype=float), {
        "loss": hist[-1][0],
        "pde": hist[-1][1],
        "center": hist[-1][2],
        "boundary": hist[-1][3],
        "mean_abs_error": hist[-1][4],
        "max_abs_error": max_abs,
        "steps": len(hist),
        "samples": int(samples),
        "r_max": float(r_max),
        "boundary_r": float(boundary_r),
    }


def infer_initial_perturbation(p: np.ndarray, q: np.ndarray, initial_curve: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    initial_curve = np.asarray(initial_curve, dtype=float)
    if len(initial_curve) < 3:
        return 0.15
    mid_index = len(initial_curve) // 2
    t_mid = mid_index / (len(initial_curve) - 1)
    base_mid = (1.0 - t_mid) * p + t_mid * q
    chord = q - p
    norm = float(np.linalg.norm(chord))
    normal = np.array([0.0, 1.0], dtype=float) if norm < EPS else np.array([-chord[1], chord[0]]) / norm
    amp = float(np.dot(initial_curve[mid_index] - base_mid, normal))
    return float(np.clip(abs(amp), 0.01, 0.8))


def relax_curve_pytorch(
    p: np.ndarray,
    q: np.ndarray,
    initial_curve: np.ndarray,
    steps: int = 700,
    lr: float = 0.015,
    update_callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None,
    callback_every: int = 20,
    early_stop: bool = True,
    patience: int = 90,
    rel_tol: float = 1.0e-10,
    metric_model: Optional[object] = None,
) -> RelaxationResult:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    n = int(len(initial_curve))
    perturb = infer_initial_perturbation(p, q, initial_curve)
    model = NeuralCurveModel(p, q, perturb=perturb)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    metric_requires_grad = None
    if metric_model is not None:
        metric_model.eval()
        metric_requires_grad = [par.requires_grad for par in metric_model.parameters()]
        for par in metric_model.parameters():
            par.requires_grad_(False)
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(-1, 1)
    dt = 1.0 / (n - 1)

    hist: List[float] = []
    best_energy = float("inf")
    best_curve: Optional[np.ndarray] = None
    stale = 0

    for step in range(1, int(steps) + 1):
        opt.zero_grad(set_to_none=True)
        curve = model(t)
        dz_dt = (curve[1:] - curve[:-1]) / dt
        mid = 0.5 * (curve[1:] + curve[:-1])
        if metric_model is not None:
            phi_mid = metric_model.phi(mid)[:, 0]
            e2phi = torch.exp(torch.clamp(2.0 * phi_mid, max=24.0))
        else:
            r2 = torch.sum(mid * mid, dim=1)
            denom = torch.clamp(1.0 - r2, min=1.0e-8)
            e2phi = (2.0 / denom) ** 2
        energy = 0.5 * torch.sum(e2phi * torch.sum(dz_dt * dz_dt, dim=1)) * dt

        # Soft interior barrier.  It is zero for ordinary curves and prevents
        # transient optimizer steps from leaving the disk.
        curve_r2 = torch.sum(curve * curve, dim=1)
        barrier = 2.0e4 * torch.mean(torch.relu(curve_r2 - 0.990) ** 2)
        loss = energy + barrier
        loss.backward()
        opt.step()

        E = float(energy.detach().cpu())
        hist.append(E)
        curve_np = curve.detach().cpu().numpy()
        if E < best_energy * (1.0 - rel_tol):
            best_energy = E
            best_curve = curve_np.copy()
            stale = 0
        else:
            stale += 1

        if update_callback is not None and (step % callback_every == 0 or step == 1 or step == steps):
            update_callback(step, curve_np, np.array(hist, dtype=float))

        if early_stop and step > max(80, patience) and stale >= patience:
            break

    if best_curve is None:
        best_curve = model(t).detach().cpu().numpy()
    if metric_model is not None and metric_requires_grad is not None:
        for par, req in zip(metric_model.parameters(), metric_requires_grad):
            par.requires_grad_(req)
    hist_np = np.array(hist, dtype=float)
    return RelaxationResult(
        curve=best_curve,
        energy_history=hist_np,
        final_energy=float(hist_np[-1]) if len(hist_np) else discrete_hyperbolic_energy(best_curve),
        final_length=discrete_hyperbolic_length(best_curve),
    )


# -----------------------------------------------------------------------------
# Matplotlib canvases
# -----------------------------------------------------------------------------

class DiskCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9.1, 6.7), constrained_layout=True)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0])
        self.ax_disk = self.fig.add_subplot(gs[0, 0])
        self.ax_energy = self.fig.add_subplot(gs[1, 0])
        super().__init__(self.fig)
        self.setParent(parent)


class MetricSurfaceWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Universal-cover metric landscape z = phi(x,y)")
        self.resize(860, 760)
        self.fig = Figure(figsize=(8.2, 7.0), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.setCentralWidget(self.canvas)

    def plot_metric(
        self,
        p: Optional[np.ndarray] = None,
        q: Optional[np.ndarray] = None,
        q_lift: Optional[np.ndarray] = None,
        exact_curve: Optional[np.ndarray] = None,
        neural_curve: Optional[np.ndarray] = None,
    ):
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")
        grid = np.linspace(-0.965, 0.965, 120)
        X, Y = np.meshgrid(grid, grid)
        R2 = X * X + Y * Y
        Phi = math.log(2.0) - np.log(np.maximum(1.0 - R2, 1.0e-12))
        Phi[R2 >= 0.965 * 0.965] = np.nan
        ax.plot_surface(X, Y, Phi, linewidth=0, alpha=0.86, antialiased=True)

        def lift_phi(curve: np.ndarray) -> np.ndarray:
            phi = poincare_phi_np(curve)
            return np.minimum(phi, 4.8)

        if exact_curve is not None:
            z = lift_phi(exact_curve)
            ax.plot(exact_curve[:, 0], exact_curve[:, 1], z + 0.05, linewidth=2.5, label="exact lifted geodesic")
        if neural_curve is not None:
            z = lift_phi(neural_curve)
            ax.plot(neural_curve[:, 0], neural_curve[:, 1], z + 0.10, linewidth=2.5, label="neural curve")
        for pt, label, marker in [(p, "p", "o"), (q, "q", "s"), (q_lift, "A^n(q)", "D")]:
            if pt is not None:
                z = float(min(poincare_phi_np(pt[None, :])[0], 4.8))
                ax.scatter([pt[0]], [pt[1]], [z + 0.16], s=60, marker=marker, label=label)

        ax.set_title("Universal-cover conformal factor; same local metric for all disk quotients")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("phi")
        ax.set_zlim(0.0, 5.0)
        ax.view_init(elev=32, azim=-55)
        ax.legend(loc="upper left", fontsize=8)
        self.canvas.draw_idle()


# -----------------------------------------------------------------------------
# Main GUI
# -----------------------------------------------------------------------------

class FuchsianGENNExplorer(QMainWindow):
    MODEL_DISK = "disk"
    MODEL_CYCLIC = "cyclic"
    MODEL_TWOGEN = "twogen"
    MODEL_OCTAGON = "octagon"
    MODEL_POLYGON = "polygon"
    MODEL_SCHOTTKY = "schottky"
    MODEL_MODULAR_FORD = "modular_ford"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fuchsian GENN Explorer v17.6 — explicit tessellation trigger")
        self.resize(1360, 720)

        self.model_kind = self.MODEL_DISK
        self.cyclic_a = 0.58
        self.current_exponent = 0
        self.current_word = ""
        self.current_gamma = DiskMobius.identity()
        self.custom_a = 0.45
        self.custom_a_angle = 0.0
        self.custom_b = 0.42
        self.custom_b_angle = 65.0
        self.surface: FuchsianSurface = TrivialDiskSurface()
        self.user_polygon_surface: UserPolygonFuchsianSurface = UserPolygonFuchsianSurface.empty()
        self.user_schottky_surface: Optional[SchottkyIdealGeodesicSurface] = None
        self.user_modular_ford_surface: Optional[ModularFordSurface] = None

        self.p: Optional[np.ndarray] = None
        self.q: Optional[np.ndarray] = None
        self.q_lift: Optional[np.ndarray] = None
        self.exact_curve: Optional[np.ndarray] = None
        self.initial_curve: Optional[np.ndarray] = None
        self.neural_curve: Optional[np.ndarray] = None
        self.energy_history: Optional[np.ndarray] = None
        self.last_search_summary = ""
        self.metric_window: Optional[MetricSurfaceWindow] = None
        self.metric_model = MetricPhiNetwork() if TORCH_AVAILABLE else None
        self.metric_loss_history: Optional[np.ndarray] = None
        self.metric_last_stats: Optional[dict] = None
        self.metric_trained = False
        self.last_relaxation_metric = "none"
        self.exact_metric_curve: Optional[np.ndarray] = None
        self.exact_metric_energy_history: Optional[np.ndarray] = None
        self.learned_metric_curve: Optional[np.ndarray] = None
        self.learned_metric_energy_history: Optional[np.ndarray] = None
        self.metric_geodesic_comparison_summary = ""
        self.tessellation_cache_key = None
        self.tessellation_cache_edges = []
        self.tessellation_cache_segments = []
        self.orbit_cache_key = None
        self.orbit_cache_points = []
        self.skip_heavy_tessellation_redraw = False
        # v17.6: load only the principal/base domain by default.
        # Finite word-patch tessellation is shown only after the user presses the Tessellate button.
        self.tessellation_requested = False
        self.last_invariants: Optional[dict] = None

        self.canvas = DiskCanvas(self)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(self.canvas, stretch=1)
        root.addWidget(self.build_control_panel(), stretch=0)

        self.refresh_model_description()
        self.update_group_state_from_ui(reset_curves=True)
        self.refresh_plot()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def build_control_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setMinimumWidth(390)
        panel.setMaximumWidth(455)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("GENN v17.6 Controls")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        brief = QLabel(
            "v17.6 keeps the geometry+spectral feedstock workspace and modular-aware audit/fingerprint reporting. It loads only the principal/base domain by default; press the Tessellate button to deliberately draw the finite word patch: "
            "classical differential-geometry samples, base-polygon side-crossing intersections, and richer fingerprint export. "
            "spectral features are point-cloud graph-Laplacian prototypes, not certified Laplace-Beltrami eigenvalues."
        )
        brief.setWordWrap(True)
        layout.addWidget(brief)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        layout.addWidget(tabs, stretch=1)

        # Points tab
        tab_points = QWidget()
        points_layout = QVBoxLayout(tab_points)
        points_layout.setContentsMargins(8, 8, 8, 8)
        points_layout.setSpacing(7)
        grid = QGridLayout()
        self.clear_btn = QPushButton("Clear points")
        self.clear_btn.clicked.connect(self.clear_points)
        grid.addWidget(self.clear_btn, 0, 0)
        self.disk_example_btn = QPushButton("Load disk example")
        self.disk_example_btn.clicked.connect(self.load_disk_example)
        grid.addWidget(self.disk_example_btn, 0, 1)
        self.cyclic_example_btn = QPushButton("Load cyclic quotient example")
        self.cyclic_example_btn.clicked.connect(self.load_cyclic_example)
        grid.addWidget(self.cyclic_example_btn, 1, 0, 1, 2)
        self.twogen_example_btn = QPushButton("Load two-generator example")
        self.twogen_example_btn.clicked.connect(self.load_twogen_example)
        grid.addWidget(self.twogen_example_btn, 2, 0, 1, 2)
        self.octagon_example_btn = QPushButton("Load certified genus-2 octagon example")
        self.octagon_example_btn.clicked.connect(self.load_octagon_example)
        grid.addWidget(self.octagon_example_btn, 3, 0, 1, 2)
        self.userpoly_example_btn = QPushButton("Load surface JSON file…")
        self.userpoly_example_btn.clicked.connect(self.load_and_apply_surface_json_file)
        grid.addWidget(self.userpoly_example_btn, 4, 0, 1, 2)
        self.exact_btn = QPushButton("Draw selected lifted geodesic")
        self.exact_btn.clicked.connect(self.draw_exact_geodesic)
        grid.addWidget(self.exact_btn, 5, 0, 1, 2)

        self.tessellate_btn = QPushButton("Tessellate finite word patch")
        self.tessellate_btn.clicked.connect(self.tessellate_finite_patch)
        grid.addWidget(self.tessellate_btn, 6, 0, 1, 2)

        self.deep_tessellation_box = QCheckBox("Deeper run mode: show finite word patch")
        self.deep_tessellation_box.setChecked(False)
        self.deep_tessellation_box.stateChanged.connect(self.on_deep_tessellation_changed)
        grid.addWidget(self.deep_tessellation_box, 7, 0, 1, 2)

        self.principal_domain_btn = QPushButton("Principal/base domain only")
        self.principal_domain_btn.clicked.connect(self.show_principal_domain_only)
        grid.addWidget(self.principal_domain_btn, 8, 0, 1, 2)

        grid.addWidget(QLabel("Word-patch/search depth"), 9, 0)
        self.points_depth_spin = QSpinBox()
        self.points_depth_spin.setRange(0, 7)
        self.points_depth_spin.setValue(4)
        self.points_depth_spin.valueChanged.connect(self.on_points_depth_changed)
        grid.addWidget(self.points_depth_spin, 9, 1)

        points_layout.addLayout(grid)
        note = QLabel(
            "Click p and q in the disk. In cyclic mode, the selected curve goes "
            "from p to A^n(q). In word-based modes, it goes from p to word(q). "
            "v17.6 shows only the principal/base domain at load time. Press Tessellate when you deliberately want the finite word patch; for high index examples, keep depth low."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        points_layout.addWidget(note)
        points_layout.addStretch(1)
        tabs.addTab(tab_points, "Points")

        # Model / group tab
        tab_group = QWidget()
        group_layout = QGridLayout(tab_group)
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setVerticalSpacing(7)

        group_layout.addWidget(QLabel("Surface model"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItem("Exact Poincare disk / no quotient", self.MODEL_DISK)
        self.model_combo.addItem("Certified cyclic quotient <A>", self.MODEL_CYCLIC)
        self.model_combo.addItem("By-fiat two-generator <A,B>", self.MODEL_TWOGEN)
        self.model_combo.addItem("Certified regular octagon genus-2 surface", self.MODEL_OCTAGON)
        self.model_combo.addItem("Advanced user-supplied polygon surface", self.MODEL_POLYGON)
        self.model_combo.addItem("Schottky ideal-geodesic domain", self.MODEL_SCHOTTKY)
        self.model_combo.addItem("Modular/Ford PSL(2,Z) orbifold domain", self.MODEL_MODULAR_FORD)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        group_layout.addWidget(self.model_combo, 0, 1)

        group_layout.addWidget(QLabel("Cyclic A(0)=a"), 1, 0)
        self.cyclic_a_spin = QDoubleSpinBox()
        self.cyclic_a_spin.setRange(0.10, 0.88)
        self.cyclic_a_spin.setDecimals(3)
        self.cyclic_a_spin.setSingleStep(0.02)
        self.cyclic_a_spin.setValue(self.cyclic_a)
        self.cyclic_a_spin.valueChanged.connect(self.on_generator_params_changed)
        group_layout.addWidget(self.cyclic_a_spin, 1, 1)

        group_layout.addWidget(QLabel("Custom A strength"), 2, 0)
        self.custom_a_spin = QDoubleSpinBox()
        self.custom_a_spin.setRange(0.05, 0.88)
        self.custom_a_spin.setDecimals(3)
        self.custom_a_spin.setSingleStep(0.02)
        self.custom_a_spin.setValue(self.custom_a)
        self.custom_a_spin.valueChanged.connect(self.on_generator_params_changed)
        group_layout.addWidget(self.custom_a_spin, 2, 1)

        group_layout.addWidget(QLabel("Custom A angle deg"), 3, 0)
        self.custom_a_angle_spin = QDoubleSpinBox()
        self.custom_a_angle_spin.setRange(-180.0, 180.0)
        self.custom_a_angle_spin.setDecimals(1)
        self.custom_a_angle_spin.setSingleStep(5.0)
        self.custom_a_angle_spin.setValue(self.custom_a_angle)
        self.custom_a_angle_spin.valueChanged.connect(self.on_generator_params_changed)
        group_layout.addWidget(self.custom_a_angle_spin, 3, 1)

        group_layout.addWidget(QLabel("Custom B strength"), 4, 0)
        self.custom_b_spin = QDoubleSpinBox()
        self.custom_b_spin.setRange(0.05, 0.88)
        self.custom_b_spin.setDecimals(3)
        self.custom_b_spin.setSingleStep(0.02)
        self.custom_b_spin.setValue(self.custom_b)
        self.custom_b_spin.valueChanged.connect(self.on_generator_params_changed)
        group_layout.addWidget(self.custom_b_spin, 4, 1)

        group_layout.addWidget(QLabel("Custom B angle deg"), 5, 0)
        self.custom_b_angle_spin = QDoubleSpinBox()
        self.custom_b_angle_spin.setRange(-180.0, 180.0)
        self.custom_b_angle_spin.setDecimals(1)
        self.custom_b_angle_spin.setSingleStep(5.0)
        self.custom_b_angle_spin.setValue(self.custom_b_angle)
        self.custom_b_angle_spin.valueChanged.connect(self.on_generator_params_changed)
        group_layout.addWidget(self.custom_b_angle_spin, 5, 1)

        group_layout.addWidget(QLabel("Cyclic exponent n"), 6, 0)
        self.exponent_spin = QSpinBox()
        self.exponent_spin.setRange(-12, 12)
        self.exponent_spin.setValue(0)
        self.exponent_spin.valueChanged.connect(self.on_exponent_changed)
        group_layout.addWidget(self.exponent_spin, 6, 1)

        group_layout.addWidget(QLabel("Word / integer"), 7, 0)
        self.word_edit = QLineEdit("")
        self.word_edit.setPlaceholderText("cyclic: AA or -3; custom: ABaB")
        self.word_edit.returnPressed.connect(self.on_word_entered)
        group_layout.addWidget(self.word_edit, 7, 1)

        group_layout.addWidget(QLabel("Orbit/search depth"), 8, 0)
        self.orbit_depth_spin = QSpinBox()
        self.orbit_depth_spin.setRange(0, 7)
        self.orbit_depth_spin.setValue(int(self.points_depth_spin.value()) if hasattr(self, "points_depth_spin") else 4)
        self.orbit_depth_spin.valueChanged.connect(self.on_model_depth_changed)
        group_layout.addWidget(self.orbit_depth_spin, 8, 1)

        self.find_short_btn = QPushButton("Find shortest searched image of q")
        self.find_short_btn.clicked.connect(self.find_shortest_orbit_candidate)
        group_layout.addWidget(self.find_short_btn, 9, 0, 1, 2)

        self.model_description = QLabel("")
        self.model_description.setWordWrap(True)
        self.model_description.setStyleSheet("color: #555;")
        desc_scroll = QScrollArea()
        desc_scroll.setWidgetResizable(True)
        desc_scroll.setFrameShape(QFrame.Shape.StyledPanel)
        desc_scroll.setWidget(self.model_description)
        group_layout.addWidget(desc_scroll, 10, 0, 1, 2)
        tabs.addTab(tab_group, "Model")

        # Neural tab
        tab_neural = QWidget()
        curve_layout = QGridLayout(tab_neural)
        curve_layout.setContentsMargins(8, 8, 8, 8)
        curve_layout.setVerticalSpacing(7)
        curve_layout.addWidget(QLabel("Energy samples"), 0, 0)
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(8, 220)
        self.n_points_spin.setValue(56)
        curve_layout.addWidget(self.n_points_spin, 0, 1)
        curve_layout.addWidget(QLabel("Initial bend"), 1, 0)
        self.perturb_spin = QDoubleSpinBox()
        self.perturb_spin.setRange(0.00, 1.00)
        self.perturb_spin.setSingleStep(0.02)
        self.perturb_spin.setValue(0.22)
        curve_layout.addWidget(self.perturb_spin, 1, 1)
        curve_layout.addWidget(QLabel("Steps"), 2, 0)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(20, 10000)
        self.steps_spin.setValue(700)
        curve_layout.addWidget(self.steps_spin, 2, 1)
        curve_layout.addWidget(QLabel("Learning rate"), 3, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0005, 0.20)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0025)
        self.lr_spin.setValue(0.015)
        curve_layout.addWidget(self.lr_spin, 3, 1)
        self.early_stop_box = QCheckBox("Early stop when energy stalls")
        self.early_stop_box.setChecked(True)
        curve_layout.addWidget(self.early_stop_box, 4, 0, 1, 2)
        self.init_curve_btn = QPushButton("Initialize neural curve")
        self.init_curve_btn.clicked.connect(self.initialize_neural_curve)
        curve_layout.addWidget(self.init_curve_btn, 5, 0, 1, 2)
        self.relax_btn = QPushButton("Relax neural curve")
        self.relax_btn.clicked.connect(self.relax_neural_curve)
        curve_layout.addWidget(self.relax_btn, 6, 0, 1, 2)
        self.perturb_relax_btn = QPushButton("Perturb and relax again")
        self.perturb_relax_btn.clicked.connect(self.perturb_and_relax_again)
        curve_layout.addWidget(self.perturb_relax_btn, 7, 0, 1, 2)
        if not TORCH_AVAILABLE:
            warn = QLabel("PyTorch is not installed. Exact geometry still works; neural relaxation disabled.")
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #9a4a00;")
            curve_layout.addWidget(warn, 8, 0, 1, 2)
        tabs.addTab(tab_neural, "Neural")

        # Metric learning tab
        tab_metric = QWidget()
        metric_layout = QGridLayout(tab_metric)
        metric_layout.setContentsMargins(8, 8, 8, 8)
        metric_layout.setVerticalSpacing(7)
        metric_layout.addWidget(QLabel("Metric training steps"), 0, 0)
        self.metric_steps_spin = QSpinBox()
        self.metric_steps_spin.setRange(20, 20000)
        self.metric_steps_spin.setValue(900)
        metric_layout.addWidget(self.metric_steps_spin, 0, 1)
        metric_layout.addWidget(QLabel("Interior samples/step"), 1, 0)
        self.metric_samples_spin = QSpinBox()
        self.metric_samples_spin.setRange(32, 4096)
        self.metric_samples_spin.setValue(256)
        metric_layout.addWidget(self.metric_samples_spin, 1, 1)
        metric_layout.addWidget(QLabel("Metric learning rate"), 2, 0)
        self.metric_lr_spin = QDoubleSpinBox()
        self.metric_lr_spin.setRange(0.0001, 0.05)
        self.metric_lr_spin.setDecimals(5)
        self.metric_lr_spin.setSingleStep(0.0005)
        self.metric_lr_spin.setValue(0.004)
        metric_layout.addWidget(self.metric_lr_spin, 2, 1)
        metric_layout.addWidget(QLabel("Interior radius"), 3, 0)
        self.metric_rmax_spin = QDoubleSpinBox()
        self.metric_rmax_spin.setRange(0.30, 0.94)
        self.metric_rmax_spin.setDecimals(3)
        self.metric_rmax_spin.setSingleStep(0.02)
        self.metric_rmax_spin.setValue(0.82)
        metric_layout.addWidget(self.metric_rmax_spin, 3, 1)
        metric_layout.addWidget(QLabel("Boundary radius"), 4, 0)
        self.metric_boundary_spin = QDoubleSpinBox()
        self.metric_boundary_spin.setRange(0.70, 0.985)
        self.metric_boundary_spin.setDecimals(3)
        self.metric_boundary_spin.setSingleStep(0.01)
        self.metric_boundary_spin.setValue(0.93)
        metric_layout.addWidget(self.metric_boundary_spin, 4, 1)
        self.train_metric_btn = QPushButton("Train neural metric phi_theta")
        self.train_metric_btn.clicked.connect(self.train_neural_metric)
        metric_layout.addWidget(self.train_metric_btn, 5, 0, 1, 2)
        self.reset_metric_btn = QPushButton("Reset neural metric")
        self.reset_metric_btn.clicked.connect(self.reset_neural_metric)
        metric_layout.addWidget(self.reset_metric_btn, 6, 0, 1, 2)
        self.use_learned_metric_box = QCheckBox("Use learned metric in neural curve relaxation")
        self.use_learned_metric_box.setChecked(False)
        metric_layout.addWidget(self.use_learned_metric_box, 7, 0, 1, 2)
        self.compare_metric_geodesics_btn = QPushButton("Compare exact-metric vs learned-metric geodesics")
        self.compare_metric_geodesics_btn.clicked.connect(self.compare_metric_geodesics)
        metric_layout.addWidget(self.compare_metric_geodesics_btn, 8, 0, 1, 2)
        self.show_learned_phi_box = QCheckBox("Show learned phi_theta background")
        self.show_learned_phi_box.setChecked(False)
        self.show_learned_phi_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        metric_layout.addWidget(self.show_learned_phi_box, 9, 0, 1, 2)
        self.show_metric_error_box = QCheckBox("Show phi_theta - phi_true error background")
        self.show_metric_error_box.setChecked(False)
        self.show_metric_error_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        metric_layout.addWidget(self.show_metric_error_box, 10, 0, 1, 2)
        self.metric_status_label = QLabel(
            "Neural metric status: not trained.  This is a disk validation: "
            "Delta phi = exp(2 phi), with center gauge and boundary asymptotics."
        )
        self.metric_status_label.setWordWrap(True)
        self.metric_status_label.setStyleSheet("color: #555;")
        metric_layout.addWidget(self.metric_status_label, 11, 0, 1, 2)
        if not TORCH_AVAILABLE:
            metric_warn = QLabel("PyTorch is not installed, so neural metric learning is disabled.")
            metric_warn.setWordWrap(True)
            metric_warn.setStyleSheet("color: #9a4a00;")
            metric_layout.addWidget(metric_warn, 12, 0, 1, 2)
        tabs.addTab(tab_metric, "Metric")

        # View tab
        tab_view = QWidget()
        view_layout = QVBoxLayout(tab_view)
        view_layout.setContentsMargins(8, 8, 8, 8)
        view_layout.setSpacing(6)
        self.show_phi_box = QCheckBox("Show universal-cover conformal-factor background")
        self.show_phi_box.setChecked(False)
        self.show_phi_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.show_phi_box)
        self.show_strips_box = QCheckBox("Show certified strip / finite word-patch boundaries if available")
        self.show_strips_box.setChecked(True)
        self.show_strips_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.show_strips_box)
        patch_note = QLabel(
            "For polygon surfaces this is a finite word patch {gamma(F): |gamma| <= depth}, "
            "not a proof that the full infinite tessellation has been rendered canonically."
        )
        patch_note.setWordWrap(True)
        patch_note.setStyleSheet("color: #666; font-size: 10px;")
        view_layout.addWidget(patch_note)
        self.fast_tessellation_box = QCheckBox("Fast word-patch rendering/cache")
        self.fast_tessellation_box.setChecked(True)
        self.fast_tessellation_box.stateChanged.connect(lambda _=None: self.invalidate_tessellation_cache_and_refresh())
        view_layout.addWidget(self.fast_tessellation_box)

        explicit_note = QLabel("v17.6: large modular domains are not tessellated automatically. Use the Points-tab Tessellate button for a deliberate finite word-patch run.")
        explicit_note.setWordWrap(True)
        explicit_note.setStyleSheet("color: #666; font-size: 10px;")
        view_layout.addWidget(explicit_note)
        self.training_hide_tessellation_box = QCheckBox("Hide word patch during neural training redraws")
        self.training_hide_tessellation_box.setChecked(True)
        view_layout.addWidget(self.training_hide_tessellation_box)

        self.debug_tessellation_box = QCheckBox("Debug finite patch: central F + depth-1 labeled neighbors")
        self.debug_tessellation_box.setChecked(False)
        self.debug_tessellation_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.debug_tessellation_box)

        self.show_tile_labels_box = QCheckBox("Show tile labels in finite-patch/debug view")
        self.show_tile_labels_box.setChecked(False)
        self.show_tile_labels_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.show_tile_labels_box)

        self.shade_base_polygon_box = QCheckBox("Lightly shade the base fundamental domain F")
        self.shade_base_polygon_box.setChecked(True)
        self.shade_base_polygon_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.shade_base_polygon_box)

        self.filter_duplicate_tiles_box = QCheckBox("Filter visually duplicate word-images by polygon center")
        self.filter_duplicate_tiles_box.setChecked(True)
        self.filter_duplicate_tiles_box.stateChanged.connect(lambda _=None: self.invalidate_tessellation_cache_and_refresh())
        view_layout.addWidget(self.filter_duplicate_tiles_box)
        self.show_orbit_box = QCheckBox("Show orbit points A^n(q)")
        self.show_orbit_box.setChecked(True)
        self.show_orbit_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.show_orbit_box)
        self.show_candidates_box = QCheckBox("Show candidate geodesics to orbit points")
        self.show_candidates_box.setChecked(False)
        self.show_candidates_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.show_candidates_box)
        self.show_exact_box = QCheckBox("Show selected exact lifted geodesic")
        self.show_exact_box.setChecked(True)
        self.show_exact_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.show_exact_box)
        self.show_initial_box = QCheckBox("Show initial neural curve")
        self.show_initial_box.setChecked(True)
        self.show_initial_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.show_initial_box)
        self.show_neural_box = QCheckBox("Show relaxed neural curve")
        self.show_neural_box.setChecked(True)
        self.show_neural_box.stateChanged.connect(lambda _=None: self.refresh_plot())
        view_layout.addWidget(self.show_neural_box)
        self.metric_btn = QPushButton("Open / update 3D universal-cover metric landscape")
        self.metric_btn.clicked.connect(self.open_metric_surface)
        view_layout.addWidget(self.metric_btn)
        honest = QLabel(
            "The 3D surface is z=phi(x,y) on the universal cover.  It is not an "
            "isometric embedding and it does not change with Gamma; Gamma changes "
            "the quotient identifications."
        )
        honest.setWordWrap(True)
        honest.setStyleSheet("color: #555;")
        view_layout.addWidget(honest)
        view_layout.addStretch(1)
        tabs.addTab(tab_view, "View")


        # Geometry audit tab
        tab_audit = QWidget()
        audit_layout = QVBoxLayout(tab_audit)
        audit_layout.setContentsMargins(8, 8, 8, 8)
        self.audit_text = QTextEdit()
        self.audit_text.setReadOnly(True)
        self.audit_text.setPlainText("Geometry audit will appear here.")
        self.audit_text.setStyleSheet("font-family: monospace;")
        audit_layout.addWidget(self.audit_text)
        self.refresh_audit_btn = QPushButton("Refresh geometry audit")
        self.refresh_audit_btn.clicked.connect(self.update_geometry_audit)
        audit_layout.addWidget(self.refresh_audit_btn)
        tabs.addTab(tab_audit, "Audit")

        # Geometry/Invariants tab
        tab_inv = QWidget()
        inv_layout = QVBoxLayout(tab_inv)
        inv_layout.setContentsMargins(8, 8, 8, 8)
        inv_note = QLabel(
            "Compute first-pass geometric fingerprints from the currently selected surface. "
            "These are finite-search numerical invariants, not proofs of global optimality."
        )
        inv_note.setWordWrap(True)
        inv_layout.addWidget(inv_note)
        inv_grid = QGridLayout()
        inv_grid.addWidget(QLabel("word depth"), 0, 0)
        self.invariants_word_depth_spin = QSpinBox()
        self.invariants_word_depth_spin.setRange(1, 6)
        self.invariants_word_depth_spin.setValue(3)
        inv_grid.addWidget(self.invariants_word_depth_spin, 0, 1)
        inv_grid.addWidget(QLabel("max rows"), 1, 0)
        self.invariants_max_rows_spin = QSpinBox()
        self.invariants_max_rows_spin.setRange(10, 400)
        self.invariants_max_rows_spin.setValue(40)
        inv_grid.addWidget(self.invariants_max_rows_spin, 1, 1)
        inv_grid.addWidget(QLabel("injectivity samples"), 2, 0)
        self.invariants_sample_spin = QSpinBox()
        self.invariants_sample_spin.setRange(5, 500)
        self.invariants_sample_spin.setValue(60)
        inv_grid.addWidget(self.invariants_sample_spin, 2, 1)

        self.invariants_deep_run_box = QCheckBox("Deep fingerprint run for large modular surfaces")
        self.invariants_deep_run_box.setChecked(False)
        self.invariants_deep_run_box.setToolTip("Unchecked: skip/cap expensive word, injectivity, and graph-spectral calculations for large modular examples.")
        inv_grid.addWidget(self.invariants_deep_run_box, 3, 0, 1, 2)
        self.compute_invariants_btn = QPushButton("Compute geometry / invariants")
        self.compute_invariants_btn.clicked.connect(self.compute_and_display_invariants)
        inv_grid.addWidget(self.compute_invariants_btn, 4, 0, 1, 2)
        self.export_fingerprint_btn = QPushButton("Export surface fingerprint JSON")
        self.export_fingerprint_btn.clicked.connect(self.export_surface_fingerprint_json)
        inv_grid.addWidget(self.export_fingerprint_btn, 5, 0, 1, 2)
        inv_layout.addLayout(inv_grid)
        self.invariants_text = QTextEdit()
        self.invariants_text.setReadOnly(True)
        self.invariants_text.setPlainText("Geometry/invariant computations will appear here.")
        self.invariants_text.setStyleSheet("font-family: monospace;")
        inv_layout.addWidget(self.invariants_text, stretch=1)
        tabs.addTab(tab_inv, "Invariants")

        # Workflow tab
        tab_workflow = QWidget()
        wf_layout = QVBoxLayout(tab_workflow)
        wf_layout.setContentsMargins(8, 8, 8, 8)
        wf_note = QLabel("Student-safe workflows set known configurations. They do not replace the mathematical diagnostics; they simply make the important examples easier to run.")
        wf_note.setWordWrap(True)
        wf_layout.addWidget(wf_note)
        for text, handler in [
            ("1. Disk: exact geodesic", self.load_disk_example),
            ("2. Cyclic quotient: shortest lift", self.workflow_cyclic_shortest),
            ("3. Genus-2 octagon: word AB", self.load_octagon_example),
            ("4. Train neural metric on disk", self.train_neural_metric),
            ("5. Compare exact vs learned metric geodesics", self.compare_metric_geodesics),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            wf_layout.addWidget(btn)
        wf_layout.addStretch(1)
        tabs.addTab(tab_workflow, "Workflow")

        # Surface JSON tab
        tab_surface_data = QWidget()
        sd_layout = QVBoxLayout(tab_surface_data)
        sd_layout.setContentsMargins(8, 8, 8, 8)
        sd_layout.setSpacing(6)
        sd_note = QLabel(
            "Advanced user-polygon mode requires externally certified quotient data: generators, "
            "a fundamental polygon F, and side-pairing metadata. The program takes these by fiat "
            "as valid Fuchsian data and then draws gamma(F). For a built-in certified example, "
            "use the regular genus-2 octagon surface instead."
        )
        sd_note.setWordWrap(True)
        sd_layout.addWidget(sd_note)
        self.surface_json_edit = QTextEdit()
        self.surface_json_edit.setPlaceholderText("Paste externally certified surface JSON here, then click Apply user polygon surface.")
        self.surface_json_edit.setPlainText(self.default_surface_json_template())
        sd_layout.addWidget(self.surface_json_edit, stretch=1)
        sd_buttons = QHBoxLayout()
        self.apply_surface_json_btn = QPushButton("Apply surface JSON")
        self.apply_surface_json_btn.clicked.connect(self.apply_user_polygon_surface)
        sd_buttons.addWidget(self.apply_surface_json_btn)
        self.template_surface_json_btn = QPushButton("Show JSON format skeleton only")
        self.template_surface_json_btn.clicked.connect(lambda: self.surface_json_edit.setPlainText(self.default_surface_json_template()))
        sd_buttons.addWidget(self.template_surface_json_btn)
        self.load_surface_json_file_btn = QPushButton("Load JSON file")
        self.load_surface_json_file_btn.clicked.connect(self.load_surface_json_file)
        sd_buttons.addWidget(self.load_surface_json_file_btn)
        self.save_octagon_json_btn = QPushButton("Export built-in octagon JSON")
        self.save_octagon_json_btn.clicked.connect(self.export_octagon_json_to_editor)
        sd_buttons.addWidget(self.save_octagon_json_btn)
        sd_layout.addLayout(sd_buttons)
        tabs.addTab(tab_surface_data, "Surface Data")

        # Diagnostics tab
        tab_diag = QWidget()
        diag_layout = QVBoxLayout(tab_diag)
        diag_layout.setContentsMargins(8, 8, 8, 8)
        self.diagnostics = QTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setPlainText("Diagnostics will appear here.")
        self.diagnostics.setStyleSheet("font-family: monospace;")
        diag_layout.addWidget(self.diagnostics)
        tabs.addTab(tab_diag, "Diagnostics")

        self.status = QLabel("Status: choose two points in the disk.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-family: monospace; color: #333;")
        layout.addWidget(self.status)
        footer = QLabel(
            "v16.3 = geometry feedstock plus base-polygon side-crossing intersections, "
            "richer JSON export, and a first point-cloud graph-Laplacian spectral prototype."
        )
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #555; font-size: 10px;")
        layout.addWidget(footer)
        return panel

    # ------------------------------------------------------------------
    # State and model helpers
    # ------------------------------------------------------------------

    def selected_model(self) -> str:
        return self.model_combo.currentData() if hasattr(self, "model_combo") else self.MODEL_DISK

    def on_points_depth_changed(self, value: int):
        """Synchronize the convenient Points-tab depth control with the model depth.

        The same integer controls orbit search depth and finite word-patch
        depth.  Large-genus regular surfaces can become expensive quickly, so this
        control is intentionally placed on the Points tab where it is easy to reach.
        """
        if hasattr(self, "orbit_depth_spin") and int(self.orbit_depth_spin.value()) != int(value):
            self.orbit_depth_spin.blockSignals(True)
            self.orbit_depth_spin.setValue(int(value))
            self.orbit_depth_spin.blockSignals(False)
        self.invalidate_tessellation_cache_and_refresh()

    def on_model_depth_changed(self, value: int):
        """Synchronize the Model-tab depth control with the Points-tab control."""
        if hasattr(self, "points_depth_spin") and int(self.points_depth_spin.value()) != int(value):
            self.points_depth_spin.blockSignals(True)
            self.points_depth_spin.setValue(int(value))
            self.points_depth_spin.blockSignals(False)
        self.invalidate_tessellation_cache_and_refresh()

    def make_surface(self) -> FuchsianSurface:
        kind = self.selected_model()
        if kind == self.MODEL_CYCLIC:
            a = float(self.cyclic_a_spin.value()) if hasattr(self, "cyclic_a_spin") else self.cyclic_a
            return CyclicQuotientSurface(a)
        if kind == self.MODEL_TWOGEN:
            a = float(self.custom_a_spin.value()) if hasattr(self, "custom_a_spin") else self.custom_a
            aa = float(self.custom_a_angle_spin.value()) if hasattr(self, "custom_a_angle_spin") else self.custom_a_angle
            b = float(self.custom_b_spin.value()) if hasattr(self, "custom_b_spin") else self.custom_b
            ba = float(self.custom_b_angle_spin.value()) if hasattr(self, "custom_b_angle_spin") else self.custom_b_angle
            return TwoGeneratorByFiatSurface(a, aa, b, ba)
        if kind == self.MODEL_OCTAGON:
            return RegularOctagonGenus2Surface()
        if kind == self.MODEL_POLYGON:
            return self.user_polygon_surface
        if kind == self.MODEL_SCHOTTKY and self.user_schottky_surface is not None:
            return self.user_schottky_surface
        if kind == self.MODEL_MODULAR_FORD and self.user_modular_ford_surface is not None:
            return self.user_modular_ford_surface
        return TrivialDiskSurface()

    def current_lift_selector(self):
        if self.selected_model() in (self.MODEL_TWOGEN, self.MODEL_OCTAGON, self.MODEL_POLYGON, self.MODEL_SCHOTTKY, self.MODEL_MODULAR_FORD):
            return reduce_fuchsian_word(self.current_word)
        return int(self.current_exponent)

    def update_group_state_from_ui(self, reset_curves: bool = True):
        self.model_kind = self.selected_model()
        self.cyclic_a = float(self.cyclic_a_spin.value()) if hasattr(self, "cyclic_a_spin") else self.cyclic_a
        if hasattr(self, "custom_a_spin"):
            self.custom_a = float(self.custom_a_spin.value())
            self.custom_a_angle = float(self.custom_a_angle_spin.value())
            self.custom_b = float(self.custom_b_spin.value())
            self.custom_b_angle = float(self.custom_b_angle_spin.value())
        self.surface = self.make_surface()
        if self.selected_model() == self.MODEL_CYCLIC:
            selector = int(self.exponent_spin.value()) if hasattr(self, "exponent_spin") else self.current_exponent
            self.current_exponent = int(selector)
            self.current_word = exponent_to_word(self.current_exponent)
        elif self.selected_model() in (self.MODEL_TWOGEN, self.MODEL_OCTAGON, self.MODEL_POLYGON, self.MODEL_SCHOTTKY, self.MODEL_MODULAR_FORD):
            selector = reduce_fuchsian_word(self.word_edit.text() if hasattr(self, "word_edit") else self.current_word)
            self.current_word = selector
        else:
            selector = 0
            self.current_exponent = 0
            self.current_word = ""
        self.current_gamma = self.surface.gamma(selector)
        self.q_lift = self.surface.lifted_endpoint(self.q, selector) if self.q is not None else None
        if self.p is not None and self.q_lift is not None:
            self.exact_curve = exact_poincare_geodesic(self.p, self.q_lift)
        else:
            self.exact_curve = None
        if reset_curves:
            self.initial_curve = None
            self.neural_curve = None
            self.energy_history = None
            self.last_relaxation_metric = "none"
            self.exact_metric_curve = None
            self.exact_metric_energy_history = None
            self.learned_metric_curve = None
            self.learned_metric_energy_history = None
            self.metric_geodesic_comparison_summary = ""

    def refresh_model_description(self):
        if not hasattr(self, "model_description"):
            return
        surface = self.make_surface()
        self.model_description.setText(surface.description())

    def clear_points(self):
        self.p = None
        self.q = None
        self.q_lift = None
        self.exact_curve = None
        self.initial_curve = None
        self.neural_curve = None
        self.energy_history = None
        self.last_search_summary = ""
        self.status.setText("Status: cleared. Choose two points in the disk.")
        self.update_diagnostics()
        self.refresh_plot()

    def load_disk_example(self):
        self.model_combo.setCurrentIndex(0)
        self.exponent_spin.setValue(0)
        self.word_edit.setText("")
        self.p = np.array([-0.55, -0.15], dtype=float)
        self.q = np.array([0.52, 0.46], dtype=float)
        self.last_search_summary = ""
        self.invalidate_tessellation_cache()
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText("Status: loaded exact disk example.")
        self.update_diagnostics()
        self.refresh_plot()

    def load_cyclic_example(self):
        self.model_combo.setCurrentIndex(1)
        self.cyclic_a_spin.setValue(0.58)
        self.exponent_spin.setValue(2)
        self.word_edit.setText("AA")
        self.p = np.array([-0.48, 0.26], dtype=float)
        self.q = np.array([0.36, -0.18], dtype=float)
        self.last_search_summary = ""
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText("Status: loaded certified cyclic quotient example p -> A^2(q).")
        self.update_diagnostics()
        self.refresh_plot()

    def load_twogen_example(self):
        self.model_combo.setCurrentIndex(2)
        self.custom_a_spin.setValue(0.45)
        self.custom_a_angle_spin.setValue(0.0)
        self.custom_b_spin.setValue(0.42)
        self.custom_b_angle_spin.setValue(65.0)
        self.word_edit.setText("AB")
        self.current_word = "AB"
        self.p = np.array([-0.42, 0.18], dtype=float)
        self.q = np.array([0.28, -0.22], dtype=float)
        self.last_search_summary = ""
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText("Status: loaded by-fiat two-generator example p -> AB(q).")
        self.update_diagnostics()
        self.refresh_plot()


    def load_octagon_example(self):
        idx = self.model_combo.findData(self.MODEL_OCTAGON)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.word_edit.setText("AB")
        self.current_word = "AB"
        self.orbit_depth_spin.setValue(min(int(self.orbit_depth_spin.value()), 3))
        self.p = np.array([-0.18, 0.10], dtype=float)
        self.q = np.array([0.16, -0.12], dtype=float)
        self.last_search_summary = ""
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText("Status: loaded certified regular octagon genus-2 example p -> AB(q).")
        self.update_diagnostics()
        self.refresh_plot()


    def default_surface_json_template(self) -> str:
        data = {
            "format": "FuchsianGENN surface JSON v12",
            "domain_type": "skeleton_not_a_surface",
            "v12_polygon_compatible": False,
            "name": "ADVANCED SKELETON ONLY - replace with externally certified Fuchsian data",
            "generators": {
                "A": {"type": "axial", "strength": 0.45, "angle_deg": 0.0},
                "B": {"type": "axial", "strength": 0.42, "angle_deg": 65.0}
            },
            "polygon_vertices": [
                [-0.35, -0.25], [0.18, -0.42], [0.50, 0.05],
                [0.18, 0.45], [-0.38, 0.28]
            ],
            "side_pairings": [
                {"side": 0, "paired_with": 2, "word": "A"},
                {"side": 1, "paired_with": 3, "word": "B"}
            ],
            "notes": "Skeleton only. These numbers are not asserted to be a certified fundamental polygon. Use the built-in regular octagon for a certified compact example, or replace every field with externally certified quotient data."
        }
        return json.dumps(data, indent=2)

    def apply_user_polygon_surface(self):
        try:
            model_key, candidate = fuchsian_surface_from_json_text(self.surface_json_edit.toPlainText())
        except Exception as exc:
            QMessageBox.critical(self, "Invalid or unsupported surface JSON", str(exc))
            self.status.setText("Status: surface JSON rejected; see error dialog.")
            return
        if model_key == self.MODEL_POLYGON:
            self.user_polygon_surface = candidate
        elif model_key == self.MODEL_SCHOTTKY:
            self.user_schottky_surface = candidate
        elif model_key == self.MODEL_MODULAR_FORD:
            self.user_modular_ford_surface = candidate
        idx = self.model_combo.findData(model_key)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        if hasattr(candidate, "available_alphabet"):
            alphabet = candidate.available_alphabet()
            if not self.word_edit.text().strip() or any(ch.upper() not in candidate.generators for ch in clean_fuchsian_word(self.word_edit.text())):
                first_gen = next((ch for ch in alphabet if ch.isupper()), "")
                self.word_edit.setText(first_gen)
        self.current_word = reduce_fuchsian_word(self.word_edit.text())
        self.refresh_model_description()
        self.last_search_summary = ""
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText("Status: applied surface JSON. Tessellation/domain lines use gamma(F) or gamma(ideal sides) as appropriate.")
        self.update_diagnostics()
        self.refresh_plot()

    def load_user_polygon_example(self):
        """Deprecated helper retained only for backwards compatibility.

        v12.2 no longer loads the skeleton as an active surface from the Points/Workflow
        buttons, because doing so creates a visually plausible but non-certified
        tessellation.  Use load_and_apply_surface_json_file() instead.
        """
        QMessageBox.information(
            self,
            "Skeleton is not a surface",
            "The built-in JSON skeleton is only a format guide, not a certified Fuchsian "
            "fundamental polygon. Use the Surface Data tab to inspect the skeleton, or "
            "load an externally certified JSON file."
        )

    def load_and_apply_surface_json_file(self):
        """Load an externally certified surface JSON file and immediately apply it.

        This is the intended entry point for JSON produced by the forthcoming
        standalone certified-domain generator.  The program validates syntax and
        internal schema, but it still assumes by fiat that the supplied
        (Gamma, F, side-pairings) are mathematically certified.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load externally certified Fuchsian surface JSON",
            "",
            "JSON files (*.json);;All files (*)"
        )
        if not path:
            self.status.setText("Status: surface JSON load cancelled.")
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            # Validate JSON syntax and route to the matching implemented Explorer mode.
            json.loads(text)
            model_key, candidate = fuchsian_surface_from_json_text(text)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load surface JSON", str(exc))
            self.status.setText("Status: surface JSON rejected; see error dialog.")
            return

        self.surface_json_edit.setPlainText(text)
        if model_key == self.MODEL_POLYGON:
            self.user_polygon_surface = candidate
        elif model_key == self.MODEL_SCHOTTKY:
            self.user_schottky_surface = candidate
        elif model_key == self.MODEL_MODULAR_FORD:
            self.user_modular_ford_surface = candidate
        idx = self.model_combo.findData(model_key)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        if hasattr(candidate, "available_alphabet"):
            alphabet = candidate.available_alphabet()
            if not self.word_edit.text().strip() or any(ch.upper() not in candidate.generators for ch in clean_fuchsian_word(self.word_edit.text())):
                first_gen = next((ch for ch in alphabet if ch.isupper()), "")
                self.word_edit.setText(first_gen)
        self.current_word = reduce_fuchsian_word(self.word_edit.text())
        self.tessellation_requested = False
        if hasattr(self, "deep_tessellation_box"):
            self.deep_tessellation_box.blockSignals(True)
            self.deep_tessellation_box.setChecked(False)
            self.deep_tessellation_box.blockSignals(False)
        self.refresh_model_description()
        self.last_search_summary = f"Loaded externally supplied surface JSON:\n  {path}"
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText("Status: loaded and applied externally supplied Fuchsian surface JSON.")
        self.update_diagnostics()
        self.refresh_plot()

    def on_model_changed(self):
        if self.selected_model() == self.MODEL_DISK:
            self.exponent_spin.setValue(0)
            self.word_edit.setText("")
            self.current_word = ""
        elif self.selected_model() == self.MODEL_CYCLIC:
            self.word_edit.setText(exponent_to_word(int(self.exponent_spin.value())))
        else:
            if not self.word_edit.text().strip():
                self.word_edit.setText("AB")
            self.current_word = reduce_fuchsian_word(self.word_edit.text())
            # Keep the first view responsive for polygonal surfaces; users can
            # deepen the finite patch deliberately after seeing the base surface.
            if self.selected_model() in (self.MODEL_OCTAGON, self.MODEL_POLYGON, self.MODEL_SCHOTTKY) and hasattr(self, "orbit_depth_spin"):
                if int(self.orbit_depth_spin.value()) > 3:
                    self.orbit_depth_spin.setValue(3)
            if self.selected_model() == self.MODEL_MODULAR_FORD:
                # v17.6: keep load-time rendering to the principal/base domain.
                self.tessellation_requested = False
        if hasattr(self, "deep_tessellation_box") and self.deep_tessellation_box.isChecked():
            self.deep_tessellation_box.blockSignals(True)
            self.deep_tessellation_box.setChecked(False)
            self.deep_tessellation_box.blockSignals(False)
        self.invalidate_tessellation_cache()
        self.refresh_model_description()
        self.last_search_summary = ""
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText("Status: surface model changed.")
        self.refresh_plot()

    def on_generator_params_changed(self):
        self.refresh_model_description()
        self.last_search_summary = ""
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText("Status: generator parameter changed; curves reset.")
        self.refresh_plot()

    def on_cyclic_a_changed(self):
        self.on_generator_params_changed()

    def on_exponent_changed(self):
        n = int(self.exponent_spin.value())
        if self.selected_model() != self.MODEL_CYCLIC:
            return
        self.word_edit.setText(exponent_to_word(n))
        self.last_search_summary = ""
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText(f"Status: selected cyclic quotient lift p -> A^{n}(q).")
        self.refresh_plot()

    def on_word_entered(self):
        if self.selected_model() in (self.MODEL_TWOGEN, self.MODEL_OCTAGON, self.MODEL_POLYGON, self.MODEL_SCHOTTKY, self.MODEL_MODULAR_FORD):
            word = reduce_fuchsian_word(self.word_edit.text())
            self.current_word = word
            self.word_edit.setText(word)
            self.update_group_state_from_ui(reset_curves=True)
            self.status.setText(f"Status: parsed custom word as '{word or 'identity'}'.")
        else:
            n = parse_cyclic_word(self.word_edit.text())
            n = max(self.exponent_spin.minimum(), min(self.exponent_spin.maximum(), n))
            self.exponent_spin.setValue(n)
            self.word_edit.setText(exponent_to_word(n))
            if self.selected_model() == self.MODEL_DISK and n != 0:
                self.model_combo.setCurrentIndex(1)
            self.update_group_state_from_ui(reset_curves=True)
            self.status.setText(f"Status: parsed cyclic word as exponent n={n}.")
        self.refresh_plot()

    def require_two_points(self) -> bool:
        if self.p is None or self.q is None:
            QMessageBox.information(self, "Need two points", "Please click two points inside the disk first.")
            return False
        return True

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------

    def on_plot_click(self, event):
        if event.inaxes != self.canvas.ax_disk or event.xdata is None or event.ydata is None:
            return
        pt = np.array([float(event.xdata), float(event.ydata)], dtype=float)
        if not inside_unit_disk(pt, margin=1.0e-5):
            self.status.setText("Status: point rejected. Click strictly inside the unit disk.")
            return
        if self.p is None or (self.p is not None and self.q is not None):
            self.p = pt
            self.q = None
            self.q_lift = None
            self.exact_curve = None
            self.initial_curve = None
            self.neural_curve = None
            self.energy_history = None
            self.last_search_summary = ""
            self.status.setText(f"Status: p set at ({pt[0]: .3f}, {pt[1]: .3f}). Choose q.")
        else:
            self.q = pt
            self.last_search_summary = ""
            self.update_group_state_from_ui(reset_curves=True)
            self.status.setText(f"Status: q set at ({pt[0]: .3f}, {pt[1]: .3f}).")
        self.update_diagnostics()
        self.refresh_plot()

    def on_deep_tessellation_changed(self, state):
        self.tessellation_requested = bool(state)
        self.invalidate_tessellation_cache()
        if self.tessellation_requested:
            depth = int(self.orbit_depth_spin.value()) if hasattr(self, "orbit_depth_spin") else int(self.points_depth_spin.value())
            self.status.setText(f"Status: deeper/tessellated run mode enabled at finite word depth {depth}.")
        else:
            self.status.setText("Status: deeper/tessellated run mode disabled; showing principal/base domain only.")
        self.refresh_plot()

    def tessellate_finite_patch(self):
        """User-triggered finite word-patch tessellation.

        v17.6 intentionally does not tessellate automatically at load time.
        This button turns on the word-patch rendering using the current depth.
        """
        self.tessellation_requested = True
        if hasattr(self, "deep_tessellation_box") and not self.deep_tessellation_box.isChecked():
            self.deep_tessellation_box.blockSignals(True)
            self.deep_tessellation_box.setChecked(True)
            self.deep_tessellation_box.blockSignals(False)
        self.invalidate_tessellation_cache()
        depth = int(self.orbit_depth_spin.value()) if hasattr(self, "orbit_depth_spin") else int(self.points_depth_spin.value())
        self.status.setText(f"Status: tessellation requested at finite word depth {depth}; this may be slow for high-index modular domains.")
        self.refresh_plot()

    def show_principal_domain_only(self):
        """Return to base/principal-domain-only display."""
        self.tessellation_requested = False
        if hasattr(self, "deep_tessellation_box") and self.deep_tessellation_box.isChecked():
            self.deep_tessellation_box.blockSignals(True)
            self.deep_tessellation_box.setChecked(False)
            self.deep_tessellation_box.blockSignals(False)
        self.invalidate_tessellation_cache()
        self.status.setText("Status: showing only the principal/base domain; finite word patch hidden.")
        self.refresh_plot()

    def draw_exact_geodesic(self):
        if not self.require_two_points():
            return
        self.update_group_state_from_ui(reset_curves=True)
        label = self.surface.lift_label(self.current_lift_selector())
        self.status.setText(f"Status: exact lifted geodesic drawn from p to {label}.")
        self.update_diagnostics()
        self.refresh_plot()

    def initialize_neural_curve(self):
        if not self.require_two_points():
            return
        self.update_group_state_from_ui(reset_curves=False)
        if self.q_lift is None:
            return
        n = int(self.n_points_spin.value())
        perturb = float(self.perturb_spin.value())
        self.initial_curve = make_initial_curve(self.p, self.q_lift, n=n, perturb=perturb)
        self.neural_curve = None
        self.energy_history = None
        self.status.setText("Status: initialized neural curve between the selected lifted endpoints.")
        self.update_diagnostics()
        self.refresh_plot()

    def relax_neural_curve(self):
        if not self.require_two_points():
            return
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "PyTorch missing", "Install PyTorch to use neural relaxation:\n\n    pip install torch")
            return
        self.update_group_state_from_ui(reset_curves=False)
        if self.q_lift is None:
            return
        if self.initial_curve is None:
            self.initialize_neural_curve()
            if self.initial_curve is None:
                return

        steps = int(self.steps_spin.value())
        lr = float(self.lr_spin.value())
        self.relax_btn.setEnabled(False)
        self.perturb_relax_btn.setEnabled(False)
        use_learned = bool(hasattr(self, "use_learned_metric_box") and self.use_learned_metric_box.isChecked() and self.learned_metric_available())
        self.last_relaxation_metric = "learned phi_theta" if use_learned else "exact Poincare phi"
        self.status.setText(f"Status: relaxing neural curve using {self.last_relaxation_metric}...")
        QApplication.processEvents()

        def callback(k: int, curve_np: np.ndarray, energy_np: np.ndarray):
            self.neural_curve = curve_np
            self.energy_history = energy_np
            self.status.setText(f"Status: relaxing... step {k}/{steps}")
            self.refresh_plot(light=True)
            QApplication.processEvents()

        try:
            self.skip_heavy_tessellation_redraw = bool(hasattr(self, "training_hide_tessellation_box") and self.training_hide_tessellation_box.isChecked())
            result = relax_curve_pytorch(
                self.p,
                self.q_lift,
                self.initial_curve,
                steps=steps,
                lr=lr,
                update_callback=callback,
                callback_every=max(5, steps // 60),
                early_stop=bool(self.early_stop_box.isChecked()),
                patience=max(40, steps // 8),
                metric_model=self.metric_model if use_learned else None,
            )
            self.neural_curve = result.curve
            self.energy_history = result.energy_history
            self.status.setText("Status: relaxation complete.")
            self.update_diagnostics()
            self.refresh_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Relaxation failed", str(exc))
            self.status.setText("Status: neural relaxation failed. See error dialog.")
        finally:
            self.skip_heavy_tessellation_redraw = False
            self.relax_btn.setEnabled(True)
            self.perturb_relax_btn.setEnabled(True)

    def compare_metric_geodesics(self):
        """Run two neural geodesic relaxations from the same initial curve.

        First run: exact Poincare metric.
        Second run: learned phi_theta.

        This is the key learned-metric validation: the learned neural metric is not merely
        plotted or compared pointwise; it drives a downstream geometric solver.
        The final curves are then assessed with the exact Poincare geometry,
        because the disk case has a known benchmark.
        """
        if not self.require_two_points():
            return
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "PyTorch missing", "PyTorch is required for this comparison.")
            return
        if not self.learned_metric_available():
            QMessageBox.information(
                self,
                "Train metric first",
                "Please train the neural metric phi_theta before comparing learned-metric geodesics.",
            )
            return
        self.update_group_state_from_ui(reset_curves=False)
        if self.q_lift is None:
            return
        if self.initial_curve is None:
            self.initialize_neural_curve()
            if self.initial_curve is None:
                return

        steps = int(self.steps_spin.value())
        lr = float(self.lr_spin.value())
        self.compare_metric_geodesics_btn.setEnabled(False)
        self.relax_btn.setEnabled(False)
        self.perturb_relax_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            self.status.setText("Status: comparison 1/2, relaxing with exact Poincare metric...")
            QApplication.processEvents()
            exact_result = relax_curve_pytorch(
                self.p,
                self.q_lift,
                self.initial_curve,
                steps=steps,
                lr=lr,
                update_callback=None,
                early_stop=bool(self.early_stop_box.isChecked()),
                patience=max(40, steps // 8),
                metric_model=None,
            )
            self.exact_metric_curve = exact_result.curve
            self.exact_metric_energy_history = exact_result.energy_history

            self.status.setText("Status: comparison 2/2, relaxing with learned phi_theta...")
            QApplication.processEvents()
            learned_result = relax_curve_pytorch(
                self.p,
                self.q_lift,
                self.initial_curve,
                steps=steps,
                lr=lr,
                update_callback=None,
                early_stop=bool(self.early_stop_box.isChecked()),
                patience=max(40, steps // 8),
                metric_model=self.metric_model,
            )
            self.learned_metric_curve = learned_result.curve
            self.learned_metric_energy_history = learned_result.energy_history

            # Make the learned-metric curve the currently displayed relaxed curve,
            # while preserving both comparison curves for overlay/diagnostics.
            self.neural_curve = learned_result.curve
            self.energy_history = learned_result.energy_history
            self.last_relaxation_metric = "learned phi_theta (comparison run)"

            # Simple pointwise curve separation; both curves use the same sample count.
            n = min(len(self.exact_metric_curve), len(self.learned_metric_curve))
            diff = self.learned_metric_curve[:n] - self.exact_metric_curve[:n]
            rms_sep = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
            max_sep = float(np.max(np.linalg.norm(diff, axis=1)))
            self.metric_geodesic_comparison_summary = (
                f"Compared exact-metric and learned-metric neural geodesics with {steps} requested steps.\n"
                f"RMS Euclidean curve separation = {rms_sep:.3e}\n"
                f"max Euclidean curve separation = {max_sep:.3e}"
            )
            self.status.setText("Status: exact-vs-learned metric geodesic comparison complete.")
            self.update_diagnostics()
            self.refresh_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Metric-geodesic comparison failed", str(exc))
            self.status.setText("Status: metric-geodesic comparison failed. See error dialog.")
        finally:
            self.compare_metric_geodesics_btn.setEnabled(True)
            self.relax_btn.setEnabled(True)
            self.perturb_relax_btn.setEnabled(True)

    def perturb_and_relax_again(self):
        if not self.require_two_points():
            return
        self.initialize_neural_curve()
        if self.initial_curve is not None:
            self.relax_neural_curve()

    def find_shortest_orbit_candidate(self):
        if not self.require_two_points():
            return
        if self.selected_model() == self.MODEL_DISK:
            self.model_combo.setCurrentIndex(1)
        depth = int(self.orbit_depth_spin.value())
        best_selector = 0 if self.selected_model() == self.MODEL_CYCLIC else ""
        best_d = float("inf")
        if isinstance(getattr(self, "surface", None), ModularFordSurface) and self.modular_safe_mode_active():
            orbit = self.guarded_modular_orbit_points(self.q, depth)
        else:
            orbit = self.surface.orbit_points(self.q, depth)
        for label, pt in orbit:
            if not inside_unit_disk(pt, margin=1.0e-8):
                continue
            d = poincare_distance(self.p, pt)
            if d < best_d:
                best_selector = label
                best_d = d
        if not np.isfinite(best_d):
            self.status.setText("Status: no finite searched orbit candidate found.")
            return
        if self.selected_model() == self.MODEL_CYCLIC:
            best_n = int(best_selector)
            self.exponent_spin.setValue(best_n)
            self.word_edit.setText(exponent_to_word(best_n))
            self.last_search_summary = (
                f"Shortest searched cyclic image: A^{best_n}(q), search n in [-{depth},{depth}]\n"
                f"searched distance = {best_d: .6f}"
            )
        elif self.selected_model() in (self.MODEL_TWOGEN, self.MODEL_OCTAGON, self.MODEL_POLYGON, self.MODEL_SCHOTTKY, self.MODEL_MODULAR_FORD):
            best_word = str(best_selector)
            self.current_word = best_word
            self.word_edit.setText(best_word)
            self.last_search_summary = (
                f"Shortest searched word image: {best_word or 'identity'}(q), reduced-word depth <= {depth}\n"
                f"searched distance = {best_d: .6f}"
            )
        self.update_group_state_from_ui(reset_curves=True)
        self.status.setText("Status: selected shortest candidate in the finite orbit search range.")
        self.update_diagnostics()
        self.refresh_plot()


    def reset_neural_metric(self):
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "PyTorch missing", "Install PyTorch to use neural metric learning.")
            return
        self.metric_model = MetricPhiNetwork()
        self.metric_loss_history = None
        self.metric_last_stats = None
        self.metric_trained = False
        self.metric_status_label.setText("Neural metric status: reset; not trained.")
        self.status.setText("Status: neural metric reset.")
        self.update_diagnostics()
        self.refresh_plot()

    def train_neural_metric(self):
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "PyTorch missing", "Install PyTorch to use neural metric learning:\n\n    pip install torch")
            return
        if self.metric_model is None:
            self.metric_model = MetricPhiNetwork()
        steps = int(self.metric_steps_spin.value())
        samples = int(self.metric_samples_spin.value())
        lr = float(self.metric_lr_spin.value())
        rmax = float(self.metric_rmax_spin.value())
        br = float(self.metric_boundary_spin.value())
        if br <= rmax:
            br = min(0.985, rmax + 0.06)
            self.metric_boundary_spin.setValue(br)

        self.train_metric_btn.setEnabled(False)
        self.reset_metric_btn.setEnabled(False)
        self.status.setText("Status: training neural metric phi_theta by Liouville residual...")
        QApplication.processEvents()

        def callback(k: int, stats: dict):
            self.metric_status_label.setText(
                f"Training neural metric... step {k}/{steps}; "
                f"loss={stats['loss']:.3e}, PDE={stats['pde']:.3e}, "
                f"mean |error|={stats['mean_abs_error']:.3e}"
            )
            self.status.setText(f"Status: metric training step {k}/{steps}")
            QApplication.processEvents()

        try:
            hist, stats = train_metric_network_pytorch(
                self.metric_model,
                steps=steps,
                samples=samples,
                lr=lr,
                r_max=rmax,
                boundary_r=br,
                callback=callback,
                callback_every=max(5, steps // 40),
            )
            self.metric_loss_history = hist
            self.metric_last_stats = stats
            self.metric_trained = True
            self.metric_status_label.setText(
                f"Neural metric trained: loss={stats['loss']:.3e}, "
                f"mean |phi-theta - phi-true|={stats['mean_abs_error']:.3e}, "
                f"max |error|={stats['max_abs_error']:.3e}."
            )
            self.status.setText("Status: neural metric training complete.")
            self.update_diagnostics()
            self.refresh_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Metric training failed", str(exc))
            self.status.setText("Status: neural metric training failed. See error dialog.")
        finally:
            self.train_metric_btn.setEnabled(True)
            self.reset_metric_btn.setEnabled(True)

    def learned_metric_available(self) -> bool:
        return bool(TORCH_AVAILABLE and self.metric_model is not None and self.metric_trained)

    def learned_phi_grid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pts = np.stack([X, Y], axis=-1)
        return evaluate_metric_network_np(self.metric_model, pts)

    def open_metric_surface(self):
        if self.metric_window is None:
            self.metric_window = MetricSurfaceWindow(self)
        self.metric_window.plot_metric(
            p=self.p,
            q=self.q,
            q_lift=self.q_lift,
            exact_curve=self.exact_curve if self.show_exact_box.isChecked() else None,
            neural_curve=self.neural_curve if self.show_neural_box.isChecked() else None,
        )
        self.metric_window.show()
        self.metric_window.raise_()
        self.metric_window.activateWindow()
        self.status.setText("Status: opened/updated universal-cover metric landscape.")

    # ------------------------------------------------------------------
    # Computations for display
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # v12 workflow, audit, and JSON helpers
    # ------------------------------------------------------------------

    def invalidate_tessellation_cache(self):
        self.tessellation_cache_key = None
        self.tessellation_cache_edges = []
        self.tessellation_cache_segments = []
        self.orbit_cache_key = None
        self.orbit_cache_points = []

    def invalidate_tessellation_cache_and_refresh(self):
        self.invalidate_tessellation_cache()
        self.refresh_plot()

    def workflow_cyclic_shortest(self):
        self.load_cyclic_example()
        self.find_shortest_orbit_candidate()

    def disk_mobius_to_json_spec(self, g: DiskMobius) -> dict:
        return {
            "type": "su11",
            "alpha": [float(g.alpha.real), float(g.alpha.imag)],
            "beta": [float(g.beta.real), float(g.beta.imag)],
        }

    def surface_to_json_data(self, surface: FuchsianSurface) -> dict:
        """Export a surface in the v12 generator-ready JSON shape.

        A future standalone certified-domain script should write this same
        structure.  The GUI treats the data by fiat as already certified.
        """
        data = {
            "format": "FuchsianGENN surface JSON v12",
            "name": getattr(surface, "name", surface.display_name),
            "model_hint": surface.key,
            "certification": "exported from a built-in certified surface" if surface.key == self.MODEL_OCTAGON else "user/external certification required",
            "generators": {},
            "polygon_vertices": [],
            "side_pairings": [],
            "notes": "Load this JSON in v12 Surface Data. The program assumes the supplied (Gamma,F,side-pairings) are certified.",
        }
        if hasattr(surface, "generators"):
            for name in sorted(k for k in surface.generators if len(k) == 1 and k.isupper()):
                data["generators"][name] = self.disk_mobius_to_json_spec(surface.generators[name])
        if hasattr(surface, "polygon_vertices"):
            data["polygon_vertices"] = [[float(x), float(y)] for x, y in np.asarray(surface.polygon_vertices, dtype=float)]
        if hasattr(surface, "side_pairings"):
            data["side_pairings"] = list(surface.side_pairings)
        if surface.key == self.MODEL_OCTAGON:
            data["genus"] = 2
            data["area"] = 4.0 * math.pi
            data["polygon_description"] = "regular {8,8} octagon, opposite-side pairings"
        return data

    def export_octagon_json_to_editor(self):
        surf = RegularOctagonGenus2Surface()
        self.surface_json_edit.setPlainText(json.dumps(self.surface_to_json_data(surf), indent=2))
        self.status.setText("Status: exported built-in certified octagon surface JSON to the editor.")

    def load_surface_json_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load certified Fuchsian surface JSON", "", "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            # Validate JSON syntax before putting it in the editor.
            json.loads(text)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load JSON", str(exc))
            return
        self.surface_json_edit.setPlainText(text)
        self.status.setText(f"Status: loaded surface JSON file {path} into the editor. Click Apply to use it, or use Points -> Load surface JSON file… to load and apply in one step.")

    def geometry_audit_text(self) -> str:
        surface = self.surface if hasattr(self, "surface") else self.make_surface()
        lines = []
        lines.append("GEOMETRY AUDIT")
        lines.append("==============")
        lines.append(f"surface: {surface.display_name}")
        lines.append(f"interface: {surface.__class__.__name__}")
        lines.append("")
        lines.append("Certification status")
        lines.append("--------------------")
        if isinstance(surface, TrivialDiskSurface):
            lines.append("CERTIFIED: exact Poincare disk with trivial group.")
        elif isinstance(surface, CyclicQuotientSurface):
            lines.append("CERTIFIED: cyclic group generated by a hyperbolic disk isometry.")
            lines.append("Fundamental domain: Dirichlet strip for basepoint 0.")
        elif isinstance(surface, RegularOctagonGenus2Surface):
            lines.append("CERTIFIED BUILT-IN: regular hyperbolic {8,8} octagon, opposite sides paired.")
            lines.append("Interior angle: pi/4.  Quotient genus: 2.  Area: 4*pi.")
        elif isinstance(surface, SchottkyIdealGeodesicSurface):
            status = surface.certification.get("status", "")
            lines.append(f"SCHOTTKY / IDEAL-GEODESIC DOMAIN: {status or 'by fiat / not supplied'}")
            construction = surface.certification.get("construction")
            if construction:
                lines.append(f"construction: {construction}")
            warning = surface.certification.get("warning")
            if warning:
                lines.append(f"note: {warning}")
            lines.append("The Explorer displays ideal geodesic sides and their images; it does not prove discreteness from scratch.")
        elif isinstance(surface, ModularFordSurface):
            status = surface.certification.get("status", "")
            md = surface.modular_metadata()
            subgroup = md.get("subgroup") or "PSL(2,Z) / not supplied"
            if status:
                lines.append(f"MODULAR / FORD METADATA: {status}")
            else:
                lines.append("MODULAR / FORD DOMAIN: certification metadata not supplied")
            construction = surface.certification.get("construction")
            if construction:
                lines.append(f"construction: {construction}")
            audit_note = surface.certification.get("audit")
            if audit_note:
                lines.append(f"audit note: {audit_note}")
            warning = surface.certification.get("warning")
            if warning:
                lines.append(f"note: {warning}")
            if md.get("torsion_free") is True:
                lines.append(f"{subgroup} is labeled torsion-free by the generator; H/Gamma is a smooth noncompact finite-area Riemann surface before cusp compactification.")
            else:
                lines.append(f"{subgroup} is treated as a modular/Ford finite-area domain; if elliptic points are present, the quotient is an orbifold.")
            lines.append("The Explorer consumes modular metadata and displays the cleaned Ford boundary; it does not independently prove subgroup arithmetic from scratch.")
        elif isinstance(surface, UserPolygonFuchsianSurface):
            status = surface.certification.get("status", "")
            if status:
                lines.append(f"EXTERNALLY CERTIFIED METADATA: {status}")
                construction = surface.certification.get("construction")
                if construction:
                    lines.append(f"construction: {construction}")
                audit_note = surface.certification.get("audit")
                if audit_note:
                    lines.append(f"audit note: {audit_note}")
                lines.append("The Explorer consumes this metadata and performs consistency checks; it does not prove discreteness/fundamentality from scratch.")
            else:
                lines.append("BY FIAT / ADVANCED: user or external script must certify (Gamma,F,side-pairings).")
                lines.append("The GUI consumes this data; it does not prove discreteness or fundamentality.")
        else:
            lines.append("No certification information for this surface class.")

        lines.append("")
        lines.append("Surface diagnostics")
        lines.append("-------------------")
        for k, v in surface.diagnostics():
            lines.append(f"{k:<28} {v}")

        # Modular/Ford-domain files produced by Domain Maker v6-v8 carry
        # arithmetic/cusp/torsion-free metadata.  This section makes the
        # noncompact modular Riemann-surface status visible in the Explorer.
        if isinstance(surface, ModularFordSurface):
            md = surface.modular_metadata()
            lines.append("")
            lines.append("Modular / Ford-domain metadata")
            lines.append("------------------------------")
            subgroup = md.get("subgroup") or "PSL(2,Z) / not supplied"
            lines.append(f"subgroup                    {subgroup}")
            if md.get("level_N") is not None:
                lines.append(f"level N                     {md.get('level_N')}")
            if md.get("subdomain_type"):
                lines.append(f"subdomain type              {md.get('subdomain_type')}")
            if md.get("index_in_psl2z") is not None:
                lines.append(f"index in PSL(2,Z)           {md.get('index_in_psl2z')}")
            if md.get("area") is not None:
                try:
                    lines.append(f"hyperbolic area             {float(md.get('area')):.9f}")
                except Exception:
                    lines.append(f"hyperbolic area             {md.get('area')}")
            lines.append(f"torsion-free                {md.get('torsion_free')}")
            lines.append(f"compact before cusps        {md.get('compact')}")
            if md.get("compactified_genus") is not None:
                lines.append(f"compactified genus          {md.get('compactified_genus')}")
            if md.get("cusp_count") is not None:
                lines.append(f"cusp count                  {md.get('cusp_count')}")
            if md.get("cusp_widths") is not None:
                lines.append(f"cusp widths                 {md.get('cusp_widths')}")
            if md.get("elliptic_orders") not in (None, [], ""):
                lines.append(f"elliptic orders/points      {md.get('elliptic_orders')}")
            if md.get("riemann_surface_status"):
                lines.append(f"Riemann-surface status      {md.get('riemann_surface_status')}")
            if md.get("kahler_status"):
                lines.append(f"Kähler status               {md.get('kahler_status')}")
            cleanup = md.get("boundary_cleanup_audit") or {}
            if isinstance(cleanup, dict) and cleanup:
                lines.append("")
                lines.append("Ford-domain boundary cleanup")
                lines.append("----------------------------")
                for key in ["construction_edges_total", "exterior_boundary_edges", "internal_construction_edges"]:
                    if cleanup.get(key) is not None:
                        lines.append(f"{key:<32} {cleanup.get(key)}")
                if md.get("construction_tile_count") is not None:
                    lines.append(f"construction tiles              {md.get('construction_tile_count')}")
                if md.get("exterior_edge_count") is not None:
                    lines.append(f"displayed exterior sides        {md.get('exterior_edge_count')}")
                if md.get("internal_edge_count") is not None:
                    lines.append(f"hidden internal scaffold edges  {md.get('internal_edge_count')}")
            tfa = md.get("torsion_free_audit") or {}
            if isinstance(tfa, dict) and tfa:
                lines.append("")
                lines.append("Torsion-free audit")
                lines.append("------------------")
                for key, val in tfa.items():
                    lines.append(f"{str(key):<32} {val}")

        # Compact-polygon files produced by Domain Maker v4 may carry a
        # triangle_source block.  This is how we represent the first
        # torsion-free compact-surface bridge from triangle-group ideas without
        # pretending that the triangle orbifold itself is a smooth surface.
        if isinstance(surface, UserPolygonFuchsianSurface):
            tri_src = surface.data.get("triangle_source", {})
            if isinstance(tri_src, dict) and tri_src:
                lines.append("")
                lines.append("Triangle-source / Hurwitz-pathway metadata")
                lines.append("-------------------------------------------")
                if tri_src.get("triangle_signature"):
                    lines.append(f"triangle signature          {tri_src.get('triangle_signature')}")
                if tri_src.get("role"):
                    lines.append(f"role                        {tri_src.get('role')}")
                if tri_src.get("surface_area_over_orbifold_area_pi_over_21") is not None:
                    try:
                        lines.append(f"area / (pi/21)              {float(tri_src.get('surface_area_over_orbifold_area_pi_over_21')):.6f}")
                    except Exception:
                        lines.append(f"area / (pi/21)              {tri_src.get('surface_area_over_orbifold_area_pi_over_21')}")
                if tri_src.get("hurwitz_automorphism_status"):
                    lines.append(f"Hurwitz automorphism status {tri_src.get('hurwitz_automorphism_status')}")

        if hasattr(surface, "polygon_vertices") and len(getattr(surface, "polygon_vertices")):
            verts = np.asarray(surface.polygon_vertices, dtype=float)
            lines.append("")
            lines.append("Fundamental polygon checks")
            lines.append("--------------------------")
            lines.append(f"vertex count                {len(verts)}")
            radii = np.sqrt(np.sum(verts * verts, axis=1))
            lines.append(f"max |vertex|               {float(np.max(radii)):.9f}")
            lines.append(f"all vertices inside disk   {bool(np.all(radii < 1.0))}")
            if isinstance(surface, RegularOctagonGenus2Surface):
                lines.append(f"expected Euclidean radius  {surface.rho:.9f}")
                lines.append(f"radius spread              {float(np.max(radii)-np.min(radii)):.3e}")
                lines.append("side-pair endpoint checks")
                for sp in surface.side_pairings:
                    i = int(sp["side"]); j = int(sp["paired_with"]); word = str(sp["word"])
                    g = surface.generators[word]
                    v_i = verts[i]; v_ip1 = verts[(i+1)%len(verts)]
                    target_1 = verts[(j+1)%len(verts)]; target_2 = verts[j]
                    err1 = np.linalg.norm(g.apply_point(v_i) - target_1)
                    err2 = np.linalg.norm(g.apply_point(v_ip1) - target_2)
                    lines.append(f"  {word}: side {i}->{j} endpoint errors {err1:.3e}, {err2:.3e}")
            elif isinstance(surface, UserPolygonFuchsianSurface):
                # If the JSON includes precomputed endpoint-audit information, show it.
                ext_audit = surface.data.get("side_pairing_endpoint_audit", [])
                if isinstance(ext_audit, list) and ext_audit:
                    lines.append("side-pair endpoint checks  supplied by external generator")
                    for item in ext_audit[:12]:
                        if isinstance(item, dict):
                            lines.append(f"  {item.get('word', '?')}: endpoint_error {float(item.get('endpoint_error', float('nan'))):.3e}")
                    if len(ext_audit) > 12:
                        lines.append(f"  ... {len(ext_audit)-12} more audit rows")
                # Also perform a lightweight local endpoint check when pairings and
                # generators use the expected reversed orientation convention.
                computed = []
                for sp in getattr(surface, "side_pairings", []):
                    if not isinstance(sp, dict):
                        continue
                    word = str(sp.get("word", ""))
                    try:
                        i = int(sp.get("side")); j = int(sp.get("paired_with"))
                        g = surface.gamma(word)
                        nverts = len(verts)
                        v_i = verts[i]; v_ip1 = verts[(i+1)%nverts]
                        target_1 = verts[(j+1)%nverts]; target_2 = verts[j]
                        err1 = np.linalg.norm(g.apply_point(v_i) - target_1)
                        err2 = np.linalg.norm(g.apply_point(v_ip1) - target_2)
                        computed.append((word, i, j, err1, err2))
                    except Exception:
                        pass
                if computed:
                    lines.append("side-pair endpoint checks  computed by Explorer (reversed orientation assumption)")
                    for word, i, j, err1, err2 in computed[:12]:
                        lines.append(f"  {word}: side {i}->{j} endpoint errors {err1:.3e}, {err2:.3e}")
                    if len(computed) > 12:
                        lines.append(f"  ... {len(computed)-12} more computed rows")
                if not (isinstance(ext_audit, list) and ext_audit) and not computed:
                    lines.append("side-pair endpoint checks  not performed; external certification required")
                vaudit = surface.data.get("vertex_angle_audit", [])
                if isinstance(vaudit, list) and vaudit:
                    lines.append("vertex angle checks       supplied by external generator")
                    for item in vaudit[:12]:
                        if isinstance(item, dict):
                            try:
                                lines.append(f"  class size {int(item.get('count', 0)):2d}: total angle {float(item.get('total_angle', float('nan'))):.9f}, error from 2pi {float(item.get('smooth_error_from_2pi', float('nan'))):.3e}")
                            except Exception:
                                lines.append(f"  {item}")
                    if len(vaudit) > 12:
                        lines.append(f"  ... {len(vaudit)-12} more vertex-angle rows")

        if isinstance(surface, SchottkyIdealGeodesicSurface):
            lines.append("")
            lines.append("Ideal-geodesic domain checks")
            lines.append("-----------------------------")
            lines.append(f"ideal geodesic side count  {len(surface.geodesic_sides)}")
            ok = True
            for side in surface.geodesic_sides[:12]:
                try:
                    idx = side.get("side", "?")
                    e0, e1 = side.get("ideal_endpoints", [])
                    r0 = float(np.linalg.norm(np.asarray(e0, dtype=float)))
                    r1 = float(np.linalg.norm(np.asarray(e1, dtype=float)))
                    lines.append(f"  side {idx}: |endpoint radii|-1 = {abs(r0-1):.3e}, {abs(r1-1):.3e}")
                    ok = ok and abs(r0-1) < 1e-6 and abs(r1-1) < 1e-6
                except Exception:
                    ok = False
            if len(surface.geodesic_sides) > 12:
                lines.append(f"  ... {len(surface.geodesic_sides)-12} more ideal sides")
            lines.append(f"endpoints on unit circle   {ok}")

        lines.append("")
        lines.append("Generator-ready JSON")
        lines.append("--------------------")
        lines.append("A standalone certified-domain generator should export the v12 JSON shape:")
        lines.append("  format, domain_type=compact_polygon, v12_polygon_compatible=true, name,")
        lines.append("  generators, polygon_vertices, side_pairings, optional genus/area/certification/audit.")
        lines.append("Use Surface Data -> Load JSON file, then Apply user surface/domain JSON.")
        return "\n".join(lines)

    def update_geometry_audit(self):
        if hasattr(self, "audit_text"):
            self.audit_text.setPlainText(self.geometry_audit_text())

    # ------------------------------------------------------------------
    # v17.6 performance guards for large modular congruence examples
    # ------------------------------------------------------------------

    def current_modular_metadata(self) -> dict:
        if isinstance(getattr(self, "surface", None), ModularFordSurface):
            try:
                return self.surface.modular_metadata()
            except Exception:
                return {}
        return {}

    def modular_complexity_estimate(self) -> dict:
        md = self.current_modular_metadata()
        gens = self.surface_uppercase_generators() if hasattr(self, "surface_uppercase_generators") else []
        exterior = md.get("exterior_edge_count")
        index = md.get("index_in_psl2z")
        try:
            exterior_i = int(exterior) if exterior is not None else len(getattr(self.surface, "ford_sides", []))
        except Exception:
            exterior_i = len(getattr(self.surface, "ford_sides", []))
        try:
            index_i = int(index) if index is not None else 0
        except Exception:
            index_i = 0
        return {
            "is_modular": isinstance(getattr(self, "surface", None), ModularFordSurface),
            "index": index_i,
            "generator_count": len(gens),
            "exterior_edge_count": exterior_i,
        }

    def modular_surface_is_large(self) -> bool:
        c = self.modular_complexity_estimate()
        return bool(c.get("is_modular") and (c.get("index", 0) >= 50 or c.get("generator_count", 0) >= 12 or c.get("exterior_edge_count", 0) >= 40))

    def modular_safe_mode_active(self) -> bool:
        # v17.6 deliberately removes automatic safe/deep modular mode switching.
        # Large domains remain responsive because the principal/base domain is drawn
        # by default; finite word-patch tessellation is user-triggered.
        return False

    def deep_modular_fingerprint_enabled(self) -> bool:
        return bool(getattr(self, "invariants_deep_run_box", None) is not None and self.invariants_deep_run_box.isChecked())

    def modular_safe_note(self) -> str:
        c = self.modular_complexity_estimate()
        if not c.get("is_modular"):
            return ""
        if not getattr(self, "tessellation_requested", False):
            return (
                f"v17.6 principal/base-domain display: index={c.get('index')}, "
                f"generators={c.get('generator_count')}, exterior sides={c.get('exterior_edge_count')}. "
                "No finite word-patch tessellation is drawn until the Tessellate button is pressed; metadata audit values are unchanged."
            )
        return "v17.6 finite word-patch tessellation requested by user; computations may be slow for high-index congruence domains."

    def limited_modular_alphabet(self, max_generators: int = 6) -> str:
        try:
            uppers = sorted(ch for ch in self.surface.generators if isinstance(ch, str) and len(ch) == 1 and ch.isupper())
        except Exception:
            uppers = []
        uppers = uppers[:max(1, int(max_generators))]
        chars = []
        for ch in uppers:
            chars.append(ch)
            if ch.lower() in getattr(self.surface, "generators", {}):
                chars.append(ch.lower())
        return "".join(chars)

    def generate_limited_word_list(self, depth: int, alphabet: str, max_words: int) -> List[str]:
        words = []
        for w in generate_reduced_words(int(depth), alphabet=alphabet):
            if w:
                words.append(w)
                if len(words) >= int(max_words):
                    break
        return words

    def modular_guarded_selectors(self, requested_depth: int, max_words: int = 2000, for_spectrum: bool = False) -> Tuple[List[str], dict]:
        if not isinstance(getattr(self, "surface", None), ModularFordSurface):
            return [], {"guard_active": False}
        guard = self.modular_safe_mode_active()
        large = self.modular_surface_is_large()
        if guard and large:
            # Depth 1 and a subset of generators keeps GUI operations responsive.
            depth = min(int(requested_depth), 1)
            alphabet = self.limited_modular_alphabet(max_generators=6 if not for_spectrum else 8)
            cap = min(int(max_words), 600 if not for_spectrum else 1200)
        elif guard:
            depth = min(int(requested_depth), 2)
            alphabet = self.limited_modular_alphabet(max_generators=10)
            cap = min(int(max_words), 1800)
        else:
            depth = int(requested_depth)
            alphabet = self.surface.available_alphabet()
            cap = int(max_words)
        words = self.generate_limited_word_list(depth, alphabet, cap) if alphabet else []
        info = {
            "guard_active": bool(guard),
            "large_modular_surface": bool(large),
            "requested_depth": int(requested_depth),
            "effective_depth": int(depth),
            "alphabet_used": alphabet,
            "word_cap": int(cap),
            "words_generated": int(len(words)),
        }
        return words, info

    def guarded_modular_orbit_points(self, q: Optional[np.ndarray], depth: int) -> List[Tuple[str, np.ndarray]]:
        if q is None:
            return []
        words, info = self.modular_guarded_selectors(depth, max_words=1200, for_spectrum=False)
        out: List[Tuple[str, np.ndarray]] = [("", q)]
        for word in words:
            try:
                pt = self.surface.gamma(word).apply_point(q)
                if inside_unit_disk(pt, margin=1.0e-8):
                    out.append((word, pt))
            except Exception:
                continue
        self.last_modular_guard_info = info
        return out

    def compute_orbit_points(self) -> List[Tuple[int, np.ndarray]]:
        depth = int(self.orbit_depth_spin.value())
        q_key = None if self.q is None else tuple(np.round(np.asarray(self.q, dtype=float), 12))
        guard_key = None
        if isinstance(getattr(self, "surface", None), ModularFordSurface):
            guard_key = (self.modular_safe_mode_active(), self.modular_surface_is_large(), self.deep_modular_fingerprint_enabled())
        key = (self.selected_model(), surface_signature(self.surface), depth, q_key, guard_key)
        if key != self.orbit_cache_key:
            if isinstance(getattr(self, "surface", None), ModularFordSurface) and self.modular_safe_mode_active():
                self.orbit_cache_points = self.guarded_modular_orbit_points(self.q, depth)
            else:
                self.orbit_cache_points = self.surface.orbit_points(self.q, depth)
            self.orbit_cache_key = key
        return self.orbit_cache_points

    def surface_has_polygon_vertices(self) -> bool:
        return hasattr(self.surface, "polygon_vertices") and len(getattr(self.surface, "polygon_vertices", [])) >= 3 and hasattr(self.surface, "gamma")

    def transformed_polygon_vertices(self, word: str = "") -> Optional[np.ndarray]:
        if not self.surface_has_polygon_vertices():
            return None
        verts = np.asarray(getattr(self.surface, "polygon_vertices"), dtype=float)
        try:
            return self.surface.gamma(word).apply_curve(verts)
        except Exception:
            return None

    def polygon_edge_curves_from_vertices(self, verts: np.ndarray, samples_per_edge: int = 36) -> List[np.ndarray]:
        verts = np.asarray(verts, dtype=float)
        if verts.ndim != 2 or len(verts) < 3:
            return []
        curves = []
        for i in range(len(verts)):
            curves.append(exact_poincare_geodesic(verts[i], verts[(i + 1) % len(verts)], n=max(8, int(samples_per_edge))))
        return curves

    def compute_polygon_tessellation_tiles(self, depth_override: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """Return transformed polygon vertices for finite gamma(F) patch.

        This is mainly a visual/debug helper.  It assumes the current surface has
        polygon_vertices and a gamma(word) method.  The optional duplicate filter
        removes visually duplicated tiles by rounded Euclidean centers; this is
        not a group-theoretic word-reduction proof, but it keeps finite patches
        from becoming misleadingly overdrawn when the supplied presentation has
        relations.
        """
        if not self.surface_has_polygon_vertices():
            return []
        depth = int(self.orbit_depth_spin.value() if depth_override is None else depth_override)
        try:
            alphabet = self.surface.available_alphabet()
        except Exception:
            alphabet = ""
        if not alphabet:
            return []
        words = generate_reduced_words(depth, alphabet=alphabet)
        use_filter = bool(hasattr(self, "filter_duplicate_tiles_box") and self.filter_duplicate_tiles_box.isChecked())
        seen = set()
        out: List[Tuple[str, np.ndarray]] = []
        for word in words:
            poly = self.transformed_polygon_vertices(word)
            if poly is None:
                continue
            if np.any(np.sum(poly * poly, axis=1) >= 1.000001):
                # Numerical safety.  True disk isometries should preserve the disk,
                # but avoid rendering pathological bad user data.
                continue
            if use_filter:
                center = np.mean(poly, axis=0)
                key = tuple(np.round(center, 5))
                if key in seen:
                    continue
                seen.add(key)
            out.append((word, poly))
        return out

    def compute_polygon_tessellation_edges(self, depth: int, samples: int) -> List[Tuple[str, np.ndarray]]:
        tiles = self.compute_polygon_tessellation_tiles(depth_override=depth)
        samples_per_edge = max(18, int(samples) // 8)
        out: List[Tuple[str, np.ndarray]] = []
        for word, poly in tiles:
            for edge in self.polygon_edge_curves_from_vertices(poly, samples_per_edge=samples_per_edge):
                out.append((word, edge))
        return out

    def draw_base_polygon_shade(self, ax):
        if not (hasattr(self, "shade_base_polygon_box") and self.shade_base_polygon_box.isChecked()):
            return
        poly = self.transformed_polygon_vertices("")
        if poly is None:
            return
        try:
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.08, zorder=1, label="base F")
        except Exception:
            pass

    def draw_tessellation_debug_view(self, ax):
        """Draw a deliberately small, labeled patch to audit the tessellation visually."""
        if not self.surface_has_polygon_vertices():
            return False
        tiles = self.compute_polygon_tessellation_tiles(depth_override=1)
        if not tiles:
            return False
        self.draw_base_polygon_shade(ax)
        for word, poly in tiles:
            is_base = (word == "")
            color = "0.10" if is_base else "0.35"
            lw = 1.6 if is_base else 0.85
            alpha = 0.92 if is_base else 0.55
            for edge in self.polygon_edge_curves_from_vertices(poly, samples_per_edge=40):
                ax.plot(edge[:, 0], edge[:, 1], color=color, linewidth=lw, alpha=alpha, zorder=2)
            if hasattr(self, "show_tile_labels_box") and self.show_tile_labels_box.isChecked():
                c = np.mean(poly, axis=0)
                lab = "F" if word == "" else word
                ax.text(c[0], c[1], lab, fontsize=8 if is_base else 7, ha="center", va="center", alpha=0.85, zorder=3)
        ax.plot([], [], color="0.10", linewidth=1.5, label="debug: central F + depth-1 neighbors")
        return True

    def compute_strip_edges(self) -> List[Tuple[int, np.ndarray]]:
        requested_depth = int(self.orbit_depth_spin.value())
        depth = requested_depth
        fast = bool(hasattr(self, "fast_tessellation_box") and self.fast_tessellation_box.isChecked())
        samples = 126 if fast else 260
        guard_info = None
        if self.surface.supports_strip_boundaries and not getattr(self, "tessellation_requested", False):
            # v17.6 default: draw only the supplied/base fundamental domain.
            # This isolates whether the previous slowdown was caused by tessellation rendering.
            depth = 0
            samples = min(samples, 126)
            guard_info = ("principal_domain_only", requested_depth, depth)
        # The cache key deliberately uses the current surface description and key
        # rather than object identity, because make_surface() may create a fresh
        # equivalent object after UI changes.
        key = (
            self.selected_model(),
            surface_signature(self.surface),
            depth,
            samples,
            fast,
            guard_info,
        )
        if key != self.tessellation_cache_key:
            if self.surface_has_polygon_vertices() and bool(hasattr(self, "filter_duplicate_tiles_box") and self.filter_duplicate_tiles_box.isChecked()):
                self.tessellation_cache_edges = self.compute_polygon_tessellation_edges(depth=depth, samples=samples)
            else:
                self.tessellation_cache_edges = self.surface.strip_edges(depth=depth, samples=samples)
            self.tessellation_cache_segments = [edge for _, edge in self.tessellation_cache_edges]
            self.tessellation_cache_key = key
        return self.tessellation_cache_edges

    def selected_distance(self) -> Optional[float]:
        if self.p is None or self.q_lift is None:
            return None
        return poincare_distance(self.p, self.q_lift)


    # ------------------------------------------------------------------
    # Geometry / invariants prototype
    # ------------------------------------------------------------------

    def surface_uppercase_generators(self) -> List[str]:
        gens = getattr(self.surface, "generators", {})
        return sorted(ch for ch in gens.keys() if isinstance(ch, str) and len(ch) == 1 and ch.isupper())

    def mobius_trace_real(self, g: DiskMobius) -> float:
        """Real trace of the SU(1,1) matrix [[alpha,beta],[conj(beta),conj(alpha)]]."""
        return float((g.alpha + g.alpha.conjugate()).real)

    def classify_mobius(self, g: DiskMobius, tol: Optional[float] = None) -> Tuple[str, float, float]:
        """Classify a disk Mobius element with cusp-aware numerical tolerances.

        Modular/cusped examples contain exact parabolic elements (|tr|=2),
        and Reidemeister-Schreier generator lists may contain word products
        that are identity relations even though they are not freely reduced by
        label.  Roundoff can otherwise make these look like tiny hyperbolic
        translations.  v17.6 separates identity/relation and parabolic/cusp
        elements from genuine hyperbolic length candidates.
        """
        if tol is None:
            tol = 1.0e-7 if isinstance(getattr(self, "surface", None), ModularFordSurface) else 1.0e-9
        tr = self.mobius_trace_real(g)
        atr = abs(tr)
        beta_abs = abs(complex(getattr(g, "beta", 0.0)))
        alpha_abs = abs(complex(getattr(g, "alpha", 1.0)))
        # In PSU(1,1), +I and -I act as the same identity transformation.
        if beta_abs <= 10.0 * tol and abs(alpha_abs - 1.0) <= 10.0 * tol:
            return "identity_or_relation", 0.0, tr
        if atr > 2.0 + tol:
            length = 2.0 * math.acosh(max(atr / 2.0, 1.0))
            return "hyperbolic", length, tr
        if abs(atr - 2.0) <= tol:
            return "parabolic_or_near_parabolic", 0.0, tr
        angle = 2.0 * math.acos(max(-1.0, min(1.0, atr / 2.0)))
        return "elliptic", angle, tr

    def word_length_spectrum(self, depth: int, max_rows: int) -> List[dict]:
        if self.selected_model() == self.MODEL_DISK:
            return []
        if self.selected_model() == self.MODEL_CYCLIC:
            words = [exponent_to_word(n) for n in range(-depth, depth + 1) if n != 0]
        else:
            try:
                if isinstance(getattr(self, "surface", None), ModularFordSurface) and self.modular_safe_mode_active():
                    words, info = self.modular_guarded_selectors(depth, max_words=max(80, int(max_rows) * 20), for_spectrum=True)
                    self.last_modular_guard_info = info
                else:
                    alphabet = self.surface.available_alphabet()
                    words = [w for w in generate_reduced_words(depth, alphabet=alphabet) if w] if alphabet else []
            except Exception:
                words = []
            if not words:
                return []
        rows = []
        seen = set()
        for word in words:
            try:
                if self.selected_model() == self.MODEL_CYCLIC:
                    n = parse_cyclic_word(word)
                    g = self.surface.gamma(n)
                    display = f"A^{n}" if n != 0 else "identity"
                else:
                    rw = reduce_fuchsian_word(word)
                    if not rw or rw in seen:
                        continue
                    seen.add(rw)
                    g = self.surface.gamma(rw)
                    display = rw
                kind, val, tr = self.classify_mobius(g)
                rows.append({
                    "word": display,
                    "type": kind,
                    "value": float(val),
                    "trace": float(tr),
                    "trace_abs_minus_2": float(abs(tr) - 2.0),
                })
            except Exception:
                continue
        # Hyperbolic length candidates first, then elliptic/parabolic/identity-like
        # rows.  This prevents modular cusp elements from becoming artificial
        # systoles just because trace roundoff is slightly above 2.
        rank = {"hyperbolic": 0, "elliptic": 1, "parabolic_or_near_parabolic": 2, "identity_or_relation": 3}
        rows.sort(key=lambda r: (rank.get(r["type"], 9), r["value"], len(r["word"]), r["word"]))
        return rows[:max_rows]

    def sample_points_for_injectivity(self, n: int) -> np.ndarray:
        n = int(max(1, n))
        rng = np.random.default_rng(12345)
        if self.surface_has_polygon_vertices():
            verts = np.asarray(getattr(self.surface, "polygon_vertices"), dtype=float)
            if len(verts) >= 3:
                c = np.mean(verts, axis=0)
                pts = []
                for _ in range(n):
                    i = int(rng.integers(0, len(verts)))
                    a = verts[i]
                    b = verts[(i + 1) % len(verts)]
                    u = rng.random()
                    v = rng.random()
                    if u + v > 1.0:
                        u = 1.0 - u
                        v = 1.0 - v
                    pt = c + u * (a - c) + v * (b - c)
                    if inside_unit_disk(pt, margin=1.0e-8):
                        pts.append(pt)
                return np.array(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)
        # Fallback for Schottky, cyclic, or by-fiat modes: sample a moderate disk.
        pts = []
        for _ in range(n):
            r = 0.72 * math.sqrt(float(rng.random()))
            th = 2.0 * math.pi * float(rng.random())
            pts.append([r * math.cos(th), r * math.sin(th)])
        return np.array(pts, dtype=float)

    def injectivity_radius_estimate(self, depth: int, samples: int) -> dict:
        pts = self.sample_points_for_injectivity(samples)
        if len(pts) == 0 or self.selected_model() == self.MODEL_DISK:
            return {"available": False, "reason": "No quotient group or no sample points."}
        if self.selected_model() == self.MODEL_CYCLIC:
            selectors = [n for n in range(-depth, depth + 1) if n != 0]
        else:
            try:
                if isinstance(getattr(self, "surface", None), ModularFordSurface) and self.modular_safe_mode_active():
                    selectors, info = self.modular_guarded_selectors(depth, max_words=1200, for_spectrum=False)
                    self.last_modular_guard_info = info
                else:
                    alphabet = self.surface.available_alphabet()
                    selectors = [w for w in generate_reduced_words(depth, alphabet=alphabet) if w] if alphabet else []
            except Exception:
                selectors = []
        if not selectors:
            return {"available": False, "reason": "No non-identity words in search range."}
        radii = []
        witnesses = []
        excluded_identity = 0
        excluded_parabolic = 0
        excluded_nonhyperbolic = 0
        is_modular = isinstance(getattr(self, "surface", None), ModularFordSurface)
        for z in pts:
            best = float("inf")
            best_sel = None
            for sel in selectors:
                try:
                    g = self.surface.gamma(sel)
                    kind, _, _ = self.classify_mobius(g)
                    if kind == "identity_or_relation":
                        excluded_identity += 1
                        continue
                    if is_modular and kind == "parabolic_or_near_parabolic":
                        excluded_parabolic += 1
                        continue
                    if is_modular and kind != "hyperbolic":
                        excluded_nonhyperbolic += 1
                        continue
                    gz = g.apply_point(z)
                    if inside_unit_disk(gz, margin=1.0e-8):
                        d = poincare_distance(z, gz)
                        if d < best and d > 1.0e-10:
                            best = d
                            best_sel = sel
                except Exception:
                    continue
            if math.isfinite(best):
                radii.append(0.5 * best)
                witnesses.append(str(best_sel))
        if not radii:
            return {
                "available": False,
                "reason": "No finite hyperbolic/thick-part distances found after excluding identity/near-parabolic elements.",
                "word_depth": int(depth),
                "samples_requested": int(samples),
                "modular_cusp_aware": bool(is_modular),
                "excluded_identity_or_relation_count": int(excluded_identity),
                "excluded_parabolic_or_near_parabolic_count": int(excluded_parabolic),
                "excluded_other_nonhyperbolic_count": int(excluded_nonhyperbolic),
            }
        arr = np.array(radii, dtype=float)
        j = int(np.argmin(arr))
        return {
            "available": True,
            "samples": int(len(arr)),
            "word_depth": int(depth),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "witness": witnesses[j],
            "modular_cusp_aware": bool(is_modular),
            "mode": "hyperbolic/thick-part estimate excluding parabolic cusp elements" if is_modular else "ordinary finite-search quotient estimate",
            "excluded_identity_or_relation_count": int(excluded_identity),
            "excluded_parabolic_or_near_parabolic_count": int(excluded_parabolic),
            "excluded_other_nonhyperbolic_count": int(excluded_nonhyperbolic),
        }

    def basic_surface_invariants(self) -> dict:
        data = getattr(self.surface, "data", {}) if hasattr(self.surface, "data") else {}
        genus = data.get("genus", None)
        area = data.get("area", None)
        gb_area = data.get("gauss_bonnet_area", None)
        if self.selected_model() == self.MODEL_OCTAGON:
            genus = 2
            area = 4.0 * math.pi
            gb_area = area
        if self.selected_model() == self.MODEL_CYCLIC:
            genus = None
            area = None
            gb_area = None
        chi = None
        if isinstance(genus, (int, float)):
            chi = 2 - 2 * int(genus)
        out = {
            "surface_name": getattr(self.surface, "name", self.surface.display_name),
            "surface_interface": self.surface.__class__.__name__,
            "domain_type": getattr(self.surface, "domain_type", self.selected_model()),
            "genus": genus,
            "euler_characteristic": chi,
            "area": area,
            "gauss_bonnet_area": gb_area,
            "generator_labels": self.surface_uppercase_generators(),
            "generator_count": len(self.surface_uppercase_generators()),
            "vertex_count": int(len(getattr(self.surface, "polygon_vertices", []))) if hasattr(self.surface, "polygon_vertices") else None,
        }
        if isinstance(self.surface, ModularFordSurface):
            md = self.surface.modular_metadata()
            modular = {k: md.get(k) for k in [
                "subdomain_type", "category", "parent_group", "subgroup", "level_N",
                "torsion_free", "compact", "index_in_psl2z", "cusp_count",
                "cusp_widths", "elliptic_orders", "compactified_genus",
                "riemann_surface_status", "kahler_status", "mathematical_object",
                "construction_tile_count", "exterior_edge_count", "internal_edge_count",
            ] if md.get(k) is not None}
            out["modular"] = modular
            # For modular curves, genus means compactified genus after adding cusps.
            if md.get("compactified_genus") is not None:
                out["compactified_genus"] = md.get("compactified_genus")
                out["genus"] = md.get("compactified_genus")
            if md.get("area") is not None:
                out["area"] = md.get("area")
                out["gauss_bonnet_area"] = md.get("area")
            if md.get("cusp_count") is not None:
                out["cusp_count"] = md.get("cusp_count")
        return out


    # ------------------------------------------------------------------
    # v16.3 geometry-feedstock cleanup additions
    # ------------------------------------------------------------------

    def exact_differential_geometry_at_point(self, z: np.ndarray) -> dict:
        """Classical differential-geometry quantities for the exact Poincare disk metric.

        The quotient surface is locally modeled on the disk, so these local values are
        the universal-cover quantities.  The global quotient data enter separately
        through side pairings, word searches, and side-crossing itineraries.
        """
        z = np.asarray(z, dtype=float)
        x = float(z[0]); y = float(z[1])
        r2 = x*x + y*y
        den = max(1.0 - r2, 1.0e-12)
        phi = math.log(2.0) - math.log(den)
        scale = 2.0 / den
        e2phi = scale * scale
        phix = 2.0 * x / den
        phiy = 2.0 * y / den
        grad_norm_euclid = math.hypot(phix, phiy)
        lap = 4.0 / (den * den)
        liouville_residual = lap - e2phi
        K = -math.exp(-2.0 * phi) * lap
        # Nonzero Christoffel symbols for g=e^{2 phi}(dx^2+dy^2).
        christoffel = {
            "Gamma^x_xx": phix,
            "Gamma^x_xy": phiy,
            "Gamma^x_yy": -phix,
            "Gamma^y_xx": -phiy,
            "Gamma^y_xy": phix,
            "Gamma^y_yy": phiy,
        }
        return {
            "point": [x, y],
            "r": math.sqrt(max(r2, 0.0)),
            "phi": phi,
            "scale_lambda": scale,
            "metric_g_xx": e2phi,
            "metric_g_yy": e2phi,
            "area_density": e2phi,
            "grad_phi": [phix, phiy],
            "grad_phi_euclidean_norm": grad_norm_euclid,
            "laplacian_phi": lap,
            "exp_2phi": e2phi,
            "liouville_residual": liouville_residual,
            "gaussian_curvature": K,
            "ricci_xx": K * e2phi,
            "ricci_yy": K * e2phi,
            "christoffel": christoffel,
        }

    def differential_geometry_feedstock(self, samples: int) -> dict:
        pts = self.sample_points_for_injectivity(max(8, int(samples)))
        selected = {}
        for name, z in [("p", self.p), ("q", self.q), ("q_lift", self.q_lift)]:
            if z is not None and inside_unit_disk(np.asarray(z, dtype=float), margin=1.0e-8):
                selected[name] = self.exact_differential_geometry_at_point(np.asarray(z, dtype=float))
        if pts is None or len(pts) == 0:
            sample_stats = {"available": False, "reason": "No sample points available."}
        else:
            vals = [self.exact_differential_geometry_at_point(z) for z in pts]
            arr_phi = np.array([v["phi"] for v in vals], dtype=float)
            arr_area = np.array([v["area_density"] for v in vals], dtype=float)
            arr_grad = np.array([v["grad_phi_euclidean_norm"] for v in vals], dtype=float)
            arr_k = np.array([v["gaussian_curvature"] for v in vals], dtype=float)
            arr_res = np.array([v["liouville_residual"] for v in vals], dtype=float)
            sample_stats = {
                "available": True,
                "samples": int(len(vals)),
                "phi_mean": float(np.mean(arr_phi)),
                "phi_min": float(np.min(arr_phi)),
                "phi_max": float(np.max(arr_phi)),
                "area_density_mean": float(np.mean(arr_area)),
                "area_density_min": float(np.min(arr_area)),
                "area_density_max": float(np.max(arr_area)),
                "grad_phi_norm_mean": float(np.mean(arr_grad)),
                "curvature_mean": float(np.mean(arr_k)),
                "curvature_min": float(np.min(arr_k)),
                "curvature_max": float(np.max(arr_k)),
                "max_abs_liouville_residual": float(np.max(np.abs(arr_res))),
            }
        learned = {"available": False, "reason": "Neural metric not trained or PyTorch unavailable."}
        if self.learned_metric_available() and TORCH_AVAILABLE and pts is not None and len(pts) > 0:
            try:
                flat = np.asarray(pts[: min(len(pts), max(8, int(samples)))], dtype=float)
                xy = torch.tensor(flat, dtype=torch.float64)
                phi_t, lap_t = metric_laplacian(self.metric_model, xy)
                phi_np = phi_t.detach().cpu().numpy()
                lap_np = lap_t.detach().cpu().numpy()
                k_np = -np.exp(-2.0 * phi_np) * lap_np
                residual_np = lap_np - np.exp(2.0 * phi_np)
                phi_true = poincare_phi_np(flat)
                learned = {
                    "available": True,
                    "samples": int(len(flat)),
                    "mean_abs_phi_error": float(np.mean(np.abs(phi_np - phi_true))),
                    "max_abs_phi_error": float(np.max(np.abs(phi_np - phi_true))),
                    "curvature_mean": float(np.mean(k_np)),
                    "curvature_min": float(np.min(k_np)),
                    "curvature_max": float(np.max(k_np)),
                    "mean_abs_curvature_plus_one": float(np.mean(np.abs(k_np + 1.0))),
                    "max_abs_liouville_residual": float(np.max(np.abs(residual_np))),
                }
            except Exception as exc:
                learned = {"available": False, "reason": f"Neural metric DG evaluation failed: {exc}"}
        return {
            "local_metric": "Poincare disk metric upstairs: ds^2 = exp(2 phi)(dx^2+dy^2), phi=log(2)-log(1-r^2)",
            "selected_points": selected,
            "sample_statistics_exact": sample_stats,
            "sample_statistics_learned_metric": learned,
            "note": "Local differential geometry is universal-cover geometry; quotient-specific information is in word, side-pairing, and itinerary data.",
        }

    @staticmethod
    def _segment_intersection_2d(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> Optional[Tuple[float, float, np.ndarray]]:
        """Return (t,u,point) for segment intersection p1+t(p2-p1)=q1+u(q2-q1)."""
        p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
        q1 = np.asarray(q1, dtype=float); q2 = np.asarray(q2, dtype=float)
        r = p2 - p1; s = q2 - q1
        den = float(r[0]*s[1] - r[1]*s[0])
        if abs(den) < 1.0e-12:
            return None
        qp = q1 - p1
        t = float((qp[0]*s[1] - qp[1]*s[0]) / den)
        u = float((qp[0]*r[1] - qp[1]*r[0]) / den)
        eps = 1.0e-9
        if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
            pt = p1 + max(0.0, min(1.0, t)) * r
            return t, u, pt
        return None

    def _polyline_intersections(self, curve: np.ndarray, edge: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        hits: List[Tuple[float, np.ndarray]] = []
        curve = np.asarray(curve, dtype=float)
        edge = np.asarray(edge, dtype=float)
        if len(curve) < 2 or len(edge) < 2:
            return hits
        nseg = max(1, len(curve) - 1)
        for i in range(len(curve) - 1):
            p1, p2 = curve[i], curve[i+1]
            for j in range(len(edge) - 1):
                ans = self._segment_intersection_2d(p1, p2, edge[j], edge[j+1])
                if ans is None:
                    continue
                t, _u, pt = ans
                global_t = (i + t) / nseg
                # Suppress numerical duplicates from adjacent edge polyline segments.
                if hits and np.linalg.norm(hits[-1][1] - pt) < 1.0e-4:
                    continue
                hits.append((float(global_t), pt))
        return hits

    def side_crossing_feedstock(self) -> dict:
        if self.p is None or self.q_lift is None:
            return {"available": False, "reason": "Choose p and q, then draw a selected lifted geodesic first."}
        if not self.surface_has_polygon_vertices():
            return {"available": False, "reason": "Side-crossing prototype currently requires a compact polygon surface."}
        try:
            curve = self.exact_curve if self.exact_curve is not None and len(self.exact_curve) >= 2 else exact_poincare_geodesic(self.p, self.q_lift, n=600)
            verts = np.asarray(getattr(self.surface, "polygon_vertices"), dtype=float)
            side_pair_word = {}
            data = getattr(self.surface, "data", {}) if hasattr(self.surface, "data") else {}
            for item in data.get("side_pairings", []) or []:
                try:
                    side_pair_word[int(item.get("side"))] = str(item.get("word", ""))
                    side_pair_word[int(item.get("paired_with"))] = str(item.get("word", "")).lower()
                except Exception:
                    continue
            crossings = []
            for side in range(len(verts)):
                a = verts[side]
                b = verts[(side + 1) % len(verts)]
                edge = exact_poincare_geodesic(a, b, n=160)
                for t, pt in self._polyline_intersections(curve, edge):
                    # Avoid counting when curve starts/ends very near a side endpoint.
                    if t < 1.0e-5 or t > 1.0 - 1.0e-5:
                        continue
                    crossings.append({
                        "t_parameter_approx": float(t),
                        "side": int(side),
                        "side_label": side_pair_word.get(side, f"side_{side}"),
                        "point": [float(pt[0]), float(pt[1])],
                    })
            crossings.sort(key=lambda r: r["t_parameter_approx"])
            # Merge very close duplicate hits from shared vertices or overdraw.
            merged = []
            for cr in crossings:
                if merged and abs(cr["t_parameter_approx"] - merged[-1]["t_parameter_approx"]) < 2.0e-3:
                    if np.linalg.norm(np.array(cr["point"]) - np.array(merged[-1]["point"])) < 4.0e-3:
                        continue
                merged.append(cr)
            return {
                "available": True,
                "curve_source": "selected lifted exact geodesic in the universal cover",
                "crossing_count": int(len(merged)),
                "itinerary_labels": [cr["side_label"] for cr in merged],
                "crossings": merged,
                "note": "Base-polygon intersections of the selected lifted universal-cover geodesic, computed by polyline intersection. This is feedstock for later folded-geodesic/symbolic-dynamics tools; it is not yet the quotient folded itinerary and is not a certified symbolic coding.",
            }
        except Exception as exc:
            return {"available": False, "reason": f"Side-crossing computation failed: {exc}"}


    def spectral_graph_laplacian_features(self, samples: int, eig_count: int = 12, quotient_depth: int = 1) -> dict:
        """Prototype spectral feedstock from a quotient-aware point-cloud graph Laplacian.

        This is intentionally a numerical fingerprint, not a certified finite-element
        Laplace-Beltrami spectrum.  We sample points in the displayed fundamental
        region, connect them with heat-kernel-like weights based on approximate
        quotient distances

            d_X([z_i],[z_j]) ~ min_{gamma in small word ball} d_D(z_i, gamma z_j),

        and report the smallest eigenvalues of the normalized graph Laplacian.
        The result is useful as ML feedstock and as a regression-test signal, but
        its eigenvalues should not be interpreted as certified eigenvalues of
        -Delta_X.
        """
        n = int(max(12, min(int(samples), 140)))
        eig_count = int(max(3, min(int(eig_count), max(3, n - 1))))
        pts = self.sample_points_for_injectivity(n)
        if pts is None or len(pts) < 8:
            return {"available": False, "reason": "Not enough sample points for graph spectral prototype."}
        pts = np.asarray(pts, dtype=float)
        n = int(len(pts))

        selectors: List[object] = [None]
        if self.selected_model() != self.MODEL_DISK:
            if self.selected_model() == self.MODEL_CYCLIC:
                d = max(1, int(quotient_depth))
                selectors.extend([k for k in range(-d, d + 1) if k != 0])
            else:
                try:
                    if isinstance(getattr(self, "surface", None), ModularFordSurface) and self.modular_safe_mode_active():
                        words, info = self.modular_guarded_selectors(max(1, int(quotient_depth)), max_words=800, for_spectrum=True)
                        self.last_modular_guard_info = info
                        selectors.extend(words)
                    else:
                        alphabet = self.surface.available_alphabet()
                        if alphabet:
                            selectors.extend([w for w in generate_reduced_words(max(1, int(quotient_depth)), alphabet=alphabet) if w])
                except Exception:
                    pass

        # Pairwise approximate quotient distances.
        Dq = np.zeros((n, n), dtype=float)
        for i in range(n):
            zi = pts[i]
            for j in range(i + 1, n):
                zj = pts[j]
                best = poincare_distance(zi, zj)
                for sel in selectors[1:]:
                    try:
                        g = self.surface.gamma(sel)
                        gzj = g.apply_point(zj)
                        if inside_unit_disk(gzj, margin=1.0e-8):
                            dij = poincare_distance(zi, gzj)
                            if dij < best:
                                best = dij
                    except Exception:
                        continue
                Dq[i, j] = Dq[j, i] = float(best)

        # Build a symmetric k-nearest-neighbor weighted graph.
        k_nn = int(max(4, min(12, n - 1)))
        nonzero = Dq[Dq > 1.0e-12]
        if nonzero.size == 0:
            return {"available": False, "reason": "All sampled quotient distances collapsed to zero."}
        # Use a local nearest-neighbor scale; robust enough for fingerprinting.
        nearest_vals = []
        for i in range(n):
            ds = np.sort(Dq[i][Dq[i] > 1.0e-12])
            if ds.size:
                nearest_vals.extend(ds[:min(k_nn, len(ds))].tolist())
        sigma = float(np.median(nearest_vals)) if nearest_vals else float(np.median(nonzero))
        sigma = max(sigma, 1.0e-6)

        W = np.zeros((n, n), dtype=float)
        for i in range(n):
            order = np.argsort(Dq[i])
            added = 0
            for j in order:
                if i == j:
                    continue
                dij = Dq[i, j]
                if dij <= 1.0e-12:
                    continue
                wij = math.exp(- (dij / sigma) ** 2)
                if wij > W[i, j]:
                    W[i, j] = W[j, i] = wij
                added += 1
                if added >= k_nn:
                    break

        deg = np.sum(W, axis=1)
        positive = deg > 1.0e-14
        if int(np.sum(positive)) < 4:
            return {"available": False, "reason": "Graph has too few non-isolated vertices."}
        # Normalized graph Laplacian, restricted to non-isolated vertices.
        idx = np.where(positive)[0]
        Wp = W[np.ix_(idx, idx)]
        degp = np.sum(Wp, axis=1)
        invsqrt = 1.0 / np.sqrt(np.maximum(degp, 1.0e-14))
        Lsym = np.eye(len(idx)) - (invsqrt[:, None] * Wp * invsqrt[None, :])
        evals = np.linalg.eigvalsh(Lsym)
        evals = np.maximum(np.sort(np.real(evals)), 0.0)
        first = evals[:min(eig_count, len(evals))]
        heat_times = [0.1, 0.5, 1.0, 2.0]
        heat_trace = {str(t): float(np.sum(np.exp(-t * evals))) for t in heat_times}
        gap = float(evals[1]) if len(evals) > 1 else None
        return {
            "available": True,
            "method": "quotient-aware point-cloud normalized graph Laplacian prototype",
            "rigor": "ML/feedstock fingerprint only; not a certified Laplace-Beltrami spectrum",
            "samples_requested": int(samples),
            "samples_used": int(n),
            "nonisolated_vertices": int(len(idx)),
            "k_nearest_neighbors": int(k_nn),
            "kernel_sigma": float(sigma),
            "quotient_word_depth_for_distance": int(quotient_depth),
            "eigenvalues_normalized_graph_laplacian": [float(x) for x in first],
            "spectral_gap_graph_lambda1": gap,
            "heat_trace_normalized_graph": heat_trace,
            "distance_summary": {
                "min_nonzero": float(np.min(nonzero)),
                "median_nonzero": float(np.median(nonzero)),
                "max": float(np.max(nonzero)),
            },
        }

    def compute_invariants_payload(self) -> dict:
        requested_depth = int(self.invariants_word_depth_spin.value()) if hasattr(self, "invariants_word_depth_spin") else 3
        max_rows = int(self.invariants_max_rows_spin.value()) if hasattr(self, "invariants_max_rows_spin") else 40
        requested_samples = int(self.invariants_sample_spin.value()) if hasattr(self, "invariants_sample_spin") else 60

        safe_guard = bool(isinstance(getattr(self, "surface", None), ModularFordSurface) and self.modular_safe_mode_active() and not self.deep_modular_fingerprint_enabled())
        large_modular = bool(isinstance(getattr(self, "surface", None), ModularFordSurface) and self.modular_surface_is_large())
        depth = requested_depth
        samples = requested_samples
        guard_notes = []
        if safe_guard:
            if large_modular:
                depth = min(depth, 1)
                samples = min(samples, 36)
                guard_notes.append("Large modular surface: capped word/injectivity depth to 1 and samples to <=36.")
            else:
                depth = min(depth, 2)
                samples = min(samples, 60)
                guard_notes.append("Modular safe mode: capped word/injectivity depth to <=2.")

        spectrum = self.word_length_spectrum(depth, max_rows)
        hyper = [r for r in spectrum if r.get("type") == "hyperbolic"]
        systole = hyper[0] if hyper else None
        inj = self.injectivity_radius_estimate(depth, samples)
        dg = self.differential_geometry_feedstock(samples)
        side = self.side_crossing_feedstock()

        if safe_guard and large_modular:
            spectral = {
                "available": False,
                "skipped_by_v17_3_guard": True,
                "reason": "Large modular surface in fast/safe mode. Enable 'Deep fingerprint run for large modular surfaces' to compute graph-spectral feedstock.",
                "rigor": "metadata unchanged; spectral feedstock intentionally skipped to keep the GUI responsive",
            }
            guard_notes.append("Graph-spectral prototype skipped in safe mode for large modular surface.")
        else:
            spectral = self.spectral_graph_laplacian_features(samples=max(24, samples), eig_count=12, quotient_depth=1)

        guard_info = getattr(self, "last_modular_guard_info", None) if isinstance(getattr(self, "surface", None), ModularFordSurface) else None
        return {
            "version": "FuchsianGENN Explorer v17.6 cusp-aware modular geometry+spectral feedstock fingerprint",
            "basic": self.basic_surface_invariants(),
            "search_parameters": {
                "requested_word_depth": requested_depth,
                "effective_word_depth": depth,
                "word_depth": depth,
                "max_rows": max_rows,
                "requested_injectivity_samples": requested_samples,
                "injectivity_samples": samples,
                "v17_4_principal_domain_guard_active": bool(safe_guard),
                "large_modular_surface_detected": bool(large_modular),
                "deep_modular_fingerprint_requested": bool(self.deep_modular_fingerprint_enabled()),
                "guard_notes": guard_notes,
                "last_modular_word_guard": guard_info,
            },
            "differential_geometry": dg,
            "side_crossing_feedstock": side,
            "length_spectrum_candidates": spectrum,
            "systole_candidate_from_search": systole,
            "sampled_injectivity_radius": inj,
            "spectral_features": spectral,
            "notes": "Geometry+spectral feedstock fingerprint. Word-length, side-crossing, injectivity, and graph-spectral values are finite numerical prototypes, not global proofs. v17.6 loads principal/base domains quickly, draws finite word-patch tessellations only when explicitly requested, and treats modular parabolic/near-parabolic elements separately from hyperbolic length candidates.",
        }

    def format_invariants_payload(self, payload: dict) -> str:
        b = payload.get("basic", {})
        lines = []
        lines.append("GEOMETRY / INVARIANTS")
        lines.append("=====================")
        lines.append("")
        lines.append("Surface")
        lines.append("-------")
        for key in ["surface_name", "surface_interface", "domain_type", "genus", "euler_characteristic", "area", "gauss_bonnet_area", "generator_count", "vertex_count"]:
            val = b.get(key, None)
            if val is not None:
                lines.append(f"{key:32s} {val}")
        labels = b.get("generator_labels", [])
        if labels:
            lines.append(f"{'generator labels':32s} {', '.join(labels)}")
        modular = b.get("modular", {})
        if isinstance(modular, dict) and modular:
            lines.append("")
            lines.append("Modular-surface metadata")
            lines.append("------------------------")
            for key in ["subgroup", "level_N", "subdomain_type", "index_in_psl2z", "torsion_free", "compact", "compactified_genus", "cusp_count", "cusp_widths", "elliptic_orders", "construction_tile_count", "exterior_edge_count", "internal_edge_count"]:
                if key in modular:
                    lines.append(f"{key:32s} {modular.get(key)}")
            if modular.get("riemann_surface_status"):
                lines.append(f"{'riemann_surface_status':32s} {modular.get('riemann_surface_status')}")
            if modular.get("kahler_status"):
                lines.append(f"{'kahler_status':32s} {modular.get('kahler_status')}")
        lines.append("")
        lines.append("Classical differential-geometry feedstock")
        lines.append("----------------------------------------")
        dg = payload.get("differential_geometry", {})
        lines.append(str(dg.get("local_metric", "exact local Poincare metric upstairs")))
        exact_stats = dg.get("sample_statistics_exact", {})
        if exact_stats.get("available"):
            lines.append(f"samples used                     {exact_stats['samples']}")
            lines.append(f"phi mean/min/max                 {exact_stats['phi_mean']:.9f} / {exact_stats['phi_min']:.9f} / {exact_stats['phi_max']:.9f}")
            lines.append(f"area density mean/min/max        {exact_stats['area_density_mean']:.9f} / {exact_stats['area_density_min']:.9f} / {exact_stats['area_density_max']:.9f}")
            lines.append(f"|grad phi| mean                  {exact_stats['grad_phi_norm_mean']:.9f}")
            lines.append(f"curvature mean/min/max           {exact_stats['curvature_mean']:.9f} / {exact_stats['curvature_min']:.9f} / {exact_stats['curvature_max']:.9f}")
            lines.append(f"max |Delta phi - exp(2phi)|      {exact_stats['max_abs_liouville_residual']:.3e}")
        else:
            lines.append(f"sample stats not available: {exact_stats.get('reason', 'unknown')}")
        selected = dg.get("selected_points", {})
        if selected:
            lines.append("selected-point local quantities")
            for name, val in selected.items():
                pt = val.get("point", [float('nan'), float('nan')])
                lines.append(f"  {name:7s} z=({pt[0]: .5f},{pt[1]: .5f}) phi={val['phi']:.6f} K={val['gaussian_curvature']:.6f} area_density={val['area_density']:.6f}")
                ch = val.get("christoffel", {})
                lines.append(f"          Gamma^x_xx={ch.get('Gamma^x_xx', float('nan')):.6f}, Gamma^x_xy={ch.get('Gamma^x_xy', float('nan')):.6f}, Gamma^y_xy={ch.get('Gamma^y_xy', float('nan')):.6f}")
        learned = dg.get("sample_statistics_learned_metric", {})
        if learned.get("available"):
            lines.append("learned-metric DG check")
            lines.append(f"  mean/max |phi_theta-phi_true|  {learned['mean_abs_phi_error']:.3e} / {learned['max_abs_phi_error']:.3e}")
            lines.append(f"  curvature mean/min/max         {learned['curvature_mean']:.9f} / {learned['curvature_min']:.9f} / {learned['curvature_max']:.9f}")
            lines.append(f"  mean |K_theta+1|               {learned['mean_abs_curvature_plus_one']:.3e}")
            lines.append(f"  max |PDE residual|             {learned['max_abs_liouville_residual']:.3e}")
        else:
            lines.append(f"learned-metric DG check          {learned.get('reason', 'not available')}")
        lines.append("")
        lines.append("Base-polygon side-crossing feedstock")
        lines.append("--------------------------------------------")
        side = payload.get("side_crossing_feedstock", {})
        if side.get("available"):
            lines.append(f"curve source                     {side.get('curve_source')}")
            lines.append(f"crossing count                   {side.get('crossing_count')}")
            labels2 = side.get("itinerary_labels", [])
            lines.append(f"base-side labels                 {' '.join(labels2) if labels2 else '(none)'}")
            rows2 = side.get("crossings", [])[:20]
            if rows2:
                lines.append(f"{'t':>9s} {'side':>6s} {'label':>8s} {'point':>24s}")
                for cr in rows2:
                    pt = cr.get("point", [float('nan'), float('nan')])
                    lines.append(f"{cr.get('t_parameter_approx', float('nan')):9.5f} {int(cr.get('side', -1)):6d} {str(cr.get('side_label','')):>8s} ({pt[0]: .5f},{pt[1]: .5f})")
            lines.append("coding note                   base-polygon intersections only; folded quotient itinerary/symbolic coding comes later")
        else:
            lines.append(f"not available                    {side.get('reason', 'unknown')}")
        lines.append("")
        lines.append("Finite word-length spectrum prototype")
        lines.append("-------------------------------------")
        sp = payload.get("search_parameters", {})
        lines.append(f"word depth searched              {sp.get('word_depth')}")
        lines.append("Values for hyperbolic elements are translation lengths.")
        lines.append("Elliptic rows report rotation-angle proxy; parabolic/near-parabolic and identity/relation rows report 0 and are excluded from systole candidates.")
        rows = payload.get("length_spectrum_candidates", [])
        if rows:
            lines.append(f"{'word':14s} {'type':12s} {'value':>14s} {'trace':>14s}")
            for r in rows:
                lines.append(f"{str(r['word']):14s} {r['type']:12s} {r['value']:14.9f} {r['trace']:14.9f}")
        else:
            lines.append("No nontrivial word spectrum available for this mode/depth.")
        lines.append("")
        lines.append("Systole candidate")
        lines.append("-----------------")
        sysrow = payload.get("systole_candidate_from_search")
        if sysrow:
            lines.append(f"shortest hyperbolic word found   {sysrow['word']}")
            lines.append(f"candidate length                 {sysrow['value']:.9f}")
            lines.append("This is a finite-search candidate, not a global proof.")
        else:
            lines.append("No hyperbolic candidate found in the finite search.")
        lines.append("")
        lines.append("Sampled injectivity-radius estimate")
        lines.append("-----------------------------------")
        inj = payload.get("sampled_injectivity_radius", {})
        if inj.get("available"):
            lines.append(f"samples used                     {inj['samples']}")
            lines.append(f"word depth                       {inj['word_depth']}")
            if inj.get("modular_cusp_aware"):
                lines.append(f"estimate mode                    {inj.get('mode', 'cusp-aware modular estimate')}")
                lines.append(f"excluded identity/relation words {inj.get('excluded_identity_or_relation_count', 0)}")
                lines.append(f"excluded parabolic/cusp words    {inj.get('excluded_parabolic_or_near_parabolic_count', 0)}")
            lines.append(f"minimum sampled inj radius       {inj['min']:.9f}")
            lines.append(f"mean sampled inj radius          {inj['mean']:.9f}")
            lines.append(f"median sampled inj radius        {inj['median']:.9f}")
            lines.append(f"maximum sampled inj radius       {inj['max']:.9f}")
            lines.append(f"minimum witness word             {inj['witness']}")
        else:
            lines.append(f"not available: {inj.get('reason', 'unknown')}")
            if inj.get("modular_cusp_aware"):
                lines.append(f"excluded identity/relation words {inj.get('excluded_identity_or_relation_count', 0)}")
                lines.append(f"excluded parabolic/cusp words    {inj.get('excluded_parabolic_or_near_parabolic_count', 0)}")
        lines.append("")
        lines.append("Spectral feedstock prototype")
        lines.append("----------------------------")
        spec = payload.get("spectral_features", {})
        if spec.get("available"):
            lines.append(f"method                           {spec.get('method')}")
            lines.append(f"rigor                            {spec.get('rigor')}")
            lines.append(f"samples used                     {spec.get('samples_used')} / requested {spec.get('samples_requested')}")
            lines.append(f"nonisolated graph vertices       {spec.get('nonisolated_vertices')}")
            lines.append(f"k nearest neighbors              {spec.get('k_nearest_neighbors')}")
            lines.append(f"kernel sigma                     {spec.get('kernel_sigma'):.9f}")
            lines.append(f"quotient distance word depth     {spec.get('quotient_word_depth_for_distance')}")
            ev = spec.get("eigenvalues_normalized_graph_laplacian", [])
            if ev:
                ev_txt = ", ".join(f"{float(x):.6f}" for x in ev[:12])
                lines.append(f"first graph eigenvalues          {ev_txt}")
            gap = spec.get("spectral_gap_graph_lambda1")
            if gap is not None:
                lines.append(f"graph spectral gap lambda_1      {float(gap):.9f}")
            ht = spec.get("heat_trace_normalized_graph", {})
            if ht:
                ht_txt = ", ".join(f"t={k}: {float(v):.6f}" for k, v in ht.items())
                lines.append(f"graph heat trace                 {ht_txt}")
            ds = spec.get("distance_summary", {})
            if ds:
                lines.append(f"quotient distance min/median/max {ds.get('min_nonzero', float('nan')):.6f} / {ds.get('median_nonzero', float('nan')):.6f} / {ds.get('max', float('nan')):.6f}")
        else:
            lines.append(f"not available                    {spec.get('reason', 'unknown')}")
        lines.append("spectral note                    normalized graph spectrum; useful ML feedstock, not a certified Laplace-Beltrami spectrum")
        lines.append("")
        lines.append("Rigor note")
        lines.append("----------")
        lines.append("These are finite-search numerical fingerprints. They are useful for")
        lines.append("exploration and second-level ML preparation, but they do not certify")
        lines.append("global systoles, exact injectivity radii, or complete length spectra.")
        return "\n".join(lines)

    def compute_and_display_invariants(self):
        try:
            payload = self.compute_invariants_payload()
            self.last_invariants = payload
            self.invariants_text.setPlainText(self.format_invariants_payload(payload))
            self.status.setText("Status: computed v17.6 modular-aware geometry+spectral feedstock fingerprint.")
        except Exception as exc:
            QMessageBox.critical(self, "Invariants failed", str(exc))
            self.status.setText("Status: invariants computation failed. See error dialog.")

    def export_surface_fingerprint_json(self):
        if self.last_invariants is None:
            self.compute_and_display_invariants()
        if self.last_invariants is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export surface fingerprint JSON",
            "surface_fingerprint_v17_4.json",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            self.status.setText("Status: fingerprint export cancelled.")
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.last_invariants, fh, indent=2)
            self.status.setText(f"Status: exported surface fingerprint JSON to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            self.status.setText("Status: fingerprint export failed. See error dialog.")

    def update_diagnostics(self):
        def pair_text(v: Optional[np.ndarray]) -> str:
            if v is None:
                return "--"
            return f"({v[0]: .6f}, {v[1]: .6f})"

        def item(label: str, value: str) -> str:
            return f"  {label:<34} {value}"

        lines: List[str] = []
        surface = self.surface if hasattr(self, "surface") else self.make_surface()
        lines.append("MODEL")
        lines.append(item("surface model", surface.display_name))
        for label, value in surface.diagnostics():
            lines.append(item(label, value))
        lines.append(item("selected lift", surface.lift_label(self.current_lift_selector())))

        lines.append("")
        lines.append("POINTS AND LIFT")
        lines.append(item("p", pair_text(self.p)))
        lines.append(item("q", pair_text(self.q)))
        if self.q_lift is not None:
            label = surface.lift_label(self.current_lift_selector())
            lines.append(item(label, pair_text(self.q_lift)))

        hyp: Optional[float] = None
        emin: Optional[float] = None
        if self.p is not None and self.q_lift is not None:
            euclid = float(np.linalg.norm(self.p - self.q_lift))
            hyp = poincare_distance(self.p, self.q_lift)
            emin = 0.5 * hyp * hyp
            lines.append("")
            lines.append("EXACT BENCHMARK FOR SELECTED LIFT")
            lines.append(item("Euclidean distance upstairs", f"{euclid:.9f}"))
            lines.append(item("hyperbolic distance d", f"{hyp:.9f}"))
            lines.append(item("minimum constant-speed energy", f"0.5 d^2 = {emin:.9f}"))

        if self.last_search_summary:
            lines.append("")
            lines.append("FINITE ORBIT SEARCH")
            for raw in self.last_search_summary.splitlines():
                lines.append("  " + raw)

        if self.initial_curve is not None:
            ilen, ienergy, ispeeds = diagnostic_length_energy_speeds(self.initial_curve)
            lines.append("")
            lines.append("INITIAL NEURAL CURVE")
            lines.append(item("diagnostic length", f"{ilen:.9f}"))
            lines.append(item("diagnostic energy", f"{ienergy:.9f}"))
            if len(ispeeds) > 1 and np.mean(ispeeds) > 0:
                lines.append(item("speed uniformity CV", f"{float(np.std(ispeeds)/np.mean(ispeeds)):.3e}"))

        if self.neural_curve is not None:
            length, energy, speeds = diagnostic_length_energy_speeds(self.neural_curve)
            lines.append("")
            lines.append("RELAXED PYTORCH NEURAL CURVE")
            lines.append(item("diagnostic length", f"{length:.9f}"))
            lines.append(item("diagnostic energy", f"{energy:.9f}"))
            if hyp is not None and emin is not None:
                signed_len_err = length - hyp
                signed_energy_err = energy - emin
                lines.append(item("length - exact distance", f"{signed_len_err:+.3e}"))
                lines.append(item("energy - exact minimum", f"{signed_energy_err:+.3e}"))
                if signed_energy_err < -1.0e-6:
                    lines.append(item("energy-bound check", "WARNING: diagnostic energy is below exact bound"))
                else:
                    lines.append(item("energy-bound check", "OK"))
            if len(speeds) > 1 and np.mean(speeds) > 0:
                lines.append(item("speed uniformity CV", f"{float(np.std(speeds)/np.mean(speeds)):.3e}"))
            if self.energy_history is not None and len(self.energy_history) > 0:
                lines.append(item("optimizer steps recorded", str(len(self.energy_history))))
                lines.append(item("training midpoint final E", f"{float(self.energy_history[-1]):.9f}"))


        if self.exact_metric_curve is not None or self.learned_metric_curve is not None:
            lines.append("")
            lines.append("V15 GEODESIC VALIDATION: EXACT METRIC VS LEARNED METRIC")
            if self.metric_geodesic_comparison_summary:
                for raw in self.metric_geodesic_comparison_summary.splitlines():
                    lines.append("  " + raw)
            if self.exact_metric_curve is not None:
                elen, eenergy, espeeds = diagnostic_length_energy_speeds(self.exact_metric_curve)
                lines.append(item("exact-metric NN length", f"{elen:.9f}"))
                if hyp is not None:
                    lines.append(item("exact-metric length error", f"{elen - hyp:+.3e}"))
                lines.append(item("exact-metric diagnostic energy", f"{eenergy:.9f}"))
            if self.learned_metric_curve is not None:
                llen, lenergy, lspeeds = diagnostic_length_energy_speeds(self.learned_metric_curve)
                lines.append(item("learned-metric NN length", f"{llen:.9f}"))
                if hyp is not None:
                    lines.append(item("learned-metric length error", f"{llen - hyp:+.3e}"))
                lines.append(item("learned-metric diagnostic energy", f"{lenergy:.9f}"))
                if emin is not None:
                    lines.append(item("learned-metric energy-bound", "OK" if lenergy >= emin - 1e-6 else "WARNING below exact bound"))


        lines.append("")
        lines.append("NEURAL METRIC VALIDATION")
        if not TORCH_AVAILABLE:
            lines.append(item("PyTorch", "not available"))
        elif self.metric_last_stats is None:
            lines.append(item("phi_theta", "not trained yet"))
            lines.append(item("metric ansatz", "phi = -log(1-r^2+eps) + u_theta(x,y)"))
        else:
            st = self.metric_last_stats
            lines.append(item("metric ansatz", "phi = -log(1-r^2+eps) + u_theta(x,y)"))
            lines.append(item("training steps", str(st.get("steps", "--"))))
            lines.append(item("interior samples/step", str(st.get("samples", "--"))))
            lines.append(item("interior radius", f"{st.get('r_max', float('nan')):.3f}"))
            lines.append(item("boundary radius", f"{st.get('boundary_r', float('nan')):.3f}"))
            lines.append(item("total loss", f"{st.get('loss', float('nan')):.3e}"))
            lines.append(item("PDE residual loss", f"{st.get('pde', float('nan')):.3e}"))
            lines.append(item("center gauge loss", f"{st.get('center', float('nan')):.3e}"))
            lines.append(item("boundary loss", f"{st.get('boundary', float('nan')):.3e}"))
            lines.append(item("mean |phi_theta-phi_true|", f"{st.get('mean_abs_error', float('nan')):.3e}"))
            lines.append(item("max |phi_theta-phi_true|", f"{st.get('max_abs_error', float('nan')):.3e}"))
            lines.append(item("last curve relaxation metric", self.last_relaxation_metric))

        lines.append("")
        lines.append("DIAGNOSTIC METHOD")
        lines.append("  The optimizer/training plot uses midpoint quadrature because it is")
        lines.append("  fast and differentiable inside PyTorch. Final length and energy")
        lines.append("  above use 24-point Gauss-Legendre quadrature on each displayed")
        lines.append("  curve segment, which is more reliable near the unit-circle boundary.")
        lines.append("")
        lines.append("RIGOR NOTE")
        lines.append("  This v17.6 app keeps the rigorous disk/cyclic modes, the built-in")
        lines.append("  certified regular genus-2 octagon, compact-polygon JSON loading, and")
        lines.append("  Schottky ideal-geodesic-domain support. Compact-polygon JSON can use")
        lines.append("  arbitrary one-letter generator labels A,B,C,... with lowercase inverses.")
        lines.append("  v17.6 also displays triangle-source metadata for compact surfaces produced")
        lines.append("  from triangle-group constructions, such as Domain Maker v4/v5/v6")
        lines.append("  (2,3,7)-compatible 14-gon. The Explorer consumes external certification")
        lines.append("  metadata and performs endpoint/angle consistency checks where supplied;")
        lines.append("  it does not prove discreteness or fundamentality from scratch.")

        self.diagnostics.setPlainText("\n".join(lines) if lines else "Diagnostics will appear here.")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def refresh_plot(self, light: bool = False):
        ax = self.canvas.ax_disk
        ae = self.canvas.ax_energy
        ax.clear()
        ae.clear()

        title = "Poincare disk"
        if self.selected_model() == self.MODEL_CYCLIC:
            title += f" with certified cyclic quotient <A>, selected lift A^{self.current_exponent}(q)"
        elif self.selected_model() == self.MODEL_TWOGEN:
            title += f" with by-fiat <A,B>, selected lift {self.surface.lift_label(self.current_lift_selector())}"
        elif self.selected_model() == self.MODEL_OCTAGON:
            title += f" with certified genus-2 octagon, selected lift {self.surface.lift_label(self.current_lift_selector())}"
        elif self.selected_model() == self.MODEL_POLYGON:
            title += f" with advanced user polygon F, selected lift {self.surface.lift_label(self.current_lift_selector())}"
        elif self.selected_model() == self.MODEL_SCHOTTKY:
            title += f" with Schottky ideal-geodesic domain, selected lift {self.surface.lift_label(self.current_lift_selector())}"
        elif self.selected_model() == self.MODEL_MODULAR_FORD:
            title += f" with modular/Ford domain, selected lift {self.surface.lift_label(self.current_lift_selector())}"
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)

        if self.show_phi_box.isChecked() or (hasattr(self, "show_learned_phi_box") and self.show_learned_phi_box.isChecked()) or (hasattr(self, "show_metric_error_box") and self.show_metric_error_box.isChecked()):
            xs = np.linspace(-0.995, 0.995, 240)
            ys = np.linspace(-0.995, 0.995, 240)
            X, Y = np.meshgrid(xs, ys)
            R2 = X * X + Y * Y
            phi_true = math.log(2.0) - np.log(np.maximum(1.0 - R2, 1.0e-12))
            phi_true[R2 >= 1.0] = np.nan
            field = phi_true
            if hasattr(self, "show_learned_phi_box") and self.show_learned_phi_box.isChecked() and self.learned_metric_available():
                phi_learned = self.learned_phi_grid(X, Y)
                phi_learned[R2 >= 1.0] = np.nan
                field = phi_learned
            if hasattr(self, "show_metric_error_box") and self.show_metric_error_box.isChecked() and self.learned_metric_available():
                phi_learned = self.learned_phi_grid(X, Y)
                phi_learned[R2 >= 1.0] = np.nan
                field = phi_learned - phi_true
                ax.imshow(np.clip(field, -0.8, 0.8), extent=[-0.995, 0.995, -0.995, 0.995], origin="lower", alpha=0.55)
            else:
                ax.imshow(np.clip(field, 0.0, 4.0), extent=[-0.995, 0.995, -0.995, 0.995], origin="lower", alpha=0.45)

        boundary = circle_points(0.0, 0.0, 1.0, 720)
        ax.plot(boundary[:, 0], boundary[:, 1], linewidth=2.0, label="unit boundary")

        if self.selected_model() == self.MODEL_CYCLIC:
            ax.plot([-1.0, 1.0], [0.0, 0.0], linestyle=":", linewidth=1.35, alpha=0.8, label="axis of A")
        elif self.selected_model() == self.MODEL_TWOGEN:
            for theta_deg, lab in [(float(self.custom_a_angle_spin.value()), "axis of A"), (float(self.custom_b_angle_spin.value()), "axis of B")]:
                th = math.radians(theta_deg)
                ax.plot([-math.cos(th), math.cos(th)], [-math.sin(th), math.sin(th)], linestyle=":", linewidth=1.1, alpha=0.65, label=lab)

        draw_tessellation = self.show_strips_box.isChecked() and self.surface.supports_strip_boundaries
        if light and getattr(self, "skip_heavy_tessellation_redraw", False):
            draw_tessellation = False
        if draw_tessellation:
            debug_mode = bool(hasattr(self, "debug_tessellation_box") and self.debug_tessellation_box.isChecked())
            if debug_mode and self.surface_has_polygon_vertices():
                self.draw_tessellation_debug_view(ax)
            else:
                if self.surface_has_polygon_vertices():
                    self.draw_base_polygon_shade(ax)
                edges = self.compute_strip_edges()
                if self.selected_model() == self.MODEL_CYCLIC:
                    # Cyclic strips are few; individual styling helps show the base strip.
                    for label, edge in edges:
                        lw = 1.35 if label in (-1, 0) else 0.72
                        alpha = 0.68 if label in (-1, 0) else 0.34
                        ax.plot(edge[:, 0], edge[:, 1], linewidth=lw, alpha=alpha, color="0.25")
                else:
                    # v14.2 fast path: draw the whole finite word patch as a single collection.
                    segments = [edge for _, edge in edges]
                    if segments:
                        lc = LineCollection(segments, colors="0.25", linewidths=0.50, alpha=0.25)
                        ax.add_collection(lc)
                        # Draw the base polygon on top for orientation.
                        for label, edge in edges:
                            if str(label) == "":
                                ax.plot(edge[:, 0], edge[:, 1], linewidth=1.35, alpha=0.88, color="0.12")
                        if hasattr(self, "show_tile_labels_box") and self.show_tile_labels_box.isChecked() and self.surface_has_polygon_vertices():
                            for word, poly in self.compute_polygon_tessellation_tiles(depth_override=min(2, int(self.orbit_depth_spin.value()))):
                                c = np.mean(poly, axis=0)
                                ax.text(c[0], c[1], "F" if word == "" else word, fontsize=6, ha="center", va="center", alpha=0.55)
                if not getattr(self, "tessellation_requested", False) and self.selected_model() != self.MODEL_CYCLIC:
                    tess_label = "principal/base domain boundary"
                else:
                    tess_label = "Gamma-generated strip boundaries" if self.selected_model() == self.MODEL_CYCLIC else "finite word patch gamma(F)"
                ax.plot([], [], linewidth=0.8, alpha=0.45, color="0.25", label=tess_label)

        orbit = self.compute_orbit_points()
        selected_selector = self.current_lift_selector()
        if self.show_candidates_box.isChecked() and self.p is not None:
            for label, pt in orbit:
                if label == selected_selector:
                    continue
                curve = exact_poincare_geodesic(self.p, pt, n=180)
                ax.plot(curve[:, 0], curve[:, 1], linewidth=0.8, alpha=0.16)
        if self.show_orbit_box.isChecked() and orbit:
            pts = np.array([pt for _, pt in orbit], dtype=float)
            orbit_label = "orbit A^n(q)" if self.selected_model() == self.MODEL_CYCLIC else "orbit words(q)"
            ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.35, label=orbit_label)
            # Label a small number of low-complexity images for clarity.
            labels_drawn = 0
            for label, pt in orbit:
                show_label = False
                if isinstance(label, int):
                    show_label = abs(label) <= 3
                else:
                    show_label = len(str(label)) <= 2 and labels_drawn < 16
                if show_label:
                    ax.text(pt[0], pt[1], f" {label if label != '' else 'e'}", fontsize=7, alpha=0.65)
                    labels_drawn += 1

        if self.p is not None:
            ax.scatter([self.p[0]], [self.p[1]], s=75, marker="o", label="p")
        if self.q is not None:
            ax.scatter([self.q[0]], [self.q[1]], s=75, marker="s", label="q")
        if self.q_lift is not None:
            label = self.surface.lift_label(self.current_lift_selector())
            ax.scatter([self.q_lift[0]], [self.q_lift[1]], s=95, marker="D", label=label)

        if self.exact_curve is not None and self.show_exact_box.isChecked():
            ax.plot(self.exact_curve[:, 0], self.exact_curve[:, 1], linewidth=2.9, label="selected exact lifted geodesic")
        if self.initial_curve is not None and self.show_initial_box.isChecked():
            ax.plot(self.initial_curve[:, 0], self.initial_curve[:, 1], linestyle="--", linewidth=1.8, label="initial neural curve")
            ax.scatter(self.initial_curve[1:-1, 0], self.initial_curve[1:-1, 1], s=8, alpha=0.4)
        if self.exact_metric_curve is not None and self.show_neural_box.isChecked():
            ax.plot(self.exact_metric_curve[:, 0], self.exact_metric_curve[:, 1], linewidth=2.2, linestyle=":", label="NN geodesic using exact metric")
        if self.learned_metric_curve is not None and self.show_neural_box.isChecked():
            ax.plot(self.learned_metric_curve[:, 0], self.learned_metric_curve[:, 1], linewidth=2.2, linestyle="--", label="NN geodesic using learned metric")
        if self.neural_curve is not None and self.show_neural_box.isChecked():
            ax.plot(self.neural_curve[:, 0], self.neural_curve[:, 1], linewidth=2.7, label="current relaxed neural curve")
            ax.scatter(self.neural_curve[1:-1, 0], self.neural_curve[1:-1, 1], s=8, alpha=0.4)

        ax.legend(loc="upper right", fontsize=7)

        if hasattr(self, "use_learned_metric_box") and self.use_learned_metric_box.isChecked() and self.learned_metric_available():
            ae.set_title("PyTorch neural relaxation energy using learned phi_theta")
        else:
            ae.set_title("PyTorch neural relaxation energy (exact Poincare metric)")
        ae.set_xlabel("optimization step")
        ae.set_ylabel("hyperbolic energy")
        ae.grid(True, alpha=0.25)
        if self.energy_history is not None and len(self.energy_history) > 0:
            steps_arr = np.arange(1, len(self.energy_history) + 1)
            ae.plot(steps_arr, self.energy_history, linewidth=2.0, label="training midpoint energy")
            hyp = self.selected_distance()
            if hyp is not None:
                ae.axhline(0.5 * hyp * hyp, linestyle="--", linewidth=1.2, alpha=0.7, label="exact 1/2 d^2")
            if self.neural_curve is not None:
                _, diag_energy, _ = diagnostic_length_energy_speeds(self.neural_curve)
                ae.scatter([steps_arr[-1]], [diag_energy], s=38, marker="o", label="final diagnostic E")
            ae.text(
                0.98,
                0.05,
                "final marker uses higher-accuracy quadrature",
                ha="right",
                va="bottom",
                transform=ae.transAxes,
                fontsize=8,
            )
            ae.legend(loc="upper right", fontsize=7)
            ae.set_ylim(bottom=0.0)
        elif self.metric_loss_history is not None and len(self.metric_loss_history) > 0:
            ae.set_title("Neural metric training loss")
            ae.set_xlabel("metric training step")
            ae.set_ylabel("loss / error")
            steps_arr = np.arange(1, len(self.metric_loss_history) + 1)
            ae.semilogy(steps_arr, self.metric_loss_history[:, 0], linewidth=2.0, label="total loss")
            ae.semilogy(steps_arr, self.metric_loss_history[:, 1], linewidth=1.2, label="PDE residual")
            ae.semilogy(steps_arr, self.metric_loss_history[:, 4], linewidth=1.2, label="mean abs phi error")
            ae.legend(loc="upper right", fontsize=7)
        else:
            ae.text(0.5, 0.5, "Energy/loss history appears\nafter neural relaxation or metric training.", ha="center", va="center", transform=ae.transAxes)

        self.canvas.draw_idle()
        if not light:
            self.update_diagnostics()
            if self.metric_window is not None and self.metric_window.isVisible():
                self.metric_window.plot_metric(
                    p=self.p,
                    q=self.q,
                    q_lift=self.q_lift,
                    exact_curve=self.exact_curve if self.show_exact_box.isChecked() else None,
                    neural_curve=self.neural_curve if self.show_neural_box.isChecked() else None,
                )


def main():
    app = QApplication(sys.argv)
    window = FuchsianGENNExplorer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
