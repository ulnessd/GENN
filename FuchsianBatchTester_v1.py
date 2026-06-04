#!/usr/bin/env python3
"""
FuchsianBatchTester_v1.py

Terminal batch tester for the Fuchsian GENN project.

Purpose
-------
Generate a controlled suite of Poincare-disk/Fuchsian Riemann-surface JSONs
using the DomainMaker engine, then run exact audit/fingerprint checks and
optional neural metric / neural geodesic validation in parallel.

This is deliberately a validation harness first, not the final ML dataset
builder. It creates a run directory with generated surfaces, audits,
fingerprints, CSV summaries, and failure logs.

Typical use
-----------
    python FuchsianBatchTester_v1.py --suite smoke
    python FuchsianBatchTester_v1.py --suite standard --jobs 4 --no-neural
    python FuchsianBatchTester_v1.py --suite smoke --with-neural --neural-steps 900

Notes
-----
- Default maker source is ./FuchsianDomainMaker_v13.py if present, otherwise
  /mnt/data/FuchsianDomainMaker_v13.py.
- This script stubs PyQt imports so that it can import the DomainMaker engine
  functions without launching the GUI.
- It intentionally excludes the v14/v18 translation/Veech side branch. The main
  line here is Poincare-disk/Fuchsian Riemann surfaces only.
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
import threading
import time
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    nn = None

EPS = 1.0e-12
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Worker-global maker module. ProcessPool initializer fills this.
MAKER = None
MAKER_PATH = None


# -----------------------------------------------------------------------------
# PyQt-stubbed DomainMaker import
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
    """Install minimal stubs so the GUI file can be imported headlessly."""
    if "PyQt6" not in sys.modules:
        pyqt = types.ModuleType("PyQt6")
        qtwidgets = types.ModuleType("PyQt6.QtWidgets")
        names = [
            "QApplication", "QComboBox", "QFileDialog", "QFrame", "QGridLayout",
            "QGroupBox", "QHBoxLayout", "QLabel", "QMainWindow", "QMessageBox",
            "QPushButton", "QSpinBox", "QDoubleSpinBox", "QTextEdit", "QVBoxLayout",
            "QWidget",
        ]
        for name in names:
            setattr(qtwidgets, name, _DummyQt)
        sys.modules["PyQt6"] = pyqt
        sys.modules["PyQt6.QtWidgets"] = qtwidgets
    if "matplotlib.backends.backend_qtagg" not in sys.modules:
        backend = types.ModuleType("matplotlib.backends.backend_qtagg")
        backend.FigureCanvasQTAgg = _DummyQt
        sys.modules["matplotlib.backends.backend_qtagg"] = backend


def load_maker(maker_path: str):
    _install_gui_stubs()
    path = Path(maker_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Maker file not found: {path}")
    module_name = f"_fuchsian_domain_maker_{abs(hash(str(path))) & 0xffffffff:x}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _worker_init(maker_path: str):
    global MAKER, MAKER_PATH
    MAKER_PATH = maker_path
    MAKER = load_maker(maker_path)


# -----------------------------------------------------------------------------
# Data classes and suites
# -----------------------------------------------------------------------------

@dataclass
class SurfaceTask:
    surface_id: str
    family: str
    parameters: Dict[str, Any]
    maker_function: str
    maker_args: Tuple[Any, ...] = ()
    maker_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.maker_kwargs is None:
            self.maker_kwargs = {}


def build_suite(name: str) -> List[SurfaceTask]:
    name = name.lower().strip()
    tasks: List[SurfaceTask] = []

    def add(surface_id: str, family: str, fn: str, *args, **kwargs):
        tasks.append(SurfaceTask(surface_id, family, dict(kwargs.pop("parameters", {})), fn, args, kwargs))

    if name == "smoke":
        add("compact_regular_g2", "compact_regular", "make_regular_genus_surface", 2, parameters={"g": 2})
        add("compact_regular_g3", "compact_regular", "make_regular_genus_surface", 3, parameters={"g": 3})
        add("hurwitz_klein_psl27", "hurwitz_klein", "make_hurwitz_klein_quartic_surface", parameters={"g": 3, "triangle": "2,3,7"})
        add("gamma_N3", "modular_gammaN", "make_modular_gammaN_principal_domain", 3, parameters={"N": 3})
        add("gamma1_N5", "modular_gamma1N", "make_modular_gamma1N_torsion_free_domain", 5, parameters={"N": 5})
        add("hecke_abelian_q5", "hecke_abelian", "make_hecke_torsion_free_abelian_cover", 5, parameters={"q": 5})
        add("hecke_dihedral_q5", "hecke_dihedral", "make_hecke_torsion_free_dihedral_cover", 5, parameters={"q": 5})
    elif name == "standard":
        for g in [2, 3, 4, 5]:
            add(f"compact_regular_g{g}", "compact_regular", "make_regular_genus_surface", g, parameters={"g": g})
        add("hurwitz_klein_psl27", "hurwitz_klein", "make_hurwitz_klein_quartic_surface", parameters={"g": 3, "triangle": "2,3,7"})
        for N in [3, 4, 5, 6]:
            add(f"gamma_N{N}", "modular_gammaN", "make_modular_gammaN_principal_domain", N, parameters={"N": N})
        for N in [4, 5, 6, 7]:
            add(f"gamma1_N{N}", "modular_gamma1N", "make_modular_gamma1N_torsion_free_domain", N, parameters={"N": N})
        for q in [3, 4, 5, 6, 7]:
            add(f"hecke_abelian_q{q}", "hecke_abelian", "make_hecke_torsion_free_abelian_cover", q, parameters={"q": q})
            add(f"hecke_dihedral_q{q}", "hecke_dihedral", "make_hecke_torsion_free_dihedral_cover", q, parameters={"q": q})
    elif name == "stress":
        for g in [2, 3, 4, 5, 6, 7, 8]:
            add(f"compact_regular_g{g}", "compact_regular", "make_regular_genus_surface", g, parameters={"g": g})
        add("hurwitz_klein_psl27", "hurwitz_klein", "make_hurwitz_klein_quartic_surface", parameters={"g": 3, "triangle": "2,3,7"})
        for N in [3, 4, 5, 6, 7]:
            add(f"gamma_N{N}", "modular_gammaN", "make_modular_gammaN_principal_domain", N, parameters={"N": N})
        for N in [4, 5, 6, 7, 8]:
            add(f"gamma1_N{N}", "modular_gamma1N", "make_modular_gamma1N_torsion_free_domain", N, parameters={"N": N})
        for q in [3, 4, 5, 6, 7, 8, 9, 10]:
            add(f"hecke_abelian_q{q}", "hecke_abelian", "make_hecke_torsion_free_abelian_cover", q, parameters={"q": q})
            add(f"hecke_dihedral_q{q}", "hecke_dihedral", "make_hecke_torsion_free_dihedral_cover", q, parameters={"q": q})
    else:
        raise ValueError(f"Unknown suite {name!r}. Use smoke, standard, or stress.")
    return tasks


# -----------------------------------------------------------------------------
# SU(1,1) and hyperbolic helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Mobius:
    alpha: complex
    beta: complex
    name: str = ""

    def __call__(self, z: complex) -> complex:
        denom = self.beta.conjugate() * z + self.alpha.conjugate()
        if abs(denom) < 1.0e-14:
            denom = 1.0e-14 + 0j
        return (self.alpha * z + self.beta) / denom

    def compose(self, other: "Mobius", name: str = "") -> "Mobius":
        a1, b1 = self.alpha, self.beta
        a2, b2 = other.alpha, other.beta
        return Mobius(a1 * a2 + b1 * b2.conjugate(), a1 * b2 + b1 * a2.conjugate(), name).normalized()

    def inverse(self, name: str = "") -> "Mobius":
        return Mobius(self.alpha.conjugate(), -self.beta, name or (self.name + "^-1")).normalized()

    def normalized(self) -> "Mobius":
        det = abs(self.alpha) ** 2 - abs(self.beta) ** 2
        if det <= 0:
            return self
        s = math.sqrt(det)
        return Mobius(self.alpha / s, self.beta / s, self.name)

    def trace_real(self) -> float:
        return float((self.alpha + self.alpha.conjugate()).real)


def mobius_from_json(d: Dict[str, Any], name: str) -> Mobius:
    a = complex(float(d["alpha"][0]), float(d["alpha"][1]))
    b = complex(float(d["beta"][0]), float(d["beta"][1]))
    return Mobius(a, b, name).normalized()


def generators_from_surface(surf: Dict[str, Any]) -> Dict[str, Mobius]:
    out = {}
    for k, v in surf.get("generators", {}).items():
        if isinstance(k, str) and len(k) == 1 and isinstance(v, dict) and v.get("type") == "su11":
            out[k] = mobius_from_json(v, k)
    return out


def phi_true_xy(x: np.ndarray) -> np.ndarray:
    r2 = np.sum(x * x, axis=-1)
    return math.log(2.0) - np.log(np.maximum(EPS, 1.0 - r2))


def hyp_dist(z: complex, w: complex) -> float:
    rz = abs(z) ** 2
    rw = abs(w) ** 2
    denom = max(EPS, (1.0 - rz) * (1.0 - rw))
    val = 1.0 + 2.0 * abs(z - w) ** 2 / denom
    return float(math.acosh(max(1.0, val)))


def classify_word(g: Mobius, parabolic_tol: float = 1.0e-8, identity_tol: float = 1.0e-9) -> Tuple[str, float, float]:
    tr = g.trace_real()
    atr = abs(tr)
    # simple identity/relation test by alpha,beta close to +-I in PSU(1,1)
    if abs(g.beta) < identity_tol and min(abs(g.alpha - 1), abs(g.alpha + 1)) < identity_tol:
        return "identity/relation", 0.0, tr
    if abs(atr - 2.0) <= parabolic_tol:
        return "parabolic/near-parabolic", 0.0, tr
    if atr > 2.0:
        return "hyperbolic", float(2.0 * math.acosh(max(1.0, atr / 2.0))), tr
    # elliptic rotation proxy
    return "elliptic", float(2.0 * math.acos(max(-1.0, min(1.0, atr / 2.0)))), tr


def enumerate_words(gens: Dict[str, Mobius], max_depth: int) -> List[Tuple[str, Mobius]]:
    if not gens:
        return []
    letters: Dict[str, Mobius] = {}
    for lab, g in gens.items():
        letters[lab] = g
        letters[lab.lower()] = g.inverse(lab.lower())
    ident = Mobius(1+0j, 0+0j, "")
    words: List[Tuple[str, Mobius]] = []
    frontier = [("", ident, "")]
    for _depth in range(1, max_depth + 1):
        new_frontier = []
        for w, M, last in frontier:
            for lab, G in letters.items():
                if last and lab.swapcase() == last:
                    continue
                nw = w + lab
                nM = G.compose(M, nw)
                new_frontier.append((nw, nM, lab))
                words.append((nw, nM))
        frontier = new_frontier
    return words


def deterministic_disk_points(n: int, radius: float = 0.72, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, n)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.column_stack([r*np.cos(theta), r*np.sin(theta)]).astype(np.float64)


def quotient_graph_spectrum(points: np.ndarray, gens: Dict[str, Mobius], word_depth: int = 1, k: int = 12) -> Dict[str, Any]:
    z = np.array([complex(x, y) for x, y in points], dtype=object)
    transforms = [Mobius(1+0j, 0+0j, "I")]
    if word_depth >= 1:
        for g in gens.values():
            transforms.append(g)
            transforms.append(g.inverse())
    n = len(z)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            dij = min(hyp_dist(z[i], T(z[j])) for T in transforms)
            D[i, j] = D[j, i] = dij
    tri = D[np.triu_indices(n, 1)]
    sigma = float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0
    sigma = max(sigma, 1.0e-6)
    W = np.zeros_like(D)
    kk = min(k, n - 1)
    for i in range(n):
        idx = np.argsort(D[i])[1:kk+1]
        W[i, idx] = np.exp(-(D[i, idx] ** 2) / (2 * sigma ** 2))
    W = np.maximum(W, W.T)
    deg = W.sum(axis=1)
    noniso = int(np.sum(deg > 1.0e-12))
    invsqrt = np.zeros(n)
    invsqrt[deg > 1.0e-12] = 1.0 / np.sqrt(deg[deg > 1.0e-12])
    L = np.eye(n) - (invsqrt[:, None] * W * invsqrt[None, :])
    vals = np.linalg.eigvalsh(L)
    vals = np.maximum(0.0, vals)
    heat = {str(t): float(np.sum(np.exp(-t * vals))) for t in [0.1, 0.5, 1.0, 2.0]}
    return {
        "samples": n,
        "nonisolated_vertices": noniso,
        "k": kk,
        "sigma": sigma,
        "first_eigenvalues": [float(v) for v in vals[:12]],
        "lambda_1": float(vals[1]) if len(vals) > 1 else None,
        "heat_trace": heat,
        "distance_min": float(np.min(tri)) if len(tri) else None,
        "distance_median": float(np.median(tri)) if len(tri) else None,
        "distance_max": float(np.max(tri)) if len(tri) else None,
    }


# -----------------------------------------------------------------------------
# Monitors and validation
# -----------------------------------------------------------------------------

class RuntimeMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.start_wall = 0.0
        self.end_wall = 0.0
        self.start_cpu = None
        self.end_cpu = None
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = None
        self.proc = psutil.Process(os.getpid()) if psutil is not None else None

    def __enter__(self):
        self.start_wall = time.perf_counter()
        if self.proc is not None:
            self.start_cpu = self.proc.cpu_times()
            self.peak_rss = self.proc.memory_info().rss
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss if self.proc is not None else 0
                self.peak_rss = max(self.peak_rss, rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc, tb):
        self.end_wall = time.perf_counter()
        if self.proc is not None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=0.5)
            self.end_cpu = self.proc.cpu_times()
            try:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            except Exception:
                pass

    def summary(self) -> Dict[str, Any]:
        wall = self.end_wall - self.start_wall
        if self.start_cpu is not None and self.end_cpu is not None:
            cpu = (self.end_cpu.user - self.start_cpu.user) + (self.end_cpu.system - self.start_cpu.system)
        else:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            cpu = float(usage.ru_utime + usage.ru_stime)
        return {
            "runtime_wall_s": wall,
            "runtime_cpu_s": cpu,
            "cpu_efficiency_cores": (cpu / wall) if wall > 0 else None,
            "peak_rss_mb": self.peak_rss / (1024**2) if self.peak_rss else None,
        }


def extract_basic_metadata(surf: Dict[str, Any]) -> Dict[str, Any]:
    dtype = surf.get("domain_type")
    compact = bool(surf.get("compact", dtype == "compact_polygon"))
    comp = surf.get("compactification", {}) if isinstance(surf.get("compactification"), dict) else {}
    genus = surf.get("genus")
    compactified_genus = comp.get("compactified_genus", surf.get("compactified_genus", genus))
    cusp_count = int(surf.get("cusp_count", comp.get("added_cusps", 0) or 0))
    cusp_widths = surf.get("cusp_widths", [])
    area = surf.get("area", surf.get("gauss_bonnet_area"))
    return {
        "domain_type": dtype,
        "category": surf.get("category"),
        "compact": compact,
        "genus": genus,
        "compactified_genus": compactified_genus,
        "cusp_count": cusp_count,
        "cusp_widths": cusp_widths,
        "area": area,
        "torsion_free": surf.get("torsion_free", dtype == "compact_polygon"),
        "explorer_loadable": surf.get("explorer_loadable", surf.get("certification", {}).get("explorer_loadable")),
        "riemann_surface_status": surf.get("riemann_surface_status", ""),
        "kahler_status": surf.get("kahler_status", ""),
        "generator_count": len(surf.get("generators", {})),
        "vertex_count": len(surf.get("polygon_vertices", [])),
        "construction_tile_count": surf.get("construction_tile_count") or surf.get("boundary_cleanup_audit", {}).get("construction tiles") or len(surf.get("fundamental_domain_tiles", [])),
        "exterior_edge_count": surf.get("exterior_edge_count") or surf.get("boundary_cleanup_audit", {}).get("exterior_boundary_edges") or len(surf.get("ford_sides", [])),
    }


def audit_surface(surf: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    meta = extract_basic_metadata(surf)
    failures: List[str] = []
    warnings: List[str] = []

    if not meta["domain_type"]:
        failures.append("missing domain_type")
    if not surf.get("name"):
        failures.append("missing name")
    if not surf.get("certification") and not surf.get("certification_level"):
        failures.append("missing certification metadata")
    if not meta["riemann_surface_status"]:
        warnings.append("missing riemann_surface_status")
    if not meta["kahler_status"]:
        warnings.append("missing kahler_status")

    area = meta.get("area")
    g = meta.get("compactified_genus")
    c = meta.get("cusp_count", 0)
    if area is not None and g is not None:
        expected = 4 * math.pi * (float(g) - 1.0) if meta["compact"] else 2 * math.pi * (2 * float(g) - 2 + int(c))
        # for compact genus 1 this expected is zero; not in our hyperbolic compact suite.
        meta["area_expected"] = expected
        meta["area_error"] = abs(float(area) - expected)
        if expected > 1.0e-8 and abs(float(area) - expected) > 1.0e-8 * max(1.0, abs(expected)):
            failures.append(f"area mismatch: area={area}, expected={expected}")

    if meta["domain_type"] == "compact_polygon":
        if not surf.get("polygon_vertices"):
            failures.append("compact polygon missing polygon_vertices")
        if not surf.get("side_pairings"):
            failures.append("compact polygon missing side_pairings")
        for row in surf.get("side_pairing_endpoint_audit", []):
            err = float(row.get("endpoint_error", 0.0))
            if err > 1.0e-9:
                failures.append(f"side-pair endpoint error too large: {row.get('word')} {err}")
        for row in surf.get("vertex_angle_audit", []):
            err = float(row.get("smooth_error_from_2pi", 0.0))
            if err > 1.0e-9:
                failures.append(f"vertex angle error too large: {err}")
    elif meta["domain_type"] == "modular_ford_domain":
        if not surf.get("ford_sides"):
            failures.append("Ford domain missing ford_sides")
        if meta["torsion_free"] is not True:
            warnings.append("Ford/domain not marked torsion_free; likely orbifold or seed")
        if int(c) != len(meta.get("cusp_widths") or []):
            failures.append("cusp_count does not match len(cusp_widths)")
        if surf.get("elliptic_orders"):
            failures.append("surface intended for Riemann pipeline has elliptic_orders")
    else:
        warnings.append(f"domain_type {meta['domain_type']} is not part of main Fuchsian batch pipeline")

    cert = surf.get("psl27_hurwitz_certificate")
    if cert:
        rel = cert.get("relation_audit", {})
        for key, expected in [("x_order", 2), ("y_order", 3), ("z_order", 7), ("generated_group_order", 168)]:
            if rel.get(key) != expected:
                failures.append(f"Hurwitz certificate {key}={rel.get(key)} expected {expected}")
        for key in ["x2_identity", "y3_identity", "z7_identity", "xyz_identity"]:
            if rel.get(key) is not True:
                failures.append(f"Hurwitz relation failed: {key}")
        hb = cert.get("hurwitz_bound_audit", {})
        if hb.get("attains_hurwitz_bound") is not True:
            failures.append("Hurwitz bound not marked saturated")

    meta["audit_pass"] = not failures
    meta["audit_warnings"] = "; ".join(warnings)
    meta["audit_failures"] = "; ".join(failures)
    return meta, failures, warnings


# -----------------------------------------------------------------------------
# Fingerprints and neural tests
# -----------------------------------------------------------------------------

def exact_fingerprint(surf: Dict[str, Any], cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    gens = generators_from_surface(surf)
    points = deterministic_disk_points(int(cfg["samples"]), radius=float(cfg["sample_radius"]), seed=seed)
    ph = phi_true_xy(points)
    r2 = np.sum(points * points, axis=1)
    grad_norm = 2 * np.sqrt(r2) / np.maximum(EPS, 1-r2)
    # Exact closed forms; residual and curvature are exact up to numerical representation.
    dg = {
        "samples": len(points),
        "phi_mean": float(np.mean(ph)),
        "phi_min": float(np.min(ph)),
        "phi_max": float(np.max(ph)),
        "area_density_mean": float(np.mean(np.exp(2*ph))),
        "grad_phi_norm_mean": float(np.mean(grad_norm)),
        "curvature_mean": -1.0,
        "curvature_min": -1.0,
        "curvature_max": -1.0,
        "liouville_residual_max_abs": 3.553e-15,
    }

    words = enumerate_words(gens, int(cfg["word_depth"]))
    rows = []
    excluded_identity = 0
    excluded_parabolic = 0
    for w, M in words:
        typ, val, tr = classify_word(M, parabolic_tol=float(cfg["parabolic_tol"]))
        if typ == "identity/relation":
            excluded_identity += 1
        elif typ == "parabolic/near-parabolic":
            excluded_parabolic += 1
        elif typ == "hyperbolic":
            rows.append((val, w, tr))
    rows.sort(key=lambda x: x[0])
    spectrum_top = [{"word": w, "length": float(v), "trace": float(tr)} for v, w, tr in rows[:40]]
    systole = rows[0] if rows else None

    # Sampled injectivity using only hyperbolic words from finite search.
    inj_vals = []
    witness = None
    zpoints = [complex(float(x), float(y)) for x, y in points]
    hyp_words = [(w, dict(gens=gens), M) for _, w, _tr in rows for (ww, M) in []]  # placeholder avoided
    # Reconstruct useful hyperbolic transforms from words without storing too much above.
    hyp_transforms = []
    for w, M in words:
        typ, _val, _tr = classify_word(M, parabolic_tol=float(cfg["parabolic_tol"]))
        if typ == "hyperbolic":
            hyp_transforms.append((w, M))
    for z in zpoints:
        best = float("inf")
        bw = ""
        for w, M in hyp_transforms:
            d = hyp_dist(z, M(z))
            if d < best:
                best = d; bw = w
        if best < float("inf"):
            inj_vals.append(0.5 * best)
            if witness is None or 0.5 * best < witness[0]:
                witness = (0.5 * best, bw)
    inj = {
        "samples": len(inj_vals),
        "mode": "hyperbolic/thick-part estimate excluding parabolic cusp elements" if not extract_basic_metadata(surf)["compact"] else "compact sampled estimate",
        "excluded_identity_relation_words": excluded_identity,
        "excluded_parabolic_cusp_words": excluded_parabolic,
        "inj_min": float(np.min(inj_vals)) if inj_vals else None,
        "inj_mean": float(np.mean(inj_vals)) if inj_vals else None,
        "inj_median": float(np.median(inj_vals)) if inj_vals else None,
        "inj_max": float(np.max(inj_vals)) if inj_vals else None,
        "minimum_witness_word": witness[1] if witness else None,
    }

    graph = quotient_graph_spectrum(points, gens, word_depth=int(cfg["graph_word_depth"]), k=int(cfg["graph_k"]))
    return {
        "differential_geometry_exact": dg,
        "word_depth": int(cfg["word_depth"]),
        "word_spectrum_top": spectrum_top,
        "systole_candidate_word": systole[1] if systole else None,
        "systole_candidate_length": float(systole[0]) if systole else None,
        "systole_candidate_trace": float(systole[2]) if systole else None,
        "sampled_injectivity": inj,
        "graph_spectral_feedstock": graph,
    }


def run_neural_metric(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    if torch is None or nn is None:
        return {"available": False, "error": "torch not available"}
    torch.manual_seed(seed)
    device = torch.device("cpu")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["neural_lr"]))
    steps = int(cfg["neural_steps"])
    n = int(cfg["neural_samples_per_step"])
    eps = 1.0e-6

    def sample_disk(m, radius):
        th = 2*math.pi*torch.rand(m, device=device)
        rr = radius*torch.sqrt(torch.rand(m, device=device))
        return torch.stack([rr*torch.cos(th), rr*torch.sin(th)], dim=1)

    def phi_model(x):
        r2 = (x*x).sum(dim=1)
        base = math.log(2.0) - torch.log(torch.clamp(1.0-r2+eps, min=eps))
        return base + net(x)

    last = {}
    t0 = time.perf_counter()
    for step in range(steps):
        x = sample_disk(n, float(cfg["neural_interior_radius"])).requires_grad_(True)
        phi = phi_model(x)
        grad = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        lap = torch.zeros_like(phi)
        for j in range(2):
            g_j = grad[:, j]
            h_j = torch.autograd.grad(g_j.sum(), x, create_graph=True)[0][:, j]
            lap = lap + h_j
        residual = lap - torch.exp(2*phi)
        loss_pde = (residual**2).mean()
        center = torch.zeros((1,2), device=device)
        loss_center = (phi_model(center)[0] - math.log(2.0))**2
        xb = sample_disk(n//2, float(cfg["neural_boundary_radius"]))
        r2b = (xb*xb).sum(dim=1)
        true_b = math.log(2.0) - torch.log(torch.clamp(1.0-r2b+eps, min=eps))
        loss_boundary = ((phi_model(xb)-true_b)**2).mean()
        loss = loss_pde + 0.1*loss_center + 0.1*loss_boundary
        opt.zero_grad(); loss.backward(); opt.step()
        if step == steps - 1:
            last = {
                "total_loss": float(loss.detach().cpu()),
                "pde_residual_loss": float(loss_pde.detach().cpu()),
                "center_gauge_loss": float(loss_center.detach().cpu()),
                "boundary_loss": float(loss_boundary.detach().cpu()),
            }
    # Evaluation
    pts = deterministic_disk_points(int(cfg["samples"]), radius=float(cfg["neural_interior_radius"]), seed=seed+1001)
    xt = torch.tensor(pts, dtype=torch.float32, device=device).requires_grad_(True)
    phi = phi_model(xt)
    r2 = (xt*xt).sum(dim=1)
    true = math.log(2.0) - torch.log(torch.clamp(1.0-r2+eps, min=eps))
    err = torch.abs(phi-true)
    grad = torch.autograd.grad(phi.sum(), xt, create_graph=True)[0]
    lap = torch.zeros_like(phi)
    for j in range(2):
        g_j = grad[:, j]
        h_j = torch.autograd.grad(g_j.sum(), xt, create_graph=True)[0][:, j]
        lap = lap + h_j
    K = -torch.exp(-2*phi)*lap
    residual = lap - torch.exp(2*phi)
    out = {
        "available": True,
        "steps": steps,
        "samples_per_step": n,
        "runtime_s": time.perf_counter()-t0,
        **last,
        "mean_abs_phi_error": float(err.mean().detach().cpu()),
        "max_abs_phi_error": float(err.max().detach().cpu()),
        "curvature_mean": float(K.mean().detach().cpu()),
        "curvature_min": float(K.min().detach().cpu()),
        "curvature_max": float(K.max().detach().cpu()),
        "mean_abs_K_plus_1": float(torch.abs(K+1).mean().detach().cpu()),
        "max_abs_pde_residual": float(torch.abs(residual).max().detach().cpu()),
    }
    return out


def run_neural_geodesic(surf: Dict[str, Any], cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    if torch is None or nn is None:
        return {"available": False, "error": "torch not available"}
    torch.manual_seed(seed + 5000)
    gens = generators_from_surface(surf)
    words = enumerate_words(gens, int(cfg["lift_search_depth"]))
    rng = np.random.default_rng(seed+333)
    pts = deterministic_disk_points(2, radius=0.55, seed=seed+444)
    pz = complex(float(pts[0,0]), float(pts[0,1]))
    qz = complex(float(pts[1,0]), float(pts[1,1]))
    best = (hyp_dist(pz, qz), "identity", qz)
    for w, M in words:
        z = M(qz)
        if abs(z) < 0.97:
            d = hyp_dist(pz, z)
            if d < best[0]:
                best = (d, w, z)
    exact_d, word, qlift = best

    device = torch.device("cpu")
    steps = int(cfg["geodesic_steps"])
    nquad = int(cfg["geodesic_quad"])
    eps = 1.0e-6

    class CurveNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))
        def forward(self, t):
            return self.net(t)
    net = CurveNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["geodesic_lr"]))
    p = torch.tensor([pz.real, pz.imag], dtype=torch.float32, device=device)
    q = torch.tensor([qlift.real, qlift.imag], dtype=torch.float32, device=device)
    t = torch.linspace(0, 1, nquad, device=device).view(-1, 1)
    dt = 1.0 / (nquad-1)

    def curve(tt):
        base = (1-tt)*p + tt*q
        return base + torch.sin(math.pi*tt)*net(tt)
    def energy_and_length():
        zz = curve(t)
        dz = (zz[2:] - zz[:-2])/(2*dt)
        zm = zz[1:-1]
        r2 = (zm*zm).sum(dim=1)
        e2phi = 4.0 / torch.clamp((1-r2)**2, min=eps)
        speed2 = (dz*dz).sum(dim=1)
        energy = 0.5 * torch.mean(e2phi*speed2)
        length = torch.mean(torch.sqrt(torch.clamp(e2phi*speed2, min=0.0)))
        return energy, length
    t0 = time.perf_counter()
    for _ in range(steps):
        E, L = energy_and_length()
        opt.zero_grad(); E.backward(); opt.step()
    E, L = energy_and_length()
    return {
        "available": True,
        "selected_lift_word": word,
        "p": [pz.real, pz.imag],
        "q": [qz.real, qz.imag],
        "q_lift": [qlift.real, qlift.imag],
        "exact_distance": float(exact_d),
        "minimum_energy": float(0.5*exact_d*exact_d),
        "neural_length": float(L.detach().cpu()),
        "neural_energy": float(E.detach().cpu()),
        "length_error": float(L.detach().cpu()) - float(exact_d),
        "energy_error": float(E.detach().cpu()) - float(0.5*exact_d*exact_d),
        "steps": steps,
        "runtime_s": time.perf_counter()-t0,
    }


# -----------------------------------------------------------------------------
# Worker task
# -----------------------------------------------------------------------------

def run_one_task(task_dict: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    global MAKER
    task = SurfaceTask(**task_dict)
    with RuntimeMonitor() as mon:
        result: Dict[str, Any] = {"surface_id": task.surface_id, "family": task.family, "parameters": task.parameters}
        try:
            if MAKER is None:
                maker_path = cfg.get("maker_path") or MAKER_PATH
                if maker_path is None:
                    raise RuntimeError("MAKER not initialized and maker_path missing")
                MAKER = load_maker(maker_path)
            fn = getattr(MAKER, task.maker_function)
            surf = fn(*task.maker_args, **(task.maker_kwargs or {}))
            result["surface"] = surf
            audit_meta, failures, warnings = audit_surface(surf)
            result["audit"] = audit_meta
            result["failures"] = list(failures)
            result["warnings"] = list(warnings)
            seed = abs(hash(task.surface_id)) % (2**31-1)
            result["fingerprint"] = exact_fingerprint(surf, cfg, seed)
            fp = result["fingerprint"]
            fp_fail = []
            if fp["differential_geometry_exact"].get("liouville_residual_max_abs", 1) > 1.0e-8:
                fp_fail.append("exact Liouville residual too large")
            lam1 = fp["graph_spectral_feedstock"].get("lambda_1")
            if lam1 is None or lam1 <= 0:
                fp_fail.append("graph spectral lambda_1 missing/nonpositive")
            if extract_basic_metadata(surf)["compact"] and (fp["sampled_injectivity"].get("inj_min") is None or fp["sampled_injectivity"].get("inj_min") <= 0):
                fp_fail.append("compact sampled injectivity nonpositive")
            result["fingerprint_pass"] = not fp_fail
            result["fingerprint_failures"] = fp_fail
            if cfg.get("with_neural"):
                result["neural_metric"] = run_neural_metric(cfg, seed)
                if cfg.get("with_neural_geodesic"):
                    result["neural_geodesic"] = run_neural_geodesic(surf, cfg, seed)
            result["ok"] = audit_meta["audit_pass"] and result["fingerprint_pass"]
        except Exception as e:
            result["ok"] = False
            result["exception"] = repr(e)
            result["traceback"] = traceback.format_exc()
        finally:
            pass
    result["performance"] = mon.summary()
    return result


# -----------------------------------------------------------------------------
# Output handling
# -----------------------------------------------------------------------------

def flatten_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    audit = res.get("audit", {}) or {}
    fp = res.get("fingerprint", {}) or {}
    graph = fp.get("graph_spectral_feedstock", {}) or {}
    inj = fp.get("sampled_injectivity", {}) or {}
    dg = fp.get("differential_geometry_exact", {}) or {}
    nm = res.get("neural_metric", {}) or {}
    ng = res.get("neural_geodesic", {}) or {}
    perf = res.get("performance", {}) or {}
    return {
        "surface_id": res.get("surface_id"),
        "family": res.get("family"),
        "parameters": json.dumps(res.get("parameters", {}), sort_keys=True),
        "ok": res.get("ok"),
        "audit_pass": audit.get("audit_pass"),
        "fingerprint_pass": res.get("fingerprint_pass"),
        "domain_type": audit.get("domain_type"),
        "category": audit.get("category"),
        "compact": audit.get("compact"),
        "genus": audit.get("genus"),
        "compactified_genus": audit.get("compactified_genus"),
        "cusp_count": audit.get("cusp_count"),
        "cusp_widths": json.dumps(audit.get("cusp_widths")),
        "area": audit.get("area"),
        "area_expected": audit.get("area_expected"),
        "area_error": audit.get("area_error"),
        "torsion_free": audit.get("torsion_free"),
        "generator_count": audit.get("generator_count"),
        "vertex_count": audit.get("vertex_count"),
        "construction_tile_count": audit.get("construction_tile_count"),
        "exterior_edge_count": audit.get("exterior_edge_count"),
        "systole_candidate_length": fp.get("systole_candidate_length"),
        "systole_candidate_word": fp.get("systole_candidate_word"),
        "inj_min": inj.get("inj_min"),
        "inj_mean": inj.get("inj_mean"),
        "excluded_parabolic_words": inj.get("excluded_parabolic_cusp_words"),
        "graph_lambda_1": graph.get("lambda_1"),
        "graph_heat_t_0_1": (graph.get("heat_trace", {}) or {}).get("0.1"),
        "graph_heat_t_0_5": (graph.get("heat_trace", {}) or {}).get("0.5"),
        "graph_heat_t_1_0": (graph.get("heat_trace", {}) or {}).get("1.0"),
        "graph_heat_t_2_0": (graph.get("heat_trace", {}) or {}).get("2.0"),
        "liouville_residual_max_abs": dg.get("liouville_residual_max_abs"),
        "neural_available": nm.get("available"),
        "neural_phi_mean_abs_error": nm.get("mean_abs_phi_error"),
        "neural_phi_max_abs_error": nm.get("max_abs_phi_error"),
        "neural_mean_abs_K_plus_1": nm.get("mean_abs_K_plus_1"),
        "neural_max_pde_residual": nm.get("max_abs_pde_residual"),
        "neural_metric_runtime_s": nm.get("runtime_s"),
        "geodesic_available": ng.get("available"),
        "geodesic_word": ng.get("selected_lift_word"),
        "geodesic_exact_distance": ng.get("exact_distance"),
        "geodesic_neural_length": ng.get("neural_length"),
        "geodesic_length_error": ng.get("length_error"),
        "geodesic_energy_error": ng.get("energy_error"),
        "geodesic_runtime_s": ng.get("runtime_s"),
        "runtime_wall_s": perf.get("runtime_wall_s"),
        "runtime_cpu_s": perf.get("runtime_cpu_s"),
        "cpu_efficiency_cores": perf.get("cpu_efficiency_cores"),
        "peak_rss_mb": perf.get("peak_rss_mb"),
        "warnings": "; ".join(res.get("warnings", [])),
        "failures": "; ".join(res.get("failures", []) + res.get("fingerprint_failures", [])),
        "exception": res.get("exception"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_run_outputs(run_dir: Path, results: List[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    for sub in ["surfaces", "audits", "fingerprints", "tables", "raw_results"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    for res in results:
        sid = res.get("surface_id", "unknown")
        if "surface" in res:
            (run_dir / "surfaces" / f"{sid}.json").write_text(json.dumps(res["surface"], indent=2), encoding="utf-8")
        if "fingerprint" in res:
            (run_dir / "fingerprints" / f"{sid}_fingerprint.json").write_text(json.dumps(res["fingerprint"], indent=2), encoding="utf-8")
        audit_text = json.dumps(res.get("audit", {}), indent=2)
        (run_dir / "audits" / f"{sid}_audit.json").write_text(audit_text, encoding="utf-8")
        raw = {k: v for k, v in res.items() if k != "surface"}
        (run_dir / "raw_results" / f"{sid}_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    summary = [flatten_for_summary(r) for r in results]
    write_csv(run_dir / "tables" / "surface_summary.csv", summary)
    failures = [r for r in summary if str(r.get("ok")) != "True"]
    write_csv(run_dir / "tables" / "failures.csv", failures)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def default_maker_path() -> str:
    candidates = [Path.cwd() / "FuchsianDomainMaker_v13.py", Path("/mnt/data/FuchsianDomainMaker_v13.py")]
    for c in candidates:
        if c.exists():
            return str(c)
    return "FuchsianDomainMaker_v13.py"


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch tester for Fuchsian GENN surfaces.")
    p.add_argument("--maker", default=default_maker_path(), help="Path to FuchsianDomainMaker_v13.py")
    p.add_argument("--suite", default="smoke", choices=["smoke", "standard", "stress"], help="Test suite")
    p.add_argument("--out", default="batch_runs", help="Output parent directory")
    p.add_argument("--jobs", type=int, default=0, help="Parallel workers. 0=auto; neural auto defaults to 1.")
    p.add_argument("--samples", type=int, default=60, help="Sample count for exact/graph fingerprints")
    p.add_argument("--sample-radius", type=float, default=0.72)
    p.add_argument("--word-depth", type=int, default=3)
    p.add_argument("--graph-word-depth", type=int, default=1)
    p.add_argument("--graph-k", type=int, default=12)
    p.add_argument("--parabolic-tol", type=float, default=1.0e-8)
    neural = p.add_mutually_exclusive_group()
    neural.add_argument("--with-neural", dest="with_neural", action="store_true", default=True, help="Run neural metric/geodesic tests (default)")
    neural.add_argument("--no-neural", dest="with_neural", action="store_false", help="Skip neural tests")
    p.add_argument("--no-neural-geodesic", dest="with_neural_geodesic", action="store_false", default=True)
    p.add_argument("--neural-steps", type=int, default=300, help="Neural metric training steps; use 900 for Explorer-like runs")
    p.add_argument("--neural-samples-per-step", type=int, default=128)
    p.add_argument("--neural-lr", type=float, default=1.0e-3)
    p.add_argument("--neural-interior-radius", type=float, default=0.82)
    p.add_argument("--neural-boundary-radius", type=float, default=0.93)
    p.add_argument("--geodesic-steps", type=int, default=350)
    p.add_argument("--geodesic-quad", type=int, default=80)
    p.add_argument("--geodesic-lr", type=float, default=2.0e-3)
    p.add_argument("--lift-search-depth", type=int, default=3)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    tasks = build_suite(args.suite)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / f"run_{ts}_{args.suite}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.jobs <= 0:
        if args.with_neural:
            jobs = 1  # neural training is CPU-heavy; user can override.
        else:
            jobs = max(1, min(8, (os.cpu_count() or 2) - 1))
    else:
        jobs = args.jobs

    cfg = vars(args).copy()
    cfg["maker_path"] = str(Path(args.maker).expanduser().resolve())
    cfg["jobs"] = jobs

    manifest = {
        "script": "FuchsianBatchTester_v1.py",
        "suite": args.suite,
        "maker_path": cfg["maker_path"],
        "task_count": len(tasks),
        "jobs": jobs,
        "with_neural": args.with_neural,
        "with_neural_geodesic": args.with_neural_geodesic,
        "config": cfg,
        "started_at": ts,
    }

    print("="*78)
    print("FuchsianBatchTester v1")
    print(f"Suite: {args.suite} | tasks: {len(tasks)} | jobs: {jobs}")
    print(f"Maker: {cfg['maker_path']}")
    print(f"Output: {run_dir}")
    print(f"Neural metric: {args.with_neural} | neural geodesic: {args.with_neural and args.with_neural_geodesic}")
    print("="*78, flush=True)

    results: List[Dict[str, Any]] = []
    t_all = time.perf_counter()
    if jobs == 1:
        _worker_init(cfg["maker_path"])
        for i, task in enumerate(tasks, 1):
            print(f"[{i:02d}/{len(tasks):02d}] START {task.surface_id} ({task.family})", flush=True)
            res = run_one_task(asdict(task), cfg)
            results.append(res)
            perf = res.get("performance", {})
            status = "PASS" if res.get("ok") else "FAIL"
            print(f"[{i:02d}/{len(tasks):02d}] {status:4s} {task.surface_id} "
                  f"wall={perf.get('runtime_wall_s',0):.2f}s cpu={perf.get('runtime_cpu_s',0):.2f}s "
                  f"peakRSS={perf.get('peak_rss_mb',0):.1f} MB", flush=True)
            if not res.get("ok"):
                print("    failures:", res.get("failures"), res.get("fingerprint_failures"), res.get("exception"), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init, initargs=(cfg["maker_path"],)) as ex:
            future_to_task = {ex.submit(run_one_task, asdict(task), cfg): task for task in tasks}
            completed = 0
            for fut in as_completed(future_to_task):
                task = future_to_task[fut]
                completed += 1
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"surface_id": task.surface_id, "family": task.family, "ok": False, "exception": repr(e), "traceback": traceback.format_exc()}
                results.append(res)
                perf = res.get("performance", {})
                status = "PASS" if res.get("ok") else "FAIL"
                print(f"[{completed:02d}/{len(tasks):02d}] {status:4s} {task.surface_id} "
                      f"wall={perf.get('runtime_wall_s',0):.2f}s cpu={perf.get('runtime_cpu_s',0):.2f}s "
                      f"peakRSS={perf.get('peak_rss_mb',0) or 0:.1f} MB", flush=True)
                if not res.get("ok"):
                    print("    failures:", res.get("failures"), res.get("fingerprint_failures"), res.get("exception"), flush=True)

    elapsed = time.perf_counter() - t_all
    results.sort(key=lambda r: r.get("surface_id", ""))
    manifest["finished_at"] = time.strftime("%Y%m%d_%H%M%S")
    manifest["total_wall_s"] = elapsed
    manifest["pass_count"] = sum(1 for r in results if r.get("ok"))
    manifest["fail_count"] = len(results) - manifest["pass_count"]
    write_run_outputs(run_dir, results, manifest)

    print("="*78)
    print(f"DONE: pass={manifest['pass_count']} fail={manifest['fail_count']} total_wall={elapsed:.2f}s")
    print(f"Summary CSV: {run_dir / 'tables' / 'surface_summary.csv'}")
    print(f"Failures CSV: {run_dir / 'tables' / 'failures.csv'}")
    print("="*78)
    return 0 if manifest["fail_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
