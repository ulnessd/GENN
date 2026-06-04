#!/usr/bin/env python3
"""
FuchsianDownstairsNet_v3.py

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
    specs += [f"hecke_ab{q}" for q in [3, 4, 5, 6, 7, 8, 9, 10, 12]]
    specs += [f"hecke_d{q}" for q in [3, 4, 5, 6, 7, 8, 9, 10, 12]]
    return specs


def smoke_surface_specs() -> List[str]:
    return ["regular_g2", "regular_g3", "hurwitz", "gamma3", "gamma1_5", "hecke_ab5", "hecke_d5"]


def standard_surface_specs() -> List[str]:
    specs: List[str] = []
    specs += [f"regular_g{g}" for g in range(2, 6)]
    specs += ["hurwitz"]
    specs += [f"gamma{N}" for N in range(3, 7)]
    specs += [f"gamma1_{N}" for N in range(4, 8)]
    specs += [f"hecke_ab{q}" for q in range(3, 8)]
    specs += [f"hecke_d{q}" for q in range(3, 8)]
    return specs


def make_surface(surface: str, maker_path: str) -> Dict[str, Any]:
    maker = load_maker(maker_path)
    surface = surface.lower().strip()
    if surface in {"hurwitz", "klein", "klein_quartic", "hurwitz_klein"}:
        return maker.make_hurwitz_klein_quartic_surface()
    if surface.startswith("regular_g"):
        return maker.make_regular_genus_surface(int(surface.replace("regular_g", "")))
    if surface.startswith("gamma1_"):
        return maker.make_modular_gamma1N_torsion_free_domain(int(surface.split("_", 1)[1]))
    if surface.startswith("gamma"):
        return maker.make_modular_gammaN_principal_domain(int(surface.replace("gamma", "")))
    if surface.startswith("hecke_ab"):
        return maker.make_hecke_torsion_free_abelian_cover(int(surface.replace("hecke_ab", "")))
    if surface.startswith("hecke_d") or surface.startswith("hecke_dihedral"):
        q = int(surface.replace("hecke_dihedral", "").replace("hecke_d", ""))
        return maker.make_hecke_torsion_free_dihedral_cover(q)
    if surface in {"g2", "genus2"}:
        return maker.make_regular_genus_surface(2)
    if surface in {"g3", "genus3"}:
        return maker.make_regular_genus_surface(3)
    raise ValueError(f"Unknown v2 surface {surface!r}. Use --list-surfaces to see supported specs.")


def surface_family(surface: str) -> str:
    s = surface.lower()
    if s.startswith("regular_g"):
        return "compact_regular_polygon"
    if s == "hurwitz" or "klein" in s:
        return "hurwitz_klein_psl27"
    if s.startswith("gamma1_"):
        return "modular_gamma1"
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
    return ch.lower() if ch.isupper() else ch.upper()


def invert_word(w: str) -> str:
    return "".join(invert_letter(c) for c in reversed(w))


def reduce_word(w: str) -> str:
    st: List[str] = []
    for c in w:
        if st and invert_letter(c) == st[-1]:
            st.pop()
        else:
            st.append(c)
    return "".join(st)


def parse_generators(surface_json: Dict[str, Any]) -> Dict[str, Mobius]:
    gens: Dict[str, Mobius] = {}
    raw = surface_json.get("generators", {})
    for label, g in raw.items():
        if not isinstance(label, str) or len(label) != 1:
            continue
        if g.get("type") != "su11":
            continue
        ar, ai = g["alpha"]
        br, bi = g["beta"]
        M = Mobius(complex(ar, ai), complex(br, bi), label).normalized()
        gens[label] = M
        gens[label.lower()] = M.inverse(label.lower())
    if not gens:
        raise ValueError("No SU(1,1) generators found in surface JSON.")
    return gens


def compose_word(word: str, gens: Dict[str, Mobius]) -> Mobius:
    M = Mobius(1.0 + 0j, 0.0 + 0j, "")
    # Convention: word abc means apply a then b then c? For label search this only
    # needs to be consistent. We use left-to-right composition on z.
    current = Mobius(1.0 + 0j, 0.0 + 0j, "")
    for ch in word:
        current = gens[ch].compose(current, word=word[: len(current.word) + 1])
    return Mobius(current.alpha, current.beta, word).normalized()


def build_word_ball(gens: Dict[str, Mobius], depth: int) -> List[Mobius]:
    letters = sorted(gens.keys(), key=lambda c: (c.lower(), c.islower()))
    words = [""]
    frontier = [""]
    for _ in range(depth):
        new_frontier: List[str] = []
        for w in frontier:
            last = w[-1] if w else ""
            for ch in letters:
                if last and invert_letter(ch) == last:
                    continue
                nw = reduce_word(w + ch)
                if len(nw) == len(w) + 1:
                    new_frontier.append(nw)
                    words.append(nw)
        frontier = new_frontier
    # Deduplicate numerically by word string. Keep identity first.
    seen = set()
    out: List[Mobius] = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        if w == "":
            out.append(Mobius(1 + 0j, 0 + 0j, ""))
        else:
            out.append(compose_word(w, gens))
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
# DownstairsNet model and training
# -----------------------------------------------------------------------------

class DownstairsMLP(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int, hidden: int, max_depth: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
        )
        self.distance_head = nn.Linear(hidden // 2, 1)
        self.depth_head = nn.Linear(hidden // 2, max_depth + 1)

    def forward(self, x):
        h = self.shared(x)
        dist = self.distance_head(h).squeeze(-1)
        depth_logits = self.depth_head(h)
        return dist, depth_logits


@dataclass(frozen=True)
class ExpertConfig:
    name: str
    hidden: int
    lr: float
    batch_size: int
    epochs: int
    patience: int
    pairs: int = 9000
    depth: int = 2


def production_committee_configs() -> List[ExpertConfig]:
    """Two-expert committee from the overnight optimizer.

    The optimizer suggested that the main signal is 9000 pairs at depth 2.
    These two protocols were the best complementary settings across the difficult
    diagnostic surfaces: a medium 256-wide model with lr=1e-3, and a heavier
    384-wide model with lr=7e-4.  Selection is by validation RMSE, not test RMSE.
    """
    return [
        ExpertConfig(
            name="expert_A_d2_morepairs_h256_lr1e-3",
            pairs=9000, depth=2, hidden=256, batch_size=256,
            lr=1.0e-3, epochs=260, patience=45,
        ),
        ExpertConfig(
            name="expert_B_d2_morepairs_h384_lr7e-4",
            pairs=9000, depth=2, hidden=384, batch_size=256,
            lr=7.0e-4, epochs=280, patience=50,
        ),
    ]


def features_from_rows(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X = []
    y_dist = []
    y_depth = []
    feature_names = [
        "p_x", "p_y", "q_x", "q_y",
        "identity_distance",
        "euclidean_pair_distance",
        "p_radius", "q_radius",
        "p_angle_cos", "p_angle_sin", "q_angle_cos", "q_angle_sin",
    ]
    for r in rows:
        px, py, qx, qy = float(r["p_x"]), float(r["p_y"]), float(r["q_x"]), float(r["q_y"])
        pr = math.hypot(px, py)
        qr = math.hypot(qx, qy)
        pa = math.atan2(py, px)
        qa = math.atan2(qy, qx)
        X.append([
            px, py, qx, qy,
            float(r["identity_distance"]),
            math.hypot(px - qx, py - qy),
            pr, qr,
            math.cos(pa), math.sin(pa), math.cos(qa), math.sin(qa),
        ])
        y_dist.append(float(r["quotient_distance"]))
        y_depth.append(int(r["shortest_lift_depth"]))
    return np.asarray(X, dtype=np.float32), np.asarray(y_dist, dtype=np.float32), np.asarray(y_depth, dtype=np.int64), feature_names


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - y
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / denom if denom > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_downstairs_net(
    rows: List[Dict[str, Any]],
    outdir: Path,
    max_depth: int,
    epochs: int,
    hidden: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str,
    patience: int = 30,
) -> Dict[str, Any]:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required for DownstairsNet training but is not available.")

    rng = np.random.default_rng(seed)
    X, y_dist, y_depth, feature_names = features_from_rows(rows)
    n = len(rows)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # Feature normalization from training split only.
    x_mean = X[train_idx].mean(axis=0)
    x_std = X[train_idx].std(axis=0)
    x_std[x_std < 1.0e-6] = 1.0
    y_mean = float(y_dist[train_idx].mean())
    y_std = float(y_dist[train_idx].std())
    if y_std < 1.0e-6:
        y_std = 1.0

    Xn = (X - x_mean) / x_std
    yn = (y_dist - y_mean) / y_std

    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] device={dev}  n={n} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
    model = DownstairsMLP(X.shape[1], hidden, max_depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)

    X_t = torch.tensor(Xn, dtype=torch.float32, device=dev)
    yd_t = torch.tensor(yn, dtype=torch.float32, device=dev)
    ydepth_t = torch.tensor(np.clip(y_depth, 0, max_depth), dtype=torch.long, device=dev)

    train_log: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_state = None
    t0, c0 = time.time(), cpu_seconds()
    best_epoch = 0
    epochs_without_improve = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = train_idx.copy()
        rng.shuffle(perm)
        total = 0.0
        nb = 0
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            xb = X_t[b]
            yb = yd_t[b]
            db = ydepth_t[b]
            pred_d, logits = model(xb)
            loss_dist = F.mse_loss(pred_d, yb)
            loss_depth = F.cross_entropy(logits, db)
            loss = loss_dist + 0.15 * loss_depth
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item())
            nb += 1
        model.eval()
        with torch.no_grad():
            pv, lv = model(X_t[val_idx])
            val_dist = F.mse_loss(pv, yd_t[val_idx]).item()
            val_depth = F.cross_entropy(lv, ydepth_t[val_idx]).item()
            val_loss = val_dist + 0.15 * val_depth
        if val_loss < best_val - 1.0e-7:
            best_val = val_loss
            best_epoch = ep
            epochs_without_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improve += 1
        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"[train] epoch {ep:4d}/{epochs}  train_loss={total/max(nb,1):.6f}  val_loss={val_loss:.6f}  val_dist_mse={val_dist:.6f}", flush=True)
            print_perf("train", t0, c0)
        train_log.append({
            "epoch": ep,
            "train_loss": total / max(nb, 1),
            "val_loss": val_loss,
            "val_dist_mse_normalized": val_dist,
            "val_depth_ce": val_depth,
            "best_epoch_so_far": best_epoch,
            "epochs_without_improve": epochs_without_improve,
        })
        if patience > 0 and epochs_without_improve >= patience:
            print(f"[train] early stopping at epoch {ep}; best_epoch={best_epoch} best_val={best_val:.6f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_all_n, depth_logits = model(X_t)
        pred_all = pred_all_n.detach().cpu().numpy() * y_std + y_mean
        depth_pred = torch.argmax(depth_logits, dim=1).detach().cpu().numpy()

    test_y = y_dist[test_idx]
    test_pred = pred_all[test_idx]
    val_y = y_dist[val_idx]
    val_pred = pred_all[val_idx]
    train_y = y_dist[train_idx]
    train_pred = pred_all[train_idx]

    # Baselines.
    id_baseline = X[:, feature_names.index("identity_distance")]
    mean_baseline = np.full_like(y_dist, y_mean)
    metrics = {
        "train": regression_metrics(train_y, train_pred),
        "val": regression_metrics(val_y, val_pred),
        "test": regression_metrics(test_y, test_pred),
        "baseline_identity_test": regression_metrics(test_y, id_baseline[test_idx]),
        "baseline_mean_test": regression_metrics(test_y, mean_baseline[test_idx]),
        "depth_accuracy_test": float(np.mean(depth_pred[test_idx] == y_depth[test_idx])),
        "shortcut_fraction_test": float(np.mean([rows[int(i)]["nontrivial_shortcut"] for i in test_idx])),
        "feature_names": feature_names,
        "target_distance_mean_train": y_mean,
        "target_distance_std_train": y_std,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(train_log[-1]["epoch"] if train_log else 0),
    }
    nn_rmse = metrics["test"]["rmse"]
    base_rmse = metrics["baseline_identity_test"]["rmse"]
    metrics["identity_baseline_rmse_improvement_fraction"] = float((base_rmse - nn_rmse) / base_rmse) if base_rmse > 0 else None

    # Save predictions on test split.
    pred_path = outdir / "predictions_test.csv"
    with pred_path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) + ["pred_quotient_distance", "pred_shortest_lift_depth", "split"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in test_idx:
            rr = dict(rows[int(i)])
            rr["pred_quotient_distance"] = float(pred_all[int(i)])
            rr["pred_shortest_lift_depth"] = int(depth_pred[int(i)])
            rr["split"] = "test"
            writer.writerow(rr)

    save_geodesic_witnesses(outdir, pred_path, max_witnesses=24)

    # Save training log.
    log_path = outdir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
        writer.writeheader()
        writer.writerows(train_log)

    # Save model checkpoint.
    ckpt_path = outdir / "downstairs_net.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "feature_names": feature_names,
        "max_depth": max_depth,
        "hidden": hidden,
    }, ckpt_path)

    print("[result] TEST quotient-distance prediction", flush=True)
    print(f"         neural RMSE        = {metrics['test']['rmse']:.6f}", flush=True)
    print(f"         identity RMSE      = {metrics['baseline_identity_test']['rmse']:.6f}", flush=True)
    print(f"         improvement        = {100.0 * metrics['identity_baseline_rmse_improvement_fraction']:.1f}%", flush=True)
    print(f"         neural R^2         = {metrics['test']['r2']:.4f}", flush=True)
    print(f"         depth accuracy     = {metrics['depth_accuracy_test']:.3f}", flush=True)
    print_perf("train-final", t0, c0)
    return metrics


# -----------------------------------------------------------------------------
# Geodesic witness helpers
# -----------------------------------------------------------------------------

def save_geodesic_witnesses(outdir: Path, prediction_csv: Path, max_witnesses: int = 24) -> None:
    """Save JSON witnesses showing the exact lifted Poincare geodesic endpoints.

    Each witness verifies the exact label geometrically: the geodesic upstairs is
    the ordinary Poincare geodesic from p to gamma*(q), with length equal to the
    quotient-distance label.  These are not plots yet; they are audit records for
    v2 and easy to plot later.
    """
    if not prediction_csv.exists():
        return
    rows: List[Dict[str, Any]] = []
    with prediction_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                true_d = float(r["quotient_distance"])
                pred_d = float(r["pred_quotient_distance"])
                r["abs_distance_error"] = abs(pred_d - true_d)
                rows.append(r)
            except Exception:
                pass
    rows.sort(key=lambda r: float(r.get("abs_distance_error", 0.0)), reverse=True)
    chosen = rows[: max_witnesses // 2]
    if len(rows) > len(chosen):
        step = max(1, len(rows) // max(1, max_witnesses - len(chosen)))
        chosen += rows[::step][: max_witnesses - len(chosen)]
    wdir = outdir / "geodesic_witnesses"
    wdir.mkdir(exist_ok=True)
    for k, r in enumerate(chosen):
        px, py = float(r["p_x"]), float(r["p_y"])
        qx, qy = float(r["q_x"]), float(r["q_y"])
        lx, ly = float(r.get("lifted_q_x", qx)), float(r.get("lifted_q_y", qy))
        witness = {
            "pair_id": int(float(r["pair_id"])),
            "p": [px, py],
            "q_base": [qx, qy],
            "q_lifted_by_exact_winning_word": [lx, ly],
            "exact_winning_word": r.get("shortest_lift_word", ""),
            "identity_distance": float(r["identity_distance"]),
            "quotient_distance_exact": float(r["quotient_distance"]),
            "neural_predicted_quotient_distance": float(r["pred_quotient_distance"]),
            "absolute_distance_error": float(r.get("abs_distance_error", 0.0)),
            "geodesic_length_check": float(r.get("geodesic_length_check", r["quotient_distance"])),
            "verification_statement": "The exact downstairs label is witnessed by the Poincare geodesic from p to q_lifted_by_exact_winning_word upstairs.",
        }
        with (wdir / f"witness_{k:04d}_pair_{witness['pair_id']:06d}.json").open("w") as f:
            json.dump(witness, f, indent=2)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact_word_ball_summary(word_ball: List[Mobius]) -> List[Dict[str, Any]]:
    out = []
    for m in word_ball:
        tr = m.trace_real()
        typ = "identity" if m.word == "" else ("hyperbolic" if abs(tr) > 2.0 + 1.0e-10 else "elliptic_or_relation")
        out.append({"word": m.word if m.word else "identity", "depth": len(m.word), "trace": tr, "type_proxy": typ})
    return out


def train_committee_on_rows(
    rows: List[Dict[str, Any]],
    outdir: Path,
    args: argparse.Namespace,
    configs: List[ExpertConfig],
) -> Dict[str, Any]:
    """Train committee experts on one labeled dataset and select by validation RMSE.

    The final test metrics are still written for every expert, but the selected
    expert is chosen using validation RMSE to avoid choosing on the test split.
    """
    committee_dir = outdir / "committee"
    committee_dir.mkdir(exist_ok=True)
    rows_summary: List[Dict[str, Any]] = []
    expert_metrics: Dict[str, Any] = {}

    for k, cfg in enumerate(configs, 1):
        print("-" * 78, flush=True)
        print(f"[committee] expert {k}/{len(configs)}: {cfg.name}", flush=True)
        print(f"[committee] hidden={cfg.hidden} lr={cfg.lr} batch={cfg.batch_size} epochs={cfg.epochs} patience={cfg.patience}", flush=True)
        ed = committee_dir / cfg.name
        ed.mkdir(parents=True, exist_ok=True)
        metrics = train_downstairs_net(
            rows=rows,
            outdir=ed,
            max_depth=cfg.depth,
            epochs=cfg.epochs,
            hidden=cfg.hidden,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            seed=args.seed + 1000 * k,
            device=args.device,
            patience=cfg.patience,
        )
        val_rmse = float(metrics.get("val", {}).get("rmse", float("inf")))
        test_rmse = float(metrics.get("test", {}).get("rmse", float("nan")))
        test_r2 = float(metrics.get("test", {}).get("r2", float("nan")))
        depth_acc = float(metrics.get("depth_accuracy_test", float("nan")))
        row = {
            "expert_name": cfg.name,
            "pairs": cfg.pairs,
            "depth": cfg.depth,
            "hidden": cfg.hidden,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "best_epoch": metrics.get("best_epoch"),
            "epochs_ran": metrics.get("epochs_ran"),
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "depth_accuracy_test": depth_acc,
            "identity_baseline_rmse": metrics.get("baseline_identity_test", {}).get("rmse"),
            "identity_baseline_improvement_fraction": metrics.get("identity_baseline_rmse_improvement_fraction"),
        }
        rows_summary.append(row)
        expert_metrics[cfg.name] = metrics

    rows_summary.sort(key=lambda r: float(r["val_rmse"]))
    selected = rows_summary[0] if rows_summary else {}
    selected_name = str(selected.get("expert_name", ""))
    selected_metrics = expert_metrics.get(selected_name, {})

    write_csv_rows(outdir / "committee_summary.csv", rows_summary)
    with (outdir / "selected_expert.json").open("w") as f:
        json.dump({
            "selection_rule": "lowest validation RMSE",
            "selected_expert": selected,
            "all_experts_ranked": rows_summary,
        }, f, indent=2)

    print("[committee] selected expert by validation RMSE:", selected_name, flush=True)
    if selected_metrics:
        print(f"[committee] selected test RMSE={selected_metrics['test']['rmse']:.6f} R2={selected_metrics['test']['r2']:.4f}", flush=True)
    return {
        "committee_enabled": True,
        "selection_rule": "lowest validation RMSE",
        "selected_expert_name": selected_name,
        "selected_expert_row": selected,
        "selected_metrics": selected_metrics,
        "all_experts_ranked": rows_summary,
    }


def run_one_surface(args: argparse.Namespace, surface_name: str) -> Path:
    run_id = f"run_{now_stamp()}_{surface_name}"
    outdir = Path(args.outdir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print(f"[start] DownstairsNet v3 surface={surface_name} outdir={outdir}", flush=True)
    print("=" * 78, flush=True)
    overall_t0, overall_c0 = time.time(), cpu_seconds()

    surface_json = make_surface(surface_name, args.maker)
    with (outdir / "surface.json").open("w") as f:
        json.dump(surface_json, f, indent=2)
    print(f"[surface] {surface_json.get('name', surface_name)}", flush=True)
    print(f"[surface] domain_type={surface_json.get('domain_type')} genus={surface_json.get('genus')} area={surface_json.get('area')}", flush=True)

    # Committee mode uses the maximum data requirement across the selected experts.
    # The production two-expert committee currently shares pairs=9000 and depth=2.
    # When --committee is off, the CLI --pairs/--depth values are used normally.
    committee_configs: List[ExpertConfig] = []
    if getattr(args, "committee", "none") in {"production", "prod", "two", "2"}:
        committee_configs = production_committee_configs()
        label_pairs = max(cfg.pairs for cfg in committee_configs)
        label_depth = max(cfg.depth for cfg in committee_configs)
        print(f"[committee] production two-expert mode: label_pairs={label_pairs} label_depth={label_depth}", flush=True)
    else:
        label_pairs = args.pairs
        label_depth = args.depth

    rows, word_ball, label_meta = generate_pair_dataset(
        surface_json=surface_json,
        n_pairs=label_pairs,
        depth=label_depth,
        seed=args.seed,
        jobs=args.jobs,
        progress_every=max(250, label_pairs // 10),
    )
    write_csv_rows(outdir / "pair_dataset.csv", rows)
    with (outdir / "word_ball.json").open("w") as f:
        json.dump(compact_word_ball_summary(word_ball), f, indent=2)

    metrics: Dict[str, Any] = {}
    if args.no_train:
        print("[train] skipped because --no-train was supplied", flush=True)
    elif committee_configs:
        metrics = train_committee_on_rows(
            rows=rows,
            outdir=outdir,
            args=args,
            configs=committee_configs,
        )
    else:
        metrics = train_downstairs_net(
            rows=rows,
            outdir=outdir,
            max_depth=args.depth,
            epochs=args.epochs,
            hidden=args.hidden,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            patience=args.patience,
        )

    manifest = {
        "program": "FuchsianDownstairsNet_v3.py",
        "surface_requested": surface_name,
        "surface_name": surface_json.get("name"),
        "domain_type": surface_json.get("domain_type"),
        "genus": surface_json.get("genus"),
        "area": surface_json.get("area"),
        "maker": args.maker,
        "pairs": label_pairs,
        "word_depth": label_depth,
        "cli_pairs": args.pairs,
        "cli_word_depth": args.depth,
        "committee_mode": getattr(args, "committee", "none"),
        "jobs": args.jobs,
        "seed": args.seed,
        "label_meta": label_meta,
        "metrics": metrics,
        "wall_seconds_total": time.time() - overall_t0,
        "cpu_seconds_total": cpu_seconds() - overall_c0,
        "rss_mb_final": rss_mb(),
        "important_interpretation": (
            "The neural network is trained on downstairs quotient-distance labels produced by a finite deck-group word search. "
            "This is a first quotient/gluing learner, not merely an upstairs Poincare metric learner."
        ),
        "v3_scope": (
            "All main Poincare/Fuchsian families are supported. Exact winning-lift words and geodesic witness JSON files are stored. Optional committee mode trains two diverse hyperparameter experts and selects the best by validation RMSE. Full explicit side-crossing itinerary through adjacent tiles is still future work."
        ),
    }
    with (outdir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (outdir / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print_perf("overall", overall_t0, overall_c0)
    print(f"[done] outputs written to {outdir}", flush=True)
    return outdir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train/test downstairs quotient-geometry neural learners across main Fuchsian surface families.")
    parser.add_argument("--surface", default="regular_g2", help="Surface spec, comma-list, all, smoke, standard, regular_g2, hurwitz, gamma3, gamma1_5, hecke_ab5, hecke_d5, etc.")
    parser.add_argument("--list-surfaces", action="store_true", help="Print supported dataset surface specs and exit")
    parser.add_argument("--maker", default=default_maker_path(), help="Path to FuchsianDomainMaker_v13.py")
    parser.add_argument("--outdir", default="downstairs_runs", help="Output directory root")
    parser.add_argument("--pairs", type=int, default=6000, help="Number of random point pairs per surface")
    parser.add_argument("--depth", type=int, default=3, help="Reduced-word search depth for quotient labels")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="Parallel jobs for label generation")
    parser.add_argument("--epochs", type=int, default=180, help="Maximum neural training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience in epochs; 0 disables")
    parser.add_argument("--hidden", type=int, default=128, help="MLP hidden width")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2.0e-3, help="Learning rate")
    parser.add_argument("--committee", default="none", choices=["none", "production", "prod", "two", "2"],
                        help="Optional two-expert committee mode. 'production' runs the two best optimizer protocols and selects by validation RMSE.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--no-train", action="store_true", help="Only generate labels/dataset; do not train")
    args = parser.parse_args(argv)

    if args.depth < 0 or args.depth > 5:
        print("[error] v3 depth should be between 0 and 5. Depth 2-3 is recommended for large-generator families.", file=sys.stderr)
        return 2
    if args.pairs < 100:
        print("[error] Use at least 100 pairs for a meaningful test.", file=sys.stderr)
        return 2
    if not args.no_train and torch is None:
        print("[error] PyTorch is not available. Re-run with --no-train or install torch.", file=sys.stderr)
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)

    if args.list_surfaces:
        print("Smoke suite:", ", ".join(smoke_surface_specs()))
        print("Standard suite:", ", ".join(standard_surface_specs()))
        print("All mainline suite:", ", ".join(main_surface_specs()))
        return 0

    surf_arg = args.surface.lower().strip()
    if surf_arg in {"both", "smoke"}:
        surfaces = smoke_surface_specs()
    elif surf_arg == "standard":
        surfaces = standard_surface_specs()
    elif surf_arg in {"all", "main", "mainline"}:
        surfaces = main_surface_specs()
    elif "," in args.surface:
        surfaces = [x.strip() for x in args.surface.split(",") if x.strip()]
    else:
        surfaces = [args.surface]

    print("FuchsianDownstairsNet v3", flush=True)
    print(f"maker={args.maker}", flush=True)
    print(f"surfaces={surfaces}", flush=True)
    print(f"pairs={args.pairs} depth={args.depth} jobs={args.jobs} epochs={args.epochs} committee={args.committee}", flush=True)
    if args.committee in {"production", "prod", "two", "2"}:
        print("committee experts:", flush=True)
        for cfg in production_committee_configs():
            print(f"  {cfg.name}: pairs={cfg.pairs} depth={cfg.depth} hidden={cfg.hidden} lr={cfg.lr} batch={cfg.batch_size} epochs={cfg.epochs} patience={cfg.patience}", flush=True)
    print(f"initial rss={rss_mb():.1f} MB", flush=True)

    try:
        completed: List[str] = []
        for s in surfaces:
            out = run_one_surface(args, s)
            completed.append(str(out))
        summary_root = Path(args.outdir) / f"summary_{now_stamp()}"
        summary_root.mkdir(parents=True, exist_ok=True)
        with (summary_root / "completed_runs.json").open("w") as f:
            json.dump({"completed_runs": completed, "surfaces": surfaces}, f, indent=2)
        print(f"[summary] completed {len(completed)} surface runs; summary={summary_root}", flush=True)
    except Exception as exc:
        print("[fatal] DownstairsNet run failed:", exc, file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
