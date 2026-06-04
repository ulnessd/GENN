#!/usr/bin/env python3
r"""FuchsianSidePairingPolygonBatchTrainer_v1_1.py

Batch generator/trainer for smooth compact Riemann surfaces built directly from
explicit side-pairings of regular hyperbolic polygons in the Poincare disk.

Scope
-----
* Smooth compact Riemann surfaces only.
* Explicit Poincare-disk fundamental polygons only.
* No orbifold records.
* No Teichmuller/Fenchel-Nielsen coordinate-only records.

Construction
------------
For genus g >= 2, start with a regular hyperbolic {4g,4g} polygon.  Its
interior angle is pi/(2g), so the full polygon has hyperbolic area

    (4g-2) pi - 4g*pi/(2g) = 4 pi (g-1).

The program chooses orientation-reversing side-pairing matchings of the 4g
sides.  A matching is accepted only when the induced vertex identifications
form one vertex cycle.  In that case the cycle contains all 4g vertices, so the
cycle angle is 4g*pi/(2g)=2pi.  The quotient has

    V=1, E=2g, F=1, chi=2-2g,

and is a smooth compact genus-g hyperbolic surface.  Each side-pairing map is
computed as the unique orientation-preserving disk isometry sending one side to
its paired side with reversed endpoint order.

This is the conservative final member of the six-program non-triangle expansion
series.  v1.1 fixes the hidden-width argument type passed to the shared GINN trainer and adds a vertex-cycle transformation consistency audit.  It deliberately restricts to regular polygons with one-vertex
orientation-reversing side-pairings, where the Poincare polygon theorem audit is
especially transparent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROGRAM = "FuchsianSidePairingPolygonBatchTrainer_v1_1.py"
VERSION = "1.1"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_label(s: str) -> str:
    out: List[str] = []
    for ch in str(s):
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(out).strip("_") or "run"


def parse_int_list(s: str, default: Sequence[int]) -> List[int]:
    if not str(s or "").strip():
        return list(default)
    vals: List[int] = []
    for tok in str(s).replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_module(path: str, name: str) -> Any:
    p = Path(path)
    if not p.exists():
        p = Path.cwd() / path
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def complex_to_xy(z: complex) -> List[float]:
    return [float(z.real), float(z.imag)]


def fingerprint_pairs(pairs: Sequence[Tuple[int, int]]) -> str:
    canon = sorted((min(a, b), max(a, b)) for a, b in pairs)
    return hashlib.sha1(json.dumps(canon, separators=(",", ":")).encode("utf-8")).hexdigest()[:10]


def parse_hidden_width(x: Any, default: int) -> int:
    """Accept either an int-like value or a legacy comma-list and return one width."""
    if x is None:
        return int(default)
    if isinstance(x, int):
        return int(x)
    txt = str(x).strip()
    if not txt:
        return int(default)
    # Earlier wrappers used strings such as "128,128".  The shared v2.4 GINN
    # trainer expects a single integer width, so take the first positive token.
    for tok in txt.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = int(float(tok))
        if val > 0:
            return val
    return int(default)


def su11_equivalence_error(a: "SU11", b: "SU11") -> float:
    """Coefficient error between two SU(1,1) maps, modulo the global sign."""
    aa = a.normalized()
    bb = b.normalized()
    err_same = max(abs(aa.alpha - bb.alpha), abs(aa.beta - bb.beta))
    err_neg = max(abs(aa.alpha + bb.alpha), abs(aa.beta + bb.beta))
    return float(min(err_same, err_neg))


# -----------------------------------------------------------------------------
# SU(1,1) disk isometries and regular polygon utilities
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SU11:
    alpha: complex
    beta: complex
    word: str = ""

    def __call__(self, z: complex) -> complex:
        den = self.beta.conjugate() * z + self.alpha.conjugate()
        if abs(den) < 1.0e-14:
            den = 1.0e-14 + 0j
        return (self.alpha * z + self.beta) / den

    def det(self) -> float:
        return float(abs(self.alpha) ** 2 - abs(self.beta) ** 2)

    def normalized(self, word: Optional[str] = None) -> "SU11":
        d = self.det()
        w = self.word if word is None else word
        if d <= 0.0 or not math.isfinite(d):
            return SU11(self.alpha, self.beta, w)
        s = math.sqrt(d)
        a = self.alpha / s
        b = self.beta / s
        # Fix harmless global sign ambiguity for stable JSON.
        if a.real < 0:
            a, b = -a, -b
        return SU11(a, b, w)

    def compose(self, other: "SU11", word: Optional[str] = None) -> "SU11":
        # self after other: self(other(z)).
        a1, b1 = self.alpha, self.beta
        a2, b2 = other.alpha, other.beta
        a = a1 * a2 + b1 * b2.conjugate()
        b = a1 * b2 + b1 * a2.conjugate()
        return SU11(a, b, self.word if word is None else word).normalized()

    def inverse(self, word: Optional[str] = None) -> "SU11":
        return SU11(self.alpha.conjugate(), -self.beta, self.word if word is None else word).normalized()

    def as_json(self) -> Dict[str, Any]:
        g = self.normalized(self.word)
        return {
            "type": "su11",
            "alpha": [float(g.alpha.real), float(g.alpha.imag)],
            "beta": [float(g.beta.real), float(g.beta.imag)],
            "name": self.word,
        }


def disk_move_to_zero(z0: complex) -> SU11:
    r2 = abs(z0) ** 2
    if r2 >= 1.0:
        raise ValueError("Point must lie inside the disk")
    scale = 1.0 / math.sqrt(1.0 - r2)
    return SU11(scale + 0j, -z0 * scale).normalized()


def disk_move_from_zero(w0: complex) -> SU11:
    r2 = abs(w0) ** 2
    if r2 >= 1.0:
        raise ValueError("Point must lie inside the disk")
    scale = 1.0 / math.sqrt(1.0 - r2)
    return SU11(scale + 0j, w0 * scale).normalized()


def disk_rotation(theta: float) -> SU11:
    return SU11(complex(math.cos(theta / 2.0), math.sin(theta / 2.0)), 0j).normalized()


def disk_isometry_from_two_point_pairs(z1: complex, z2: complex, w1: complex, w2: complex, name: str) -> SU11:
    """Unique orientation-preserving disk isometry sending z1->w1 and z2->w2."""
    Mz = disk_move_to_zero(z1)
    Mw_inv = disk_move_from_zero(w1)
    u = Mz(z2)
    Mw = disk_move_to_zero(w1)
    v = Mw(w2)
    if abs(u) < 1.0e-14 or abs(v) < 1.0e-14:
        raise ValueError("Degenerate side endpoint data")
    lam = v / u
    lam = lam / abs(lam)
    R = disk_rotation(math.atan2(lam.imag, lam.real))
    return Mw_inv.compose(R.compose(Mz), word=name).normalized(name)


def regular_hyperbolic_polygon_radius(p: int, q: int) -> float:
    cosh_R = math.cos(math.pi / q) / math.sin(math.pi / p)
    if cosh_R <= 1.0:
        raise ValueError("The requested {p,q} is not hyperbolic")
    return math.tanh(0.5 * math.acosh(cosh_R))


def regular_polygon_vertices(genus: int, rotation_deg: float) -> List[complex]:
    n = 4 * genus
    rho = regular_hyperbolic_polygon_radius(n, n)
    rot = math.radians(rotation_deg)
    return [rho * complex(math.cos(rot + 2.0 * math.pi * k / n), math.sin(rot + 2.0 * math.pi * k / n)) for k in range(n)]


def polygon_area(n: int, interior_angle: float) -> float:
    return (n - 2) * math.pi - n * interior_angle


# -----------------------------------------------------------------------------
# Side-pairing matching and audits
# -----------------------------------------------------------------------------


def vertex_cycles_for_pairings(n: int, pairs: Sequence[Tuple[int, int]]) -> List[List[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        x %= n
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Orientation-reversing side gluing: side i=(i,i+1) glued to side j=(j+1,j).
    for i, j in pairs:
        union(i, j + 1)
        union(i + 1, j)

    classes: Dict[int, List[int]] = {}
    for v in range(n):
        classes.setdefault(find(v), []).append(v)
    return [sorted(c) for c in classes.values()]


def canonical_opposite_pairs(genus: int) -> List[Tuple[int, int]]:
    n = 4 * genus
    h = 2 * genus
    return [(i, i + h) for i in range(h)]


def random_matching(n: int, rng: random.Random) -> List[Tuple[int, int]]:
    sides = list(range(n))
    rng.shuffle(sides)
    pairs = []
    for k in range(0, n, 2):
        a, b = sides[k], sides[k + 1]
        pairs.append((min(a, b), max(a, b)))
    pairs.sort()
    return pairs


def generate_pairings_for_genus(
    genus: int,
    variants: int,
    rng: random.Random,
    include_opposite: bool,
    max_attempts: int,
) -> List[Tuple[str, List[Tuple[int, int]]]]:
    n = 4 * genus
    out: List[Tuple[str, List[Tuple[int, int]]]] = []
    seen: set[str] = set()

    if include_opposite:
        p = canonical_opposite_pairs(genus)
        fp = fingerprint_pairs(p)
        out.append(("opposite_control", p))
        seen.add(fp)

    attempts = 0
    while len(out) < variants + (1 if include_opposite else 0) and attempts < max_attempts:
        attempts += 1
        p = random_matching(n, rng)
        fp = fingerprint_pairs(p)
        if fp in seen:
            continue
        cycles = vertex_cycles_for_pairings(n, p)
        if len(cycles) != 1:
            continue
        seen.add(fp)
        out.append((f"one_vertex_random{len(out):02d}", p))
    return out[: variants + (1 if include_opposite else 0)]


def build_side_pairing_surface(genus: int, pairs: List[Tuple[int, int]], profile: str, variant_index: int, rotation_deg: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if genus < 2:
        raise ValueError("Compact side-pairing polygons require genus >= 2")
    n = 4 * genus
    if len(pairs) != 2 * genus:
        raise ValueError("Need exactly 2g side pairs for a 4g-gon")
    if 2 * genus > len(LETTERS):
        raise ValueError(f"At most genus {len(LETTERS)//2} supported with one-letter generator labels")

    vertices = regular_polygon_vertices(genus, rotation_deg)
    interior_angle = math.pi / (2 * genus)
    area = polygon_area(n, interior_angle)
    gb = 4.0 * math.pi * (genus - 1)
    cycles = vertex_cycles_for_pairings(n, pairs)
    chi = len(cycles) - len(pairs) + 1
    inferred_genus = (2.0 - chi) / 2.0

    generators: Dict[str, Any] = {}
    side_pairings: List[Dict[str, Any]] = []
    endpoint_audit: List[Dict[str, Any]] = []
    # Directed vertex-identification edges carrying the actual side-pairing
    # transformation.  These are used below to audit the vertex-cycle
    # transformation, not merely the combinatorial vertex equivalence class.
    directed_vertex_edges: List[Tuple[int, int, SU11]] = []
    max_endpoint_error = 0.0
    max_det_error = 0.0
    bad_generator_count = 0

    for k, (i, j) in enumerate(pairs):
        label = LETTERS[k]
        z1 = vertices[i]
        z2 = vertices[(i + 1) % n]
        w1 = vertices[(j + 1) % n]
        w2 = vertices[j % n]
        gmap = disk_isometry_from_two_point_pairs(z1, z2, w1, w2, label)
        e1 = abs(gmap(z1) - w1)
        e2 = abs(gmap(z2) - w2)
        ep = max(e1, e2)
        det_err = abs(gmap.det() - 1.0)
        max_endpoint_error = max(max_endpoint_error, ep)
        max_det_error = max(max_det_error, det_err)
        if det_err > 1.0e-8 or not math.isfinite(det_err):
            bad_generator_count += 1
        generators[label] = gmap.as_json()
        inv_gmap = gmap.inverse(word=f"{label}^-1")
        # T_i maps side i=(i,i+1) to side j=(j+1,j).  The inverse maps
        # the paired side back.  These four directed edges encode all vertex
        # identifications induced by this side-pairing transformation.
        directed_vertex_edges.extend([
            (i % n, (j + 1) % n, gmap),
            ((i + 1) % n, j % n, gmap),
            ((j + 1) % n, i % n, inv_gmap),
            (j % n, (i + 1) % n, inv_gmap),
        ])
        side_pairings.append({
            "side": int(i),
            "paired_with": int(j),
            "word": label,
            "orientation": "reversed",
            "maps_vertices": [[int(i), int((j + 1) % n)], [int((i + 1) % n), int(j % n)]],
        })
        endpoint_audit.append({
            "word": label,
            "side": int(i),
            "paired_with": int(j),
            "endpoint_error": float(ep),
            "det_error": float(det_err),
            "maps": [[int(i), int((j + 1) % n)], [int((i + 1) % n), int(j % n)]],
        })

    # Vertex-cycle transformation consistency audit.
    # If the side-pairing data are genuinely compatible, transporting a local
    # chart around any closed loop in the vertex-identification graph returns
    # the same SU(1,1) transformation, modulo the harmless global sign of the
    # matrix representation.  This is a stronger audit than merely counting
    # vertex cycles and checking the angle sum.
    identity = SU11(1.0 + 0.0j, 0.0 + 0.0j, "id")
    vertex_transport: Dict[int, SU11] = {0: identity}
    queue: List[int] = [0]
    adjacency: Dict[int, List[Tuple[int, SU11]]] = {}
    for u, v, tr in directed_vertex_edges:
        adjacency.setdefault(u, []).append((v, tr))
    max_cycle_transform_consistency_error = 0.0
    while queue:
        u = queue.pop(0)
        Tu = vertex_transport[u]
        for v, edge_tr in adjacency.get(u, []):
            cand = edge_tr.compose(Tu, word=f"v{v}_transport")
            if v not in vertex_transport:
                vertex_transport[v] = cand
                queue.append(v)
            else:
                max_cycle_transform_consistency_error = max(
                    max_cycle_transform_consistency_error,
                    su11_equivalence_error(cand, vertex_transport[v]),
                )
    max_vertex_transport_image_error = 0.0
    if 0 in vertex_transport:
        z0 = vertices[0]
        for v, Tv in vertex_transport.items():
            max_vertex_transport_image_error = max(max_vertex_transport_image_error, abs(Tv(z0) - vertices[v]))
    transported_vertices_count = len(vertex_transport)

    vertex_angle_audit: List[Dict[str, Any]] = []
    max_angle_error = 0.0
    for ci, cls in enumerate(cycles):
        angle_sum = len(cls) * interior_angle
        err = abs(angle_sum - 2.0 * math.pi)
        max_angle_error = max(max_angle_error, err)
        vertex_angle_audit.append({
            "cycle_index": ci,
            "vertices": cls,
            "cycle_length": len(cls),
            "angle_sum": float(angle_sum),
            "smooth_error_from_2pi": float(err),
        })

    pairing_covers_sides = sorted([x for pair in pairs for x in pair]) == list(range(n))
    one_vertex = len(cycles) == 1
    genus_ok = abs(inferred_genus - genus) < 1.0e-12
    area_error = abs(area - gb)
    pass_audit = bool(
        pairing_covers_sides
        and one_vertex
        and genus_ok
        and area_error < 1.0e-10
        and max_angle_error < 1.0e-10
        and max_endpoint_error < 1.0e-10
        and bad_generator_count == 0
        and max_det_error < 1.0e-10
        and transported_vertices_count == n
        and max_vertex_transport_image_error < 1.0e-10
        and max_cycle_transform_consistency_error < 1.0e-8
    )

    fp = fingerprint_pairs(pairs)
    sid = f"sidepair_reg4g_g{genus}_{safe_label(profile)}_{fp}_n{variant_index:03d}"
    sj: Dict[str, Any] = {
        "format": "FuchsianGENN surface JSON v12",
        "surface_id": sid,
        "name": f"Certified regular 4g-gon side-pairing compact genus-{genus} surface ({profile})",
        "domain_type": "compact_polygon",
        "subdomain_type": "regular_4g_gon_one_vertex_side_pairing",
        "category": "compact_side_pairing_polygon",
        "surface_family": "compact_side_pairing_polygon",
        "surface_subfamily": "regular_one_vertex_pairing",
        "compact": True,
        "finite_area": True,
        "torsion_free": True,
        "smooth_riemann_surface": True,
        "orbifold_cone_points": 0,
        "cusp_count": 0,
        "genus": int(genus),
        "compactified_genus": int(genus),
        "inferred_genus_from_vertex_pairing": float(inferred_genus),
        "area": float(area),
        "gauss_bonnet_area": float(gb),
        "sides": int(n),
        "generators_count": int(len(generators)),
        "regular_tiling_symbol": f"{{{n},{n}}}",
        "interior_angle": float(interior_angle),
        "rotation_deg": float(rotation_deg),
        "side_pairing_fingerprint": fp,
        "side_pairing_pairs": [[int(a), int(b)] for a, b in pairs],
        "generators": generators,
        "polygon_vertices": [complex_to_xy(z) for z in vertices],
        "side_pairings": side_pairings,
        "vertex_equivalence_classes": cycles,
        "side_pairing_endpoint_audit": endpoint_audit,
        "vertex_angle_audit": vertex_angle_audit,
        "vertex_cycle_transform_audit": {
            "transported_vertices_count": int(transported_vertices_count),
            "max_vertex_transport_image_error": float(max_vertex_transport_image_error),
            "max_cycle_transform_consistency_error": float(max_cycle_transform_consistency_error),
            "interpretation": "closed vertex-identification transports are identity modulo SU(1,1) global sign",
        },
        "certification": {
            "status": "poincare_polygon_theorem_audited_regular_one_vertex_pairing",
            "construction": "regular {4g,4g} hyperbolic polygon with orientation-reversing one-vertex side-pairing matching",
            "audit": "one vertex cycle, cycle angle 2pi, Gauss-Bonnet area, endpoint pairing, SU(1,1) determinant checks, and vertex-cycle transformation consistency",
            "caveat": "The exported generator set comes from the selected side pairings. It is not a Teichmuller parameter sweep and not an orbifold record.",
        },
        "compatibility": {
            "fuchsian_explorer_v12_2": "loadable in advanced compact-polygon mode",
            "word_letters": LETTERS[: len(generators)],
        },
        "sampling_status": "supported by polygon sampler",
        "fundamental_domain_status": "explicit compact regular polygon fundamental domain with audited side-pairing data",
        "riemann_surface_status": "smooth compact hyperbolic Riemann surface D/Gamma",
        "kahler_status": "complex dimension one; automatically Kähler",
        "mainline_dataset_eligible": bool(pass_audit),
        "source_program": PROGRAM,
        "source_version": VERSION,
        "construction_parameters": {
            "genus": genus,
            "profile": profile,
            "variant_index": variant_index,
            "rotation_deg": rotation_deg,
            "pairing_pairs": [[int(a), int(b)] for a, b in pairs],
        },
    }

    audit = {
        "surface_id": sid,
        "surface": sid,
        "genus": genus,
        "profile": profile,
        "side_pairing_fingerprint": fp,
        "sides": n,
        "generators": len(generators),
        "pairing_covers_sides": pairing_covers_sides,
        "vertex_cycle_count": len(cycles),
        "one_vertex_cycle": one_vertex,
        "max_vertex_angle_error": max_angle_error,
        "inferred_genus": inferred_genus,
        "genus_formula_error": abs(inferred_genus - genus),
        "area": area,
        "gauss_bonnet_area": gb,
        "gauss_bonnet_error": area_error,
        "max_endpoint_error": max_endpoint_error,
        "su11_max_det_error": max_det_error,
        "bad_generator_count": bad_generator_count,
        "transported_vertices_count": transported_vertices_count,
        "max_vertex_transport_image_error": max_vertex_transport_image_error,
        "max_cycle_transform_consistency_error": max_cycle_transform_consistency_error,
        "poincare_polygon_conditions_pass": pass_audit,
        "smooth_riemann_surface": True,
        "orbifold_cone_points": 0,
        "cusps": 0,
        "compact": True,
        "finite_area": True,
    }
    return sj, audit


# -----------------------------------------------------------------------------
# Direct GINN integration
# -----------------------------------------------------------------------------


def run_direct_ginn(ginn: Any, sj: Dict[str, Any], run_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    sid = str(sj.get("surface_id"))
    outdir = run_root / "ginn_runs" / sid
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        rows, X, D, word_ball, label_meta, feature_names = ginn.generate_ginn_dataset(
            sj, args.pairs, args.depth, args.seed, max_word_ball=args.max_word_ball
        )
        write_json(outdir / "surface.json", sj)
        write_json(outdir / "label_meta.json", label_meta)
        pair_hidden = parse_hidden_width(args.pair_hidden, 128)
        score_hidden = parse_hidden_width(args.score_hidden, 64)
        metrics = ginn.train_ginn(
            rows, X, D, word_ball, outdir, args.depth, args.epochs,
            pair_hidden, score_hidden, args.lr,
            args.batch_size, args.seed, args.device, args.patience,
            args.ce_weight, args.soft_distance_weight, args.temperature,
            candidate_chunk_size=args.candidate_chunk_size,
            auto_chunk_threshold_mb=args.auto_chunk_threshold_mb,
        )
        return {
            "surface_id": sid,
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
            "surface_id": sid,
            "returncode": 1,
            "wall_seconds": time.time() - t0,
            "pass_ginn_training": False,
            "error": f"{type(e).__name__}: {e}",
        }


# -----------------------------------------------------------------------------
# Reports and CLI
# -----------------------------------------------------------------------------


def profile_defaults(profile: str) -> Dict[str, Any]:
    if profile == "smoke":
        return {"genera": [2], "variants_per_genus": 1, "include_opposite": False, "pairs": 500, "epochs": 3}
    if profile == "small":
        return {"genera": [2, 3], "variants_per_genus": 3, "include_opposite": True, "pairs": 3000, "epochs": 30}
    if profile == "balanced":
        return {"genera": [2, 3, 4, 5], "variants_per_genus": 6, "include_opposite": True, "pairs": 9000, "epochs": 80}
    if profile == "wide":
        return {"genera": [2, 3, 4, 5, 6, 7, 8], "variants_per_genus": 8, "include_opposite": True, "pairs": 9000, "epochs": 80}
    return {"genera": [2], "variants_per_genus": 1, "include_opposite": False, "pairs": 9000, "epochs": 80}


def write_report(path: Path, args: argparse.Namespace, surfaces: List[Dict[str, Any]], audits: List[Dict[str, Any]], train_rows: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Side-Pairing Polygon Batch Trainer Report\n")
    lines.append(f"Program: `{PROGRAM}` v{VERSION}\n")
    lines.append("## Scope\n")
    lines.append("Smooth compact Riemann surfaces only. Each animal is built from a regular hyperbolic `{4g,4g}` polygon with an audited orientation-reversing side-pairing matching. No orbifold records and no Teichmuller-space-only records are included.\n")
    lines.append("## Summary\n")
    lines.append(f"- Generated/attempted surfaces: **{len(audits) + len(failures)}**")
    lines.append(f"- Successfully generated surfaces: **{len(audits)}**")
    lines.append(f"- Run GINN: `{args.run_ginn}`")
    lines.append(f"- Dry run: `{args.dry_run}`")
    if args.run_ginn and not args.dry_run:
        ok = sum(1 for r in train_rows if r.get("pass_ginn_training"))
        lines.append(f"- Training successes: **{ok}/{len(train_rows)}**")
    lines.append("")
    lines.append("## Surfaces and audit fields\n")
    lines.append("| surface_id | genus | profile | sides | generators | one vertex? | GB error | angle error | endpoint error | vertex transport error | cycle transform error | SU(1,1) det error | Poincare audit? |")
    lines.append("| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for a in audits:
        lines.append(
            f"| `{a['surface_id']}` | {a['genus']} | {a['profile']} | {a['sides']} | {a['generators']} | "
            f"{a['one_vertex_cycle']} | {a['gauss_bonnet_error']:.3g} | {a['max_vertex_angle_error']:.3g} | "
            f"{a['max_endpoint_error']:.3g} | {a.get('max_vertex_transport_image_error', float('nan')):.3g} | "
            f"{a.get('max_cycle_transform_consistency_error', float('nan')):.3g} | {a['su11_max_det_error']:.3g} | {a['poincare_polygon_conditions_pass']} |"
        )
    lines.append("")
    lines.append("## Mathematical audit notes\n")
    lines.append("- `one vertex?` checks that the side-pairing vertex identifications have one vertex cycle.")
    lines.append("- For a regular `{4g,4g}` polygon, the angle at each vertex is `pi/(2g)`. One vertex cycle therefore has total angle `2*pi`.")
    lines.append("- `GB error` is `|area - 4*pi*(g-1)|`.")
    lines.append("- Endpoint and SU(1,1) determinant errors audit the exported side-pairing maps.")
    lines.append("- `vertex transport error` checks that side-pairing transformations carry a chosen base vertex to every equivalent vertex as expected.")
    lines.append("- `cycle transform error` checks closed vertex-cycle transports return the same SU(1,1) map, modulo the global matrix sign.")
    lines.append("- These are side-pairing generators for the chosen polygon presentation; they need not be canonical in any moduli-space sense.")
    lines.append("")
    if train_rows:
        lines.append("## Training results\n")
        lines.append("| surface_id | pass | word ball | top1 | top5 | shortcut fraction | seconds | error |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for r in train_rows:
            lines.append(
                f"| `{r.get('surface_id')}` | {r.get('pass_ginn_training')} | {r.get('word_ball_size','')} | "
                f"{r.get('winning_lift_accuracy_test','')} | {r.get('winning_lift_top5_accuracy_test','')} | "
                f"{r.get('shortcut_fraction_test','')} | {r.get('wall_seconds','')} | {str(r.get('error','')).replace('|','/')} |"
            )
        lines.append("")
    if failures:
        lines.append("## Failures\n")
        lines.append("| surface | error |")
        lines.append("| --- | --- |")
        for f in failures:
            lines.append(f"| `{f.get('surface')}` | {str(f.get('error','')).replace('|','/')} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compact side-pairing polygon batch trainer")
    ap.add_argument("--profile", default="balanced", choices=["smoke", "small", "balanced", "wide", "manual"])
    ap.add_argument("--genera", default="", help="comma-list override, e.g. 2,3,4")
    ap.add_argument("--variants-per-genus", type=int, default=0, help="random one-vertex pairings per genus; 0 uses profile default")
    ap.add_argument("--include-opposite-control", action="store_true", help="include the usual opposite-side regular_g control for each genus")
    ap.add_argument("--no-opposite-control", dest="include_opposite_control", action="store_false")
    ap.set_defaults(include_opposite_control=None)
    ap.add_argument("--max-surfaces", type=int, default=0, help="0 means no cap")
    ap.add_argument("--max-attempts-per-genus", type=int, default=200000)
    ap.add_argument("--rotation-deg", type=float, default=22.5)
    ap.add_argument("--outroot", default="side_pairing_polygon_batch_runs")
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-ginn", action="store_true")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-word-ball", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--candidate-chunk-size", type=int, default=8192)
    ap.add_argument("--pair-hidden", default="128", help="integer hidden width; legacy comma-lists are accepted by taking the first value")
    ap.add_argument("--score-hidden", default="64", help="integer score/context width; legacy comma-lists are accepted by taking the first value")
    ap.add_argument("--lr", type=float, default=2.0e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--soft-distance-weight", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--auto-chunk-threshold-mb", type=float, default=256.0)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)

    defs = profile_defaults(args.profile)
    genera = parse_int_list(args.genera, defs["genera"])
    variants_per_genus = args.variants_per_genus if args.variants_per_genus > 0 else int(defs["variants_per_genus"])
    include_opposite = bool(defs["include_opposite"] if args.include_opposite_control is None else args.include_opposite_control)
    if args.pairs <= 0:
        args.pairs = int(defs["pairs"])
    if args.epochs <= 0:
        args.epochs = int(defs["epochs"])

    rng = random.Random(args.seed)
    run_label = f"run_{now_stamp()}" + (f"_{safe_label(args.label)}" if args.label else "")
    run_root = Path(args.outroot) / run_label
    run_root.mkdir(parents=True, exist_ok=True)

    ginn = None
    if args.run_ginn and not args.dry_run:
        ginn = load_module(args.ginn_script, "ginn_v24_side_pairing_polygon")

    surfaces: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    try:
        for genus in genera:
            specs = generate_pairings_for_genus(
                genus,
                variants=variants_per_genus,
                rng=rng,
                include_opposite=include_opposite,
                max_attempts=args.max_attempts_per_genus,
            )
            for profile_name, pairs in specs:
                if args.max_surfaces and len(audits) >= args.max_surfaces:
                    break
                try:
                    sj, audit = build_side_pairing_surface(genus, pairs, profile_name, len(audits), args.rotation_deg)
                    surfaces.append(sj)
                    audits.append(audit)
                    write_json(run_root / "surfaces" / f"{sj['surface_id']}.json", sj)
                    if args.run_ginn and not args.dry_run and ginn is not None:
                        tr = run_direct_ginn(ginn, sj, run_root, args)
                        train_rows.append(tr)
                except Exception as e:
                    failures.append({"surface": f"g{genus}_{profile_name}", "error_type": type(e).__name__, "error": str(e)})
            if args.max_surfaces and len(audits) >= args.max_surfaces:
                break
    finally:
        manifest = {
            "program": PROGRAM,
            "version": VERSION,
            "args": vars(args),
            "generated_surfaces": len(audits),
            "failures": len(failures),
            "scope": "smooth compact Riemann surfaces from explicit audited side-pairing polygons",
        }
        write_json(run_root / "manifest.json", manifest)
        write_csv(run_root / "tables" / "side_pairing_audit.csv", audits)
        write_csv(run_root / "tables" / "ginn_training_summary.csv", train_rows)
        write_csv(run_root / "tables" / "failures.csv", failures)
        write_report(run_root / "side_pairing_polygon_batch_report.md", args, surfaces, audits, train_rows, failures)

    report_src = run_root / "side_pairing_polygon_batch_report.md"
    # Convenience copy for quick upload/inspection from the current directory.
    try:
        Path("side_pairing_polygon_batch_report.md").write_text(report_src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    print(f"[done] report={report_src}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
