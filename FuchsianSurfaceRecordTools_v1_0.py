#!/usr/bin/env python3
"""FuchsianSurfaceRecordTools_v1_0.py

Shared normalization/audit helpers for Fuchsian GENN family testers.
This is intentionally lightweight: it does not construct surfaces.  It only
adds a stable master-builder-ready metadata contract to surface JSON records
and provides CSV/JSON helpers used by the v1.2/v1.4 family testers.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CONTRACT_VERSION = "fuchsian_surface_record_contract_v1"

GEOMETRY_AUDIT_FIELDS = [
    "surface", "surface_id", "surface_family", "surface_subfamily",
    "mainline_dataset_eligible", "exclusion_reason", "riemann_surface_status",
    "domain_type", "subdomain_type", "torsion_free", "orbifold_excluded",
    "compact", "finite_area", "genus", "compactified_genus", "area",
    "cusp_count", "generator_count", "generator_truncated",
    "pass_geometry_audit", "source_program", "source_version",
]

GINN_SMOKE_FIELDS = [
    "surface", "surface_id", "pairs", "word_depth", "word_ball_size",
    "shortcut_fraction", "mean_winner_depth", "max_word_ball",
    "pass_ginn_preflight", "error",
]

GINN_TRAINING_FIELDS = [
    "surface", "surface_id", "returncode", "wall_seconds", "pass_ginn_training",
    "cmd", "stdout_tail", "stderr_tail",
]

FAILURE_FIELDS = ["surface", "surface_id", "error_type", "error"]


def finite_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        y = int(x)
        return y
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    if fieldnames:
        keys.extend(fieldnames)
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


def generator_su11_audit(surface_json: Dict[str, Any]) -> Dict[str, Any]:
    gens = surface_json.get("generators") or {}
    max_det_err = 0.0
    bad: List[str] = []
    for name, g in gens.items():
        try:
            a = g.get("alpha")
            b = g.get("beta")
            alpha = complex(float(a[0]), float(a[1]))
            beta = complex(float(b[0]), float(b[1]))
            det = abs(alpha) ** 2 - abs(beta) ** 2
            max_det_err = max(max_det_err, abs(det - 1.0))
            if det <= 0.0 or not math.isfinite(det):
                bad.append(str(name))
        except Exception:
            bad.append(str(name))
    return {
        "generator_count": len(gens),
        "su11_max_det_error": max_det_err,
        "bad_generator_count": len(bad),
        "bad_generators": ";".join(bad[:20]),
    }


def _status_string_for_surface(sj: Dict[str, Any], compact: bool, finite_area: bool, torsion_free: bool) -> str:
    if not torsion_free:
        return "not a smooth Riemann surface record; elliptic torsion/orbifold status present or not cleared"
    existing = sj.get("riemann_surface_status")
    if existing:
        return str(existing)
    if compact:
        return "smooth compact hyperbolic Riemann surface D/Gamma"
    if finite_area:
        return "smooth noncompact finite-area hyperbolic Riemann surface D/Gamma with cusps"
    return "smooth noncompact hyperbolic Riemann surface D/Gamma; finite-area status not asserted"


def _kahler_status(sj: Dict[str, Any], is_surface: bool) -> str:
    if sj.get("kahler_status"):
        return str(sj["kahler_status"])
    if is_surface:
        return "complex dimension one; automatically Kähler"
    return "not asserted because smooth Riemann-surface status is not cleared"


def _default_domain_status(sj: Dict[str, Any]) -> str:
    dt = str(sj.get("domain_type", ""))
    if dt == "compact_polygon" and sj.get("polygon_vertices"):
        return "explicit compact polygon fundamental domain with side-pairing data"
    if dt == "modular_ford_domain" and sj.get("fundamental_domain_tiles"):
        return "explicit Ford-tile union domain model in the Poincare disk"
    if dt == "triangle_kernel_tile_union" and sj.get("fundamental_domain_tiles"):
        return "triangle coset-tile sampling scaffold for torsion-free kernel surface"
    if sj.get("fundamental_domain_tiles"):
        return "explicit tile-union domain model in the Poincare disk"
    return "not available or not asserted"


def _default_sampling_status(sj: Dict[str, Any]) -> str:
    dt = str(sj.get("domain_type", ""))
    if dt == "compact_polygon" and sj.get("polygon_vertices"):
        return "supported by polygon sampler"
    if sj.get("fundamental_domain_tiles"):
        return "supported by disk tile-union sampler"
    return "not supported by current GINN sampler"


def normalize_surface_record(
    sj: Dict[str, Any],
    *,
    surface_spec: str,
    surface_family: str,
    surface_subfamily: str = "",
    source_program: str,
    source_version: str,
    construction_parameters: Optional[Dict[str, Any]] = None,
    geometry_audit_pass: Optional[bool] = None,
    finite_area: Optional[bool] = None,
    torsion_free: Optional[bool] = None,
    mainline_dataset_eligible: Optional[bool] = None,
    exclusion_reason: str = "",
) -> Dict[str, Any]:
    """Return a copy of a surface JSON with stable master-builder fields.

    This function deliberately avoids proving the mathematics.  It records the
    status established by the family-specific tester and fills in common fields
    so the master builder can ingest all family outputs consistently.
    """
    out = dict(sj)
    dt = str(out.get("domain_type", ""))
    compact = bool(out.get("compact")) if out.get("compact") is not None else (dt == "compact_polygon")
    if torsion_free is None:
        if out.get("torsion_free") is not None:
            torsion_free = bool(out.get("torsion_free"))
        elif compact and dt == "compact_polygon":
            torsion_free = True
        elif out.get("surface_type") == "hurwitz_triangle_kernel_surface":
            torsion_free = True
        else:
            torsion_free = False
    if finite_area is None:
        if out.get("finite_area") is not None:
            finite_area = bool(out.get("finite_area"))
        elif compact:
            finite_area = True
        elif out.get("area") is not None and out.get("cusp_count") is not None:
            finite_area = True
        else:
            finite_area = False
    orbifold_excluded = not bool(torsion_free)
    sampling_status = out.get("sampling_status") or _default_sampling_status(out)
    domain_status = out.get("fundamental_domain_status") or _default_domain_status(out)
    is_surface = bool(torsion_free) and not orbifold_excluded
    if mainline_dataset_eligible is None:
        mainline_dataset_eligible = bool(
            is_surface
            and (geometry_audit_pass is None or geometry_audit_pass)
            and sampling_status != "not supported by current GINN sampler"
            and len(out.get("generators") or {}) > 0
        )
    if not mainline_dataset_eligible and not exclusion_reason:
        if not torsion_free:
            exclusion_reason = "torsion/orbifold status not cleared"
        elif sampling_status == "not supported by current GINN sampler":
            exclusion_reason = "no supported sampling model"
        elif len(out.get("generators") or {}) == 0:
            exclusion_reason = "no exported SU(1,1) generators"
        elif geometry_audit_pass is False:
            exclusion_reason = "family-specific geometry audit failed"
        else:
            exclusion_reason = "not selected for mainline dataset"

    surface_id = str(out.get("surface_id") or surface_spec.replace("/", "_").replace(" ", "_"))
    genus = out.get("genus")
    if genus is None and compact:
        genus = out.get("compactified_genus")
    gea = out.get("generator_export_audit") or {}
    gen_count = len(out.get("generators") or {})
    generator_truncated = bool(
        out.get("generator_truncated")
        or out.get("generator_truncated_by_cli_max_generators")
        or gea.get("generator_truncated_by_cli_max_generators")
    )
    master = {
        "contract_version": CONTRACT_VERSION,
        "surface_id": surface_id,
        "surface_spec": surface_spec,
        "surface_family": surface_family,
        "surface_subfamily": surface_subfamily,
        "source_program": source_program,
        "source_version": source_version,
        "construction_parameters": construction_parameters or {},
        "domain_type": dt,
        "subdomain_type": out.get("subdomain_type"),
        "fundamental_domain_status": domain_status,
        "sampling_status": sampling_status,
        "riemann_surface_status": _status_string_for_surface(out, compact, bool(finite_area), bool(torsion_free)),
        "kahler_status": _kahler_status(out, is_surface),
        "torsion_free": bool(torsion_free),
        "orbifold_excluded": bool(orbifold_excluded),
        "compact": bool(compact),
        "finite_area": bool(finite_area),
        "cusp_count": out.get("cusp_count"),
        "genus": genus,
        "compactified_genus": out.get("compactified_genus"),
        "area": out.get("area"),
        "generator_count": gen_count,
        "generator_truncated": generator_truncated,
        "word_ball_recommended_depth": out.get("word_ball_recommended_depth", 2),
        "mainline_dataset_eligible": bool(mainline_dataset_eligible),
        "exclusion_reason": exclusion_reason,
        "geometry_audit_pass": geometry_audit_pass,
    }
    out.update(master)
    out["master_record"] = master
    out["mathematical_object"] = out.get("mathematical_object") or master["riemann_surface_status"]
    return out


def audit_row_from_surface(sj: Dict[str, Any], pass_geometry_audit: Optional[bool] = None) -> Dict[str, Any]:
    return {
        "surface": sj.get("surface_spec") or sj.get("surface_id"),
        "surface_id": sj.get("surface_id"),
        "surface_family": sj.get("surface_family"),
        "surface_subfamily": sj.get("surface_subfamily"),
        "mainline_dataset_eligible": sj.get("mainline_dataset_eligible"),
        "exclusion_reason": sj.get("exclusion_reason"),
        "riemann_surface_status": sj.get("riemann_surface_status"),
        "domain_type": sj.get("domain_type"),
        "subdomain_type": sj.get("subdomain_type"),
        "torsion_free": sj.get("torsion_free"),
        "orbifold_excluded": sj.get("orbifold_excluded"),
        "compact": sj.get("compact"),
        "finite_area": sj.get("finite_area"),
        "genus": sj.get("genus"),
        "compactified_genus": sj.get("compactified_genus"),
        "area": sj.get("area"),
        "cusp_count": sj.get("cusp_count"),
        "generator_count": sj.get("generator_count"),
        "generator_truncated": sj.get("generator_truncated"),
        "pass_geometry_audit": sj.get("geometry_audit_pass") if pass_geometry_audit is None else pass_geometry_audit,
        "source_program": sj.get("source_program"),
        "source_version": sj.get("source_version"),
    }
