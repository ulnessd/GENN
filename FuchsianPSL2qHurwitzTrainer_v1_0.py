#!/usr/bin/env python3
"""
FuchsianPSL2qHurwitzTrainer_v1_0.py

PSL(2,q) Hurwitz quotient -> Fuchsian kernel surface -> exact finite
word-ball atlas -> candidate-pool GINN reranker.

This is a cleaned-up 2,q sibling of the Janko trainer.  It reuses the fast
Big-Hurwitz atlas/training machinery, but replaces the old PSL kernel export
with a Janko-style selected-kernel-generator surface builder:

  --kernel-generator-mode all       export every nonidentity Schreier generator
  --kernel-generator-mode first     export the first N nonidentity generators
  --kernel-generator-mode shortest  scan all kernel edges and export the N
                                    shortest by SU(1,1) identity displacement

Interpretation warning
----------------------
When --kernel-generator-mode shortest/first is used, the word ball is the exact
reduced length-<=depth word ball in the selected generator pool.  It is not the
full word ball in the complete Reidemeister-Schreier generating set.  This is
precisely the same convention used in the Janko v1.1 trainer.

For q=13 you may use --kernel-generator-mode all to reproduce the old fully
exported-generator interpretation.  For q=29, shortest-256 is the recommended
first serious run.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import json
import math
import platform
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROGRAM = "FuchsianPSL2qHurwitzTrainer_v1_0.py"
VERSION = "1.0"


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
                    seen.add(k)
                    keys.append(k)
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


def default_hurwitz_script() -> str:
    return default_local("FuchsianHurwitzTester_v1_6.py")


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


def psl2_order(q: int) -> int:
    return int(q) * (int(q) * int(q) - 1) // 2


def psl2_hurwitz_prime_condition(q: int) -> bool:
    q = int(q)
    return q == 7 or (q % 7 in (1, 6))


def psl2_hurwitz_genus(q: int) -> Optional[int]:
    order = psl2_order(q)
    if order % 84:
        return None
    return 1 + order // 84


def install_leq_depth_word_ball_builder(ginn: Any) -> None:
    """Install a local, explicit <=depth reduced word-ball builder.

    For m oriented letters and depth 2, the raw reduced word ball has
    1 + m + m(m-1) elements.  This pins the convention used by this trainer.
    """
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

    build_word_ball_leq_depth._psl2q_word_ball_convention = "reduced_length_leq_depth"  # type: ignore[attr-defined]
    build_word_ball_leq_depth._psl2q_depth2_formula = "1 + m + m*(m-1)"  # type: ignore[attr-defined]
    ginn.build_word_ball = build_word_ball_leq_depth


def normalize_su11_pair(alpha: complex, beta: complex) -> Tuple[complex, complex]:
    det = abs(alpha) ** 2 - abs(beta) ** 2
    if det > 0 and math.isfinite(det):
        scale = 1.0 / math.sqrt(det)
        alpha *= scale
        beta *= scale
    if alpha.real < -1.0e-14 or (abs(alpha.real) <= 1.0e-14 and alpha.imag < 0):
        alpha, beta = -alpha, -beta
    return alpha, beta


def su11_pair_inverse(alpha: complex, beta: complex) -> Tuple[complex, complex]:
    return normalize_su11_pair(alpha.conjugate(), -beta)


def su11_pair_compose(a: complex, b: complex, c: complex, d: complex) -> Tuple[complex, complex]:
    # Matrix product [[a,b],[b*,a*]] * [[c,d],[d*,c*]].
    return normalize_su11_pair(a * c + b * d.conjugate(), a * d + b * c.conjugate())


def su11_pair_apply(alpha: complex, beta: complex, z: complex) -> complex:
    return (alpha * z + beta) / (beta.conjugate() * z + alpha.conjugate())


def su11_pair_displacement_from_identity(alpha: complex, beta: complex) -> float:
    return math.hypot(alpha.real - 1.0, alpha.imag) + abs(beta)


def su11_pair_as_json(alpha: complex, beta: complex) -> Dict[str, Any]:
    return {
        "type": "su11",
        "alpha": [float(alpha.real), float(alpha.imag)],
        "beta": [float(beta.real), float(beta.imag)],
    }


def maybe_add_inverse_generators(gens: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # Surfaces store only positive selected generators; GINN parse_generators adds inverses.
    # Kept as a named hook in case older local GINN versions behave differently.
    return gens


def build_schreier_kernel_surface_psl2q(group: Any, triple: Dict[str, Any], args: argparse.Namespace, run_id: str, hurwitz: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Janko-style selected-generator Schreier surface builder for PSL(2,q).

    v1.0 uses a fast coset-transversal SU(1,1) computation.  Instead of
    composing the long token word for every Schreier edge, it stores the
    SU(1,1) transform M_h for each coset representative and computes

        K(h,s) = M_h S_s M_{hs}^{-1}

    directly in alpha/beta coordinates.  This makes q=29 shortest-mode scans
    practical.
    """
    t0 = time.perf_counter()
    delta_gens, delta_audit = hurwitz.build_delta_237_su11()
    Xf = hurwitz.finite_from_json_matrix(triple["x"])
    Yf = hurwitz.finite_from_json_matrix(triple["y"])
    finite_gens = {"X": Xf, "Y": Yf, "Y^-1": hurwitz.mat_inv(Yf, group.p)}
    token_order = ["X", "Y", "Y^-1"]

    # Schreier transversal by BFS in the quotient group G using X,Y,Y^-1.
    # Store IDs, parents, and parent tokens so that we can build SU(1,1)
    # representatives without repeatedly recomposing long words.
    rep_id: Dict[Any, int] = {group.identity: 0}
    rep_mats: List[Any] = [group.identity]
    rep_words: List[List[str]] = [[]]
    parent_id: List[int] = [-1]
    parent_tok: List[str] = [""]
    qd: deque[int] = deque([0])
    while qd:
        hid = qd.popleft()
        h = rep_mats[hid]
        for tok in token_order:
            nh = hurwitz.mat_mul(h, finite_gens[tok], group.p)
            if nh not in rep_id:
                nid = len(rep_mats)
                rep_id[nh] = nid
                rep_mats.append(nh)
                rep_words.append(rep_words[hid] + [tok])
                parent_id.append(hid)
                parent_tok.append(tok)
                qd.append(nid)
    if len(rep_mats) != group.order:
        raise RuntimeError(f"Schreier transversal incomplete: {len(rep_mats)} of {group.order}")

    # Coset representative transforms in SU(1,1), built by parent chain.
    coset_alpha = np.empty(group.order, dtype=np.complex128)
    coset_beta = np.empty(group.order, dtype=np.complex128)
    coset_alpha[0] = 1.0 + 0j
    coset_beta[0] = 0.0 + 0j
    delta_pair = {tok: (complex(delta_gens[tok].alpha), complex(delta_gens[tok].beta)) for tok in token_order}
    for i in range(1, group.order):
        pid = parent_id[i]
        tok = parent_tok[i]
        a, b = su11_pair_compose(complex(coset_alpha[pid]), complex(coset_beta[pid]), delta_pair[tok][0], delta_pair[tok][1])
        coset_alpha[i] = a
        coset_beta[i] = b

    mode = str(args.kernel_generator_mode)
    limit = int(args.kernel_generator_limit)
    if mode == "all":
        limit = 0
    if mode not in {"all", "first", "shortest"}:
        raise ValueError(f"Unknown kernel_generator_mode={mode}")

    raw_nonidentity = 0
    identity_like = 0
    nonidentity_image_failures = 0
    scanned_slots = 0
    selected_all: List[Dict[str, Any]] = []
    selected_heap: List[Tuple[float, int, Dict[str, Any]]] = []
    progress_every = max(1, int(getattr(args, "kernel_progress_every", 10000)))

    def _keep_record(rec: Dict[str, Any]) -> None:
        nonlocal selected_all, selected_heap
        if mode in ("all", "first"):
            if mode == "all" or limit <= 0 or len(selected_all) < limit:
                selected_all.append(rec)
            return
        # shortest mode: maintain max-heap by displacement using negative key.
        disp = float(rec["identity_displacement"])
        item = (-disp, int(rec["scan_index"]), rec)
        if limit <= 0:
            selected_all.append(rec)
        elif len(selected_heap) < limit:
            heapq.heappush(selected_heap, item)
        elif disp < -selected_heap[0][0]:
            heapq.heapreplace(selected_heap, item)

    for hid, h in enumerate(rep_mats):
        ah = complex(coset_alpha[hid]); bh = complex(coset_beta[hid])
        for tok in token_order:
            scanned_slots += 1
            hs = hurwitz.mat_mul(h, finite_gens[tok], group.p)
            hs_id = rep_id[hs]

            # The finite image of h*s*rep(hs)^-1 is identity by construction.
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
            _keep_record(rec)
        if mode == "first" and limit > 0 and len(selected_all) >= limit:
            break
        if getattr(args, "verbose", False) and hid > 0 and hid % progress_every == 0:
            kept = len(selected_all) if mode != "shortest" else len(selected_heap)
            print(f"[kernel PSL2,{group.p}] scanned cosets={hid:,}/{group.order:,} slots={scanned_slots:,} raw_nonidentity={raw_nonidentity:,} kept={kept:,}", flush=True)

    if mode == "shortest" and limit > 0:
        selected = [item[2] for item in selected_heap]
        selected.sort(key=lambda r: (float(r["identity_displacement"]), int(r["scan_index"])))
    else:
        selected = selected_all
        if mode == "shortest":
            selected.sort(key=lambda r: (float(r["identity_displacement"]), int(r["scan_index"])))

    gen_json: Dict[str, Dict[str, Any]] = {}
    meanings: Dict[str, str] = {}
    audit_sample: List[Dict[str, Any]] = []
    for idx, rec in enumerate(selected):
        label = f"h{idx:06d}"
        gen_json[label] = rec["su11"]
        meanings[label] = (
            f"PSL(2,{group.p}) selected Schreier kernel edge generator {label}: "
            f"coset {rec['source_coset_id']} --{rec['edge_token']}--> {rec['target_coset_id']}; "
            f"selection={mode}; displacement={float(rec['identity_displacement']):.8g}; trace={float(rec['trace_real']):.8g}"
        )
        if len(audit_sample) < int(args.kernel_audit_sample_rows):
            audit_sample.append({k: v for k, v in rec.items() if k != "su11"})

    kernel_generator_scan_complete = not (mode == "first" and limit > 0 and len(selected_all) >= limit and scanned_slots < group.order * 3)
    generator_export_complete = (mode == "all" and nonidentity_image_failures == 0 and kernel_generator_scan_complete)

    # Fundamental-domain sampling scaffold: images of the base triangle by all/capped coset reps.
    base_verts = [complex(x, y) for x, y in delta_audit["base_orbifold_triangle_vertices"]]
    max_tiles = int(args.max_tiles) if int(args.max_tiles) > 0 else group.order
    tiles = []
    for tile_idx in range(min(group.order, max_tiles)):
        a = complex(coset_alpha[tile_idx]); b = complex(coset_beta[tile_idx])
        verts = [su11_pair_apply(a, b, z) for z in base_verts]
        tiles.append({
            "tile_index": tile_idx,
            "coset_id": tile_idx,
            "coset_matrix": hurwitz.matrix_to_list(rep_mats[tile_idx]),
            "vertices": [[float(z.real), float(z.imag)] for z in verts],
        })
        if getattr(args, "verbose", False) and tile_idx > 0 and tile_idx % max(1, int(args.tile_progress_every)) == 0:
            print(f"[tiles PSL2,{group.p}] {tile_idx:,}/{min(group.order,max_tiles):,}", flush=True)

    tile_scaffold_complete = len(tiles) == group.order
    relation_max_error = max(delta_audit["relations_numerical"].values())
    finite_certificate_complete = (
        triple.get("surjective") is True
        and triple.get("x_order") == 2
        and triple.get("y_order") == 3
        and triple.get("z_order") == 7
        and triple.get("xyz_identity") is True
    )
    atlas_training_ready = bool(gen_json) and bool(tiles) and finite_certificate_complete and relation_max_error < 1.0e-8 and nonidentity_image_failures == 0
    pass_geometry_audit = atlas_training_ready and generator_export_complete and tile_scaffold_complete

    partial_reasons: List[str] = []
    if not generator_export_complete:
        partial_reasons.append(f"kernel generator export is selected/truncated: mode={mode}, selected={len(gen_json)}, raw_nonidentity={raw_nonidentity}")
    if not kernel_generator_scan_complete:
        partial_reasons.append("kernel edge scan stopped early")
    if not tile_scaffold_complete:
        partial_reasons.append("triangle-tile scaffold truncated by --max-tiles")
    if nonidentity_image_failures:
        partial_reasons.append("some Schreier generators did not map to identity in the finite quotient")
    if relation_max_error >= 1.0e-8:
        partial_reasons.append("PSU(1,1) triangle relation numerical error above tolerance")
    if not finite_certificate_complete:
        partial_reasons.append("finite quotient certificate failed")
    if not gen_json:
        partial_reasons.append("no nonidentity kernel generators exported")
    exclusion_reason = "; ".join(partial_reasons)

    genus = triple["genus"]
    sid = f"hurwitz_PSL2_{group.p}_triple_{int(triple['triple_index']):04d}_kernel_{mode}{len(gen_json)}"
    surface = {
        "format": "FuchsianGENN surface JSON v1.0 psl2q-selected-hurwitz-tokenized-kernel",
        "surface_id": sid,
        "name": f"Hurwitz PSU(1,1) kernel surface from PSL(2,{group.p}) triple {triple['triple_index']:04d}",
        "surface_type": "psl2q_hurwitz_triangle_kernel_surface",
        "domain_type": "triangle_kernel_tile_union",
        "compact": True,
        "finite_area": True,
        "torsion_free": True,
        "orbifold_excluded": False,
        "mainline_dataset_eligible": pass_geometry_audit,
        "atlas_training_ready": atlas_training_ready,
        "riemann_surface_status": "smooth compact Hurwitz Riemann surface D/Gamma, with Gamma the torsion-free kernel of Delta^+(2,3,7) -> PSL(2,q)",
        "kahler_status": "complex dimension one; automatically Kähler",
        "genus": genus,
        "area": 4.0 * math.pi * (genus - 1) if genus is not None else None,
        "gauss_bonnet_area": 4.0 * math.pi * (genus - 1) if genus is not None else None,
        "triangle_group": "Delta^+(2,3,7)",
        "triangle_signature": [2, 3, 7],
        "finite_quotient": f"PSL(2,{group.p})",
        "finite_quotient_order": group.order,
        "quotient_order": group.order,
        "ginn_ready": atlas_training_ready,
        "explorer_loadable": False,
        "v1_0_psl2q_tokenized_generators": True,
        "generator_count": len(gen_json),
        "generator_selection_mode": mode,
        "generator_selection_limit": int(args.kernel_generator_limit),
        "generator_truncated": not generator_export_complete,
        "generator_export_complete": generator_export_complete,
        "kernel_generator_scan_complete": kernel_generator_scan_complete,
        "raw_schreier_slots": group.order * 3,
        "raw_nonidentity_schreier_generators": raw_nonidentity,
        "identity_like_generators_filtered": identity_like,
        "tile_scaffold_complete": tile_scaffold_complete,
        "tiles_truncated_by_cli_max_tiles": not tile_scaffold_complete,
        "tile_count": len(tiles),
        "expected_tile_count": group.order,
        "exclusion_reason": exclusion_reason,
        "generators": maybe_add_inverse_generators(gen_json),
        "generator_meanings": meanings,
        "kernel_generator_audit_sample": audit_sample,
        "fundamental_domain_tiles": tiles,
        "tile_scaffold_warning": "Tile union is built from Delta(2,3,7) orbifold triangle coset representatives. Full scaffold means all quotient cosets are present. It is a computational sampling scaffold, not a polished side-paired compact polygon.",
        "finite_group_triple": triple,
        "psu11_triangle_audit": delta_audit,
        "schreier_audit": {
            "transversal_size": len(rep_mats),
            "expected_transversal_size": group.order,
            "raw_schreier_slots": group.order * 3,
            "raw_nonidentity_schreier_generators": raw_nonidentity,
            "selected_kernel_generators": len(gen_json),
            "identity_like_generators_filtered": identity_like,
            "nonidentity_image_failures": nonidentity_image_failures,
            "kernel_generator_export_complete": generator_export_complete,
            "kernel_generator_scan_complete": kernel_generator_scan_complete,
            "generator_selection_mode": mode,
            "generator_selection_limit": int(args.kernel_generator_limit),
            "tile_scaffold_complete": tile_scaffold_complete,
            "tile_count": len(tiles),
            "expected_tile_count": group.order,
            "all_selected_generators_map_to_identity_in_quotient": nonidentity_image_failures == 0,
        },
        "certification": {
            "status": "complete_ginn_ready_psl2q_hurwitz_kernel_surface" if pass_geometry_audit else "selected_generator_psl2q_hurwitz_kernel_surface",
            "finite_quotient_certificate": "exact PSL(2,p) relation and BFS generation order check",
            "psu11_triangle_certificate": "numerical SU(1,1) Delta(2,3,7) relation checks",
            "kernel_certificate": "selected Reidemeister-Schreier generators k=t*s*rep(ts)^-1 map to identity in the finite quotient",
            "remaining_caveat": "Unless --kernel-generator-mode all is used, the exported generator set is a selected computational pool, not a complete kernel generator set or minimal side-paired polygon presentation.",
        },
        "maker_run_id": run_id,
    }
    audit = {
        "surface_id": sid,
        "q": group.p,
        "triple_index": triple["triple_index"],
        "genus": genus,
        "quotient_order": group.order,
        "transversal_size": len(rep_mats),
        "raw_schreier_slots": group.order * 3,
        "raw_nonidentity_schreier_generators": raw_nonidentity,
        "kernel_generators_exported": len(gen_json),
        "kernel_generator_export_complete": generator_export_complete,
        "kernel_generator_scan_complete": kernel_generator_scan_complete,
        "generator_truncated": not generator_export_complete,
        "identity_like_filtered": identity_like,
        "tile_count": len(tiles),
        "expected_tile_count": group.order,
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


def build_surfaces_psl2q(args: argparse.Namespace, hurwitz: Any, run_root: Path, perf: Optional[Any] = None) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    print(f"[group] building PSL(2,{args.q})", flush=True)
    group = hurwitz.PSL2PrimeGroup.build(args.q, verbose=args.verbose)
    triples, stats = hurwitz.find_hurwitz_triples(
        group,
        max_triples=args.max_triples,
        random_trials=0,
        seed=args.seed + args.q,
        triple_search="conjugacy_orbit",
        triple_equivalence=args.triple_equivalence,
        verbose=args.verbose,
    )
    if not triples:
        raise RuntimeError(f"No Hurwitz triples found for PSL(2,{args.q})")
    print(f"[triples] found {len(triples)} {args.triple_equivalence}-reduced triple(s)", flush=True)
    write_json(run_root / "group" / f"PSL2_{args.q}_triple_search_stats.json", stats)
    write_json(run_root / "group" / f"PSL2_{args.q}_triples.json", {"q": args.q, "triples": triples, "stats": stats})
    if perf is not None:
        perf.log("group_and_triples_built", q=args.q, n_triples=len(triples), quotient_order=group.order)

    surfaces: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    run_id = run_root.name.replace("run_", "")
    for tr in triples:
        print(f"[surface] building selected kernel for triple {tr['triple_index']:04d}", flush=True)
        if perf is not None:
            perf.log("surface_build_start", triple_index=tr.get("triple_index"))
        surface, audit = build_schreier_kernel_surface_psl2q(group, tr, args, run_id, hurwitz)
        surface["psl2q_hurwitz_trainer_record"] = True
        surface["psl2q_hurwitz_program"] = PROGRAM
        surface["psl2q_hurwitz_version"] = VERSION
        sid = str(surface.get("surface_id"))
        write_json(run_root / "surfaces" / f"{sid}.json", surface)
        write_json(run_root / "kernel_audits" / f"{sid}_audit.json", audit)
        surfaces.append(surface)
        audits.append(audit)
        if perf is not None:
            perf.log("surface_build_done", surface_id=sid, triple_index=tr.get("triple_index"), genus=surface.get("genus"), generator_count=surface.get("generator_count"), tile_count=surface.get("tile_count"), raw_nonidentity=surface.get("raw_nonidentity_schreier_generators"))
    return group, triples, surfaces, audits


def write_psl2q_report(run_root: Path, args: argparse.Namespace, surface_rows: List[Dict[str, Any]], atlas_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Fuchsian PSL(2,q) Hurwitz Trainer v1.0 Report")
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This run builds PSL(2,q) Hurwitz kernel surfaces, generates exact finite-word top-k atlases, and trains candidate-pool GINN rerankers. The finite quotient/tile scaffold may be complete while the kernel generator set may be selected rather than complete.")
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
    for k in ["q", "triple_equivalence", "max_triples", "kernel_generator_mode", "kernel_generator_limit", "depth", "pairs", "top_k_max", "train_pool_size", "epochs", "engine", "candidate_chunk_size", "pair_batch_size", "target_vram_mb", "stream_huge_word_ball"]:
        lines.append(f"- `{k}`: `{getattr(args, k, '')}`")
    lines.append("")
    lines.append("## Surface summary")
    lines.append("")
    if surface_rows:
        cols = ["surface_id", "q", "triple_index", "quotient_order", "genus", "kernel_generators_exported", "raw_nonidentity_schreier_generators", "tile_count", "tile_scaffold_complete", "atlas_training_ready"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in surface_rows:
            lines.append("| " + " | ".join(str(round(r.get(c), 5)) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Atlas summary")
    lines.append("")
    if atlas_rows:
        cols = ["surface_id", "word_ball_size_raw", "word_ball_size_unique", "n_pairs", "engine", "wall_seconds", "evals_per_second", "shortcut_fraction", "median_gap12"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in atlas_rows:
            lines.append("| " + " | ".join(str(round(r.get(c), 5)) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Training summary")
    lines.append("")
    if train_rows:
        cols = ["surface_id", "unique_word_ball_size", "pool_size", "train_pairs", "val_pairs", "test_pairs", "device", "epochs_ran", "test_recall_at_1", "test_recall_at_5", "test_recall_at_20", "test_top5_pruned_rmse", "train_seconds"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in train_rows:
            lines.append("| " + " | ".join(str(round(r.get(c), 5)) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Recall@1 is the correct-lift win rate relative to the exact atlas for this run's finite word ball. If the generator mode is selected/truncated, this is a selected-atlas result, not a complete-generator word-ball result.")
    (run_root / "report").mkdir(parents=True, exist_ok=True)
    (run_root / "report" / "psl2q_hurwitz_training_report.md").write_text("\n".join(lines), encoding="utf-8")
    # Compatibility name so older habits find it.
    (run_root / "report" / "big_hurwitz_training_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build PSL(2,q) Hurwitz selected-kernel atlases and train candidate-pool GINN rerankers.")
    ap.add_argument("--q", type=int, default=13, help="Prime q=p for PSL(2,p). Prime powers are not supported by this finite-field engine.")
    ap.add_argument("--triple-equivalence", choices=["inner", "pgl"], default="pgl")
    ap.add_argument("--max-triples", type=int, default=3)
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
    ap.add_argument("--allow-huge-word-ball", action="store_true", help="Bypass the pre-build word-ball estimate guard and allocate full raw word list.")
    ap.add_argument("--stream-huge-word-ball", action="store_true", help="For depth 2 only: stream reduced depth-2 words virtually. This remains experimental and uses local top-k dedupe rather than global geometric dedupe.")
    ap.add_argument("--virtual-topk-buffer", type=int, default=5000)
    ap.add_argument("--virtual-topk-dedupe-tol", type=float, default=0.0)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--dedupe-tol", type=float, default=1.0e-10)
    ap.add_argument("--alias-summary-rows", type=int, default=500)
    ap.add_argument("--alias-sample-limit", type=int, default=8)
    ap.add_argument("--pool-sizes", type=str, default="128,256,512,1024")
    ap.add_argument("--frequency-rows", type=int, default=200)
    ap.add_argument("--write-word-ball-summary", action="store_true")
    ap.add_argument("--outroot", type=str, default="psl2q_hurwitz_training_runs")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--zoo-script", type=str, default=default_zoo_script())
    ap.add_argument("--big-trainer-script", type=str, default=default_big_trainer_script())
    ap.add_argument("--hurwitz-script", type=str, default=default_hurwitz_script())
    ap.add_argument("--ginn-script", type=str, default=default_ginn_script())
    # New selected-kernel controls.
    ap.add_argument("--kernel-generator-mode", choices=["all", "first", "shortest"], default="shortest")
    ap.add_argument("--kernel-generator-limit", type=int, default=256, help="Number of selected kernel generators for first/shortest. Ignored for all.")
    ap.add_argument("--kernel-audit-sample-rows", type=int, default=200)
    ap.add_argument("--kernel-progress-every", type=int, default=10000)
    ap.add_argument("--tile-progress-every", type=int, default=25000)
    # Backward-compatible old cap; if supplied and new limit not supplied by user, it can still be used.
    ap.add_argument("--max-kernel-generators", type=int, default=0, help="Backward-compatible alias for --kernel-generator-limit with --kernel-generator-mode first if explicitly used.")
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--identity-tol", type=float, default=1.0e-9)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-perf-log", action="store_true")
    ap.add_argument("--skip-atlas", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    # Training options.
    ap.add_argument("--train-pool-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--min-epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=24)
    ap.add_argument("--min-delta", type=float, default=1.0e-5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--context-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--soft-distance-weight", type=float, default=0.25)
    ap.add_argument("--soft-distance-tau", type=float, default=0.50)
    ap.add_argument("--train-device", type=str, default="auto")
    ap.add_argument("--no-train-gpu", action="store_true")
    ap.add_argument("--cache-tensors-gpu", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--feature-batch-size", type=int, default=2048)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--no-shuffle-pool", action="store_true")
    ap.add_argument("--print-every", type=int, default=5)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.max_kernel_generators and args.kernel_generator_mode == "shortest" and args.kernel_generator_limit == 256:
        # Compatibility for old commands: --max-kernel-generators N used to mean first N.
        # Do not override if the user explicitly gave the new options.
        args.kernel_generator_mode = "first"
        args.kernel_generator_limit = int(args.max_kernel_generators)

    if args.mode == "smoke":
        args.pairs = int(args.smoke_pairs)
        args.top_k_max = min(int(args.top_k_max), 20)
        args.csv_top_k = min(int(args.csv_top_k), int(args.top_k_max))
        args.epochs = min(int(args.epochs), 4)
        args.min_epochs = min(int(args.min_epochs), 1)
        args.patience = min(int(args.patience), 3)
        args.train_pool_size = min(int(args.train_pool_size), 64)
        if args.kernel_generator_mode == "shortest":
            # For smoke, avoid scanning very large quotient generators unless requested.
            args.kernel_generator_mode = "first"
            args.kernel_generator_limit = min(int(args.kernel_generator_limit), 64)
        if not args.label:
            args.label = "smoke_train"

    pool_sizes = sorted(set([x for x in parse_int_list(args.pool_sizes) if x > 0] + [int(args.train_pool_size)]))
    args.pool_sizes = ",".join(str(x) for x in pool_sizes)
    if not args.label:
        args.label = f"psl2_{args.q}_kg{args.kernel_generator_mode}{args.kernel_generator_limit}_depth{args.depth}_pairs{args.pairs}"

    if not psl2_hurwitz_prime_condition(int(args.q)):
        print(f"[q-preflight warn] q={args.q} does not satisfy the prime PSL(2,p) Hurwitz condition p=7 or p ≡ ±1 mod 7. Triple search may fail.", flush=True)
    print(f"[q-preflight] q={args.q} PSL2_order={psl2_order(args.q)} genus={psl2_hurwitz_genus(args.q)}", flush=True)

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
    print(f"q={args.q} triples={args.max_triples} depth={args.depth} pairs={args.pairs} kernel_mode={args.kernel_generator_mode} kernel_limit={args.kernel_generator_limit} train_pool={args.train_pool_size}")
    print("-" * 78)

    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "args": vars(args).copy(),
        "python": sys.version,
        "platform": platform.platform(),
        "purpose": "PSL(2,q) selected/complete Schreier-kernel atlas generation plus candidate-pool GINN reranker training.",
    }
    write_json(run_root / "manifest.json", manifest)

    try:
        bt.require_torch()
        perf.log("module_load_start", hurwitz_script=args.hurwitz_script, ginn_script=args.ginn_script, zoo_script=args.zoo_script, big_trainer_script=args.big_trainer_script)
        hurwitz = zoo.load_module(args.hurwitz_script, "hurwitz_v16")
        ginn = zoo.load_module(args.ginn_script, "ginn_v24")
        install_leq_depth_word_ball_builder(ginn)
        perf.log("module_load_done")

        group, triples, surfaces, audits = build_surfaces_psl2q(args, hurwitz, run_root, perf=perf)
        surface_rows: List[Dict[str, Any]] = []
        for s, a in zip(surfaces, audits):
            surface_rows.append({
                "surface_id": s.get("surface_id"),
                "q": args.q,
                "triple_index": s.get("finite_group_triple", {}).get("triple_index"),
                "genus": s.get("genus"),
                "quotient_order": s.get("quotient_order"),
                "kernel_generators_exported": a.get("kernel_generators_exported"),
                "raw_nonidentity_schreier_generators": a.get("raw_nonidentity_schreier_generators"),
                "generator_count": s.get("generator_count"),
                "tile_count": s.get("tile_count"),
                "tile_scaffold_complete": a.get("tile_scaffold_complete"),
                "mainline_dataset_eligible": s.get("mainline_dataset_eligible"),
                "ginn_ready": s.get("ginn_ready"),
                "atlas_training_ready": s.get("atlas_training_ready"),
                "kernel_generator_export_complete": a.get("kernel_generator_export_complete"),
                "kernel_generator_scan_complete": a.get("kernel_generator_scan_complete"),
            })
        write_csv(run_root / "tables" / "psl2q_surface_summary.csv", surface_rows)
        write_csv(run_root / "tables" / "big_hurwitz_surface_summary.csv", surface_rows)

        atlas_results: List[Any] = []
        failure_rows: List[Dict[str, Any]] = []
        if not args.skip_atlas:
            old_mode = args.mode
            args.mode = "atlas" if old_mode == "train" else "smoke"
            for s in surfaces:
                sid = str(s.get("surface_id"))
                tridx = int(s.get("finite_group_triple", {}).get("triple_index", len(atlas_results)))
                try:
                    atlas_results.append(zoo.atlas_for_surface(args, ginn, s, run_root, tridx, perf=perf))
                except Exception as e:
                    print(f"[atlas fail] {sid}: {type(e).__name__}: {e}", flush=True)
                    failure_rows.append({"stage": "atlas", "surface_id": sid, "error_type": type(e).__name__, "error": str(e)})
                    continue
            args.mode = old_mode
        atlas_rows = [r.__dict__ for r in atlas_results]
        write_csv(run_root / "tables" / "psl2q_atlas_summary.csv", atlas_rows)
        write_csv(run_root / "tables" / "big_hurwitz_atlas_summary.csv", atlas_rows)

        if args.no_train:
            write_csv(run_root / "tables" / "failures.csv", failure_rows, ["stage", "surface_id", "error_type", "error"])
            write_psl2q_report(run_root, args, surface_rows, atlas_rows, [])
            summary = {"completed": datetime.now().isoformat(timespec="seconds"), "wall_seconds": time.perf_counter() - t_all, "surfaces_built": len(surfaces), "atlases_completed": len(atlas_results), "trained_surfaces": 0, "failures": len(failure_rows), "run_root": str(run_root), "process_peak_rss_mb": perf.peak_rss_mb}
            write_json(run_root / "run_summary.json", summary)
            perf.log("run_done", **summary)
            perf.write()
            return 0 if not failure_rows else 1

        train_results: List[Any] = []
        for s in surfaces:
            sid = str(s.get("surface_id"))
            if failure_rows and any(r.get("surface_id") == sid and r.get("stage") == "atlas" for r in failure_rows):
                continue
            try:
                train_results.append(bt.train_surface_reranker(args, zoo, ginn, s, run_root, perf=perf))
            except Exception as e:
                print(f"[train fail] {sid}: {type(e).__name__}: {e}", flush=True)
                failure_rows.append({"stage": "train", "surface_id": sid, "error_type": type(e).__name__, "error": str(e)})
                continue
        train_rows = [r.__dict__ for r in train_results]
        write_csv(run_root / "tables" / "psl2q_training_summary.csv", train_rows)
        write_csv(run_root / "tables" / "big_hurwitz_training_summary.csv", train_rows)
        write_csv(run_root / "tables" / "failures.csv", failure_rows, ["stage", "surface_id", "error_type", "error"])
        write_psl2q_report(run_root, args, surface_rows, atlas_rows, train_rows)
        summary = {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": time.perf_counter() - t_all,
            "surfaces_built": len(surfaces),
            "atlases_completed": len(atlas_results),
            "trained_surfaces": len(train_results),
            "failures": len(failure_rows),
            "run_root": str(run_root),
            "process_peak_rss_mb": perf.peak_rss_mb,
        }
        write_json(run_root / "run_summary.json", summary)
        perf.log("run_done", surfaces_built=len(surfaces), atlases_completed=len(atlas_results), trained_surfaces=len(train_results), failures=len(failure_rows))
        perf.write()
        print("=" * 78)
        print(f"[done] surfaces={len(surfaces)} atlases={len(atlas_results)} trained={len(train_results)} failures={len(failure_rows)}")
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
