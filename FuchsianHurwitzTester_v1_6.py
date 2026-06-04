#!/usr/bin/env python3
"""
FuchsianHurwitzTester_v1_6.py

Terminal tester for Hurwitz quotients of the orientation-preserving triangle group

    Delta^+(2,3,7) = < x,y,z | x^2 = y^3 = z^7 = xyz = 1 >.

The program currently focuses on finite quotients PSL(2,p) over prime fields F_p.
For a finite group G, a generating triple (x,y,z) of exact orders (2,3,7)
with xyz=1 gives a quotient Delta^+(2,3,7) -> G.  The kernel is torsion-free
by the standard triangle-group torsion theorem, so the quotient surface is a
Hurwitz surface of genus

    g = 1 + |G|/84.

This is a finite-group certificate engine first.  It exports abstract Hurwitz
surface certificates and triple audits.  It can optionally construct explicit PSU(1,1) triangle generators and Reidemeister-Schreier kernel generators for certified triples, exporting preliminary GINN-loadable Hurwitz surface JSON files. v1.6 also adds an optional compatibility bridge to FuchsianDownstairsGINN_v2_4.py: it can build a word ball, sample the triangle-tile scaffold, generate finite-search pair labels, and optionally run a small branch-ranker training smoke test. The finite quotient certificate remains the primary truth anchor.

No hidden q cap is imposed.  Runtime/memory are controlled only by explicit CLI
choices such as --max-triples, --random-trials, --dedupe-conjugacy-max-order.

Examples
--------
  python FuchsianHurwitzTester_v1_6.py --q 5 7 11 13
  python FuchsianHurwitzTester_v1_6.py --scan-primes 5 50 --max-triples 3
  python FuchsianHurwitzTester_v1_6.py --q 29 --random-trials 20000 --max-triples 5

Notes
-----
- PSL(2,p) is represented as SL(2,p)/{+/-I}.  Each matrix is canonicalized by
  choosing the lexicographically smaller of M and -M.
- v1.6 supports prime p only.  Prime powers q=p^n require finite-field
  arithmetic and are intentionally reported as unsupported rather than faked.
"""

from __future__ import annotations

import argparse
import cmath
import csv
import json
import math
import importlib.util
import os
import random
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from FuchsianSurfaceRecordTools_v1_0 import (
    GEOMETRY_AUDIT_FIELDS,
    GINN_SMOKE_FIELDS,
    normalize_surface_record,
    write_csv as write_contract_csv,
    write_json as write_contract_json,
)

Matrix = Tuple[int, int, int, int]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def primes_between(a: int, b: int) -> List[int]:
    lo, hi = min(a, b), max(a, b)
    return [n for n in range(lo, hi + 1) if is_prime(n)]


def factor_int(n: int) -> Dict[int, int]:
    d: Dict[int, int] = {}
    x = n
    f = 2
    while f * f <= x:
        while x % f == 0:
            d[f] = d.get(f, 0) + 1
            x //= f
        f += 1 if f == 2 else 2
    if x > 1:
        d[x] = d.get(x, 0) + 1
    return d


def inv_mod(a: int, p: int) -> int:
    return pow(a % p, p - 2, p)


def neg_mat(m: Matrix, p: int) -> Matrix:
    return tuple(((-x) % p) for x in m)  # type: ignore


def canon(m: Matrix, p: int) -> Matrix:
    m = tuple((x % p) for x in m)  # type: ignore
    nm = neg_mat(m, p)
    return min(m, nm)


def mat_mul(a: Matrix, b: Matrix, p: int) -> Matrix:
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return canon(((a0*b0 + a1*b2) % p, (a0*b1 + a1*b3) % p,
                  (a2*b0 + a3*b2) % p, (a2*b1 + a3*b3) % p), p)


def mat_inv(a: Matrix, p: int) -> Matrix:
    a0, a1, a2, a3 = a
    # determinant is 1 in SL(2,p); inverse is [[d,-b],[-c,a]]
    return canon((a3, -a1, -a2, a0), p)


def mat_trace(a: Matrix, p: int) -> int:
    return (a[0] + a[3]) % p


def trace_square(a: Matrix, p: int) -> int:
    t = mat_trace(a, p)
    return (t * t) % p


@dataclass
class PSL2PrimeGroup:
    p: int
    elements: List[Matrix]
    element_set: Set[Matrix]
    identity: Matrix
    order: int
    order_factors: Dict[int, int]
    element_orders: Dict[Matrix, int]
    order_classes: Dict[int, List[Matrix]]

    @classmethod
    def build(cls, p: int, verbose: bool = False) -> "PSL2PrimeGroup":
        if p == 2:
            raise ValueError("v1.6 expects odd prime p for PSL(2,p); p=2 is exceptional.")
        if not is_prime(p):
            raise ValueError(f"q={p} is not prime. v1.6 supports prime fields only.")
        t0 = time.perf_counter()
        elems: Set[Matrix] = set()
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    if a % p != 0:
                        d = ((1 + b * c) * inv_mod(a, p)) % p
                        elems.add(canon((a, b, c, d), p))
                    else:
                        # Need -b*c = 1 mod p; d is free.
                        if (-b * c - 1) % p == 0:
                            for d in range(p):
                                elems.add(canon((a, b, c, d), p))
        elements = sorted(elems)
        expected = p * (p*p - 1) // 2
        if len(elements) != expected:
            raise RuntimeError(f"PSL(2,{p}) enumeration mismatch: got {len(elements)}, expected {expected}")
        identity = canon((1, 0, 0, 1), p)
        group = cls(
            p=p,
            elements=elements,
            element_set=set(elements),
            identity=identity,
            order=len(elements),
            order_factors=factor_int(len(elements)),
            element_orders={},
            order_classes={},
        )
        group.compute_element_orders(verbose=verbose)
        if verbose:
            print(f"[group] PSL(2,{p}) order={group.order} built in {time.perf_counter()-t0:.2f}s")
        return group

    def pow(self, a: Matrix, n: int) -> Matrix:
        result = self.identity
        base = a
        k = n
        while k > 0:
            if k & 1:
                result = mat_mul(result, base, self.p)
            base = mat_mul(base, base, self.p)
            k >>= 1
        return result

    def order_of(self, a: Matrix) -> int:
        if a in self.element_orders:
            return self.element_orders[a]
        n = self.order
        r = n
        for prime, exp in self.order_factors.items():
            for _ in range(exp):
                if r % prime == 0 and self.pow(a, r // prime) == self.identity:
                    r //= prime
                else:
                    break
        self.element_orders[a] = r
        return r

    def compute_element_orders(self, verbose: bool = False) -> None:
        self.element_orders = {}
        self.order_classes = {}
        for i, e in enumerate(self.elements):
            o = self.order_of(e)
            self.order_classes.setdefault(o, []).append(e)
            if verbose and (i + 1) % max(1, len(self.elements)//10) == 0:
                print(f"[orders] {i+1}/{len(self.elements)}")

    def generated_subgroup_order(self, gens: Sequence[Matrix], stop_at_full: bool = True) -> int:
        gen_list = []
        for g in gens:
            gen_list.append(g)
            inv = mat_inv(g, self.p)
            if inv not in gen_list:
                gen_list.append(inv)
        seen: Set[Matrix] = {self.identity}
        q: deque[Matrix] = deque([self.identity])
        while q:
            h = q.popleft()
            for g in gen_list:
                hg = mat_mul(h, g, self.p)
                if hg not in seen:
                    seen.add(hg)
                    if stop_at_full and len(seen) == self.order:
                        return len(seen)
                    q.append(hg)
        return len(seen)

    def conjugate(self, g: Matrix, x: Matrix) -> Matrix:
        return mat_mul(mat_mul(g, x, self.p), mat_inv(g, self.p), self.p)


def matrix_to_list(m: Matrix) -> List[List[int]]:
    return [[m[0], m[1]], [m[2], m[3]]]


def triangle_signature_for_triple(group: PSL2PrimeGroup, x: Matrix, y: Matrix, z: Matrix) -> Dict[str, int]:
    # A cheap invariant useful for grouping, not a complete isomorphism/triple equivalence test.
    return {
        "tr2_x": trace_square(x, group.p),
        "tr2_y": trace_square(y, group.p),
        "tr2_z": trace_square(z, group.p),
        "tr2_xy": trace_square(mat_mul(x, y, group.p), group.p),
        "tr2_xz": trace_square(mat_mul(x, z, group.p), group.p),
        "tr2_yz": trace_square(mat_mul(y, z, group.p), group.p),
    }


def conjugacy_canonical_signature(group: PSL2PrimeGroup, x: Matrix, y: Matrix, z: Matrix) -> str:
    triples = []
    for g in group.elements:
        cx = group.conjugate(g, x)
        cy = group.conjugate(g, y)
        cz = group.conjugate(g, z)
        triples.append((cx, cy, cz))
    return repr(min(triples))


def mat_mul_raw(a: Matrix, b: Matrix, p: int) -> Matrix:
    """Raw 2x2 product mod p, without PSL +/- canonicalization."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return ((a0*b0 + a1*b2) % p, (a0*b1 + a1*b3) % p,
            (a2*b0 + a3*b2) % p, (a2*b1 + a3*b3) % p)


def det_raw(a: Matrix, p: int) -> int:
    return (a[0]*a[3] - a[1]*a[2]) % p


def gl_inv_raw(a: Matrix, p: int) -> Matrix:
    d = det_raw(a, p)
    if d == 0:
        raise ZeroDivisionError("singular GL/PGL matrix")
    dinv = inv_mod(d, p)
    return ((a[3]*dinv) % p, (-a[1]*dinv) % p,
            (-a[2]*dinv) % p, (a[0]*dinv) % p)


def projective_canon_gl(a: Matrix, p: int) -> Matrix:
    """Canonical representative of a nonzero projective GL(2,p) matrix."""
    vals = tuple(x % p for x in a)
    first = None
    for x in vals:
        if x % p != 0:
            first = x % p
            break
    if first is None:
        raise ValueError("zero matrix has no projective class")
    scale = inv_mod(first, p)
    return tuple((x * scale) % p for x in vals)  # type: ignore


def pgl_representatives(p: int) -> List[Matrix]:
    """Representatives for PGL(2,p)=GL(2,p)/F_p^*.  Size p(p^2-1)."""
    reps: Set[Matrix] = set()
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    m = (a, b, c, d)
                    if det_raw(m, p) != 0:
                        reps.add(projective_canon_gl(m, p))
    return sorted(reps)


def pgl_conjugate(pgl_g: Matrix, x: Matrix, p: int) -> Matrix:
    """Conjugate PSL element x by a PGL representative, then project to PSL."""
    hxh = mat_mul_raw(mat_mul_raw(pgl_g, x, p), gl_inv_raw(pgl_g, p), p)
    return canon(hxh, p)


def triple_canonical_signature_with_conjugators(
    group: PSL2PrimeGroup,
    x: Matrix,
    y: Matrix,
    z: Matrix,
    conjugators: Sequence[Matrix],
    mode: str = "inner",
) -> str:
    triples = []
    if mode == "pgl":
        for g in conjugators:
            triples.append((pgl_conjugate(g, x, group.p), pgl_conjugate(g, y, group.p), pgl_conjugate(g, z, group.p)))
    else:
        for g in conjugators:
            triples.append((group.conjugate(g, x), group.conjugate(g, y), group.conjugate(g, z)))
    return repr(min(triples))


def centralizer(group: PSL2PrimeGroup, x: Matrix) -> List[Matrix]:
    return [g for g in group.elements if group.conjugate(g, x) == x]


def conjugacy_class_reps(elements: Sequence[Matrix], conjugators: Sequence[Matrix], group: PSL2PrimeGroup) -> Tuple[List[Matrix], Dict[Matrix, int]]:
    """Return representatives and an element->class-index map under supplied conjugators."""
    remaining: Set[Matrix] = set(elements)
    reps: List[Matrix] = []
    class_map: Dict[Matrix, int] = {}
    while remaining:
        rep = min(remaining)
        idx = len(reps)
        reps.append(rep)
        orbit = {group.conjugate(g, rep) for g in conjugators}
        for h in orbit:
            class_map[h] = idx
        remaining.difference_update(orbit)
    return reps, class_map


def orbit_reps_under_centralizer(elements: Sequence[Matrix], centralizer_elems: Sequence[Matrix], group: PSL2PrimeGroup) -> Tuple[List[Matrix], Dict[Matrix, int]]:
    """Representatives for the action y -> c y c^-1, c in C_G(x)."""
    remaining: Set[Matrix] = set(elements)
    reps: List[Matrix] = []
    orbit_sizes: Dict[Matrix, int] = {}
    while remaining:
        rep = min(remaining)
        orbit = {group.conjugate(c, rep) for c in centralizer_elems}
        reps.append(rep)
        orbit_sizes[rep] = len(orbit)
        remaining.difference_update(orbit)
    return reps, orbit_sizes


def find_hurwitz_triples_raw(
    group: PSL2PrimeGroup,
    max_triples: int = 10,
    random_trials: int = 0,
    seed: int = 12345,
    dedupe_conjugacy_max_order: int = 2500,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Legacy v1.5 raw pair search.  Kept for comparison/debugging."""
    rng = random.Random(seed)
    order2 = group.order_classes.get(2, [])
    order3 = group.order_classes.get(3, [])
    order7 = group.order_classes.get(7, [])
    stats = {
        "order2_count": len(order2),
        "order3_count": len(order3),
        "order7_count": len(order7),
        "pair_tests": 0,
        "relation_candidates": 0,
        "surjective_candidates": 0,
        "dedupe_mode": "conjugacy" if group.order <= dedupe_conjugacy_max_order else "trace_signature_plus_exact",
        "triple_search_mode": "raw",
        "triple_equivalence": "inner_if_small_else_raw_signature",
    }
    triples: List[Dict] = []
    seen_keys: Set[str] = set()

    if not order2 or not order3 or not order7:
        return triples, stats

    def add_triple(x: Matrix, y: Matrix) -> bool:
        stats["pair_tests"] += 1
        xy = mat_mul(x, y, group.p)
        if group.order_of(xy) != 7:
            return False
        z = mat_inv(xy, group.p)
        stats["relation_candidates"] += 1
        gen_order = group.generated_subgroup_order([x, y], stop_at_full=True)
        if gen_order != group.order:
            return False
        stats["surjective_candidates"] += 1
        if group.order <= dedupe_conjugacy_max_order:
            key = conjugacy_canonical_signature(group, x, y, z)
        else:
            sig = triangle_signature_for_triple(group, x, y, z)
            key = json.dumps({"x": x, "y": y, "z": z, "sig": sig}, sort_keys=True)
        if key in seen_keys:
            return False
        seen_keys.add(key)
        triples.append(make_triple_record(group, x, y, z, len(triples), equivalence_key=key))
        if verbose:
            print(f"[triple/raw] found #{len(triples)} for PSL(2,{group.p})")
        return max_triples > 0 and len(triples) >= max_triples

    if random_trials and random_trials > 0:
        for i in range(random_trials):
            if add_triple(rng.choice(order2), rng.choice(order3)):
                break
            if verbose and (i + 1) % max(1, random_trials // 10) == 0:
                print(f"[search/raw] random {i+1}/{random_trials}; found={len(triples)}")
    else:
        total = len(order2) * len(order3)
        k = 0
        for x in order2:
            for y in order3:
                k += 1
                if add_triple(x, y):
                    return triples, stats
                if verbose and total >= 10 and k % max(1, total // 10) == 0:
                    print(f"[search/raw] pairs {k}/{total}; candidates={stats['relation_candidates']} found={len(triples)}")
    return triples, stats


def make_triple_record(
    group: PSL2PrimeGroup,
    x: Matrix,
    y: Matrix,
    z: Matrix,
    triple_index: int,
    equivalence_key: str = "",
    x_conjugacy_class: Optional[int] = None,
    z_order7_class: Optional[int] = None,
    y_centralizer_orbit_size: Optional[int] = None,
) -> Dict:
    xy = mat_mul(x, y, group.p)
    gen_order = group.generated_subgroup_order([x, y], stop_at_full=True)
    genus = 1 + group.order // 84 if group.order % 84 == 0 else None
    rec = {
        "triple_index": triple_index,
        "x": matrix_to_list(x),
        "y": matrix_to_list(y),
        "z": matrix_to_list(z),
        "x_order": group.order_of(x),
        "y_order": group.order_of(y),
        "z_order": group.order_of(z),
        "xy_order": group.order_of(xy),
        "xyz_identity": mat_mul(mat_mul(x, y, group.p), z, group.p) == group.identity,
        "generated_subgroup_order": gen_order,
        "group_order": group.order,
        "surjective": gen_order == group.order,
        "genus": genus,
        "trace_square_signature": triangle_signature_for_triple(group, x, y, z),
        "equivalence_key": equivalence_key,
    }
    if x_conjugacy_class is not None:
        rec["x_conjugacy_class"] = x_conjugacy_class
    if z_order7_class is not None:
        rec["z_order7_class"] = z_order7_class
    if y_centralizer_orbit_size is not None:
        rec["y_centralizer_orbit_size"] = y_centralizer_orbit_size
    return rec


def find_hurwitz_triples_conjugacy_reduced(
    group: PSL2PrimeGroup,
    max_triples: int = 10,
    seed: int = 12345,
    equivalence: str = "pgl",
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Conjugacy/orbit-reduced (2,3,7) generating triple search.

    This is the v1.6 production search.  It uses the standard reduction for
    simultaneous conjugacy:

      1. choose representatives for conjugacy classes of order-2 elements;
      2. fix such an x;
      3. let the centralizer C_G(x) act on order-3 elements and test only orbit
         representatives y;
      4. form z=(xy)^-1, require order(z)=7 and <x,y>=G;
      5. finally deduplicate candidate triples by exact simultaneous conjugacy
         under either PSL (inner) or PGL (the natural prime-field automorphism
         enlargement for PSL(2,p)).

    The PGL option is useful because the outer diagonal automorphism of PSL(2,p)
    can identify generating vectors that are not PSL-inner conjugate.  It is not
    a proof of full Riemann-surface isomorphism under all source-group/Nielsen
    equivalences, but it is a much better production representative filter than
    the v1.5 raw search.
    """
    order2 = group.order_classes.get(2, [])
    order3 = group.order_classes.get(3, [])
    order7 = group.order_classes.get(7, [])
    stats = {
        "order2_count": len(order2),
        "order3_count": len(order3),
        "order7_count": len(order7),
        "pair_tests": 0,
        "relation_candidates": 0,
        "surjective_candidates": 0,
        "reduced_candidates_before_equivalence": 0,
        "equivalence_duplicates_removed": 0,
        "triple_search_mode": "conjugacy_orbit",
        "triple_equivalence": equivalence,
        "dedupe_mode": f"exact_{equivalence}_simultaneous_conjugacy_after_centralizer_orbit_presort",
    }
    triples: List[Dict] = []
    seen_keys: Set[str] = set()

    if not order2 or not order3 or not order7:
        return triples, stats

    x_reps, x_class_map = conjugacy_class_reps(order2, group.elements, group)
    z_reps, z_class_map = conjugacy_class_reps(order7, group.elements, group)
    stats["order2_conjugacy_class_count"] = len(x_reps)
    stats["order7_conjugacy_class_count"] = len(z_reps)

    if equivalence == "pgl":
        conjugators = pgl_representatives(group.p)
        stats["conjugator_count"] = len(conjugators)
    elif equivalence == "inner":
        conjugators = group.elements
        stats["conjugator_count"] = len(conjugators)
    else:
        raise ValueError(f"Unknown triple equivalence mode {equivalence!r}; use 'inner' or 'pgl'.")

    for x in x_reps:
        Cx = centralizer(group, x)
        y_reps, y_orbit_sizes = orbit_reps_under_centralizer(order3, Cx, group)
        stats.setdefault("centralizer_sizes", []).append(len(Cx))
        stats.setdefault("order3_centralizer_orbit_counts", []).append(len(y_reps))
        if verbose:
            print(f"[presort] x_class={x_class_map.get(x,0)} |C(x)|={len(Cx)} order3_orbits={len(y_reps)}")
        for y in y_reps:
            stats["pair_tests"] += 1
            xy = mat_mul(x, y, group.p)
            if group.order_of(xy) != 7:
                continue
            z = mat_inv(xy, group.p)
            stats["relation_candidates"] += 1
            gen_order = group.generated_subgroup_order([x, y], stop_at_full=True)
            if gen_order != group.order:
                continue
            stats["surjective_candidates"] += 1
            stats["reduced_candidates_before_equivalence"] += 1
            key = triple_canonical_signature_with_conjugators(group, x, y, z, conjugators, mode=equivalence)
            if key in seen_keys:
                stats["equivalence_duplicates_removed"] += 1
                continue
            seen_keys.add(key)
            rec = make_triple_record(
                group, x, y, z, len(triples), equivalence_key=key,
                x_conjugacy_class=x_class_map.get(x),
                z_order7_class=z_class_map.get(z),
                y_centralizer_orbit_size=y_orbit_sizes.get(y),
            )
            triples.append(rec)
            if verbose:
                print(f"[triple/reduced] found #{len(triples)} z_class={rec.get('z_order7_class')} for PSL(2,{group.p})")
            if max_triples > 0 and len(triples) >= max_triples:
                return triples, stats
    return triples, stats


def find_hurwitz_triples(
    group: PSL2PrimeGroup,
    max_triples: int = 10,
    random_trials: int = 0,
    seed: int = 12345,
    dedupe_conjugacy_max_order: int = 2500,
    triple_search: str = "conjugacy_orbit",
    triple_equivalence: str = "inner",
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    if triple_search == "raw":
        return find_hurwitz_triples_raw(
            group,
            max_triples=max_triples,
            random_trials=random_trials,
            seed=seed,
            dedupe_conjugacy_max_order=dedupe_conjugacy_max_order,
            verbose=verbose,
        )
    if random_trials and random_trials > 0:
        raise ValueError("--random-trials is only supported with --triple-search raw in v1.6.")
    return find_hurwitz_triples_conjugacy_reduced(
        group,
        max_triples=max_triples,
        seed=seed,
        equivalence=triple_equivalence,
        verbose=verbose,
    )



def group_audit(group: PSL2PrimeGroup) -> Dict:
    hist = {str(k): len(v) for k, v in sorted(group.order_classes.items())}
    order_divisible_by_84 = (group.order % 84 == 0)
    possible_genus = 1 + group.order // 84 if order_divisible_by_84 else None
    return {
        "group_family": "PSL(2,p)",
        "p": group.p,
        "q": group.p,
        "field": f"F_{group.p}",
        "group_order": group.order,
        "order_formula": "p(p^2-1)/2 for odd prime p",
        "order_factorization": {str(k): v for k, v in group.order_factors.items()},
        "element_order_histogram": hist,
        "has_order_2": bool(group.order_classes.get(2)),
        "has_order_3": bool(group.order_classes.get(3)),
        "has_order_7": bool(group.order_classes.get(7)),
        "order_divisible_by_84": order_divisible_by_84,
        "genus_if_hurwitz": possible_genus,
        "immediate_rejection_reasons": [
            reason for reason, flag in [
                ("no elements of order 2", not bool(group.order_classes.get(2))),
                ("no elements of order 3", not bool(group.order_classes.get(3))),
                ("no elements of order 7", not bool(group.order_classes.get(7))),
                ("group order not divisible by 84", not order_divisible_by_84),
            ] if flag
        ],
    }



# -----------------------------------------------------------------------------
# PSU(1,1) model of Delta^+(2,3,7) and Reidemeister-Schreier bridge
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SU11Mobius:
    """Orientation-preserving disk isometry represented by an SU(1,1) matrix.

        [[alpha, beta], [conj(beta), conj(alpha)]]

    acting by z -> (alpha z + beta)/(conj(beta) z + conj(alpha)).
    """
    alpha: complex
    beta: complex
    label: str = ""

    def normalized(self, label: Optional[str] = None) -> "SU11Mobius":
        det = abs(self.alpha)**2 - abs(self.beta)**2
        if det <= 0 or not math.isfinite(det):
            return self
        scale = 1.0 / math.sqrt(det)
        a = self.alpha * scale
        b = self.beta * scale
        # Fix the harmless central sign for stable JSON output.
        if a.real < -1.0e-14 or (abs(a.real) <= 1.0e-14 and a.imag < 0):
            a, b = -a, -b
        return SU11Mobius(a, b, self.label if label is None else label)

    def compose(self, other: "SU11Mobius", label: Optional[str] = None) -> "SU11Mobius":
        # self after other, matching matrix product self * other.
        a = self.alpha * other.alpha + self.beta * other.beta.conjugate()
        b = self.alpha * other.beta + self.beta * other.alpha.conjugate()
        return SU11Mobius(a, b, self.label + other.label if label is None else label).normalized()

    def inverse(self, label: Optional[str] = None) -> "SU11Mobius":
        return SU11Mobius(self.alpha.conjugate(), -self.beta, self.label + "^-1" if label is None else label).normalized()

    def apply(self, z: complex) -> complex:
        den = self.beta.conjugate() * z + self.alpha.conjugate()
        return (self.alpha * z + self.beta) / den

    def trace_real(self) -> float:
        return float(2.0 * self.alpha.real)

    def displacement_from_identity(self) -> float:
        return math.hypot(self.alpha.real - 1.0, self.alpha.imag) + abs(self.beta)

    def as_json(self) -> Dict:
        return {
            "type": "su11",
            "alpha": [float(self.alpha.real), float(self.alpha.imag)],
            "beta": [float(self.beta.real), float(self.beta.imag)],
        }


def su11_identity(label: str = "I") -> SU11Mobius:
    return SU11Mobius(1.0 + 0.0j, 0.0 + 0.0j, label)


def su11_rotation_about_zero(theta: float, label: str) -> SU11Mobius:
    # Matrix diag(e^{i theta/2}, e^{-i theta/2}) acts as z -> e^{i theta} z.
    return SU11Mobius(cmath.exp(0.5j * theta), 0.0j, label).normalized()


def su11_real_translation_to(r: float, label: str = "T") -> SU11Mobius:
    if not (abs(r) < 1.0):
        raise ValueError("Disk translation parameter must satisfy |r| < 1.")
    a = 1.0 / math.sqrt(1.0 - r*r)
    return SU11Mobius(a + 0.0j, r * a + 0.0j, label).normalized()


def su11_rotation_about_real_point(r: float, theta: float, label: str) -> SU11Mobius:
    T = su11_real_translation_to(r, "T")
    R = su11_rotation_about_zero(theta, "R")
    return T.compose(R).compose(T.inverse(), label=label).normalized(label)


def build_delta_237_su11() -> Tuple[Dict[str, SU11Mobius], Dict]:
    """Construct a concrete PSU(1,1) representation of Delta^+(2,3,7).

    X is a half-turn about 0.  Y is a 2*pi/3 rotation about a point r on the
    real axis.  The parameter r is solved so that X*Y has elliptic trace with
    order 7.  Then Z=(XY)^-1.
    """
    target = 2.0 * math.cos(math.pi / 7.0)
    X = su11_rotation_about_zero(math.pi, "X")

    def val(r: float) -> float:
        Y = su11_rotation_about_real_point(r, 2.0 * math.pi / 3.0, "Y")
        return abs(X.compose(Y).trace_real())

    lo, hi = 0.0, 0.95
    # Find a bracket.  For the intended triangle, the small root occurs early.
    prev = val(lo)
    found = False
    grid = [i / 1000.0 for i in range(1, 950)]
    a = lo
    for b in grid:
        vb = val(b)
        if (prev - target) * (vb - target) <= 0:
            lo, hi = a, b
            found = True
            break
        a, prev = b, vb
    if not found:
        raise RuntimeError("Could not bracket the Delta(2,3,7) SU(1,1) parameter.")
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if (val(lo) - target) * (val(mid) - target) <= 0:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    Y = su11_rotation_about_real_point(r, 2.0 * math.pi / 3.0, "Y")
    XY = X.compose(Y, label="XY")
    Z = XY.inverse(label="Z")
    gens = {"X": X, "Y": Y, "Y^-1": Y.inverse("Y^-1")}
    audit = {
        "representation": "PSU(1,1) concrete Delta^+(2,3,7) representation",
        "X_rotation_angle": math.pi,
        "Y_rotation_angle": 2.0 * math.pi / 3.0,
        "Z_rotation_angle_expected_abs": 2.0 * math.pi / 7.0,
        "Y_fixed_point_real_r": r,
        "hyperbolic_distance_between_XY_centers": 2.0 * math.atanh(r),
        "target_abs_trace_order7": target,
        "abs_trace_XY": abs(XY.trace_real()),
        "trace_error": abs(abs(XY.trace_real()) - target),
        "relations_numerical": {
            "X2_identity_error": X.compose(X).displacement_from_identity(),
            "Y3_identity_error": Y.compose(Y).compose(Y).displacement_from_identity(),
            "XY_order7_identity_error": (XY.compose(XY).compose(XY).compose(XY).compose(XY).compose(XY).compose(XY)).displacement_from_identity(),
        },
        "X_fixed_point": [0.0, 0.0],
        "Y_fixed_point": [r, 0.0],
    }
    # fixed point of Z, for sampling triangle vertex.
    fpz = fixed_point_inside_disk(Z)
    audit["Z_fixed_point"] = [float(fpz.real), float(fpz.imag)]
    audit["base_orbifold_triangle_vertices"] = [[0.0, 0.0], [float(r), 0.0], [float(fpz.real), float(fpz.imag)]]
    return gens, audit


def fixed_point_inside_disk(M: SU11Mobius) -> complex:
    # beta_bar z^2 + (alpha_bar-alpha) z - beta = 0
    A = M.beta.conjugate()
    B = M.alpha.conjugate() - M.alpha
    C = -M.beta
    if abs(A) < 1.0e-14:
        return 0.0j
    disc = B*B - 4*A*C
    roots = [(-B + cmath.sqrt(disc))/(2*A), (-B - cmath.sqrt(disc))/(2*A)]
    inside = sorted(roots, key=lambda z: abs(z))
    return inside[0]


def finite_from_json_matrix(m: List[List[int]]) -> Matrix:
    return (int(m[0][0]), int(m[0][1]), int(m[1][0]), int(m[1][1]))


def tokens_to_word(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def invert_tokens(tokens: Sequence[str]) -> List[str]:
    inv = []
    for t in reversed(tokens):
        inv.append("Y^-1" if t == "Y" else "Y" if t == "Y^-1" else "X")
    return inv


def mobius_from_tokens(tokens: Sequence[str], delta_gens: Dict[str, SU11Mobius]) -> SU11Mobius:
    M = su11_identity()
    for tok in tokens:
        M = M.compose(delta_gens[tok])
    return M.normalized(tokens_to_word(tokens))


def build_schreier_kernel_surface(group: PSL2PrimeGroup, triple: Dict, args, run_id: str) -> Tuple[Dict, Dict]:
    """Build a GINN-loadable Hurwitz kernel surface from one PSL(2,p) triple.

    A saved kernel surface is considered ``mainline_dataset_eligible`` only when
    the Reidemeister-Schreier generator export is complete and the triangle-tile
    scaffold includes all quotient cosets.  Explicit CLI caps are still allowed
    for experimentation, but capped records are labeled partial and excluded
    from the production master dataset.
    """
    delta_gens, delta_audit = build_delta_237_su11()
    Xf = finite_from_json_matrix(triple["x"])
    Yf = finite_from_json_matrix(triple["y"])
    finite_gens = {"X": Xf, "Y": Yf, "Y^-1": mat_inv(Yf, group.p)}

    # Schreier transversal by BFS in the quotient group G using X,Y,Y^-1.
    rep_words: Dict[Matrix, List[str]] = {group.identity: []}
    q: deque[Matrix] = deque([group.identity])
    while q:
        h = q.popleft()
        for tok in ["X", "Y", "Y^-1"]:
            nh = mat_mul(h, finite_gens[tok], group.p)
            if nh not in rep_words:
                rep_words[nh] = rep_words[h] + [tok]
                q.append(nh)
    if len(rep_words) != group.order:
        raise RuntimeError(f"Schreier transversal incomplete: {len(rep_words)} of {group.order}")

    # Build kernel generators k_{t,s}=t*s*rep(ts)^-1.
    raw = []
    labels_seen: Set[str] = set()
    gen_json: Dict[str, Dict] = {}
    meanings: Dict[str, str] = {}
    idx = 0
    identity_like = 0
    nonidentity_image_failures = 0
    for h, w in rep_words.items():
        for tok in ["X", "Y", "Y^-1"]:
            hs = mat_mul(h, finite_gens[tok], group.p)
            rw = rep_words[hs]
            kw = w + [tok] + invert_tokens(rw)
            # finite audit
            fh = group.identity
            for kt in kw:
                fh = mat_mul(fh, finite_gens[kt], group.p)
            if fh != group.identity:
                nonidentity_image_failures += 1
                continue
            KM = mobius_from_tokens(kw, delta_gens).normalized()
            if KM.displacement_from_identity() < args.identity_tol:
                identity_like += 1
                continue
            label = f"h{idx:04d}"
            idx += 1
            if label in labels_seen:
                raise RuntimeError("internal label collision")
            labels_seen.add(label)
            gen_json[label] = KM.as_json()
            meanings[label] = f"Schreier kernel generator {label}: {tokens_to_word(kw)}"
            raw.append({
                "label": label,
                "word_tokens": kw,
                "word": tokens_to_word(kw),
                "trace_real": KM.trace_real(),
                "is_hyperbolic_by_trace": abs(KM.trace_real()) > 2.0 + 1.0e-8,
                "finite_image_identity": True,
            })
            if args.max_kernel_generators and len(raw) >= args.max_kernel_generators:
                break
        if args.max_kernel_generators and len(raw) >= args.max_kernel_generators:
            break

    # Fundamental-domain sampling scaffold: images of the base triangle by coset reps.
    base_verts = [complex(x, y) for x, y in delta_audit["base_orbifold_triangle_vertices"]]
    tiles = []
    max_tiles = args.max_tiles if args.max_tiles > 0 else group.order
    for tile_idx, (h, w) in enumerate(list(rep_words.items())[:max_tiles]):
        M = mobius_from_tokens(w, delta_gens)
        verts = [M.apply(z) for z in base_verts]
        tiles.append({
            "tile_index": tile_idx,
            "coset_matrix": matrix_to_list(h),
            "coset_word": tokens_to_word(w),
            "vertices": [[float(z.real), float(z.imag)] for z in verts],
        })

    generator_export_complete = not bool(args.max_kernel_generators and len(raw) >= args.max_kernel_generators)
    tile_scaffold_complete = len(tiles) == group.order
    relation_max_error = max(delta_audit["relations_numerical"].values())
    pass_geometry_audit = (
        generator_export_complete
        and tile_scaffold_complete
        and nonidentity_image_failures == 0
        and relation_max_error < 1.0e-8
        and len(gen_json) > 0
        and triple.get("surjective") is True
    )
    partial_reasons = []
    if not generator_export_complete:
        partial_reasons.append("kernel generator export truncated by --max-kernel-generators")
    if not tile_scaffold_complete:
        partial_reasons.append("triangle-tile scaffold truncated by --max-tiles")
    if nonidentity_image_failures:
        partial_reasons.append("some Schreier generators did not map to identity in the finite quotient")
    if relation_max_error >= 1.0e-8:
        partial_reasons.append("PSU(1,1) triangle relation numerical error above tolerance")
    if not gen_json:
        partial_reasons.append("no nonidentity kernel generators exported")
    exclusion_reason = "; ".join(partial_reasons)

    genus = triple["genus"]
    surf_id = f"hurwitz_PSL2_{group.p}_triple_{triple['triple_index']:04d}_kernel"
    surface = {
        "format": "FuchsianGENN surface JSON v1.6 hurwitz-tokenized-kernel",
        "surface_id": surf_id,
        "name": f"Hurwitz PSU(1,1) kernel surface from PSL(2,{group.p}) triple {triple['triple_index']:04d}",
        "surface_type": "hurwitz_triangle_kernel_surface",
        "domain_type": "triangle_kernel_tile_union",
        "compact": True,
        "finite_area": True,
        "torsion_free": True,
        "orbifold_excluded": False,
        "mainline_dataset_eligible": pass_geometry_audit,
        "riemann_surface_status": "smooth compact Hurwitz Riemann surface D/Gamma, with Gamma a torsion-free kernel of Delta^+(2,3,7)",
        "kahler_status": "complex dimension one; automatically Kähler",
        "genus": genus,
        "area": 4.0 * math.pi * (genus - 1) if genus is not None else None,
        "gauss_bonnet_area": 4.0 * math.pi * (genus - 1) if genus is not None else None,
        "triangle_group": "Delta^+(2,3,7)",
        "triangle_signature": [2, 3, 7],
        "finite_quotient": f"PSL(2,{group.p})",
        "quotient_order": group.order,
        "ginn_ready": pass_geometry_audit,
        "explorer_loadable": False,
        "v1_6_tokenized_generators": True,
        "generator_count": len(gen_json),
        "generator_truncated": not generator_export_complete,
        "generator_truncated_by_cli_max_kernel_generators": not generator_export_complete,
        "generator_export_complete": generator_export_complete,
        "tile_scaffold_complete": tile_scaffold_complete,
        "tiles_truncated_by_cli_max_tiles": not tile_scaffold_complete,
        "tile_count": len(tiles),
        "expected_tile_count": group.order,
        "exclusion_reason": exclusion_reason,
        "generators": gen_json,
        "generator_meanings": meanings,
        "kernel_generator_audit_sample": raw[:min(len(raw), 200)],
        "fundamental_domain_tiles": tiles,
        "tile_scaffold_warning": "Tile union is built from Delta(2,3,7) orbifold triangle coset representatives. When tile_scaffold_complete is true, it includes all quotient cosets and is the current computational sampling domain for the kernel surface. It is not yet a polished side-paired compact polygon certificate.",
        "finite_group_triple": triple,
        "psu11_triangle_audit": delta_audit,
        "schreier_audit": {
            "transversal_size": len(rep_words),
            "expected_transversal_size": group.order,
            "raw_schreier_slots": group.order * 3,
            "identity_like_generators_filtered": identity_like,
            "nonidentity_image_failures": nonidentity_image_failures,
            "kernel_generators_exported": len(gen_json),
            "kernel_generator_export_complete": generator_export_complete,
            "max_kernel_generators_cli": args.max_kernel_generators,
            "tile_scaffold_complete": tile_scaffold_complete,
            "tile_count": len(tiles),
            "expected_tile_count": group.order,
            "all_exported_generators_map_to_identity_in_quotient": nonidentity_image_failures == 0,
        },
        "certification": {
            "status": "complete_ginn_ready_hurwitz_kernel_surface" if pass_geometry_audit else "partial_hurwitz_kernel_surface_record",
            "finite_quotient_certificate": "exact PSL(2,p) brute-force relation and surjectivity checks",
            "psu11_triangle_certificate": "numerical SU(1,1) Delta(2,3,7) relation checks",
            "kernel_certificate": "Reidemeister-Schreier generators k=t*s*rep(ts)^-1 map to identity in finite quotient",
            "remaining_caveat": "v1.6 exports a complete Reidemeister-Schreier kernel generator set and full coset-tile sampling scaffold when no CLI caps are used. It still does not reduce to a minimal side-paired polygon or certify distinct isomorphism class among triples.",
        },
        "maker_run_id": run_id,
    }
    audit = {
        "surface_id": surf_id,
        "q": group.p,
        "triple_index": triple["triple_index"],
        "genus": genus,
        "quotient_order": group.order,
        "transversal_size": len(rep_words),
        "kernel_generators_exported": len(gen_json),
        "kernel_generator_export_complete": generator_export_complete,
        "generator_truncated": not generator_export_complete,
        "identity_like_filtered": identity_like,
        "tile_count": len(tiles),
        "expected_tile_count": group.order,
        "tile_scaffold_complete": tile_scaffold_complete,
        "tiles_truncated": not tile_scaffold_complete,
        "psu11_relation_max_error": relation_max_error,
        "pass_geometry_audit": pass_geometry_audit,
        "mainline_dataset_eligible": pass_geometry_audit,
        "exclusion_reason": exclusion_reason,
        "ginn_ready": pass_geometry_audit,
    }
    return surface, audit


# -----------------------------------------------------------------------------
# Optional bridge to FuchsianDownstairsGINN v2.4
# -----------------------------------------------------------------------------

def default_ginn_script_path() -> str:
    local = Path("FuchsianDownstairsGINN_v2_4.py")
    if local.exists():
        return str(local)
    return "/mnt/data/FuchsianDownstairsGINN_v2_4.py"


def load_ginn_module(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"GINN script not found: {p}")
    mod_name = f"_fuchsian_downstairs_ginn_v24_{abs(hash(str(p))) & 0xffffffff:x}"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import GINN script from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def rough_reduced_word_ball_size(num_positive_generators: int, depth: int) -> int:
    """Reduced word count for m positive gens plus formal inverses, no immediate inverse cancellation."""
    letters = 2 * int(num_positive_generators)
    if depth <= 0:
        return 1
    total = 1
    frontier = 1
    for d in range(1, depth + 1):
        frontier *= letters if d == 1 else max(letters - 1, 0)
        total += frontier
    return total


def summarize_word_ball_rows(ginn, word_ball) -> List[Dict]:
    rows = []
    for i, m in enumerate(word_ball):
        w = m.word if getattr(m, 'word', '') else 'identity'
        try:
            depth = ginn.word_depth_string(m.word)
        except Exception:
            depth = 0 if w == 'identity' else len(str(w).split())
        tr = float(m.trace_real()) if hasattr(m, 'trace_real') else float('nan')
        rows.append({
            'index': i,
            'word': w,
            'depth': depth,
            'trace_real': tr,
            'type_proxy': 'identity' if w == 'identity' else ('hyperbolic' if abs(tr) > 2.0 + 1.0e-10 else 'elliptic_or_relation'),
        })
    return rows


def run_ginn_bridge(surface: Dict, args, run_root: Path) -> Dict:
    """Run a GINN compatibility preflight and optional training smoke test on a kernel surface JSON."""
    ginn = load_ginn_module(args.ginn_script)
    sid = surface.get('surface_id', 'hurwitz_kernel_surface')
    outdir = run_root / ('ginn_runs' if args.run_ginn else 'ginn_preflight') / str(sid)
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / 'surface.json', surface)

    positive_gens = int(surface.get('generator_count', len(surface.get('generators', {}))))
    rough_W = rough_reduced_word_ball_size(positive_gens, args.ginn_depth)
    pre = {
        'surface_id': sid,
        'finite_quotient': surface.get('finite_quotient'),
        'genus': surface.get('genus'),
        'positive_generator_count': positive_gens,
        'ginn_depth': args.ginn_depth,
        'rough_word_ball_size': rough_W,
        'ginn_pairs': args.ginn_pairs,
        'ginn_max_word_ball': args.ginn_max_word_ball,
        'status': 'STARTED',
        'warnings': [],
    }
    if rough_W > args.ginn_max_word_ball > 0:
        pre['status'] = 'SKIPPED_ROUGH_WORD_BALL_TOO_LARGE'
        pre['warnings'].append('Rough reduced word-ball estimate exceeds --ginn-max-word-ball. Use depth 1, increase max, or develop staged pruning.')
        write_json(outdir / 'ginn_bridge_summary.json', pre)
        return pre

    t0 = time.perf_counter()
    rows, X, D, word_ball, meta, feature_names = ginn.generate_ginn_dataset(
        surface, args.ginn_pairs, args.ginn_depth, args.seed, max_word_ball=args.ginn_max_word_ball
    )
    write_csv(outdir / 'pair_dataset.csv', rows)
    write_csv(outdir / 'word_ball_summary.csv', summarize_word_ball_rows(ginn, word_ball))
    pre.update({
        'status': 'PASS_GINN_LABEL_PREFLIGHT',
        'actual_word_ball_size': len(word_ball),
        'sampler_kind': meta.get('sampler_kind'),
        'shortcut_fraction': meta.get('shortcut_fraction'),
        'mean_shortest_lift_depth': meta.get('mean_shortest_lift_depth'),
        'max_shortest_lift_depth': meta.get('max_shortest_lift_depth'),
        'feature_names': feature_names,
        'label_matrix_shape': list(D.shape),
        'preflight_wall_seconds': time.perf_counter() - t0,
    })

    if args.run_ginn:
        train_metrics = ginn.train_ginn(
            rows=rows,
            X=X,
            D=D,
            word_ball=word_ball,
            outdir=outdir,
            depth=args.ginn_depth,
            epochs=args.ginn_epochs,
            pair_hidden=args.ginn_pair_hidden,
            score_hidden=args.ginn_score_hidden,
            lr=args.ginn_lr,
            batch_size=args.ginn_batch_size,
            seed=args.seed,
            device=args.ginn_device,
            patience=args.ginn_patience,
            ce_weight=args.ginn_ce_weight,
            soft_distance_weight=0.0,
            temperature=0.5,
            candidate_chunk_size=args.ginn_candidate_chunk_size,
            auto_chunk_threshold_mb=args.ginn_auto_chunk_threshold_mb,
        )
        write_json(outdir / 'metrics.json', train_metrics)
        pre['status'] = 'PASS_GINN_TRAINING_SMOKE'
        pre['train_metrics_excerpt'] = {
            'word_ball_size': train_metrics.get('word_ball_size'),
            'test_hard_rmse': train_metrics.get('test', {}).get('hard_selected_distance', {}).get('rmse'),
            'baseline_identity_rmse': train_metrics.get('baseline_identity_test', {}).get('rmse'),
            'winning_lift_accuracy_test': train_metrics.get('winning_lift_accuracy_test'),
            'top5_pruned_rmse': train_metrics.get('topk_pruned_search', {}).get('top5', {}).get('pruned_exact_distance_test', {}).get('rmse'),
        }

    write_json(outdir / 'ginn_bridge_summary.json', pre)
    return pre


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_surface_certificate(group: PSL2PrimeGroup, triple: Dict, run_id: str) -> Dict:
    p = group.p
    return {
        "format": "Fuchsian Hurwitz abstract surface certificate v1.6",
        "surface_type": "abstract_hurwitz_triangle_kernel_surface",
        "ginn_ready": False,
        "explorer_loadable": False,
        "reason_not_ginn_ready": "v1.6 certifies the finite quotient; it does not yet construct PSU(1,1) kernel generators by Reidemeister-Schreier rewriting in Delta+(2,3,7).",
        "name": f"Hurwitz candidate from PSL(2,{p}) triple {triple['triple_index']:04d}",
        "triangle_group": "Delta^+(2,3,7)",
        "triangle_presentation": "<x,y,z | x^2=y^3=z^7=xyz=1>",
        "finite_quotient": f"PSL(2,{p})",
        "quotient_order": group.order,
        "genus": triple["genus"],
        "area": 4.0 * math.pi * (triple["genus"] - 1) if triple["genus"] is not None else None,
        "hurwitz_bound_84_g_minus_1": 84 * (triple["genus"] - 1) if triple["genus"] is not None else None,
        "hurwitz_bound_saturated": (84 * (triple["genus"] - 1) == group.order) if triple["genus"] is not None else False,
        "finite_group_triple": triple,
        "certification": {
            "status": "finite_quotient_hurwitz_certificate" if triple["surjective"] else "not_surjective",
            "relation_checks": {
                "x_order_2": triple["x_order"] == 2,
                "y_order_3": triple["y_order"] == 3,
                "z_order_7": triple["z_order"] == 7,
                "xyz_identity": bool(triple["xyz_identity"]),
                "generated_group_order_equals_quotient_order": triple["generated_subgroup_order"] == group.order,
            },
            "torsion_free_kernel_reasoning": "For a hyperbolic triangle group, every finite-order element is conjugate into an elliptic vertex stabilizer. If the images of x,y,z retain exact orders 2,3,7 in the quotient, the kernel contains no nontrivial elliptic element and is torsion-free.",
            "pure_math_caveat": "This is a finite quotient certificate and genus/area audit, not a proof of distinct isomorphism class among all triples unless stronger triple equivalence tests are added.",
        },
        "maker_run_id": run_id,
    }


def process_q(q: int, args, run_root: Path, run_id: str) -> Dict:
    t0 = time.perf_counter()
    group_label = f"PSL2_{q}"
    result = {
        "q": q,
        "group": f"PSL(2,{q})",
        "supported": is_prime(q) and q != 2,
        "status": "",
        "group_order": None,
        "genus_if_hurwitz": None,
        "triple_count": 0,
        "relation_candidate_count": 0,
        "surjective_candidate_count": 0,
        "pair_tests": 0,
        "wall_seconds": None,
        "notes": "",
    }
    if not result["supported"]:
        result["status"] = "UNSUPPORTED_NONPRIME_OR_EXCEPTIONAL"
        result["notes"] = "v1.6 supports PSL(2,p) for odd prime p only."
        return result
    print("=" * 78)
    print(f"[group] PSL(2,{q})")
    try:
        group = PSL2PrimeGroup.build(q, verbose=args.verbose)
        audit = group_audit(group)
        result["group_order"] = group.order
        result["genus_if_hurwitz"] = audit["genus_if_hurwitz"]
        write_json(run_root / "group_audits" / f"{group_label}_group_audit.json", audit)
        if audit["immediate_rejection_reasons"]:
            result["status"] = "PASS_NOT_HURWITZ_IMMEDIATE_REJECT"
            result["notes"] = "; ".join(audit["immediate_rejection_reasons"])
            print(f"[reject] {result['notes']}")
            return result
        triples, stats = find_hurwitz_triples(
            group,
            max_triples=args.max_triples,
            random_trials=args.random_trials,
            seed=args.seed + q,
            dedupe_conjugacy_max_order=args.dedupe_conjugacy_max_order,
            triple_search=args.triple_search,
            triple_equivalence=args.triple_equivalence,
            verbose=args.verbose,
        )
        result["triple_count"] = len(triples)
        result["relation_candidate_count"] = stats["relation_candidates"]
        result["surjective_candidate_count"] = stats["surjective_candidates"]
        result["pair_tests"] = stats["pair_tests"]
        write_json(run_root / "triples" / f"{group_label}_triples.json", {
            "group": f"PSL(2,{q})",
            "audit": audit,
            "search_stats": stats,
            "triples": triples,
        })
        if triples:
            result["status"] = "PASS_HURWITZ_CERTIFIED"
            print(f"[pass] found {len(triples)} generating (2,3,7) triple(s)")
            kernel_rows: List[Dict] = []
            for tr in triples:
                cert = make_surface_certificate(group, tr, run_id)
                surf_id = f"hurwitz_PSL2_{q}_triple_{tr['triple_index']:04d}"
                write_json(run_root / "surface_certificates" / f"{surf_id}.json", cert)
                if args.build_kernel:
                    try:
                        surface, kaudit = build_schreier_kernel_surface(group, tr, args, run_id)
                        surface = normalize_surface_record(
                            surface,
                            surface_spec=surface.get('surface_id', f"hurwitz_PSL2_{q}_kernel"),
                            surface_family="hurwitz_triangle_kernel",
                            surface_subfamily="PSL2_prime_quotient",
                            source_program="FuchsianHurwitzTester_v1_6.py",
                            source_version="1.6",
                            construction_parameters={"q": q, "finite_quotient": f"PSL(2,{q})", "triple_index": tr.get("triple_index")},
                            geometry_audit_pass=kaudit.get("pass_geometry_audit", False),
                            finite_area=True,
                            torsion_free=True,
                            mainline_dataset_eligible=kaudit.get("mainline_dataset_eligible", False),
                            exclusion_reason=kaudit.get("exclusion_reason", ""),
                        )
                        write_json(run_root / "surfaces" / f"{surface['surface_id']}.json", surface)
                        write_json(run_root / "kernel_surfaces" / f"{surface['surface_id']}.json", surface)
                        write_json(run_root / "kernel_audits" / f"{surface['surface_id']}_audit.json", kaudit)
                        if args.ginn_smoke or args.run_ginn:
                            try:
                                bridge = run_ginn_bridge(surface, args, run_root)
                                kaudit['ginn_bridge_status'] = bridge.get('status')
                                kaudit['ginn_word_ball_size'] = bridge.get('actual_word_ball_size', bridge.get('rough_word_ball_size'))
                                kaudit['ginn_shortcut_fraction'] = bridge.get('shortcut_fraction')
                                if 'train_metrics_excerpt' in bridge:
                                    kaudit['ginn_test_hard_rmse'] = bridge['train_metrics_excerpt'].get('test_hard_rmse')
                                    kaudit['ginn_top5_pruned_rmse'] = bridge['train_metrics_excerpt'].get('top5_pruned_rmse')
                                print(f"[ginn] {surface['surface_id']}: {kaudit['ginn_bridge_status']} W={kaudit.get('ginn_word_ball_size')}")
                            except Exception as ge:
                                kaudit['ginn_bridge_status'] = 'FAIL'
                                kaudit['ginn_bridge_error'] = repr(ge)
                                print(f"[ginn fail] {surface['surface_id']}: {ge}")
                        kernel_rows.append(kaudit)
                        print(f"[kernel] {surface['surface_id']}: gens={kaudit['kernel_generators_exported']} tiles={kaudit['tile_count']} rel_err={kaudit['psu11_relation_max_error']:.2e}")
                    except Exception as ke:
                        kernel_rows.append({
                            "q": q, "triple_index": tr.get("triple_index"), "ginn_ready": False,
                            "error": repr(ke),
                        })
                        print(f"[kernel fail] triple {tr.get('triple_index')}: {ke}")
            if kernel_rows:
                write_csv(run_root / "tables" / f"PSL2_{q}_kernel_summary.csv", kernel_rows)
        else:
            result["status"] = "PASS_NO_GENERATING_TRIPLE_FOUND"
            result["notes"] = "Group has required element orders and |G| divisible by 84, but no generating triple was found under the selected search mode."
            print("[no triple] no generating (2,3,7) triple found")
    except Exception as e:
        result["status"] = "FAIL"
        result["notes"] = repr(e)
        print(f"[fail] PSL(2,{q}): {e}")
    finally:
        result["wall_seconds"] = time.perf_counter() - t0
    return result


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description="Hurwitz finite quotient tester for Delta^+(2,3,7) -> PSL(2,p)")
    ap.add_argument("--q", nargs="*", type=int, default=[], help="Prime q=p values for PSL(2,p). v1.6 supports odd primes only.")
    ap.add_argument("--scan-primes", nargs=2, type=int, metavar=("START", "END"), help="Scan all primes in inclusive range.")
    ap.add_argument("--max-triples", type=int, default=10, help="Maximum nonduplicate triples to save per group. Use 0 for no explicit cap.")
    ap.add_argument("--random-trials", type=int, default=0, help="If >0, use random pair search instead of exhaustive order-2/order-3 pair scan. Only supported with --triple-search raw.")
    ap.add_argument("--triple-search", choices=["conjugacy_orbit", "raw"], default="conjugacy_orbit", help="v1.6 default uses centralizer-orbit presorting to find conjugacy-reduced triples. 'raw' restores the v1.5 behavior.")
    ap.add_argument("--triple-equivalence", choices=["inner", "pgl"], default="inner", help="Exact final triple equivalence used by conjugacy_orbit search. 'inner' means simultaneous PSL conjugacy and is the default conjugacy-class count; 'pgl' additionally quotients by the natural PGL outer automorphism.")
    ap.add_argument("--dedupe-conjugacy-max-order", type=int, default=2500, help="Legacy raw-search option: use exact simultaneous-conjugacy dedupe only up to this group order; larger raw searches use cheaper signatures/exact duplicate avoidance.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default="hurwitz_tester_runs")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--build-kernel", action="store_true", help="Also build preliminary PSU(1,1) Reidemeister-Schreier kernel surface JSONs for each saved triple.")
    ap.add_argument("--max-kernel-generators", type=int, default=0, help="Explicit cap on exported nonidentity Schreier kernel generators. 0 means no cap.")
    ap.add_argument("--max-tiles", type=int, default=0, help="Explicit cap on exported coset triangle tiles for sampling. 0 means all quotient cosets.")
    ap.add_argument("--identity-tol", type=float, default=1.0e-9, help="Numerical tolerance for filtering identity-like PSU(1,1) Schreier generators.")
    ap.add_argument("--ginn-smoke", action="store_true", help="After building each kernel surface, import FuchsianDownstairsGINN_v2_4.py and run a label-generation compatibility preflight.")
    ap.add_argument("--run-ginn", action="store_true", help="After the GINN label preflight, run a small branch-ranker training smoke test. Implies --ginn-smoke.")
    ap.add_argument("--ginn-script", type=str, default=default_ginn_script_path(), help="Path to FuchsianDownstairsGINN_v2_4.py used for compatibility preflight/training.")
    ap.add_argument("--ginn-pairs", type=int, default=300, help="Number of random point pairs for GINN preflight/training smoke.")
    ap.add_argument("--ginn-depth", type=int, default=1, help="Deck word-ball depth for Hurwitz kernel GINN smoke. Depth 1 is the safe default for high-generator kernels.")
    ap.add_argument("--ginn-max-word-ball", type=int, default=50000, help="Safety cap for GINN word ball size.")
    ap.add_argument("--ginn-epochs", type=int, default=20)
    ap.add_argument("--ginn-pair-hidden", type=int, default=128)
    ap.add_argument("--ginn-score-hidden", type=int, default=64)
    ap.add_argument("--ginn-lr", type=float, default=1.0e-3)
    ap.add_argument("--ginn-batch-size", type=int, default=64)
    ap.add_argument("--ginn-device", type=str, default="auto")
    ap.add_argument("--ginn-patience", type=int, default=8)
    ap.add_argument("--ginn-ce-weight", type=float, default=1.0)
    ap.add_argument("--ginn-candidate-chunk-size", type=int, default=-1, help="Candidate chunk size passed to v2.4 train_ginn; -1/0 leaves auto behavior.")
    ap.add_argument("--ginn-auto-chunk-threshold-mb", type=float, default=1024.0)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.run_ginn:
        args.ginn_smoke = True
    qs: List[int] = []
    if args.scan_primes:
        qs.extend(primes_between(args.scan_primes[0], args.scan_primes[1]))
    qs.extend(args.q)
    # preserve order but unique
    seen = set()
    qs = [q for q in qs if not (q in seen or seen.add(q))]
    if not qs:
        print("No q values supplied. Try --q 5 7 11 13 or --scan-primes 5 50.")
        return 2
    run_id = now_stamp() + (f"_{args.label}" if args.label else "")
    run_root = Path(args.out) / f"run_{run_id}"
    for sub in ["group_audits", "triples", "surface_certificates", "surfaces", "kernel_surfaces", "kernel_audits", "ginn_preflight", "ginn_runs", "tables"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    manifest = {
        "program": "FuchsianHurwitzTester_v1_6.py",
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "q_values": qs,
        "supports": "PSL(2,p) for odd prime p",
        "triangle_group": "Delta^+(2,3,7)",
        "no_hidden_q_cap": True,
        "cli_args": vars(args),
    }
    write_json(run_root / "manifest.json", manifest)
    print("Fuchsian Hurwitz Tester v1.6")
    print(f"run_root={run_root}")
    print(f"q_values={qs}")
    print("triangle group: Delta^+(2,3,7)")
    print("-" * 78)
    rows: List[Dict] = []
    t_all = time.perf_counter()
    for q in qs:
        row = process_q(q, args, run_root, run_id)
        rows.append(row)
        write_csv(run_root / "tables" / "group_scan_summary.csv", rows)
    # triple summary
    triple_rows: List[Dict] = []
    for q in qs:
        path = run_root / "triples" / f"PSL2_{q}_triples.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for tr in data.get("triples", []):
            triple_rows.append({
                "q": q,
                "group": f"PSL(2,{q})",
                "triple_index": tr["triple_index"],
                "group_order": tr["group_order"],
                "genus": tr["genus"],
                "x_order": tr["x_order"],
                "y_order": tr["y_order"],
                "z_order": tr["z_order"],
                "xyz_identity": tr["xyz_identity"],
                "generated_subgroup_order": tr["generated_subgroup_order"],
                "surjective": tr["surjective"],
                "tr2_x": tr["trace_square_signature"]["tr2_x"],
                "tr2_y": tr["trace_square_signature"]["tr2_y"],
                "tr2_z": tr["trace_square_signature"]["tr2_z"],
                "z_order7_class": tr.get("z_order7_class"),
                "x_conjugacy_class": tr.get("x_conjugacy_class"),
                "y_centralizer_orbit_size": tr.get("y_centralizer_orbit_size"),
            })
    write_csv(run_root / "tables" / "hurwitz_triple_summary.csv", triple_rows)
    # Combined kernel-surface summary, if --build-kernel was used.
    kernel_rows: List[Dict] = []
    for apath in sorted((run_root / "kernel_audits").glob("*_audit.json")):
        try:
            kernel_rows.append(json.loads(apath.read_text(encoding="utf-8")))
        except Exception:
            pass
    write_csv(run_root / "tables" / "kernel_surface_summary.csv", kernel_rows)
    # Generic contract-facing audit table for the future master builder.
    generic_rows = []
    for spath in sorted((run_root / "surfaces").glob("*.json")):
        try:
            sj = json.loads(spath.read_text())
            generic_rows.append({
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
                "pass_geometry_audit": sj.get("geometry_audit_pass", True),
                "source_program": sj.get("source_program"),
                "source_version": sj.get("source_version"),
            })
        except Exception:
            pass
    write_contract_csv(run_root / "tables" / "geometry_audit.csv", generic_rows, GEOMETRY_AUDIT_FIELDS)

    # Generic contract-facing GINN smoke table.
    ginn_rows = []
    for base in [run_root / "ginn_preflight", run_root / "ginn_runs"]:
        for bpath in sorted(base.glob("*/ginn_bridge_summary.json")):
            try:
                gd = json.loads(bpath.read_text(encoding="utf-8"))
                status = str(gd.get("status", ""))
                ginn_rows.append({
                    "surface": gd.get("surface_id"),
                    "surface_id": gd.get("surface_id"),
                    "pairs": gd.get("ginn_pairs"),
                    "word_depth": gd.get("ginn_depth"),
                    "word_ball_size": gd.get("actual_word_ball_size", gd.get("rough_word_ball_size")),
                    "shortcut_fraction": gd.get("shortcut_fraction"),
                    "mean_winner_depth": gd.get("mean_shortest_lift_depth"),
                    "max_word_ball": gd.get("ginn_max_word_ball"),
                    "pass_ginn_preflight": status.startswith("PASS"),
                    "error": "; ".join(gd.get("warnings", [])) if gd.get("warnings") else "",
                })
            except Exception:
                pass
    write_contract_csv(run_root / "tables" / "ginn_smoke_summary.csv", ginn_rows, GINN_SMOKE_FIELDS)

    failure_rows = []
    for r in rows:
        if str(r.get("status")) == "FAIL":
            failure_rows.append({"surface": r.get("group"), "surface_id": r.get("group"), "error_type": "group_processing_failure", "error": r.get("notes")})
    for apath in sorted((run_root / "kernel_audits").glob("*_audit.json")):
        try:
            ad = json.loads(apath.read_text(encoding="utf-8"))
            if ad.get("pass_geometry_audit") is False and ad.get("error"):
                failure_rows.append({"surface": ad.get("surface_id"), "surface_id": ad.get("surface_id"), "error_type": "kernel_surface_failure", "error": ad.get("error")})
        except Exception:
            pass
    write_contract_csv(run_root / "tables" / "failures.csv", failure_rows, ["surface", "surface_id", "error_type", "error"])

    summary = {
        "completed": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": time.perf_counter() - t_all,
        "groups_processed": len(rows),
        "hurwitz_groups_found": sum(1 for r in rows if r.get("triple_count", 0) > 0),
        "triples_saved": len(triple_rows),
        "run_root": str(run_root),
    }
    write_json(run_root / "run_summary.json", summary)
    print("=" * 78)
    print(f"[done] groups={len(rows)} hurwitz_groups={summary['hurwitz_groups_found']} triples={len(triple_rows)}")
    print(f"[done] run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
