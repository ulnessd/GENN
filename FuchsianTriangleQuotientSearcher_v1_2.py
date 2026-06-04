#!/usr/bin/env python3
"""
FuchsianTriangleQuotientSearcher_v1_0.py

Search finite quotients of orientation-preserving hyperbolic triangle groups
Delta^+(p,q,r)=<X,Y | X^p=Y^q=(XY)^r=1>.

Purpose for the GENN zoo project:
  * scan a controlled list of signatures and finite group families;
  * find exact (p,q,r)-generating pairs;
  * estimate the complete Reidemeister-Schreier kernel generator count;
  * classify candidates as:
        - COMPLETE_BALL_READY: complete depth<=2 word ball is small enough;
        - SELECTED_ATLAS_READY: full quotient is modest (default <= PSL(2,43))
          but complete depth<=2 ball is too large, so use shortest-256/512 style;
        - TOO_LARGE_FOR_CURRENT_PIPELINE.

Implemented finite families:
  * built-in anchors: GL(2,3) for (2,3,8), S5 for (2,4,5);
  * symmetric groups S_n;
  * alternating groups A_n;
  * PSL(2,p) over prime fields p.

This is a search/planning tool, not a trainer. It writes JSON/CSV/Markdown reports
that can guide later trainer additions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROGRAM = "FuchsianTriangleQuotientSearcher_v1_2.py"
VERSION = "1.2"

Element = Tuple[int, ...]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_signatures(s: str) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    if not s.strip():
        return out
    for part in s.replace(";", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        nums = [int(x.strip()) for x in part.replace("x", ",").replace(" ", ",").split(",") if x.strip()]
        if len(nums) != 3:
            raise ValueError(f"Bad signature '{part}'. Use e.g. 2,3,8;2,4,5")
        a, b, c = sorted(nums)
        if 1/a + 1/b + 1/c >= 1:
            # Still allow if the user explicitly put it in; but mark later.
            pass
        out.append((a, b, c))
    # de-duplicate preserving order
    seen = set(); ans = []
    for sig in out:
        if sig not in seen:
            seen.add(sig); ans.append(sig)
    return ans



def generate_signatures_by_delta(delta_max: float, p_max: int, q_max: int, r_max: int) -> List[Tuple[int, int, int]]:
    """Generate sorted hyperbolic signatures 2 <= p <= q <= r with defect <= delta_max."""
    out: List[Tuple[int, int, int]] = []
    for p in range(2, int(p_max) + 1):
        for q in range(p, int(q_max) + 1):
            for r in range(q, int(r_max) + 1):
                sig = (p, q, r)
                if is_hyperbolic(sig) and defect(sig) <= float(delta_max) + 1e-15:
                    out.append(sig)
    return out


def merge_signatures(*lists: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    seen = set()
    ans: List[Tuple[int, int, int]] = []
    for lst in lists:
        for sig in lst:
            sig = tuple(sorted(sig))  # type: ignore
            if sig not in seen:
                seen.add(sig)
                ans.append(sig)
    return ans


def is_hyperbolic(sig: Tuple[int, int, int]) -> bool:
    p, q, r = sig
    return 1/p + 1/q + 1/r < 1


def defect(sig: Tuple[int, int, int]) -> float:
    p, q, r = sig
    return 1.0 - 1.0/p - 1.0/q - 1.0/r


def genus_for_order(sig: Tuple[int, int, int], order: int, tol: float = 1e-9) -> Optional[int]:
    g = 1.0 + float(order) * defect(sig) / 2.0
    gi = int(round(g))
    if abs(g - gi) < tol:
        return gi
    return None


def psl2_order(q: int) -> int:
    return int(q) * (int(q) * int(q) - 1) // math.gcd(2, int(q) - 1)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def stable_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

# -----------------------------------------------------------------------------
# Finite group wrapper
# -----------------------------------------------------------------------------

@dataclass
class FiniteGroup:
    name: str
    family: str
    elements: List[Element]
    identity: Element
    metadata: Dict[str, Any]

    @property
    def order(self) -> int:
        return len(self.elements)

    def mul(self, a: Element, b: Element) -> Element:
        if self.family in {"perm", "Sn", "An"}:
            # product a*b acts as b then a, matching the trainer convention.
            return tuple(a[i] for i in b)
        if self.family == "psl2p":
            p = int(self.metadata["field_p"])
            return canonical_psl2p(((a[0]*b[0] + a[1]*b[2]) % p,
                                    (a[0]*b[1] + a[1]*b[3]) % p,
                                    (a[2]*b[0] + a[3]*b[2]) % p,
                                    (a[2]*b[1] + a[3]*b[3]) % p), p)
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
        if self.family in {"psl2p", "gl2p"}:
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
        raise RuntimeError(f"order computation failed in {self.name}")

    def generated_subgroup_order(self, gens: Sequence[Element], stop_at_full: bool = True, max_seen: Optional[int] = None) -> int:
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
                    if max_seen is not None and len(seen) > max_seen:
                        return len(seen)
                    qd.append(hg)
        return len(seen)

    def element_to_json(self, a: Element) -> Any:
        if self.family in {"perm", "Sn", "An"}:
            return [int(x) + 1 for x in a]
        if self.family in {"psl2p", "gl2p"}:
            return [[int(a[0]), int(a[1])], [int(a[2]), int(a[3])]]
        return list(a)

# -----------------------------------------------------------------------------
# Group builders
# -----------------------------------------------------------------------------

def perm_parity(p: Element) -> int:
    inv = 0
    n = len(p)
    for i in range(n):
        pi = p[i]
        for j in range(i+1, n):
            if pi > p[j]:
                inv += 1
    return inv % 2


def build_symmetric(n: int) -> FiniteGroup:
    elems = [tuple(p) for p in itertools.permutations(range(n))]
    return FiniteGroup(name=f"S{n}", family="Sn", elements=elems, identity=tuple(range(n)), metadata={"degree": n})


def build_alternating(n: int) -> FiniteGroup:
    elems = [tuple(p) for p in itertools.permutations(range(n)) if perm_parity(tuple(p)) == 0]
    return FiniteGroup(name=f"A{n}", family="An", elements=elems, identity=tuple(range(n)), metadata={"degree": n})


def canonical_psl2p(m: Element, p: int) -> Element:
    a = tuple(x % p for x in m)
    neg = tuple((-x) % p for x in a)
    # Choose lexicographically smaller representative of {M,-M}.
    return min(a, neg)


def build_psl2p(p: int) -> FiniteGroup:
    elems_set = set()
    # Enumerate determinant-1 matrices. For p <= 100 this is fine.
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # solve a*d - b*c = 1 if possible
                if a % p != 0:
                    d = ((1 + b*c) * pow(a, -1, p)) % p
                    elems_set.add(canonical_psl2p((a, b, c, d), p))
                else:
                    # then -b*c=1, so b and c nonzero and d arbitrary
                    if b % p != 0 and c % p != 0 and (-b*c) % p == 1:
                        for d in range(p):
                            elems_set.add(canonical_psl2p((a, b, c, d), p))
    elems = sorted(elems_set)
    ident = canonical_psl2p((1,0,0,1), p)
    G = FiniteGroup(name=f"PSL2_{p}", family="psl2p", elements=elems, identity=ident, metadata={"field_p": p})
    expected = psl2_order(p)
    if len(elems) != expected:
        raise RuntimeError(f"PSL2({p}) enumeration produced {len(elems)} not {expected}")
    return G


def build_gl2_3_anchor() -> Tuple[FiniteGroup, Element, Element, Tuple[int,int,int]]:
    p = 3
    elems: List[Element] = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    if (a*d - b*c) % p != 0:
                        elems.append((a,b,c,d))
    G = FiniteGroup(name="GL2_3_Bolza", family="gl2p", elements=sorted(elems), identity=(1,0,0,1), metadata={"field_p": 3, "anchor": "Bolza"})
    x = (0,1,1,0)
    y = (1,0,1,1)
    return G, x, y, (2,3,8)


def build_s5_anchor() -> Tuple[FiniteGroup, Element, Element, Tuple[int,int,int]]:
    G = build_symmetric(5)
    G.name = "S5_Bring"
    G.metadata["anchor"] = "Bring"
    x = (0,1,2,4,3)
    y = (1,2,3,0,4)
    return G, x, y, (2,4,5)

# -----------------------------------------------------------------------------
# Search and estimates
# -----------------------------------------------------------------------------

@dataclass
class SearchResult:
    signature: Tuple[int,int,int]
    group_name: str
    family: str
    order: int
    genus: Optional[int]
    x: Element
    y: Element
    x_order: int
    y_order: int
    xy_order: int
    generated_order: int
    raw_nonidentity_schreier_generators: int
    complete_depth2_raw_word_ball: int
    category: str
    search_method: str
    pair_checks: int
    elapsed_seconds: float
    notes: str
    metadata: Dict[str, Any]


def compute_schreier_raw_count(G: FiniteGroup, x: Element, y: Element, sig: Tuple[int,int,int], identity_tol: float = 1e-10) -> Tuple[int,int,str]:
    """Compute the number of nonidentity Schreier edge generators.

    Preferred path: use the same SU(1,1) triangle geometry as the trainer, so
    triangle relations such as X^p, Y^q, and (XY)^r that collapse to the true
    identity are filtered correctly. This matches the trainer's
    raw_nonidentity_schreier_generators much better than the free-Schreier
    non-tree-edge estimate.

    Fallback path: if the triangle trainer module is not importable, return the
    conservative free-Schreier non-tree-edge estimate slots-(|G|-1).
    """
    p, q, r = sig
    finite_gens: Dict[str, Element] = {"X": x, "Y": y, "Y^-1": G.inv(y)}
    token_order = ["X", "Y", "Y^-1"]
    if p != 2:
        finite_gens["X^-1"] = G.inv(x)
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
                rep_id[nh] = len(rep_elems)
                rep_elems.append(nh)
                parent_id.append(hid)
                parent_tok.append(tok)
                qd.append(rep_id[nh])
    if len(rep_elems) != G.order:
        return 0, len(rep_elems), "incomplete_transversal"

    # Try exact geometric count using the triangle trainer's SU(1,1) realization.
    try:
        try:
            import FuchsianTriangleQuotientTrainer_v1_1 as tri
        except Exception:
            import FuchsianTriangleQuotientTrainer_v1_0 as tri  # type: ignore
        delta_gens, _delta_audit = tri.build_delta_pqr_su11(sig)
        coset_alpha = [0j] * G.order
        coset_beta = [0j] * G.order
        coset_alpha[0] = 1.0 + 0j
        coset_beta[0] = 0.0 + 0j
        delta_pair = {tok: (complex(delta_gens[tok].alpha), complex(delta_gens[tok].beta)) for tok in token_order}
        for i in range(1, G.order):
            pid = parent_id[i]
            tok = parent_tok[i]
            a, b = tri.su11_pair_compose(complex(coset_alpha[pid]), complex(coset_beta[pid]), delta_pair[tok][0], delta_pair[tok][1])
            coset_alpha[i] = a
            coset_beta[i] = b
        raw = 0
        identity_like = 0
        for hid, h in enumerate(rep_elems):
            ah = complex(coset_alpha[hid]); bh = complex(coset_beta[hid])
            for tok in token_order:
                hs = G.mul(h, finite_gens[tok])
                hs_id = rep_id[hs]
                asg, bsg = delta_pair[tok]
                a1, b1 = tri.su11_pair_compose(ah, bh, asg, bsg)
                ai, bi = tri.su11_pair_inverse(complex(coset_alpha[hs_id]), complex(coset_beta[hs_id]))
                ak, bk = tri.su11_pair_compose(a1, b1, ai, bi)
                disp = tri.su11_pair_displacement_from_identity(ak, bk)
                if float(disp) < float(identity_tol):
                    identity_like += 1
                else:
                    raw += 1
        return int(raw), G.order, f"geometric_exact_identity_filtered; identity_like={identity_like}"
    except Exception as e:
        scanned = G.order * len(token_order)
        raw_est = scanned - (G.order - 1)
        return int(raw_est), G.order, f"fallback_free_schreier_estimate; reason={type(e).__name__}:{e}"

def complete_depth2_raw_from_positive_generators(k: int) -> int:
    m = 2 * int(k)
    if m <= 0:
        return 1
    return 1 + m + m * (m - 1)


def classify_candidate(order: int, complete_raw: int, max_complete_raw: int, medium_order_limit: int) -> str:
    if complete_raw <= max_complete_raw:
        return "COMPLETE_BALL_READY"
    if order <= medium_order_limit:
        return "SELECTED_ATLAS_READY"
    return "TOO_LARGE_FOR_CURRENT_PIPELINE"


def compute_order_classes(G: FiniteGroup, wanted: Sequence[int], verbose: bool = False) -> Dict[int, List[Element]]:
    wanted_set = set(int(w) for w in wanted)
    classes: Dict[int, List[Element]] = {w: [] for w in wanted_set}
    # Quick exact test: order exactly k iff g^k=e and g^d != e for proper divisors d.
    divisors: Dict[int, List[int]] = {}
    for k in wanted_set:
        divisors[k] = [d for d in range(1, k) if k % d == 0]
    for idx, a in enumerate(G.elements):
        if a == G.identity:
            continue
        for k in wanted_set:
            if G.pow(a, k) == G.identity and all(G.pow(a, d) != G.identity for d in divisors[k]):
                classes[k].append(a)
        if verbose and idx and idx % 50000 == 0:
            print(f"  [order-classes {G.name}] {idx:,}/{G.order:,}", flush=True)
    return classes


def pair_key_json(G: FiniteGroup, x: Element, y: Element) -> Dict[str, Any]:
    return {"x": G.element_to_json(x), "y": G.element_to_json(y)}


def validate_pair(G: FiniteGroup, sig: Tuple[int,int,int], x: Element, y: Element, require_full: bool = True) -> Optional[Tuple[int,int,int,int]]:
    p, q, r = sig
    xy = G.mul(x, y)
    # Fast exact order checks using powers.
    if G.pow(x, p) != G.identity or any(G.pow(x, d) == G.identity for d in range(1, p) if p % d == 0):
        return None
    if G.pow(y, q) != G.identity or any(G.pow(y, d) == G.identity for d in range(1, q) if q % d == 0):
        return None
    if G.pow(xy, r) != G.identity or any(G.pow(xy, d) == G.identity for d in range(1, r) if r % d == 0):
        return None
    gen = G.generated_subgroup_order([x, y], stop_at_full=True)
    if require_full and gen != G.order:
        return None
    return (p, q, r, gen)


def find_triangle_pair(
    G: FiniteGroup,
    sig: Tuple[int,int,int],
    rng: random.Random,
    max_pair_checks: int,
    exhaustive_pair_limit: int,
    verbose: bool = False,
) -> Tuple[Optional[Tuple[Element, Element, int]], str, int, str]:
    p, q, r = sig
    classes = compute_order_classes(G, [p, q], verbose=verbose)
    Xs = classes.get(p, [])
    Ys = classes.get(q, [])
    if not Xs or not Ys:
        return None, "no_order_class", 0, f"order classes: |X_p|={len(Xs)} |Y_q|={len(Ys)}"

    total_pairs = len(Xs) * len(Ys)
    checks = 0
    # Exhaustive for small pair sets.
    if total_pairs <= exhaustive_pair_limit:
        method = "exhaustive"
        for x in Xs:
            for y in Ys:
                checks += 1
                xy = G.mul(x, y)
                # Check xy order r first.
                if G.pow(xy, r) != G.identity:
                    continue
                if any(G.pow(xy, d) == G.identity for d in range(1, r) if r % d == 0):
                    continue
                gen = G.generated_subgroup_order([x, y], stop_at_full=True)
                if gen == G.order:
                    return (x, y, gen), method, checks, f"total_pairs={total_pairs}"
        return None, method, checks, f"total_pairs={total_pairs}; no generating exact pair found"

    # Random sampling otherwise.
    method = "random"
    seen_pairs = set()
    trials = min(max_pair_checks, total_pairs)
    for _ in range(trials):
        xi = rng.randrange(len(Xs)); yi = rng.randrange(len(Ys))
        if (xi, yi) in seen_pairs:
            continue
        seen_pairs.add((xi, yi))
        x = Xs[xi]; y = Ys[yi]
        checks += 1
        xy = G.mul(x, y)
        if G.pow(xy, r) != G.identity:
            continue
        if any(G.pow(xy, d) == G.identity for d in range(1, r) if r % d == 0):
            continue
        gen = G.generated_subgroup_order([x, y], stop_at_full=True)
        if gen == G.order:
            return (x, y, gen), method, checks, f"sampled={checks} from total_pairs={total_pairs}"
    return None, method, checks, f"sampled={checks} from total_pairs={total_pairs}; no generating exact pair found"


def make_result(G: FiniteGroup, sig: Tuple[int,int,int], x: Element, y: Element, gen: int, method: str, checks: int, elapsed: float, notes: str, args: argparse.Namespace) -> SearchResult:
    raw_k, trans_size, raw_method = compute_schreier_raw_count(G, x, y, sig, identity_tol=float(getattr(args, "identity_tol", 1e-10)))
    complete_raw = complete_depth2_raw_from_positive_generators(raw_k)
    cat = classify_candidate(G.order, complete_raw, int(args.max_complete_depth2_raw), int(args.medium_order_limit))
    return SearchResult(
        signature=sig,
        group_name=G.name,
        family=G.family,
        order=G.order,
        genus=genus_for_order(sig, G.order),
        x=x,
        y=y,
        x_order=sig[0],
        y_order=sig[1],
        xy_order=sig[2],
        generated_order=gen,
        raw_nonidentity_schreier_generators=raw_k,
        complete_depth2_raw_word_ball=complete_raw,
        category=cat,
        search_method=method,
        pair_checks=checks,
        elapsed_seconds=elapsed,
        notes=notes + f"; transversal_size={trans_size}; schreier_count_method={raw_method}",
        metadata={**G.metadata, **pair_key_json(G, x, y)},
    )


def result_to_row(r: SearchResult) -> Dict[str, Any]:
    p, q, rr = r.signature
    return {
        "signature": f"({p},{q},{rr})",
        "group": r.group_name,
        "family": r.family,
        "order": r.order,
        "genus": r.genus if r.genus is not None else "",
        "x_order": r.x_order,
        "y_order": r.y_order,
        "xy_order": r.xy_order,
        "generated_order": r.generated_order,
        "raw_nonidentity_schreier_generators_est": r.raw_nonidentity_schreier_generators,
        "complete_depth2_raw_word_ball_est": r.complete_depth2_raw_word_ball,
        "category": r.category,
        "search_method": r.search_method,
        "pair_checks": r.pair_checks,
        "elapsed_seconds": round(r.elapsed_seconds, 4),
        "notes": r.notes,
    }


def result_to_json(r: SearchResult) -> Dict[str, Any]:
    d = result_to_row(r)
    d["signature_tuple"] = list(r.signature)
    d["x"] = r.metadata.get("x")
    d["y"] = r.metadata.get("y")
    d["metadata"] = r.metadata
    return d

# -----------------------------------------------------------------------------
# Main search driver
# -----------------------------------------------------------------------------

def add_anchor_results(signatures: List[Tuple[int,int,int]], args: argparse.Namespace, results: List[SearchResult]) -> None:
    anchors = [build_gl2_3_anchor(), build_s5_anchor()]
    for G, x, y, sig in anchors:
        if sig not in signatures:
            continue
        t0 = time.perf_counter()
        val = validate_pair(G, sig, x, y)
        if val is None:
            print(f"[anchor fail] {G.name} for {sig}", flush=True)
            continue
        res = make_result(G, sig, x, y, val[3], "built_in_anchor", 1, time.perf_counter() - t0, "known named surface anchor", args)
        print(f"[found anchor] {sig} -> {G.name} order={G.order} genus={res.genus} category={res.category}", flush=True)
        results.append(res)


def iter_groups(args: argparse.Namespace) -> Iterable[FiniteGroup]:
    fams = {x.strip().lower() for x in str(args.families).replace(";", ",").split(",") if x.strip()}
    if "sn" in fams:
        for n in range(int(args.n_min), int(args.n_max) + 1):
            if math.factorial(n) > int(args.max_group_order_to_build):
                print(f"[skip] S{n} order={math.factorial(n)} exceeds --max-group-order-to-build", flush=True)
                continue
            print(f"[build] S{n}", flush=True)
            yield build_symmetric(n)
    if "an" in fams:
        for n in range(max(3, int(args.n_min)), int(args.n_max) + 1):
            order = math.factorial(n) // 2
            if order > int(args.max_group_order_to_build):
                print(f"[skip] A{n} order={order} exceeds --max-group-order-to-build", flush=True)
                continue
            print(f"[build] A{n}", flush=True)
            yield build_alternating(n)
    if "psl2" in fams or "psl2p" in fams:
        for p in range(max(3, int(args.psl_p_min)), int(args.psl_p_max) + 1):
            if not is_prime(p):
                continue
            order = psl2_order(p)
            if order > int(args.max_group_order_to_build):
                print(f"[skip] PSL2({p}) order={order} exceeds --max-group-order-to-build", flush=True)
                continue
            print(f"[build] PSL2({p}) order={order}", flush=True)
            yield build_psl2p(p)


def write_report(run_root: Path, args: argparse.Namespace, results: List[SearchResult], failures: List[Dict[str, Any]]) -> None:
    rows = [result_to_row(r) for r in results]
    by_cat: Dict[str, List[SearchResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    lines: List[str] = []
    lines.append(f"# Triangle Quotient Searcher v{VERSION} Report")
    lines.append("")
    lines.append(f"Created: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Search finite quotients of hyperbolic triangle groups `Delta^+(p,q,r)` and classify the resulting Riemann-surface candidates by expected feasibility for complete depth-2 word-ball training or selected-atlas training.")
    lines.append("")
    lines.append("## Run parameters")
    lines.append("")
    for k in ["signatures", "delta_max", "p_max", "q_max", "r_max", "families", "n_min", "n_max", "psl_p_min", "psl_p_max", "max_group_order_to_build", "max_complete_depth2_raw", "medium_order_limit", "max_pair_checks", "exhaustive_pair_limit", "seed", "identity_tol"]:
        lines.append(f"- `{k}`: `{getattr(args, k)}`")
    lines.append("")
    lines.append("## Category definitions")
    lines.append("")
    lines.append("- `COMPLETE_BALL_READY`: estimated complete all-Schreier depth-2 raw word ball is at or below `--max-complete-depth2-raw`.")
    lines.append("- `SELECTED_ATLAS_READY`: full quotient order is at or below `--medium-order-limit`, but the complete depth-2 ball is too large; use shortest-256/512 selected atlas first.")
    lines.append("- `TOO_LARGE_FOR_CURRENT_PIPELINE`: likely requires a new sampled/local quotient architecture.")
    lines.append("")
    lines.append("## Summary by category")
    lines.append("")
    lines.append("| category | count |")
    lines.append("| --- | ---: |")
    for cat in ["COMPLETE_BALL_READY", "SELECTED_ATLAS_READY", "TOO_LARGE_FOR_CURRENT_PIPELINE"]:
        lines.append(f"| {cat} | {len(by_cat.get(cat, []))} |")
    lines.append("")
    lines.append("## Candidates")
    lines.append("")
    if rows:
        cols = ["signature", "group", "family", "order", "genus", "raw_nonidentity_schreier_generators_est", "complete_depth2_raw_word_ball_est", "category", "search_method", "pair_checks"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in sorted(rows, key=lambda x: (str(x["category"]), str(x["signature"]), int(x["order"]))):
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    else:
        lines.append("No candidates found.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("The Schreier generator count is an estimate based on the non-tree edge count of the finite quotient Schreier graph with the standard triangle generators. It is intended for planning complete-vs-selected atlas feasibility, not as a minimal side-pairing presentation.")
    lines.append("")
    run_root.joinpath("report").mkdir(parents=True, exist_ok=True)
    (run_root / "report" / "triangle_quotient_search_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(run_root / "tables" / "triangle_quotient_candidates.csv", rows)
    write_csv(run_root / "tables" / "triangle_quotient_failures.csv", failures)
    write_json(run_root / "candidates" / "triangle_quotient_candidates.json", [result_to_json(r) for r in results])


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_sigs = "2,3,7;2,3,8;2,4,5;2,3,9;2,3,10;2,4,6;3,3,4;3,3,5;2,5,5"
    ap = argparse.ArgumentParser(description="Search finite quotients of hyperbolic triangle groups for the GENN zoo.")
    ap.add_argument("--signatures", type=str, default=default_sigs, help="Semicolon-separated signatures, e.g. '2,3,8;2,4,5'. If --delta-max is given, these are unioned with the auto-generated signatures.")
    ap.add_argument("--delta-max", type=float, default=None, help="Auto-generate all hyperbolic signatures 2<=p<=q<=r with defect delta <= this value.")
    ap.add_argument("--p-max", type=int, default=8, help="Maximum p for --delta-max auto-signature generation.")
    ap.add_argument("--q-max", type=int, default=12, help="Maximum q for --delta-max auto-signature generation.")
    ap.add_argument("--r-max", type=int, default=40, help="Maximum r for --delta-max auto-signature generation.")
    ap.add_argument("--families", type=str, default="anchors,Sn,An,PSL2", help="Families to scan: anchors,Sn,An,PSL2")
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--psl-p-min", type=int, default=5)
    ap.add_argument("--psl-p-max", type=int, default=43)
    ap.add_argument("--max-group-order-to-build", type=int, default=50000)
    ap.add_argument("--max-complete-depth2-raw", type=int, default=1000000)
    ap.add_argument("--medium-order-limit", type=int, default=39732, help="Default is |PSL(2,43)|")
    ap.add_argument("--max-pair-checks", type=int, default=50000)
    ap.add_argument("--exhaustive-pair-limit", type=int, default=250000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--identity-tol", type=float, default=1e-10, help="Tolerance for SU(1,1) identity filtering in exact Schreier count")
    ap.add_argument("--outroot", type=str, default="triangle_quotient_search_runs")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    explicit_signatures = parse_signatures(args.signatures)
    auto_signatures: List[Tuple[int, int, int]] = []
    if args.delta_max is not None:
        auto_signatures = generate_signatures_by_delta(float(args.delta_max), int(args.p_max), int(args.q_max), int(args.r_max))
    signatures = merge_signatures(explicit_signatures, auto_signatures)
    if not signatures:
        raise SystemExit("No signatures provided")
    nonhyp = [sig for sig in signatures if not is_hyperbolic(sig)]
    if nonhyp:
        print(f"[warn] non-hyperbolic signatures included: {nonhyp}", flush=True)

    label = args.label or "triangle_quotient_search"
    run_root = Path(args.outroot) / ("run_" + time.strftime("%Y%m%d_%H%M%S") + "_" + stable_slug(label))
    for sub in ["report", "tables", "candidates", "logs"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    print(f"{PROGRAM} v{VERSION}", flush=True)
    print(f"run_root={run_root}", flush=True)
    print(f"signatures={signatures}", flush=True)
    if args.delta_max is not None:
        print(f"auto_signatures={len(auto_signatures)} from delta<= {args.delta_max} with p<= {args.p_max}, q<= {args.q_max}, r<= {args.r_max}", flush=True)
    print(f"families={args.families} max_complete_depth2_raw={args.max_complete_depth2_raw} medium_order_limit={args.medium_order_limit}", flush=True)
    print("-" * 78, flush=True)

    rng = random.Random(int(args.seed))
    results: List[SearchResult] = []
    failures: List[Dict[str, Any]] = []

    fams = {x.strip().lower() for x in str(args.families).replace(";", ",").split(",") if x.strip()}
    if "anchors" in fams or "anchor" in fams:
        add_anchor_results(signatures, args, results)

    for G in iter_groups(args):
        tG = time.perf_counter()
        print(f"[group] {G.name} order={G.order}", flush=True)
        wanted_orders = sorted({k for sig in signatures for k in sig[:2]})
        # We compute order classes per signature inside find_triangle_pair. This is redundant
        # but simpler and okay for our intended size. A future v1.1 can cache them.
        for sig in signatures:
            if not is_hyperbolic(sig):
                continue
            # If group order cannot yield integer genus, skip early. This is a strong filter.
            if genus_for_order(sig, G.order) is None:
                failures.append({"group": G.name, "signature": str(sig), "reason": "noninteger_genus_for_order", "order": G.order})
                continue
            print(f"  [search] {G.name} signature={sig}", flush=True)
            t0 = time.perf_counter()
            try:
                pair, method, checks, notes = find_triangle_pair(
                    G, sig, rng=rng, max_pair_checks=int(args.max_pair_checks), exhaustive_pair_limit=int(args.exhaustive_pair_limit), verbose=bool(args.verbose)
                )
            except Exception as e:
                failures.append({"group": G.name, "signature": str(sig), "reason": type(e).__name__, "message": str(e), "order": G.order})
                print(f"    [fail] {type(e).__name__}: {e}", flush=True)
                continue
            elapsed = time.perf_counter() - t0
            if pair is None:
                failures.append({"group": G.name, "signature": str(sig), "reason": "no_pair_found", "method": method, "pair_checks": checks, "notes": notes, "order": G.order})
                print(f"    [none] method={method} checks={checks} {notes}", flush=True)
                continue
            x, y, gen = pair
            res = make_result(G, sig, x, y, gen, method, checks, elapsed, notes, args)
            results.append(res)
            print(f"    [found] {sig}->{G.name} order={G.order} genus={res.genus} rawK~{res.raw_nonidentity_schreier_generators:,} depth2~{res.complete_depth2_raw_word_ball:,} {res.category}", flush=True)
        print(f"[group done] {G.name} elapsed={time.perf_counter()-tG:.2f}s", flush=True)

    write_report(run_root, args, results, failures)
    print("=" * 78, flush=True)
    print(f"[done] candidates={len(results)} failures={len(failures)}", flush=True)
    print(f"[done] report={run_root/'report'/'triangle_quotient_search_report.md'}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
