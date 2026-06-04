#!/usr/bin/env python3
"""
FuchsianJankoHurwitzTrainer_v1_1.py

Janko Hurwitz quotient -> Fuchsian kernel surface -> exact finite word-ball
atlas -> candidate-pool GINN reranker.

This is a sibling of the Big-Hurwitz PSL(2,q) trainer.  It keeps the same
concrete PSU(1,1) model of Delta^+(2,3,7) and the same atlas/training machinery,
but replaces the finite quotient with the explicit Janko generators from
Pellegrini--Tamburini Bellani:

  J1: 7x7 matrices over GF(11)
  J2: 6x6 matrices over GF(4), omega^2 + omega + 1 = 0

Important practical note
------------------------
A complete Reidemeister-Schreier generator export for J1 has roughly 3|J1|
slots before identity/redundancy filtering, and J2 is larger.  The script can
export all such generators with --kernel-generator-mode all, but the default is
--kernel-generator-mode shortest with a finite --kernel-generator-limit so that
one can actually build depth-2 atlases and train.  Outputs are labeled honestly:
truncated/selected kernel generator pools are atlas/training-ready but are not
minimal side-pairing presentations and are not complete kernel generator exports.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROGRAM = "FuchsianJankoHurwitzTrainer_v1_1.py"
VERSION = "1.1"

Matrix = Tuple[int, ...]

EXPECTED_ORDERS = {"J1": 175560, "J2": 604800}
EXPECTED_GENUS = {"J1": 1 + 175560 // 84, "J2": 1 + 604800 // 84}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_slug(label: str) -> str:
    s = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(label))
    return s.strip('_') or 'run'


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def default_trainer_script() -> str:
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


def install_leq_depth_word_ball_builder(ginn: Any) -> None:
    """Install a local, explicit <=depth reduced word-ball builder.

    The upstream GINN module currently already uses the <=depth convention, but
    v1.1 pins the convention here so the Janko trainer is self-auditing and does
    not depend on a silently changed helper.  For m oriented letters and depth 2,
    the raw reduced word-ball has size 1 + m + m(m-1).
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
        """Build reduced words of length <= depth, including identity and all shorter shells."""
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
        seen: set[Tuple[str, ...]] = set()
        out: List[Any] = []
        for toks in words:
            if toks in seen:
                continue
            seen.add(toks)
            if not toks:
                out.append(ginn.Mobius(1.0 + 0j, 0.0 + 0j, ""))
            else:
                out.append(_compose_token_word(toks, gens))
        return out

    # Small self-check on the combinatorics.  This verifies the <=2 convention,
    # independent of any particular surface.
    def _expected(m: int, d: int) -> int:
        if d <= 0 or m <= 0:
            return 1
        total = 1
        shell = 1
        for k in range(1, d + 1):
            shell = m if k == 1 else shell * max(0, m - 1)
            total += shell
        return int(total)

    build_word_ball_leq_depth._janko_word_ball_convention = "reduced_length_leq_depth"  # type: ignore[attr-defined]
    build_word_ball_leq_depth._janko_depth2_formula = "1 + m + m*(m-1)"  # type: ignore[attr-defined]
    build_word_ball_leq_depth._janko_expected_size = _expected  # type: ignore[attr-defined]
    ginn.build_word_ball = build_word_ball_leq_depth


def factor_int(n: int) -> Dict[int, int]:
    d: Dict[int, int] = {}
    x = int(n)
    f = 2
    while f * f <= x:
        while x % f == 0:
            d[f] = d.get(f, 0) + 1
            x //= f
        f += 1 if f == 2 else 2
    if x > 1:
        d[x] = d.get(x, 0) + 1
    return d


# -----------------------------------------------------------------------------
# GF(4): values are 0, 1, 2=omega, 3=omega^2=omega+1.
# Addition is XOR; omega^2 + omega + 1 = 0.
# -----------------------------------------------------------------------------

def gf4_add(a: int, b: int) -> int:
    return int(a) ^ int(b)


def gf4_mul(a: int, b: int) -> int:
    a = int(a); b = int(b)
    a0, a1 = a & 1, (a >> 1) & 1
    b0, b1 = b & 1, (b >> 1) & 1
    c0 = (a0 & b0) ^ (a1 & b1)
    c1 = (a0 & b1) ^ (a1 & b0) ^ (a1 & b1)
    return c0 | (c1 << 1)


def gf4_str(a: int) -> str:
    return {0: "0", 1: "1", 2: "w", 3: "w^2"}[int(a)]


@dataclass
class JankoMatrixGroup:
    name: str
    field: str                 # "prime" or "gf4"
    field_order: int
    dim: int
    x: Matrix
    y: Matrix
    expected_order: int
    identity: Matrix = field(init=False)
    order: int = 0
    order_factors: Dict[int, int] = field(default_factory=dict)
    elements: List[Matrix] = field(default_factory=list)
    elem_to_id: Dict[Matrix, int] = field(default_factory=dict)
    parent_id: List[int] = field(default_factory=list)
    parent_tok: List[str] = field(default_factory=list)
    element_orders: Dict[Matrix, int] = field(default_factory=dict)
    transitions: Dict[str, List[int]] = field(default_factory=dict)
    _right_gen_columns: Dict[str, List[List[Tuple[int, int]]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.identity = tuple(1 if i == j else 0 for i in range(self.dim) for j in range(self.dim))
        self.order_factors = factor_int(self.expected_order)
        # Sparse right-multiplication tables are filled after y^-1 is known in
        # enumerate_from_xy.  They make Cayley BFS much faster than generic n^3
        # multiplication because the Pellegrini generators are sparse.

    def _generator_columns(self, G: Matrix) -> List[List[Tuple[int, int]]]:
        n = self.dim
        cols: List[List[Tuple[int, int]]] = []
        for j in range(n):
            entries: List[Tuple[int, int]] = []
            for k in range(n):
                c = G[k*n + j]
                if c:
                    entries.append((k, c))
            cols.append(entries)
        return cols

    def prepare_right_generators(self, y_inv: Matrix) -> None:
        self._right_gen_columns = {
            "X": self._generator_columns(self.x),
            "Y": self._generator_columns(self.y),
            "Y^-1": self._generator_columns(y_inv),
        }

    def right_mul_token(self, A: Matrix, tok: str) -> Matrix:
        """Fast A * generator(tok) using sparse columns of the fixed generator."""
        n = self.dim
        cols = self._right_gen_columns.get(tok)
        if cols is None:
            raise KeyError(f"right generator {tok!r} has not been prepared")
        out = [0] * (n * n)
        if self.field == "prime":
            p = self.field_order
            for i in range(n):
                row = i * n
                for j, col_entries in enumerate(cols):
                    s = 0
                    for k, coeff in col_entries:
                        s += A[row + k] * coeff
                    out[row + j] = s % p
        else:
            for i in range(n):
                row = i * n
                for j, col_entries in enumerate(cols):
                    s = 0
                    for k, coeff in col_entries:
                        a = A[row + k]
                        if a:
                            s ^= gf4_mul(a, coeff)
                    out[row + j] = s
        return tuple(out)

    def add(self, a: int, b: int) -> int:
        if self.field == "prime":
            return (a + b) % self.field_order
        return gf4_add(a, b)

    def mul_scalar(self, a: int, b: int) -> int:
        if self.field == "prime":
            return (a * b) % self.field_order
        return gf4_mul(a, b)

    def matmul(self, A: Matrix, B: Matrix) -> Matrix:
        n = self.dim
        if self.field == "prime":
            p = self.field_order
            out: List[int] = [0] * (n * n)
            for i in range(n):
                row = i * n
                for k in range(n):
                    aik = A[row + k]
                    if aik:
                        brow = k * n
                        for j in range(n):
                            out[row + j] = (out[row + j] + aik * B[brow + j]) % p
            return tuple(out)
        else:
            out = [0] * (n * n)
            for i in range(n):
                row = i * n
                for k in range(n):
                    aik = A[row + k]
                    if aik:
                        brow = k * n
                        for j in range(n):
                            bj = B[brow + j]
                            if bj:
                                out[row + j] ^= gf4_mul(aik, bj)
            return tuple(out)

    def pow(self, A: Matrix, e: int) -> Matrix:
        R = self.identity
        B = A
        k = int(e)
        while k > 0:
            if k & 1:
                R = self.matmul(R, B)
            B = self.matmul(B, B)
            k >>= 1
        return R

    def order_of(self, A: Matrix) -> int:
        if A in self.element_orders:
            return self.element_orders[A]
        if self.order <= 0:
            # Before enumeration, use a conservative relation scan for small orders.
            R = self.identity
            for k in range(1, 1000):
                R = self.matmul(R, A)
                if R == self.identity:
                    self.element_orders[A] = k
                    return k
            raise RuntimeError("order_of called before enumeration and no small order found")
        r = self.order
        for prime, exp in self.order_factors.items():
            for _ in range(exp):
                if r % prime == 0 and self.pow(A, r // prime) == self.identity:
                    r //= prime
                else:
                    break
        self.element_orders[A] = r
        return r

    def inv_known_small(self, A: Matrix) -> Matrix:
        o = self.order_of(A)
        return self.pow(A, o - 1)

    def matrix_to_nested(self, A: Matrix) -> List[List[int]]:
        return [list(A[i*self.dim:(i+1)*self.dim]) for i in range(self.dim)]

    def matrix_entries_as_strings(self, A: Matrix) -> List[List[str]]:
        if self.field == "gf4":
            return [[gf4_str(A[i*self.dim+j]) for j in range(self.dim)] for i in range(self.dim)]
        return self.matrix_to_nested(A)  # type: ignore[return-value]

    def relation_audit(self) -> Dict[str, Any]:
        xy = self.matmul(self.x, self.y)
        return {
            "x2_identity": self.pow(self.x, 2) == self.identity,
            "y3_identity": self.pow(self.y, 3) == self.identity,
            "xy7_identity": self.pow(xy, 7) == self.identity,
            "x_order_before_enumeration": self.order_of(self.x),
            "y_order_before_enumeration": self.order_of(self.y),
            "xy_order_before_enumeration": self.order_of(xy),
        }

    def enumerate_from_xy(self, verbose: bool = False, progress_every: int = 25000) -> None:
        """Enumerate <x,y> by BFS using X,Y,Y^-1.

        Stores parent pointers for the same transversal used by the later
        Reidemeister-Schreier kernel construction.
        """
        t0 = time.perf_counter()
        y_inv = self.pow(self.y, 2)  # y^3 = I
        gens: List[Tuple[str, Matrix]] = [("X", self.x), ("Y", self.y), ("Y^-1", y_inv)]
        self.prepare_right_generators(y_inv)
        self.elements = [self.identity]
        self.elem_to_id = {self.identity: 0}
        self.parent_id = [-1]
        self.parent_tok = [""]
        self.transitions = {"X": [], "Y": [], "Y^-1": []}
        q: deque[int] = deque([0])
        last_report = 0
        while q:
            hid = q.popleft()
            h = self.elements[hid]
            if len(self.transitions["X"]) != hid:
                raise RuntimeError("internal transition table ordering error")
            for tok, g in gens:
                nh = self.right_mul_token(h, tok)
                nid = self.elem_to_id.get(nh)
                if nid is None:
                    nid = len(self.elements)
                    self.elem_to_id[nh] = nid
                    self.elements.append(nh)
                    self.parent_id.append(hid)
                    self.parent_tok.append(tok)
                    q.append(nid)
                    if len(self.elements) > self.expected_order:
                        raise RuntimeError(
                            f"{self.name} enumeration exceeded expected order {self.expected_order}; "
                            "check the finite-field matrices."
                        )
                self.transitions[tok].append(nid)
            if verbose and len(self.elements) - last_report >= progress_every:
                last_report = len(self.elements)
                print(f"[group {self.name}] enumerated {len(self.elements):,}/{self.expected_order:,} elements; queue={len(q):,}", flush=True)
        self.order = len(self.elements)
        self.order_factors = factor_int(self.order)
        if self.order != self.expected_order:
            raise RuntimeError(f"{self.name} generated order {self.order}, expected {self.expected_order}")
        if verbose:
            print(f"[group {self.name}] order={self.order:,} complete in {time.perf_counter()-t0:.2f}s", flush=True)

    def rep_word_tokens(self, element_id: int) -> List[str]:
        toks: List[str] = []
        i = int(element_id)
        while i > 0:
            toks.append(self.parent_tok[i])
            i = self.parent_id[i]
        toks.reverse()
        return toks

    def transition_id(self, element_id: int, tok: str) -> int:
        if self.transitions and tok in self.transitions and len(self.transitions[tok]) > element_id:
            return self.transitions[tok][element_id]
        if not self._right_gen_columns:
            self.prepare_right_generators(self.pow(self.y, 2))
        return self.elem_to_id[self.right_mul_token(self.elements[element_id], tok)]


# -----------------------------------------------------------------------------
# Pellegrini Janko matrices.
# -----------------------------------------------------------------------------

def flat_mod(rows: List[List[int]], p: int) -> Matrix:
    return tuple((int(v) % p) for row in rows for v in row)


def build_j1_group() -> JankoMatrixGroup:
    x_rows = [
        [0, 0, 0, 1, 0, 0, -1],
        [0, 0, 0, 0, 1, 0, 5],
        [0, 0, 0, 0, 0, 1, 2],
        [1, 0, 0, 0, 0, 0, -1],
        [0, 1, 0, 0, 0, 0, 5],
        [0, 0, 1, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, -1],
    ]
    y_rows = [
        [1, 0, 0, 0, 4, 0, 3],
        [0, 1, 0, 0, 8, 0, -1],
        [0, 0, 1, 1, 0, 0, 9],
        [0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 1, -1],
    ]
    return JankoMatrixGroup(
        name="J1", field="prime", field_order=11, dim=7,
        x=flat_mod(x_rows, 11), y=flat_mod(y_rows, 11), expected_order=EXPECTED_ORDERS["J1"],
    )


def build_j2_group() -> JankoMatrixGroup:
    w, w2 = 2, 3
    x_rows = [
        [0, 0, 1, 0, w, 0],
        [0, 0, 0, 1, 1, w2],
        [1, 0, 0, 0, 0, w],
        [0, 1, 0, 0, w2, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
    ]
    y_rows = [
        [1, 0, 0, 0, w2, w2],
        [0, 1, 0, 0, w, w2],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
    ]
    return JankoMatrixGroup(
        name="J2", field="gf4", field_order=4, dim=6,
        x=tuple(v for row in x_rows for v in row),
        y=tuple(v for row in y_rows for v in row),
        expected_order=EXPECTED_ORDERS["J2"],
    )


def build_group_by_name(name: str) -> JankoMatrixGroup:
    name = name.upper()
    if name == "J1":
        return build_j1_group()
    if name == "J2":
        return build_j2_group()
    raise ValueError(f"Unknown Janko group {name!r}; use J1, J2, or both.")


# -----------------------------------------------------------------------------
# Schreier kernel surface builder for Janko matrix quotients.
# -----------------------------------------------------------------------------

def invert_delta_tokens(tokens: Sequence[str]) -> List[str]:
    inv = []
    for t in reversed(tokens):
        inv.append("Y^-1" if t == "Y" else "Y" if t == "Y^-1" else "X")
    return inv


def tokens_to_word(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def build_coset_mobius_transversal(group: JankoMatrixGroup, delta_gens: Dict[str, Any], hurwitz: Any) -> List[Any]:
    M: List[Any] = [None] * group.order
    M[0] = hurwitz.su11_identity()
    for i in range(1, group.order):
        pid = group.parent_id[i]
        tok = group.parent_tok[i]
        M[i] = M[pid].compose(delta_gens[tok]).normalized()
    return M


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


def build_janko_triple_record(group: JankoMatrixGroup) -> Dict[str, Any]:
    xy = group.matmul(group.x, group.y)
    z = group.pow(xy, 6)  # (xy)^7=I, so inverse is (xy)^6
    return {
        "triple_index": 0,
        "source": "Pellegrini-Tamburini-Bellani explicit Janko Hurwitz generators",
        "group_name": group.name,
        "field": "GF(11)" if group.name == "J1" else "GF(4), omega^2 + omega + 1 = 0; encoding 0,1,2=omega,3=omega^2",
        "x": group.matrix_to_nested(group.x),
        "y": group.matrix_to_nested(group.y),
        "z": group.matrix_to_nested(z),
        "x_pretty": group.matrix_entries_as_strings(group.x),
        "y_pretty": group.matrix_entries_as_strings(group.y),
        "x_order": group.order_of(group.x),
        "y_order": group.order_of(group.y),
        "xy_order": group.order_of(xy),
        "z_order": group.order_of(z),
        "xyz_identity": group.matmul(xy, z) == group.identity,
        "generated_subgroup_order": group.order,
        "group_order": group.order,
        "surjective": group.order == group.expected_order,
        "genus": 1 + group.order // 84 if group.order % 84 == 0 else None,
    }


def build_schreier_kernel_surface_janko(
    group: JankoMatrixGroup,
    triple: Dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    hurwitz: Any,
    perf: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.perf_counter()
    delta_gens, delta_audit = hurwitz.build_delta_237_su11()
    coset_mobius = build_coset_mobius_transversal(group, delta_gens, hurwitz)
    coset_alpha = np.asarray([complex(m.alpha) for m in coset_mobius], dtype=np.complex128)
    coset_beta = np.asarray([complex(m.beta) for m in coset_mobius], dtype=np.complex128)
    delta_pair = {tok: (complex(delta_gens[tok].alpha), complex(delta_gens[tok].beta)) for tok in ["X", "Y", "Y^-1"]}
    base_verts = [complex(x, y) for x, y in delta_audit["base_orbifold_triangle_vertices"]]

    if perf is not None:
        perf.log("janko_coset_mobius_done", group_name=group.name, quotient_order=group.order)

    # Scan Schreier directed edges h --tok--> hs.  The finite image of
    # M_h * tok * M_hs^{-1} is identity by construction.
    selected_heap: List[Tuple[float, int, Dict[str, Any]]] = []
    selected_all: List[Dict[str, Any]] = []
    identity_like = 0
    scanned_slots = 0
    nonidentity_image_failures = 0
    raw_nonidentity = 0
    token_order = ["X", "Y", "Y^-1"]
    mode = str(args.kernel_generator_mode)
    limit = int(args.kernel_generator_limit)
    if mode == "all":
        limit = 0

    def keep_candidate(rec: Dict[str, Any]) -> None:
        nonlocal selected_heap, selected_all
        if mode == "all" or mode == "first":
            if mode == "all" or limit <= 0 or len(selected_all) < limit:
                selected_all.append(rec)
            return
        # shortest mode: maintain a max-heap by displacement using negative key.
        disp = float(rec["identity_displacement"])
        item = (-disp, int(rec["scan_index"]), rec)
        if limit <= 0:
            selected_all.append(rec)
        elif len(selected_heap) < limit:
            heapq.heappush(selected_heap, item)
        elif disp < -selected_heap[0][0]:
            heapq.heapreplace(selected_heap, item)

    progress_every = max(1, int(args.kernel_scan_progress_every))
    for hid, Mh in enumerate(coset_mobius):
        for tok in token_order:
            scanned_slots += 1
            hs_id = group.transition_id(hid, tok)
            # finite audit: h * s * rep(hs)^-1 maps to identity.
            # Since rep(hs) is the BFS representative of the finite element hs,
            # the equality is exact in the quotient construction.
            # Fast SU(1,1) computation of K = M_h * S_tok * M_hs^{-1}
            ah = complex(coset_alpha[hid]); bh = complex(coset_beta[hid])
            asg, bsg = delta_pair[tok]
            a1, b1 = su11_pair_compose(ah, bh, asg, bsg)
            ai, bi = su11_pair_inverse(complex(coset_alpha[hs_id]), complex(coset_beta[hs_id]))
            ak, bk = su11_pair_compose(a1, b1, ai, bi)
            disp = su11_pair_displacement_from_identity(ak, bk)
            if disp < args.identity_tol:
                identity_like += 1
                continue
            raw_nonidentity += 1
            trace_real = float(2.0 * ak.real)
            rec = {
                "scan_index": scanned_slots - 1,
                "source_coset_id": hid,
                "edge_token": tok,
                "target_coset_id": hs_id,
                "trace_real": trace_real,
                "identity_displacement": disp,
                "is_hyperbolic_by_trace": abs(trace_real) > 2.0 + 1.0e-8,
                "finite_image_identity": True,
                "su11": su11_pair_as_json(ak, bk),
            }
            keep_candidate(rec)
        if args.kernel_generator_mode == "first" and limit > 0 and len(selected_all) >= limit:
            break
        if args.verbose and hid > 0 and hid % progress_every == 0:
            kept = len(selected_all) if mode != "shortest" else len(selected_heap)
            print(f"[kernel {group.name}] scanned cosets={hid:,}/{group.order:,} slots={scanned_slots:,} raw_nonidentity={raw_nonidentity:,} kept={kept:,}", flush=True)

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
            f"{group.name} selected Schreier kernel edge generator {label}: "
            f"coset {rec['source_coset_id']} --{rec['edge_token']}--> {rec['target_coset_id']}; "
            f"selection={mode}; trace={float(rec['trace_real']):.8g}"
        )
        if len(audit_sample) < int(args.kernel_audit_sample_rows):
            audit_sample.append({k: v for k, v in rec.items() if k != "su11"})

    generator_export_complete = (mode == "all" and nonidentity_image_failures == 0)
    kernel_generator_scan_complete = not (mode == "first" and limit > 0 and len(selected_all) >= limit and scanned_slots < group.order * 3)

    max_tiles = int(args.max_tiles) if int(args.max_tiles) > 0 else group.order
    tiles = []
    for tile_idx in range(min(group.order, max_tiles)):
        a = complex(coset_alpha[tile_idx]); b = complex(coset_beta[tile_idx])
        verts = [su11_pair_apply(a, b, z) for z in base_verts]
        tiles.append({
            "tile_index": tile_idx,
            "coset_id": tile_idx,
            "vertices": [[float(z.real), float(z.imag)] for z in verts],
        })
        if args.verbose and tile_idx > 0 and tile_idx % max(1, int(args.tile_progress_every)) == 0:
            print(f"[tiles {group.name}] {tile_idx:,}/{min(group.order,max_tiles):,}", flush=True)

    tile_scaffold_complete = len(tiles) == group.order
    relation_max_error = max(delta_audit["relations_numerical"].values())
    genus = triple["genus"]
    finite_certificate_complete = (
        triple.get("surjective") is True
        and triple.get("x_order") == 2
        and triple.get("y_order") == 3
        and triple.get("xy_order") == 7
        and triple.get("xyz_identity") is True
    )

    atlas_training_ready = bool(gen_json) and bool(tiles) and finite_certificate_complete and relation_max_error < 1.0e-8
    pass_geometry_audit = atlas_training_ready and generator_export_complete and tile_scaffold_complete
    partial_reasons = []
    if not generator_export_complete:
        partial_reasons.append(f"kernel generator export is selected/truncated: mode={mode}, selected={len(gen_json)}, raw_nonidentity={raw_nonidentity}")
    if not kernel_generator_scan_complete:
        partial_reasons.append("kernel edge scan stopped early")
    if not tile_scaffold_complete:
        partial_reasons.append("triangle-tile scaffold truncated by --max-tiles")
    if relation_max_error >= 1.0e-8:
        partial_reasons.append("PSU(1,1) triangle relation numerical error above tolerance")
    if not finite_certificate_complete:
        partial_reasons.append("finite quotient certificate failed")
    if not gen_json:
        partial_reasons.append("no nonidentity kernel generators exported")
    exclusion_reason = "; ".join(partial_reasons)

    sid = f"hurwitz_{group.name}_kernel_{mode}{len(gen_json)}"
    surface = {
        "format": "FuchsianGENN surface JSON v1.0 janko-hurwitz-tokenized-kernel",
        "surface_id": sid,
        "name": f"Hurwitz PSU(1,1) kernel surface from {group.name} explicit Janko quotient",
        "surface_type": "janko_hurwitz_triangle_kernel_surface",
        "domain_type": "triangle_kernel_tile_union",
        "compact": True,
        "finite_area": True,
        "torsion_free": True,
        "orbifold_excluded": False,
        "mainline_dataset_eligible": pass_geometry_audit,
        "atlas_training_ready": atlas_training_ready,
        "riemann_surface_status": "smooth compact Hurwitz Riemann surface D/Gamma, with Gamma the torsion-free kernel of Delta^+(2,3,7) -> finite Janko quotient",
        "kahler_status": "complex dimension one; automatically Kähler",
        "genus": genus,
        "area": 4.0 * math.pi * (genus - 1) if genus is not None else None,
        "gauss_bonnet_area": 4.0 * math.pi * (genus - 1) if genus is not None else None,
        "triangle_group": "Delta^+(2,3,7)",
        "triangle_signature": [2, 3, 7],
        "finite_quotient": group.name,
        "finite_quotient_order": group.order,
        "quotient_order": group.order,
        "field": "GF(11)" if group.name == "J1" else "GF(4)",
        "ginn_ready": atlas_training_ready,
        "explorer_loadable": False,
        "v1_0_janko_tokenized_generators": True,
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
        "generators": gen_json,
        "generator_meanings": meanings,
        "kernel_generator_audit_sample": audit_sample,
        "fundamental_domain_tiles": tiles,
        "tile_scaffold_warning": "Tile union is built from Delta(2,3,7) orbifold triangle coset representatives. Full scaffold means all quotient cosets are present. It is a computational sampling scaffold, not a polished side-paired compact polygon.",
        "finite_group_triple": triple,
        "psu11_triangle_audit": delta_audit,
        "schreier_audit": {
            "transversal_size": group.order,
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
            "status": "complete_ginn_ready_janko_hurwitz_kernel_surface" if pass_geometry_audit else "selected_generator_janko_hurwitz_kernel_surface",
            "finite_quotient_certificate": "exact finite-field matrix relations and BFS generation order check",
            "psu11_triangle_certificate": "numerical SU(1,1) Delta(2,3,7) relation checks",
            "kernel_certificate": "selected Reidemeister-Schreier generators k=t*s*rep(ts)^-1 map to identity in the finite quotient",
            "remaining_caveat": "Unless --kernel-generator-mode all is used, the exported generator set is a selected computational pool, not a complete kernel generator set or minimal side-paired polygon presentation.",
        },
        "maker_run_id": run_id,
    }
    audit = {
        "surface_id": sid,
        "group": group.name,
        "field_order": group.field_order,
        "quotient_order": group.order,
        "genus": genus,
        "transversal_size": group.order,
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


# -----------------------------------------------------------------------------
# Reports and CLI.
# -----------------------------------------------------------------------------

def write_janko_report(run_root: Path, args: argparse.Namespace, surface_rows: List[Dict[str, Any]], atlas_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Fuchsian Janko Hurwitz Trainer v1.1 Report")
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This run uses explicit matrix Hurwitz generators for J1 and/or J2, constructs the corresponding Delta^+(2,3,7) kernel surface scaffold, builds an exact finite word-ball top-k atlas, and optionally trains a candidate-pool GINN reranker.")
    lines.append("")
    lines.append("## Practical caveat")
    lines.append("")
    lines.append("For Janko quotients, the complete Reidemeister-Schreier generator set is huge. Unless `--kernel-generator-mode all` was used, the exported generators are a selected computational pool, normally the shortest nonidentity Schreier edge generators by SU(1,1) displacement. The atlas is exact for the selected finite word ball, not for the full infinite kernel.")
    lines.append("")
    lines.append("## Word-ball convention")
    lines.append("")
    lines.append("Version 1.1 explicitly installs a local reduced word-ball builder using the length `<= depth` convention. Thus for `m` oriented letters and `depth = 2`, the raw reduced word-ball size is `1 + m + m(m-1)`, including the identity, all length-1 words, and all reduced length-2 words.")
    lines.append("")
    lines.append("## Run parameters")
    for k in ["group", "kernel_generator_mode", "kernel_generator_limit", "depth", "pairs", "top_k_max", "train_pool_size", "epochs", "engine", "candidate_chunk_size", "pair_batch_size", "target_vram_mb", "stream_huge_word_ball"]:
        lines.append(f"- `{k}`: `{getattr(args, k, '')}`")
    lines.append("")
    lines.append("## Surface summary")
    if surface_rows:
        cols = ["surface_id", "group", "quotient_order", "genus", "kernel_generators_exported", "raw_nonidentity_schreier_generators", "tile_count", "tile_scaffold_complete", "atlas_training_ready"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in surface_rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Atlas summary")
    if atlas_rows:
        cols = ["surface_id", "word_ball_size_raw", "word_ball_size_unique", "n_pairs", "engine", "wall_seconds", "evals_per_second", "shortcut_fraction", "median_gap12"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in atlas_rows:
            lines.append("| " + " | ".join(str(round(r.get(c), 5)) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Training summary")
    if train_rows:
        cols = ["surface_id", "unique_word_ball_size", "pool_size", "train_pairs", "val_pairs", "test_pairs", "device", "epochs_ran", "test_recall_at_1", "test_recall_at_5", "test_recall_at_20", "test_top5_pruned_rmse", "train_seconds"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in train_rows:
            lines.append("| " + " | ".join(str(round(r.get(c), 5)) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    (run_root / "report").mkdir(parents=True, exist_ok=True)
    (run_root / "report" / "janko_hurwitz_training_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build J1/J2 Hurwitz-kernel atlases and train candidate-pool GINN rerankers.")
    ap.add_argument("--group", choices=["J1", "J2", "both"], default="J1")
    ap.add_argument("--mode", choices=["smoke", "atlas", "train", "surface-only", "relations"], default="train")
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--smoke-pairs", type=int, default=24)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--top-k-max", type=int, default=100)
    ap.add_argument("--top-k-list", type=parse_int_list, default=[1, 3, 5, 10, 20, 50, 100])
    ap.add_argument("--csv-top-k", type=int, default=20)
    ap.add_argument("--candidate-chunk-size", type=int, default=8192)
    ap.add_argument("--pair-batch-size", type=int, default=0)
    ap.add_argument("--engine", choices=["auto", "gpu_torch", "cpu_vec", "cpu_loop"], default="auto")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--target-ram-mb", type=float, default=16384.0)
    ap.add_argument("--target-vram-mb", type=float, default=8192.0)
    ap.add_argument("--max-word-ball", type=int, default=5000000)
    ap.add_argument("--max-unique-word-ball", type=int, default=5000000)
    ap.add_argument("--allow-huge-word-ball", action="store_true")
    ap.add_argument("--stream-huge-word-ball", action="store_true", help="Stream virtual reduced depth-2 word ball instead of allocating the explicit raw Python word list when the preflight estimate exceeds --max-word-ball.")
    ap.add_argument("--virtual-topk-buffer", type=int, default=5000)
    ap.add_argument("--virtual-topk-dedupe-tol", type=float, default=0.0)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--dedupe-tol", type=float, default=1.0e-10)
    ap.add_argument("--alias-summary-rows", type=int, default=500)
    ap.add_argument("--alias-sample-limit", type=int, default=8)
    ap.add_argument("--pool-sizes", type=str, default="128,256,512,1024")
    ap.add_argument("--frequency-rows", type=int, default=200)
    ap.add_argument("--write-word-ball-summary", action="store_true")
    ap.add_argument("--outroot", type=str, default="janko_hurwitz_training_runs")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--zoo-script", type=str, default=default_zoo_script())
    ap.add_argument("--trainer-script", type=str, default=default_trainer_script())
    ap.add_argument("--hurwitz-script", type=str, default=default_hurwitz_script())
    ap.add_argument("--ginn-script", type=str, default=default_ginn_script())
    ap.add_argument("--identity-tol", type=float, default=1.0e-9)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-perf-log", action="store_true")
    ap.add_argument("--skip-atlas", action="store_true")
    ap.add_argument("--no-train", action="store_true")

    # Janko-specific surface construction controls.
    ap.add_argument("--kernel-generator-mode", choices=["shortest", "first", "all"], default="shortest")
    ap.add_argument("--kernel-generator-limit", type=int, default=1024, help="Number of positive Schreier edge generators to export for shortest/first modes. 0 means unlimited for shortest; all mode ignores this and exports all.")
    ap.add_argument("--kernel-audit-sample-rows", type=int, default=200)
    ap.add_argument("--max-tiles", type=int, default=0, help="0 means full quotient coset tile scaffold; set a smaller number only for smoke/debugging.")
    ap.add_argument("--kernel-scan-progress-every", type=int, default=25000)
    ap.add_argument("--tile-progress-every", type=int, default=50000)

    # Training options, aligned with BigHurwitzTrainer v1.7.
    ap.add_argument("--train-pool-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=20)
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


def maybe_smoke_adjust(args: argparse.Namespace) -> None:
    if args.mode == "smoke":
        args.pairs = int(args.smoke_pairs)
        args.top_k_max = min(int(args.top_k_max), 20)
        args.csv_top_k = min(int(args.csv_top_k), int(args.top_k_max))
        args.epochs = 1
        args.min_epochs = 1
        args.patience = 1
        args.train_pool_size = min(int(args.train_pool_size), 64)
        args.kernel_generator_limit = min(int(args.kernel_generator_limit), 64)
        # Smoke mode should test the full pipeline quickly; use first rather
        # than shortest so we do not scan every Janko Schreier edge just to
        # keep a tiny pilot generator pool.
        if args.kernel_generator_mode == "shortest":
            args.kernel_generator_mode = "first"
        if args.max_tiles == 0:
            args.max_tiles = 512
        if not args.label:
            args.label = "smoke"


def run_one_group(args: argparse.Namespace, group_name: str, run_root: Path, zoo: Any, trainer_mod: Any, hurwitz: Any, ginn: Any, perf: Optional[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    surface_rows: List[Dict[str, Any]] = []
    atlas_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    print("=" * 78)
    print(f"[janko] building {group_name}", flush=True)
    group = build_group_by_name(group_name)
    rel = group.relation_audit()
    write_json(run_root / "group" / f"{group_name}_relation_audit.json", rel)
    print(f"[relations {group_name}] x2={rel['x2_identity']} y3={rel['y3_identity']} xy7={rel['xy7_identity']}", flush=True)
    if args.mode == "relations":
        return surface_rows, atlas_rows, train_rows, failures

    if not (rel["x2_identity"] and rel["y3_identity"] and rel["xy7_identity"]):
        raise RuntimeError(f"{group_name} matrix relation audit failed: {rel}")
    group.enumerate_from_xy(verbose=args.verbose)
    triple = build_janko_triple_record(group)
    write_json(run_root / "group" / f"{group_name}_group_summary.json", {
        "group": group.name,
        "field": group.field,
        "field_order": group.field_order,
        "matrix_dimension": group.dim,
        "order": group.order,
        "expected_order": group.expected_order,
        "order_factors": factor_int(group.order),
        "genus": triple["genus"],
        "triple": triple,
    })

    if perf is not None:
        perf.log("janko_group_enumerated", group_name=group.name, quotient_order=group.order, genus=triple["genus"])

    surface, audit = build_schreier_kernel_surface_janko(group, triple, args, run_root.name, hurwitz, perf=perf)
    sid = str(surface["surface_id"])
    write_json(run_root / "surfaces" / f"{sid}.json", surface)
    write_json(run_root / "kernel_audits" / f"{sid}_audit.json", audit)
    surface_rows.append({
        "surface_id": sid,
        "group": group.name,
        "quotient_order": group.order,
        "genus": triple["genus"],
        "kernel_generators_exported": audit["kernel_generators_exported"],
        "raw_nonidentity_schreier_generators": audit["raw_nonidentity_schreier_generators"],
        "tile_count": audit["tile_count"],
        "tile_scaffold_complete": audit["tile_scaffold_complete"],
        "atlas_training_ready": audit["atlas_training_ready"],
        "mainline_dataset_eligible": audit["mainline_dataset_eligible"],
        "exclusion_reason": audit["exclusion_reason"],
    })
    if args.mode == "surface-only":
        return surface_rows, atlas_rows, train_rows, failures

    # BigHurwitzZoo expects args.q as an int for seed/reporting.  Use field_order;
    # surface_id and finite_quotient keep the real Janko identity.
    old_q = getattr(args, "q", None)
    args.q = int(group.field_order)

    atlas_result = None
    if not args.skip_atlas:
        try:
            atlas_result = zoo.atlas_for_surface(args, ginn, surface, run_root, 0, perf=perf)
            atlas_rows.append(atlas_result.__dict__)
        except Exception as e:
            print(f"[atlas fail] {sid}: {type(e).__name__}: {e}", flush=True)
            failures.append({"stage": "atlas", "surface_id": sid, "error_type": type(e).__name__, "error": str(e)})
    if args.no_train or args.mode == "atlas":
        if old_q is not None:
            args.q = old_q
        return surface_rows, atlas_rows, train_rows, failures

    try:
        train_result = trainer_mod.train_surface_reranker(args, zoo, ginn, surface, run_root, perf=perf)
        row = train_result.__dict__
        row["group"] = group.name
        train_rows.append(row)
    except Exception as e:
        print(f"[train fail] {sid}: {type(e).__name__}: {e}", flush=True)
        failures.append({"stage": "train", "surface_id": sid, "error_type": type(e).__name__, "error": str(e)})
    if old_q is not None:
        args.q = old_q
    return surface_rows, atlas_rows, train_rows, failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    maybe_smoke_adjust(args)
    pool_sizes = sorted(set([x for x in parse_int_list(args.pool_sizes) if x > 0] + [int(args.train_pool_size)]))
    args.pool_sizes = ",".join(str(x) for x in pool_sizes)
    if not args.label:
        args.label = f"{args.group}_depth{args.depth}_kg{args.kernel_generator_mode}{args.kernel_generator_limit}_pairs{args.pairs}"

    run_root = Path(args.outroot) / f"run_{now_stamp()}_{stable_slug(args.label)}"
    for sub in ["group", "surfaces", "kernel_audits", "atlas", "training", "tables", "report"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    print(f"{PROGRAM} v{VERSION}")
    print(f"run_root={run_root}")
    print(f"group={args.group} depth={args.depth} pairs={args.pairs} kernel_mode={args.kernel_generator_mode} kernel_limit={args.kernel_generator_limit}")
    print("-" * 78)

    zoo = load_module(args.zoo_script, "big_hurwitz_zoo_for_janko")
    trainer_mod = load_module(args.trainer_script, "big_hurwitz_trainer_for_janko")
    hurwitz = load_module(args.hurwitz_script, "hurwitz_for_janko")
    ginn = load_module(args.ginn_script, "ginn_for_janko")
    install_leq_depth_word_ball_builder(ginn)
    perf = zoo.PerfTracker(run_root / "tables" / "performance_log.csv", enabled=(not args.no_perf_log))

    # Let imported trainer write Janko program name in checkpoints where it uses globals.
    try:
        trainer_mod.PROGRAM = PROGRAM
        trainer_mod.VERSION = VERSION
    except Exception:
        pass

    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "args": vars(args).copy(),
        "python": sys.version,
        "platform": platform.platform(),
        "purpose": "Janko Hurwitz quotient finite-field certification plus Fuchsian atlas plus candidate-pool GINN training.",
    }
    write_json(run_root / "manifest.json", manifest)

    t_all = time.perf_counter()
    all_surface_rows: List[Dict[str, Any]] = []
    all_atlas_rows: List[Dict[str, Any]] = []
    all_train_rows: List[Dict[str, Any]] = []
    all_failures: List[Dict[str, Any]] = []
    try:
        group_names = ["J1", "J2"] if args.group == "both" else [args.group]
        for gname in group_names:
            srows, arows, trows, fails = run_one_group(args, gname, run_root, zoo, trainer_mod, hurwitz, ginn, perf)
            all_surface_rows.extend(srows)
            all_atlas_rows.extend(arows)
            all_train_rows.extend(trows)
            all_failures.extend(fails)

        write_csv(run_root / "tables" / "janko_surface_summary.csv", all_surface_rows)
        write_csv(run_root / "tables" / "janko_atlas_summary.csv", all_atlas_rows)
        write_csv(run_root / "tables" / "janko_training_summary.csv", all_train_rows)
        if all_failures:
            write_csv(run_root / "tables" / "failures.csv", all_failures)
        write_janko_report(run_root, args, all_surface_rows, all_atlas_rows, all_train_rows)
        summary = {
            "program": PROGRAM,
            "version": VERSION,
            "wall_seconds_total": time.perf_counter() - t_all,
            "surfaces": len(all_surface_rows),
            "atlases_completed": len(all_atlas_rows),
            "trained_surfaces": len(all_train_rows),
            "failures": len(all_failures),
            "run_root": str(run_root),
            "process_peak_rss_mb": getattr(perf, "peak_rss_mb", None),
        }
        write_json(run_root / "run_summary.json", summary)
        perf.log("run_done", surfaces_built=len(all_surface_rows), atlases_completed=len(all_atlas_rows), trained_surfaces=len(all_train_rows), failures=len(all_failures))
        perf.write()
        print("=" * 78)
        print(f"[done] surfaces={len(all_surface_rows)} atlases={len(all_atlas_rows)} trained={len(all_train_rows)} failures={len(all_failures)}")
        print(f"[done] run_root={run_root}")
        return 0 if not all_failures else 1
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
