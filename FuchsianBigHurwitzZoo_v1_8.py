#!/usr/bin/env python3
"""
FuchsianBigHurwitzZoo_v1_4.py

Big-group atlas generator for Hurwitz surfaces coming from PSL(2,p) quotients
of Delta^+(2,3,7).  Version 1.2 is deliberately NOT a GINN trainer and does
NOT benchmark classical search.  Its job is to build large certified Hurwitz
kernel surfaces and generate compact exact finite-word branch-atlas data.

Version 1.2 added the first production-oriented optimizations:
  * projective PSU(1,1) geometric deduplication of the word ball;
  * contiguous alpha/beta arrays rather than Python Mobius-object loops;
  * dense NumPy CPU streaming and optional PyTorch CUDA streaming;
  * performance logging for wall time, RAM, VRAM, CPU, threads, and throughput.

Default first target: PSL(2,13), PGL-reduced generating triples, depth 2.
Version 1.7 adds local top-k geometric deduplication to the streamed depth-2 virtual-word-ball engine for large PSL(2,p) cases such as p=29. It avoids allocating the full raw Python word list and streams reduced depth-2 products in chunks.

For each selected surface, the program:
  1. builds a complete Schreier-kernel surface using FuchsianHurwitzTester_v1_6;
  2. builds the requested finite word ball using FuchsianDownstairsGINN_v2_4;
  3. samples fresh point pairs in the kernel's triangle-tile scaffold;
  4. deduplicates word aliases representing the same geometric lift;
  5. streams exact hyperbolic distances over the deduplicated finite word ball;
  6. saves only compact top-k atlas data and optional hard-negative pools.

The full distance matrix is never written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROGRAM = "FuchsianBigHurwitzZoo_v1_8.py"
VERSION = "1.8"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    # Make imports local to the module work, e.g. FuchsianSurfaceRecordTools_v1_0.
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


def default_hurwitz_script() -> str:
    local = Path.cwd() / "FuchsianHurwitzTester_v1_6.py"
    if local.exists():
        return str(local)
    return "FuchsianHurwitzTester_v1_6.py"


def default_ginn_script() -> str:
    local = Path.cwd() / "FuchsianDownstairsGINN_v2_4.py"
    if local.exists():
        return str(local)
    return "FuchsianDownstairsGINN_v2_4.py"


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    if not s:
        return out
    for part in str(s).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def is_prime_int(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def psl2_prime_hurwitz_eligible(q: int) -> bool:
    """Macbeath condition for prime q in the PSL(2,q) family used here."""
    q = int(q)
    return q == 7 or (is_prime_int(q) and (q % 7 in (1, 6)))


def psl2_prime_order(q: int) -> int:
    q = int(q)
    return q * (q * q - 1) // 2


def estimate_reduced_word_ball_size(oriented_generator_count: int, depth: int) -> int:
    """Estimate reduced word-ball size before building a huge Python word list."""
    m = int(oriented_generator_count)
    d = int(depth)
    if d <= 0 or m <= 0:
        return 1
    total = 1
    shell = 1
    for k in range(1, d + 1):
        if k == 1:
            shell = m
        else:
            shell *= max(0, m - 1)
        total += shell
    return int(total)


def print_q_preflight(q: int) -> None:
    q = int(q)
    order = psl2_prime_order(q) if is_prime_int(q) and q > 2 else None
    genus = 1 + order // 84 if order is not None and order % 84 == 0 else None
    eligible = psl2_prime_hurwitz_eligible(q)
    print(f"[q-preflight] q={q} prime={is_prime_int(q)} PSL2_order={order if order is not None else 'n/a'} hurwitz_prime_condition={eligible} genus={genus if genus is not None else 'n/a'}", flush=True)
    if not is_prime_int(q):
        print("[q-preflight warn] This implementation currently uses PSL2PrimeGroup and expects prime q. Prime powers such as q=8 or 27 need a different finite-field engine.", flush=True)
    if is_prime_int(q) and not eligible:
        print("[q-preflight warn] Prime q does not satisfy the PSL(2,p) Hurwitz congruence condition p=7 or p ≡ ±1 mod 7; triple search may find no surfaces.", flush=True)


def sha1_strings(vals: Sequence[str], sample: Optional[int] = None) -> str:
    h = hashlib.sha1()
    if sample is None or len(vals) <= 2 * sample:
        iterable = vals
    else:
        iterable = list(vals[:sample]) + ["..."] + list(vals[-sample:])
    for v in iterable:
        h.update(str(v).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def ensure_topk(k_list: Sequence[int], top_k_max: int) -> List[int]:
    """Return a safe sorted reporting top-k list.

    v1.0 raised if the default reporting list contained 50 or 100 after smoke
    mode reduced --top-k-max to 20.  For this atlas program the list is only a
    reporting list, so v1.2 clips it to the available saved top-k depth and
    prints a warning instead of aborting.
    """
    top_k_max = max(1, int(top_k_max))
    requested = sorted({int(k) for k in k_list if int(k) > 0})
    if not requested:
        requested = [1, 3, 5, 10, 20]
    omitted = [k for k in requested if k > top_k_max]
    out = [k for k in requested if k <= top_k_max]
    if not out:
        out = [top_k_max]
    if omitted:
        print(
            f"[top-k warn] reporting list entries {omitted} exceed --top-k-max={top_k_max}; "
            f"using {out}",
            flush=True,
        )
    return out


def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _proc_rss_mb_fallback() -> Optional[float]:
    # Linux /proc fallback; returns resident set size in MB.
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        pass
    return None


def _system_mem_mb_fallback() -> Dict[str, Optional[float]]:
    vals: Dict[str, float] = {}
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as f:
            for line in f:
                if ':' not in line:
                    continue
                key, rest = line.split(':', 1)
                m = re.search(r'(\d+)', rest)
                if m:
                    vals[key] = float(m.group(1)) / 1024.0
        total = vals.get('MemTotal')
        avail = vals.get('MemAvailable')
        used = total - avail if total is not None and avail is not None else None
        return {'system_ram_total_mb': total, 'system_ram_available_mb': avail, 'system_ram_used_mb': used}
    except Exception:
        return {'system_ram_total_mb': None, 'system_ram_available_mb': None, 'system_ram_used_mb': None}


def _query_nvidia_smi() -> Dict[str, Optional[float]]:
    if shutil.which('nvidia-smi') is None:
        return {
            'gpu_count': 0.0,
            'gpu_vram_used_mb': None,
            'gpu_vram_total_mb': None,
            'gpu_util_percent': None,
        }
    cmd = [
        'nvidia-smi',
        '--query-gpu=memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=3.0)
        used_vals: List[float] = []
        total_vals: List[float] = []
        util_vals: List[float] = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 3:
                used_vals.append(float(parts[0]))
                total_vals.append(float(parts[1]))
                util_vals.append(float(parts[2]))
        if not used_vals:
            raise RuntimeError('nvidia-smi returned no GPU rows')
        return {
            'gpu_count': float(len(used_vals)),
            'gpu_vram_used_mb': float(sum(used_vals)),
            'gpu_vram_total_mb': float(sum(total_vals)),
            'gpu_util_percent': float(sum(util_vals) / len(util_vals)),
        }
    except Exception:
        return {
            'gpu_count': None,
            'gpu_vram_used_mb': None,
            'gpu_vram_total_mb': None,
            'gpu_util_percent': None,
        }


class PerfTracker:
    """Small terminal/CSV performance logger for long atlas runs."""

    def __init__(self, csv_path: Path, enabled: bool = True):
        self.enabled = bool(enabled)
        self.csv_path = csv_path
        self.autosave = True
        self.t0 = time.perf_counter()
        self.psutil = _try_import_psutil()
        self.proc = None
        if self.psutil is not None:
            try:
                self.proc = self.psutil.Process(os.getpid())
            except Exception:
                self.proc = None
        self.rows: List[Dict[str, Any]] = []
        self.peak_rss_mb = 0.0
        if self.enabled:
            self.log('perf_start')

    def snapshot(self, label: str, **extra: Any) -> Dict[str, Any]:
        wall = time.perf_counter() - self.t0
        rss = None
        sys_mem: Dict[str, Optional[float]]
        if self.proc is not None:
            try:
                rss = float(self.proc.memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                rss = None
        if rss is None:
            rss = _proc_rss_mb_fallback()
        if self.psutil is not None:
            try:
                vm = self.psutil.virtual_memory()
                sys_mem = {
                    'system_ram_total_mb': float(vm.total) / (1024.0 * 1024.0),
                    'system_ram_available_mb': float(vm.available) / (1024.0 * 1024.0),
                    'system_ram_used_mb': float(vm.used) / (1024.0 * 1024.0),
                }
            except Exception:
                sys_mem = _system_mem_mb_fallback()
        else:
            sys_mem = _system_mem_mb_fallback()
        gpu = _query_nvidia_smi()
        proc_cpu_percent = None
        process_threads = None
        system_cpu_percent = None
        load1 = load5 = load15 = None
        if self.proc is not None:
            try:
                proc_cpu_percent = float(self.proc.cpu_percent(interval=None))
            except Exception:
                proc_cpu_percent = None
            try:
                process_threads = int(self.proc.num_threads())
            except Exception:
                process_threads = None
        if self.psutil is not None:
            try:
                system_cpu_percent = float(self.psutil.cpu_percent(interval=None))
            except Exception:
                system_cpu_percent = None
        try:
            la = os.getloadavg()
            load1, load5, load15 = float(la[0]), float(la[1]), float(la[2])
        except Exception:
            pass
        if rss is not None:
            self.peak_rss_mb = max(self.peak_rss_mb, float(rss))
        row: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'label': label,
            'wall_seconds': wall,
            'process_rss_mb': rss,
            'process_peak_rss_mb': self.peak_rss_mb if self.peak_rss_mb > 0 else None,
            'process_cpu_percent': proc_cpu_percent,
            'process_threads': process_threads,
            'system_cpu_percent': system_cpu_percent,
            'load1': load1,
            'load5': load5,
            'load15': load15,
        }
        row.update(sys_mem)
        row.update(gpu)
        row.update(extra)
        return row

    def log(self, label: str, **extra: Any) -> Dict[str, Any]:
        row = self.snapshot(label, **extra)
        self.rows.append(row)
        if self.enabled:
            def fmt(x: Any, nd: int = 1) -> str:
                try:
                    if x is None or x == '':
                        return 'NA'
                    return f'{float(x):.{nd}f}'
                except Exception:
                    return str(x)
            sys_total = row.get('system_ram_total_mb')
            gpu_total = row.get('gpu_vram_total_mb')
            sys_str = (
                f"sys_used={fmt(row.get('system_ram_used_mb'), 0)}/{fmt(sys_total, 0)}MB"
                if sys_total is not None else
                f"sys_used={fmt(row.get('system_ram_used_mb'), 0)}MB"
            )
            vram_str = (
                f"vram={fmt(row.get('gpu_vram_used_mb'), 0)}/{fmt(gpu_total, 0)}MB"
                if gpu_total is not None else
                "vram=NA"
            )
            gpu_util_str = (
                f"gpu_util={fmt(row.get('gpu_util_percent'), 0)}%"
                if row.get('gpu_util_percent') is not None else
                "gpu_util=NA"
            )
            cpu_str = (
                f"proc_cpu={fmt(row.get('process_cpu_percent'), 0)}% threads={row.get('process_threads')} "
                f"sys_cpu={fmt(row.get('system_cpu_percent'), 0)}% load1={fmt(row.get('load1'), 2)}"
            )
            print(
                '[perf] '
                f"{label}  t={fmt(row.get('wall_seconds'), 1)}s  "
                f"rss={fmt(row.get('process_rss_mb'), 1)}MB  "
                f"peak={fmt(row.get('process_peak_rss_mb'), 1)}MB  "
                f"{sys_str}  {vram_str}  {gpu_util_str}  {cpu_str}",
                flush=True,
            )
        if self.autosave:
            try:
                self.write()
            except Exception:
                pass
        return row

    def write(self) -> None:
        if self.rows:
            write_csv(self.csv_path, self.rows)


@dataclass
class AtlasResult:
    surface_id: str
    q: int
    triple_index: int
    genus: int
    generator_count: int
    word_ball_size: int
    word_ball_size_raw: int
    word_ball_size_unique: int
    word_ball_alias_reduction_fraction: float
    word_ball_alias_max_count: int
    word_ball_alias_mean_count: float
    n_pairs: int
    depth: int
    top_k_max: int
    engine: str
    pair_batch_size: int
    candidate_chunk_size: int
    wall_seconds: float
    exact_candidate_evaluations: int
    exact_candidate_evaluations_raw: int
    evals_per_second: float
    identity_wins_fraction: float
    shortcut_fraction: float
    mean_winner_depth: float
    max_winner_depth: int
    median_gap12: float
    mean_gap12: float
    near_seam_0p02: float
    near_seam_0p05: float
    outdir: str


@dataclass
class WordBallData:
    raw_size: int
    unique_size: int
    alpha: np.ndarray
    beta: np.ndarray
    words: List[str]
    depths: np.ndarray
    traces: np.ndarray
    alias_counts: np.ndarray
    representative_raw_index: np.ndarray
    alias_samples: List[List[str]]
    dedupe_enabled: bool
    dedupe_tol: float

    @property
    def alias_reduction_fraction(self) -> float:
        if self.raw_size <= 0:
            return 0.0
        return float(1.0 - (self.unique_size / self.raw_size))


def fake_hurwitz_args(args: argparse.Namespace) -> SimpleNamespace:
    """Minimal args object expected by build_schreier_kernel_surface."""
    return SimpleNamespace(
        max_kernel_generators=args.max_kernel_generators,
        max_tiles=args.max_tiles,
        identity_tol=args.identity_tol,
    )


def build_surfaces(args: argparse.Namespace, hurwitz, run_root: Path, perf: Optional[PerfTracker] = None) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        perf.log("group_and_triples_built", q=args.q, n_triples=len(triples))

    surfaces: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    hargs = fake_hurwitz_args(args)
    run_id = run_root.name.replace("run_", "")
    for tr in triples:
        print(f"[surface] building kernel for triple {tr['triple_index']:04d}", flush=True)
        if perf is not None:
            perf.log("surface_build_start", triple_index=tr.get("triple_index"))
        surface, audit = hurwitz.build_schreier_kernel_surface(group, tr, hargs, run_id)
        # Versioned Big-Zoo annotations.
        surface["big_hurwitz_zoo_record"] = True
        surface["big_hurwitz_program"] = PROGRAM
        surface["big_hurwitz_version"] = VERSION
        surface["big_hurwitz_role"] = "exact_branch_atlas_surface"
        sid = str(surface.get("surface_id"))
        write_json(run_root / "surfaces" / f"{sid}.json", surface)
        write_json(run_root / "kernel_audits" / f"{sid}_audit.json", audit)
        surfaces.append(surface)
        audits.append(audit)
        if perf is not None:
            perf.log("surface_build_done", surface_id=sid, triple_index=tr.get("triple_index"), genus=surface.get("genus"), generator_count=surface.get("generator_count"), tile_count=surface.get("tile_count"))
    return group, triples, surfaces, audits


def sample_pairs(ginn, surface: Dict[str, Any], n_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray, str]:
    rng = random.Random(seed)
    pts, sampler_kind = ginn.sample_points_on_surface(surface, 2 * n_pairs, rng)
    p = pts[0::2, 0].astype(np.float64) + 1j * pts[0::2, 1].astype(np.float64)
    q = pts[1::2, 0].astype(np.float64) + 1j * pts[1::2, 1].astype(np.float64)
    if p.shape[0] != n_pairs or q.shape[0] != n_pairs:
        raise RuntimeError("point sampling produced the wrong number of pairs")
    return p, q, str(sampler_kind)


def canonical_psu_key(alpha: complex, beta: complex, tol: float) -> Tuple[int, int, int, int]:
    """Quantized PSU(1,1) key, identifying M and -M.

    SU(1,1) matrices +/-[[alpha,beta],[bar(beta),bar(alpha)]] give the same
    Mobius transformation.  The atlas should rank distinct lifts, not aliases
    of the same lift, so v1.2 deduplicates with this projective key.
    """
    tol = max(float(tol), 1.0e-14)
    a = complex(alpha)
    b = complex(beta)
    # Normalize small SU(1,1) drift where possible.
    norm = (abs(a) ** 2 - abs(b) ** 2)
    if np.isfinite(norm) and norm > 0:
        scale = math.sqrt(norm)
        if scale > 0:
            a /= scale
            b /= scale
    vals = [a.real, a.imag, b.real, b.imag]
    sign = 1.0
    for x in vals:
        if abs(x) > tol:
            if x < 0:
                sign = -1.0
            break
    a *= sign
    b *= sign
    return (
        int(round(a.real / tol)),
        int(round(a.imag / tol)),
        int(round(b.real / tol)),
        int(round(b.imag / tol)),
    )


def build_word_ball_data(
    ginn,
    word_ball: List[Any],
    dedupe: bool = True,
    tol: float = 1.0e-10,
    alias_sample_limit: int = 8,
) -> WordBallData:
    """Convert a word-ball object list into contiguous arrays, with optional
    geometric deduplication.

    This is the central v1.2 optimization/correction.  The raw PSL(2,13)
    kernel generator word ball can contain many word aliases of the same
    transformation.  Deduplication avoids artificial zero gaps and makes all
    downstream streaming operate on distinct geometric lifts.
    """
    raw_size = len(word_ball)
    alpha_list: List[complex] = []
    beta_list: List[complex] = []
    words: List[str] = []
    depths: List[int] = []
    traces: List[float] = []
    alias_counts: List[int] = []
    rep_raw: List[int] = []
    alias_samples: List[List[str]] = []
    key_to_unique: Dict[Tuple[int, int, int, int], int] = {}

    for raw_idx, m in enumerate(word_ball):
        word = str(getattr(m, 'word', '') or 'identity')
        alpha = complex(getattr(m, 'alpha'))
        beta = complex(getattr(m, 'beta'))
        depth = int(ginn.word_depth_string(getattr(m, 'word', '')))
        try:
            trace = float(m.trace_real()) if hasattr(m, 'trace_real') else float((2.0 * alpha.real))
        except Exception:
            trace = float('nan')
        if dedupe:
            key = canonical_psu_key(alpha, beta, tol)
            if key in key_to_unique:
                u = key_to_unique[key]
                alias_counts[u] += 1
                if len(alias_samples[u]) < alias_sample_limit:
                    alias_samples[u].append(word)
                # Prefer a shallower/lexicographically smaller representative if found later.
                if (depth, len(word), word) < (depths[u], len(words[u]), words[u]):
                    alpha_list[u] = alpha
                    beta_list[u] = beta
                    words[u] = word
                    depths[u] = depth
                    traces[u] = trace
                    rep_raw[u] = raw_idx
                continue
            key_to_unique[key] = len(alpha_list)
        alpha_list.append(alpha)
        beta_list.append(beta)
        words.append(word)
        depths.append(depth)
        traces.append(trace)
        alias_counts.append(1)
        rep_raw.append(raw_idx)
        alias_samples.append([word])

    return WordBallData(
        raw_size=raw_size,
        unique_size=len(alpha_list),
        alpha=np.asarray(alpha_list, dtype=np.complex128),
        beta=np.asarray(beta_list, dtype=np.complex128),
        words=words,
        depths=np.asarray(depths, dtype=np.int16),
        traces=np.asarray(traces, dtype=np.float32),
        alias_counts=np.asarray(alias_counts, dtype=np.int32),
        representative_raw_index=np.asarray(rep_raw, dtype=np.int32),
        alias_samples=alias_samples,
        dedupe_enabled=bool(dedupe),
        dedupe_tol=float(tol),
    )




def su11_compose_arrays(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compose SU(1,1) maps represented by (alpha,beta) arrays.

    The resulting transformation is the product of the two matrices.  For the
    depth-2 virtual word ball we include all ordered reduced pairs, so the exact
    convention for word spelling is not important for the candidate set, but the
    matrix product must be a valid PSU(1,1) element.
    """
    a = a1 * a2 + b1 * np.conjugate(b2)
    b = a1 * b2 + b1 * np.conjugate(a2)
    # Renormalize tiny floating drift.
    norm = (np.abs(a) ** 2 - np.abs(b) ** 2)
    good = np.isfinite(norm) & (norm > 0)
    scale = np.ones_like(norm, dtype=np.float64)
    scale[good] = np.sqrt(norm[good])
    return (a / scale).astype(np.complex128, copy=False), (b / scale).astype(np.complex128, copy=False)


def generator_arrays_from_gens(ginn, gens: Any) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert parsed Mobius generators to contiguous arrays.

    FuchsianDownstairsGINN_v2_4.parse_generators returns a dictionary
    label -> Mobius, with each raw generator followed immediately by its
    formal inverse.  Earlier streamed-depth2 versions iterated over the
    dictionary itself, which yielded string labels and caused:

        AttributeError: 'str' object has no attribute 'alpha'

    Accept both dict and list/tuple inputs here.  For dict inputs, preserve
    insertion order, since the virtual reduced-word indexer assumes the
    oriented letters are arranged as generator,inverse,generator,inverse,... .
    """
    if isinstance(gens, dict):
        gen_list = list(gens.values())
    else:
        gen_list = list(gens)
    alpha: List[complex] = []
    beta: List[complex] = []
    words: List[str] = []
    for i, g in enumerate(gen_list):
        alpha.append(complex(getattr(g, 'alpha')))
        beta.append(complex(getattr(g, 'beta')))
        words.append(str(getattr(g, 'word', f'g{i}')) or f'g{i}')
    return np.asarray(alpha, dtype=np.complex128), np.asarray(beta, dtype=np.complex128), words


def inverse_letter_index(i: np.ndarray | int) -> np.ndarray | int:
    """The GENN generator list is oriented as generator,inverse,generator,inverse,..."""
    return np.asarray(i) ^ 1 if not isinstance(i, int) else (i ^ 1)


def virtual_reduced_depth2_size(oriented_letters: int, depth: int) -> int:
    return estimate_reduced_word_ball_size(oriented_letters, depth)


def virtual_depth2_chunk_from_indices(
    indices: np.ndarray,
    gen_alpha: np.ndarray,
    gen_beta: np.ndarray,
    gen_words: Optional[List[str]] = None,
    want_words: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[List[str]]]:
    """Return alpha,beta,depth,trace,optional labels for virtual reduced words.

    Index convention for depth<=2:
      0                    identity
      1..m                 oriented letters
      m+1..m+m*(m-1)       reduced ordered pairs (i,j), j != inverse(i)
    """
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    m = int(len(gen_alpha))
    a = np.empty(idx.shape[0], dtype=np.complex128)
    b = np.empty(idx.shape[0], dtype=np.complex128)
    depth = np.empty(idx.shape[0], dtype=np.int16)
    labels: Optional[List[str]] = [] if want_words else None

    mask0 = idx == 0
    if np.any(mask0):
        a[mask0] = 1.0 + 0j
        b[mask0] = 0.0 + 0j
        depth[mask0] = 0
    mask1 = (idx >= 1) & (idx <= m)
    if np.any(mask1):
        gi = idx[mask1] - 1
        a[mask1] = gen_alpha[gi]
        b[mask1] = gen_beta[gi]
        depth[mask1] = 1
    mask2 = idx > m
    if np.any(mask2):
        off = idx[mask2] - (m + 1)
        i = off // (m - 1)
        js = off % (m - 1)
        inv = inverse_letter_index(i)
        j = js + (js >= inv)
        aa, bb = su11_compose_arrays(gen_alpha[i], gen_beta[i], gen_alpha[j], gen_beta[j])
        a[mask2] = aa
        b[mask2] = bb
        depth[mask2] = 2
    trace = (2.0 * a.real).astype(np.float32)
    if want_words and labels is not None:
        gw = gen_words or [f'g{i}' for i in range(m)]
        for x in idx.tolist():
            if x == 0:
                labels.append('identity')
            elif 1 <= x <= m:
                labels.append(gw[x-1])
            else:
                off = x - (m + 1)
                i = off // (m - 1)
                js = off % (m - 1)
                inv = inverse_letter_index(int(i))
                j = int(js + (js >= inv))
                labels.append(f'{gw[int(i)]}*{gw[j]}')
    return a, b, depth, trace, labels


def update_topk_np_from_chunk(top_d: np.ndarray, top_i: np.ndarray, chunk_d: np.ndarray, chunk_indices: np.ndarray, kmax: int) -> Tuple[np.ndarray, np.ndarray]:
    n, c = chunk_d.shape
    cand_i = np.broadcast_to(chunk_indices.astype(np.int32)[None, :], (n, c))
    all_d = np.concatenate([top_d, chunk_d], axis=1)
    all_i = np.concatenate([top_i, cand_i], axis=1)
    part = np.argpartition(all_d, kth=kmax - 1, axis=1)[:, :kmax] if all_d.shape[1] > kmax else np.arange(all_d.shape[1])[None, :].repeat(n, axis=0)
    new_d = np.take_along_axis(all_d, part, axis=1)
    new_i = np.take_along_axis(all_i, part, axis=1)
    order = np.argsort(new_d, axis=1)
    return np.take_along_axis(new_d, order, axis=1).astype(np.float32), np.take_along_axis(new_i, order, axis=1).astype(np.int32)


def compute_distance_arrays_numpy(alpha: np.ndarray, beta: np.ndarray, p_batch: np.ndarray, q_batch: np.ndarray) -> np.ndarray:
    p2 = np.asarray(p_batch, dtype=np.complex128)[:, None]
    q2 = np.asarray(q_batch, dtype=np.complex128)[:, None]
    a = np.asarray(alpha, dtype=np.complex128)[None, :]
    b = np.asarray(beta, dtype=np.complex128)[None, :]
    z = (a * q2 + b) / (np.conjugate(b) * q2 + np.conjugate(a))
    az = np.abs(z)
    z = np.where(az >= 1.0, z / (az + 1.0e-12) * (1.0 - 1.0e-12), z)
    num = 2.0 * np.abs(p2 - z) ** 2
    den = np.maximum((1.0 - np.abs(p2) ** 2) * (1.0 - np.abs(z) ** 2), 1.0e-300)
    return np.arccosh(np.maximum(1.0, 1.0 + num / den)).astype(np.float32, copy=False)


def dedupe_virtual_topk_rows(
    top_i_buf: np.ndarray,
    top_d_buf: np.ndarray,
    gen_alpha: np.ndarray,
    gen_beta: np.ndarray,
    kmax: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate each pair's oversized virtual top-k buffer by PSU(1,1) key.

    This is a local, top-buffer deduplication.  It is not a global dedupe of the
    enormous depth-2 virtual word ball, but it removes the alias pathology that
    makes gap12 artificially zero and fills the final top-k with repeated lifts.
    """
    n, b = top_i_buf.shape
    out_i = np.full((n, kmax), -1, dtype=np.int64)
    out_d = np.full((n, kmax), np.inf, dtype=np.float32)
    unique_counts = np.zeros(n, dtype=np.int32)
    for r in range(n):
        seen = set()
        kept_i: List[int] = []
        kept_d: List[float] = []
        order = np.argsort(top_d_buf[r])
        for jj in order.tolist():
            idx = int(top_i_buf[r, jj])
            if idx < 0 or not np.isfinite(float(top_d_buf[r, jj])):
                continue
            aa, bb, _, _, _ = virtual_depth2_chunk_from_indices(np.asarray([idx], dtype=np.int64), gen_alpha, gen_beta)
            key = canonical_psu_key(complex(aa[0]), complex(bb[0]), tol)
            if key in seen:
                continue
            seen.add(key)
            kept_i.append(idx)
            kept_d.append(float(top_d_buf[r, jj]))
            if len(kept_i) >= kmax:
                break
        unique_counts[r] = len(kept_i)
        if not kept_i:
            # Defensive fallback: should not happen because identity is always in
            # the virtual ball.  Keep the raw best candidate so downstream code
            # has a valid winner.
            best = int(np.nanargmin(top_d_buf[r]))
            kept_i = [int(top_i_buf[r, best])]
            kept_d = [float(top_d_buf[r, best])]
            unique_counts[r] = 1
        # If the buffer was too alias-heavy, pad by repeating the last unique
        # candidate.  The report records unique_counts so this is visible.
        while len(kept_i) < kmax:
            kept_i.append(kept_i[-1])
            kept_d.append(kept_d[-1])
        out_i[r, :] = np.asarray(kept_i[:kmax], dtype=np.int64)
        out_d[r, :] = np.asarray(kept_d[:kmax], dtype=np.float32)
    return out_i.astype(np.int32), out_d.astype(np.float32), unique_counts


def stream_exact_topk_virtual_depth2(
    args: argparse.Namespace,
    ginn,
    gens: List[Any],
    p: np.ndarray,
    q: np.ndarray,
    kmax: int,
    progress_label: str = '',
    perf: Optional[PerfTracker] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any], Dict[str, Any]]:
    """Stream reduced words of depth<=2 virtually for q=29+.

    This avoids constructing a 50M-entry Python word list.  It still cannot do a
    global geometric dedupe of the full virtual word ball.  Instead, v1.7 keeps
    an oversized raw top-k buffer per pair and then performs local PSU(1,1)
    geometric deduplication on that buffer before returning the final top-k.
    """
    gen_alpha, gen_beta, gen_words = generator_arrays_from_gens(ginn, gens)
    m = len(gen_alpha)
    W = virtual_reduced_depth2_size(m, int(args.depth))
    n = int(p.shape[0])
    csize = max(1024, int(args.candidate_chunk_size))
    engine = resolve_engine(args)
    if engine == 'gpu_torch' and not _torch_cuda_available():
        engine = 'cpu_vec'
    pbsize = auto_pair_batch_size(args, n, csize, engine)
    requested_buf = int(getattr(args, 'virtual_topk_buffer', 5000) or 5000)
    kbuf = max(int(kmax), min(int(W), requested_buf))
    dedupe_tol = float(getattr(args, 'virtual_topk_dedupe_tol', 0.0) or getattr(args, 'dedupe_tol', 1.0e-10))
    print(f"[stream-virtual] depth={args.depth} oriented_letters={m} virtual_W={W} engine={engine} candidate_chunk={csize} pair_batch={pbsize} global_dedupe=False local_topk_dedupe=True topk_buffer={kbuf}", flush=True)
    if perf is not None:
        perf.log('virtual_depth2_stream_start', surface_id=progress_label, virtual_word_ball_size=W, engine=engine, candidate_chunk_size=csize, pair_batch_size=pbsize, topk_buffer=kbuf, local_topk_dedupe=True)
    top_i_all = np.full((n, kbuf), -1, dtype=np.int32)
    top_d_all = np.full((n, kbuf), np.inf, dtype=np.float32)
    t0 = time.perf_counter()
    total_done = 0
    last_log = 0.0
    if engine == 'gpu_torch':
        import torch  # type: ignore
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        one_minus_eps = torch.tensor(1.0 - 1.0e-7, dtype=torch.float32, device=device)
        for ps in range(0, n, pbsize):
            pe = min(n, ps + pbsize)
            pb = pe - ps
            p_t = torch.as_tensor(p[ps:pe].astype(np.complex64), device=device).view(-1, 1)
            q_t = torch.as_tensor(q[ps:pe].astype(np.complex64), device=device).view(-1, 1)
            p_abs2 = torch.abs(p_t) ** 2
            top_d = torch.full((pb, kbuf), float('inf'), dtype=torch.float32, device=device)
            top_i = torch.full((pb, kbuf), -1, dtype=torch.int64, device=device)
            for start in range(0, W, csize):
                end = min(W, start + csize)
                ind = np.arange(start, end, dtype=np.int64)
                aa, bb, _, _, _ = virtual_depth2_chunk_from_indices(ind, gen_alpha, gen_beta)
                a = torch.as_tensor(aa.astype(np.complex64), device=device).view(1, -1)
                b = torch.as_tensor(bb.astype(np.complex64), device=device).view(1, -1)
                z = (a * q_t + b) / (torch.conj(b) * q_t + torch.conj(a))
                az = torch.abs(z)
                z = torch.where(az >= 1.0, z / (az + 1.0e-7) * one_minus_eps, z)
                num = 2.0 * torch.abs(p_t - z) ** 2
                den = torch.clamp((1.0 - p_abs2) * (1.0 - torch.abs(z) ** 2), min=1.0e-30)
                D = torch.acosh(torch.clamp(1.0 + num / den, min=1.0)).to(torch.float32)
                all_d = torch.cat((top_d, D), dim=1)
                idx_t = torch.as_tensor(ind.astype(np.int64), device=device).view(1, -1).expand(pb, -1)
                all_i = torch.cat((top_i, idx_t), dim=1)
                vals, pos = torch.topk(all_d, k=kbuf, dim=1, largest=False, sorted=True)
                top_i = torch.gather(all_i, 1, pos)
                top_d = vals
                total_done += pb * (end - start)
                elapsed = time.perf_counter() - t0
                if elapsed - last_log >= 20.0 or (ps == 0 and end == min(W, csize)) or (pe == n and end == W):
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - t0
                    last_log = elapsed
                    rate = total_done / max(elapsed, 1e-9)
                    print(f"[atlas {progress_label}] engine=virtual_gpu pairs {pe}/{n} candidates {end}/{W} evals={total_done} elapsed={elapsed:.1f}s rate={rate:.1f} eval/s", flush=True)
                    if perf is not None:
                        perf.log('atlas_progress_virtual_gpu', surface_id=progress_label, pairs_done=pe, pairs_total=n, candidates_done=end, candidates_total=W, evals_done=total_done, eval_rate_per_s=rate, candidate_chunk_size=csize, pair_batch_size=pbsize)
            torch.cuda.synchronize(device)
            top_i_all[ps:pe, :] = top_i.detach().cpu().numpy().astype(np.int32)
            top_d_all[ps:pe, :] = top_d.detach().cpu().numpy().astype(np.float32)
    else:
        for ps in range(0, n, pbsize):
            pe = min(n, ps + pbsize)
            pb = pe - ps
            top_d = np.full((pb, kbuf), np.inf, dtype=np.float32)
            top_i = np.full((pb, kbuf), -1, dtype=np.int32)
            for start in range(0, W, csize):
                end = min(W, start + csize)
                ind = np.arange(start, end, dtype=np.int64)
                aa, bb, _, _, _ = virtual_depth2_chunk_from_indices(ind, gen_alpha, gen_beta)
                D = compute_distance_arrays_numpy(aa, bb, p[ps:pe], q[ps:pe])
                top_d, top_i = update_topk_np_from_chunk(top_d, top_i, D, ind, kbuf)
                total_done += pb * (end - start)
                elapsed = time.perf_counter() - t0
                if elapsed - last_log >= 20.0 or (ps == 0 and end == min(W, csize)) or (pe == n and end == W):
                    last_log = elapsed
                    rate = total_done / max(elapsed, 1e-9)
                    print(f"[atlas {progress_label}] engine=virtual_cpu pairs {pe}/{n} candidates {end}/{W} evals={total_done} elapsed={elapsed:.1f}s rate={rate:.1f} eval/s", flush=True)
                    if perf is not None:
                        perf.log('atlas_progress_virtual_cpu', surface_id=progress_label, pairs_done=pe, pairs_total=n, candidates_done=end, candidates_total=W, evals_done=total_done, eval_rate_per_s=rate, candidate_chunk_size=csize, pair_batch_size=pbsize)
            top_i_all[ps:pe, :] = top_i
            top_d_all[ps:pe, :] = top_d
    # Locally deduplicate only the oversized top-k buffer.  This avoids storing
    # a global 50M-entry alias hash table while giving a geometrically meaningful
    # final top-k list for each pair.
    top_i_final, top_d_final, local_unique_counts = dedupe_virtual_topk_rows(
        top_i_all, top_d_all, gen_alpha, gen_beta, int(kmax), dedupe_tol
    )
    wall = time.perf_counter() - t0
    if perf is not None:
        perf.log('virtual_topk_local_dedupe_done', surface_id=progress_label, topk_buffer=kbuf, top_k_returned=kmax, min_unique_in_buffer=int(np.min(local_unique_counts)), median_unique_in_buffer=float(np.median(local_unique_counts)), mean_unique_in_buffer=float(np.mean(local_unique_counts)))
    print(f"[stream-virtual] local top-k dedupe done: buffer={kbuf} returned={kmax} min_unique={int(np.min(local_unique_counts))} median_unique={float(np.median(local_unique_counts)):.1f}", flush=True)
    meta = {
        'engine_used': f'virtual_{engine}_localdedupe',
        'pair_batch_size_used': pbsize,
        'candidate_chunk_size_used': csize,
        'topk_buffer': int(kbuf),
        'local_topk_dedupe': True,
        'local_dedupe_tol': float(dedupe_tol),
        'min_unique_in_buffer': int(np.min(local_unique_counts)),
        'median_unique_in_buffer': float(np.median(local_unique_counts)),
        'mean_unique_in_buffer': float(np.mean(local_unique_counts)),
        'evals_per_second': float((n * W) / max(wall, 1e-9)),
    }
    vmeta = {
        'virtual_word_ball': True,
        'global_dedupe': False,
        'local_topk_dedupe': True,
        'topk_buffer': int(kbuf),
        'word_ball_size_raw': int(W),
        'word_ball_size_unique': int(W),
        'alias_reduction_fraction': 0.0,
        'generator_words': gen_words,
        'local_unique_counts': local_unique_counts.astype(np.int32),
    }
    return top_i_final, top_d_final, wall, meta, vmeta


def virtual_pool_feature_arrays(pool: np.ndarray, gen_alpha: np.ndarray, gen_beta: np.ndarray, max_depth: int) -> Dict[str, np.ndarray]:
    flat = pool.reshape(-1).astype(np.int64)
    aa, bb, dd, tt, _ = virtual_depth2_chunk_from_indices(flat, gen_alpha, gen_beta)
    shape = pool.shape
    return {
        'pool_alpha_real': aa.real.astype(np.float32).reshape(shape),
        'pool_alpha_imag': aa.imag.astype(np.float32).reshape(shape),
        'pool_beta_real': bb.real.astype(np.float32).reshape(shape),
        'pool_beta_imag': bb.imag.astype(np.float32).reshape(shape),
        'pool_depths': dd.astype(np.int16).reshape(shape),
        'pool_traces': tt.astype(np.float32).reshape(shape),
        'pool_alias_counts': np.ones(shape, dtype=np.int16),
    }

def write_alias_summary(path: Path, wb: WordBallData, max_rows: int = 500) -> None:
    order = np.argsort(-wb.alias_counts)
    rows: List[Dict[str, Any]] = []
    for rank, u in enumerate(order[:max_rows], start=1):
        rows.append({
            'alias_rank': rank,
            'unique_index': int(u),
            'representative_raw_index': int(wb.representative_raw_index[u]),
            'representative_word': wb.words[int(u)],
            'depth': int(wb.depths[int(u)]),
            'alias_count': int(wb.alias_counts[int(u)]),
            'alias_sample': ' ; '.join(wb.alias_samples[int(u)]),
        })
    write_csv(path, rows, ['alias_rank', 'unique_index', 'representative_raw_index', 'representative_word', 'depth', 'alias_count', 'alias_sample'])


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_engine(args: argparse.Namespace) -> str:
    engine = str(getattr(args, 'engine', 'auto')).lower()
    if engine == 'auto':
        if _torch_cuda_available() and not bool(getattr(args, 'no_gpu', False)):
            return 'gpu_torch'
        return 'cpu_vec'
    return engine


def auto_pair_batch_size(args: argparse.Namespace, n_pairs: int, candidate_chunk_size: int, engine: str) -> int:
    requested = int(getattr(args, 'pair_batch_size', 0) or 0)
    if requested > 0:
        return max(1, min(n_pairs, requested))
    c = max(1, int(candidate_chunk_size))
    # Very conservative bytes/evaluation estimates because vectorized Mobius + distance
    # builds several temporaries.  This keeps runs aggressive but not reckless.
    if engine == 'gpu_torch':
        target_mb = float(getattr(args, 'target_vram_mb', 8192.0) or 8192.0)
        bytes_per_eval = 96.0
    else:
        target_mb = float(getattr(args, 'target_ram_mb', 8192.0) or 8192.0)
        bytes_per_eval = 112.0
    b = int((target_mb * 1024.0 * 1024.0) // (bytes_per_eval * c))
    b = max(1, min(n_pairs, b))
    return b


def compute_distance_chunk_numpy(alpha: np.ndarray, beta: np.ndarray, p_batch: np.ndarray, q_batch: np.ndarray) -> np.ndarray:
    """Dense vectorized distance block: rows=pairs, cols=candidates."""
    p2 = np.asarray(p_batch, dtype=np.complex128)[:, None]
    q2 = np.asarray(q_batch, dtype=np.complex128)[:, None]
    a = np.asarray(alpha, dtype=np.complex128)[None, :]
    b = np.asarray(beta, dtype=np.complex128)[None, :]
    z = (a * q2 + b) / (np.conjugate(b) * q2 + np.conjugate(a))
    az = np.abs(z)
    z = np.where(az >= 1.0, z / (az + 1.0e-12) * (1.0 - 1.0e-12), z)
    num = 2.0 * np.abs(p2 - z) ** 2
    den = np.maximum((1.0 - np.abs(p2) ** 2) * (1.0 - np.abs(z) ** 2), 1.0e-300)
    arg = 1.0 + num / den
    return np.arccosh(np.maximum(1.0, arg)).astype(np.float32, copy=False)


def stream_exact_topk_cpu_vec(
    wb: WordBallData,
    p: np.ndarray,
    q: np.ndarray,
    kmax: int,
    candidate_chunk_size: int,
    pair_batch_size: int,
    progress_label: str = '',
    perf: Optional[PerfTracker] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    n = p.shape[0]
    W = wb.unique_size
    top_i_all = np.full((n, kmax), -1, dtype=np.int32)
    top_d_all = np.full((n, kmax), np.inf, dtype=np.float32)
    csize = max(1, int(candidate_chunk_size))
    pbsize = max(1, min(n, int(pair_batch_size)))
    t0 = time.perf_counter()
    total_done = 0
    last_log = 0.0
    for ps in range(0, n, pbsize):
        pe = min(n, ps + pbsize)
        pb = pe - ps
        top_d = np.full((pb, kmax), np.inf, dtype=np.float32)
        top_i = np.full((pb, kmax), -1, dtype=np.int32)
        for start in range(0, W, csize):
            end = min(W, start + csize)
            D = compute_distance_chunk_numpy(wb.alpha[start:end], wb.beta[start:end], p[ps:pe], q[ps:pe])
            top_d, top_i = update_topk(top_d, top_i, D, start, kmax)
            total_done += pb * (end - start)
            elapsed = time.perf_counter() - t0
            if elapsed - last_log >= 20.0 or (ps == 0 and end == min(W, csize)) or (pe == n and end == W):
                last_log = elapsed
                rate = total_done / max(elapsed, 1.0e-9)
                print(f"[atlas {progress_label}] engine=cpu_vec pairs {pe}/{n} candidates {end}/{W} evals={total_done} elapsed={elapsed:.1f}s rate={rate:.1f} eval/s", flush=True)
                if perf is not None:
                    perf.log('atlas_progress_cpu_vec', surface_id=progress_label, pairs_done=pe, pairs_total=n, candidates_done=end, candidates_total=W, evals_done=total_done, eval_rate_per_s=rate, pair_batch_size=pbsize, candidate_chunk_size=csize)
        top_i_all[ps:pe, :] = top_i
        top_d_all[ps:pe, :] = top_d
    wall = time.perf_counter() - t0
    meta = {'engine_used': 'cpu_vec', 'pair_batch_size_used': pbsize, 'candidate_chunk_size_used': csize, 'evals_per_second': float((n * W) / max(wall, 1.0e-9))}
    return top_i_all, top_d_all, wall, meta


def stream_exact_topk_cpu_loop(
    wb: WordBallData,
    p: np.ndarray,
    q: np.ndarray,
    kmax: int,
    candidate_chunk_size: int,
    progress_label: str = '',
    perf: Optional[PerfTracker] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Low-memory fallback loop over candidates. Slower, but robust."""
    n = p.shape[0]
    W = wb.unique_size
    top_d = np.full((n, kmax), np.inf, dtype=np.float32)
    top_i = np.full((n, kmax), -1, dtype=np.int32)
    t0 = time.perf_counter()
    csize = max(1, int(candidate_chunk_size))
    for start in range(0, W, csize):
        end = min(W, start + csize)
        c = end - start
        D = np.empty((n, c), dtype=np.float32)
        for jj, idx in enumerate(range(start, end)):
            D[:, jj] = compute_distance_chunk_numpy(wb.alpha[idx:idx+1], wb.beta[idx:idx+1], p, q)[:, 0]
        top_d, top_i = update_topk(top_d, top_i, D, start, kmax)
        elapsed = time.perf_counter() - t0
        rate = (n * end) / max(elapsed, 1.0e-9)
        print(f"[atlas {progress_label}] engine=cpu_loop candidates {end}/{W} elapsed={elapsed:.1f}s rate={rate:.1f} eval/s", flush=True)
        if perf is not None:
            perf.log('atlas_progress_cpu_loop', surface_id=progress_label, candidates_done=end, candidates_total=W, eval_rate_per_s=rate)
    wall = time.perf_counter() - t0
    meta = {'engine_used': 'cpu_loop', 'pair_batch_size_used': n, 'candidate_chunk_size_used': csize, 'evals_per_second': float((n * W) / max(wall, 1.0e-9))}
    return top_i, top_d, wall, meta


def stream_exact_topk_gpu_torch(
    wb: WordBallData,
    p: np.ndarray,
    q: np.ndarray,
    kmax: int,
    candidate_chunk_size: int,
    pair_batch_size: int,
    progress_label: str = '',
    perf: Optional[PerfTracker] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    import torch  # type: ignore
    if not torch.cuda.is_available():
        raise RuntimeError('torch CUDA is not available')
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    n = p.shape[0]
    W = wb.unique_size
    csize = max(1, int(candidate_chunk_size))
    pbsize = max(1, min(n, int(pair_batch_size)))
    top_i_all = np.full((n, kmax), -1, dtype=np.int32)
    top_d_all = np.full((n, kmax), np.inf, dtype=np.float32)
    # Candidate transforms are tiny relative to image/model workloads; cache once on GPU.
    alpha_t = torch.as_tensor(wb.alpha.astype(np.complex64), device=device)
    beta_t = torch.as_tensor(wb.beta.astype(np.complex64), device=device)
    t0 = time.perf_counter()
    total_done = 0
    last_log = 0.0
    one_minus_eps = torch.tensor(1.0 - 1.0e-7, dtype=torch.float32, device=device)
    for ps in range(0, n, pbsize):
        pe = min(n, ps + pbsize)
        pb = pe - ps
        p_t = torch.as_tensor(p[ps:pe].astype(np.complex64), device=device).view(-1, 1)
        q_t = torch.as_tensor(q[ps:pe].astype(np.complex64), device=device).view(-1, 1)
        p_abs2 = torch.abs(p_t) ** 2
        top_d = torch.full((pb, kmax), float('inf'), dtype=torch.float32, device=device)
        top_i = torch.full((pb, kmax), -1, dtype=torch.int64, device=device)
        for start in range(0, W, csize):
            end = min(W, start + csize)
            a = alpha_t[start:end].view(1, -1)
            b = beta_t[start:end].view(1, -1)
            z = (a * q_t + b) / (torch.conj(b) * q_t + torch.conj(a))
            az = torch.abs(z)
            z = torch.where(az >= 1.0, z / (az + 1.0e-7) * one_minus_eps, z)
            num = 2.0 * torch.abs(p_t - z) ** 2
            den = torch.clamp((1.0 - p_abs2) * (1.0 - torch.abs(z) ** 2), min=1.0e-30)
            D = torch.acosh(torch.clamp(1.0 + num / den, min=1.0)).to(torch.float32)
            all_d = torch.cat((top_d, D), dim=1)
            idx = torch.arange(start, end, dtype=torch.int64, device=device).view(1, -1).expand(pb, -1)
            all_i = torch.cat((top_i, idx), dim=1)
            vals, pos = torch.topk(all_d, k=kmax, dim=1, largest=False, sorted=True)
            top_i = torch.gather(all_i, 1, pos)
            top_d = vals
            total_done += pb * (end - start)
            elapsed = time.perf_counter() - t0
            if elapsed - last_log >= 20.0 or (ps == 0 and end == min(W, csize)) or (pe == n and end == W):
                # synchronize only when logging, not every chunk
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - t0
                last_log = elapsed
                rate = total_done / max(elapsed, 1.0e-9)
                print(f"[atlas {progress_label}] engine=gpu_torch pairs {pe}/{n} candidates {end}/{W} evals={total_done} elapsed={elapsed:.1f}s rate={rate:.1f} eval/s", flush=True)
                if perf is not None:
                    perf.log('atlas_progress_gpu_torch', surface_id=progress_label, pairs_done=pe, pairs_total=n, candidates_done=end, candidates_total=W, evals_done=total_done, eval_rate_per_s=rate, pair_batch_size=pbsize, candidate_chunk_size=csize)
        torch.cuda.synchronize(device)
        top_i_all[ps:pe, :] = top_i.detach().cpu().numpy().astype(np.int32)
        top_d_all[ps:pe, :] = top_d.detach().cpu().numpy().astype(np.float32)
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    meta = {'engine_used': 'gpu_torch', 'pair_batch_size_used': pbsize, 'candidate_chunk_size_used': csize, 'evals_per_second': float((n * W) / max(wall, 1.0e-9))}
    return top_i_all, top_d_all, wall, meta


def stream_exact_topk_optimized(
    args: argparse.Namespace,
    wb: WordBallData,
    p: np.ndarray,
    q: np.ndarray,
    kmax: int,
    progress_label: str = '',
    perf: Optional[PerfTracker] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    engine = resolve_engine(args)
    csize = max(1, int(args.candidate_chunk_size))
    pbsize = auto_pair_batch_size(args, p.shape[0], csize, engine)
    print(f"[engine] requested={args.engine} resolved={engine} candidate_chunk={csize} pair_batch={pbsize}", flush=True)
    if perf is not None:
        perf.log('atlas_engine_selected', surface_id=progress_label, engine_requested=args.engine, engine_resolved=engine, candidate_chunk_size=csize, pair_batch_size=pbsize)
    if engine == 'gpu_torch':
        try:
            return stream_exact_topk_gpu_torch(wb, p, q, kmax, csize, pbsize, progress_label=progress_label, perf=perf)
        except Exception as e:
            if str(getattr(args, 'engine', 'auto')).lower() == 'auto':
                print(f"[engine warn] gpu_torch failed ({type(e).__name__}: {e}); falling back to cpu_vec", flush=True)
                if perf is not None:
                    perf.log('atlas_gpu_fallback_to_cpu', surface_id=progress_label, error_type=type(e).__name__, error=str(e))
                return stream_exact_topk_cpu_vec(wb, p, q, kmax, csize, auto_pair_batch_size(args, p.shape[0], csize, 'cpu_vec'), progress_label=progress_label, perf=perf)
            raise
    if engine == 'cpu_loop':
        return stream_exact_topk_cpu_loop(wb, p, q, kmax, csize, progress_label=progress_label, perf=perf)
    if engine in ('cpu_vec', 'cpu_numpy', 'numpy'):
        return stream_exact_topk_cpu_vec(wb, p, q, kmax, csize, pbsize, progress_label=progress_label, perf=perf)
    raise ValueError(f"Unknown --engine {args.engine!r}. Use auto, gpu_torch, cpu_vec, or cpu_loop.")


def update_topk(
    top_d: np.ndarray,
    top_i: np.ndarray,
    chunk_d: np.ndarray,
    start_index: int,
    kmax: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n, c = chunk_d.shape
    cand_i = np.arange(start_index, start_index + c, dtype=np.int32)
    cand_i = np.broadcast_to(cand_i[None, :], (n, c))
    all_d = np.concatenate([top_d, chunk_d], axis=1)
    all_i = np.concatenate([top_i, cand_i], axis=1)
    if all_d.shape[1] <= kmax:
        part = np.arange(all_d.shape[1])[None, :].repeat(n, axis=0)
    else:
        part = np.argpartition(all_d, kth=kmax - 1, axis=1)[:, :kmax]
    new_d = np.take_along_axis(all_d, part, axis=1)
    new_i = np.take_along_axis(all_i, part, axis=1)
    order = np.argsort(new_d, axis=1)
    new_d = np.take_along_axis(new_d, order, axis=1)
    new_i = np.take_along_axis(new_i, order, axis=1)
    return new_d.astype(np.float32, copy=False), new_i.astype(np.int32, copy=False)


def stream_exact_topk(
    ginn,
    word_ball: List[Any],
    p: np.ndarray,
    q: np.ndarray,
    kmax: int,
    candidate_chunk_size: int,
    progress_label: str = "",
    perf: Optional[PerfTracker] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n = p.shape[0]
    W = len(word_ball)
    top_d = np.full((n, kmax), np.inf, dtype=np.float32)
    top_i = np.full((n, kmax), -1, dtype=np.int32)
    t0 = time.perf_counter()
    csize = max(1, int(candidate_chunk_size))
    for start in range(0, W, csize):
        end = min(W, start + csize)
        c = end - start
        D = np.empty((n, c), dtype=np.float32)
        for jj, m in enumerate(word_ball[start:end]):
            gq = ginn.apply_mobius_to_array(m.alpha, m.beta, q)
            D[:, jj] = ginn.disk_distance_array(p, gq).astype(np.float32)
        top_d, top_i = update_topk(top_d, top_i, D, start, kmax)
        done = end
        if done == W or done == c or done % max(csize, max(1, W // 10)) < csize:
            elapsed = time.perf_counter() - t0
            rate = done / max(elapsed, 1.0e-9)
            print(f"[atlas {progress_label}] candidates {done}/{W}  elapsed={elapsed:.1f}s  rate={rate:.1f}/s", flush=True)
            if perf is not None:
                perf.log("atlas_candidates_progress", surface_id=progress_label, candidates_done=done, candidates_total=W, candidate_rate_per_s=rate)
    return top_i, top_d, time.perf_counter() - t0


def build_random_pool_indices(
    rng: np.random.Generator,
    top_indices: np.ndarray,
    W: int,
    pool_size: int,
) -> np.ndarray:
    n, k = top_indices.shape
    pool = np.full((n, pool_size), -1, dtype=np.int32)
    for i in range(n):
        chosen: List[int] = []
        seen = set()
        for idx in top_indices[i].tolist():
            if idx < 0:
                continue
            if idx not in seen:
                seen.add(int(idx)); chosen.append(int(idx))
            if len(chosen) >= pool_size:
                break
        while len(chosen) < pool_size:
            need = pool_size - len(chosen)
            # Oversample to avoid many Python iterations when collisions occur.
            cand = rng.integers(0, W, size=max(need * 2, 16), dtype=np.int64)
            for x in cand.tolist():
                x = int(x)
                if x not in seen:
                    seen.add(x); chosen.append(x)
                    if len(chosen) >= pool_size:
                        break
        pool[i, :] = np.asarray(chosen[:pool_size], dtype=np.int32)
    return pool


def atlas_for_surface(args: argparse.Namespace, ginn, surface: Dict[str, Any], run_root: Path, triple_index: int, perf: Optional[PerfTracker] = None) -> AtlasResult:
    sid = str(surface.get("surface_id"))
    outdir = run_root / "atlas" / sid
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / "atlas_manifest.json"
    if meta_path.exists() and not args.overwrite:
        old = read_json(meta_path)
        print(f"[skip] existing atlas for {sid}; use --overwrite to recompute", flush=True)
        return AtlasResult(**old["atlas_result"])

    print("=" * 78)
    print(f"[atlas] {sid}", flush=True)
    print(f"[word-ball] depth={args.depth}", flush=True)
    gens = ginn.parse_generators(surface)
    oriented_letters = len(gens)
    estimated_raw_W = estimate_reduced_word_ball_size(oriented_letters, args.depth)
    print(f"[word-ball preflight] oriented_letters={oriented_letters} side_pair_generators={oriented_letters//2} estimated_raw_W={estimated_raw_W}", flush=True)
    use_virtual_stream = False
    if args.depth == 2 and args.max_word_ball > 0 and estimated_raw_W > args.max_word_ball:
        if getattr(args, 'stream_huge_word_ball', False):
            use_virtual_stream = True
            print(f"[word-ball preflight] using streamed virtual depth-2 engine because estimated_raw_W={estimated_raw_W} exceeds --max-word-ball={args.max_word_ball}", flush=True)
        elif not getattr(args, 'allow_huge_word_ball', False):
            raise RuntimeError(
                f"{sid}: estimated raw word_ball_size={estimated_raw_W} exceeds --max-word-ball={args.max_word_ball}. "
                f"For q=29+ depth 2, pass --stream-huge-word-ball to stream reduced depth-2 products without allocating the full raw word list, "
                f"or pass --allow-huge-word-ball only if you really want to allocate the full raw word list."
            )

    if use_virtual_stream:
        p, q, sampler_kind = sample_pairs(ginn, surface, args.pairs, args.seed + int(triple_index) * 100003 + int(args.q))
        local_top_k_max = min(int(args.top_k_max), int(estimated_raw_W))
        print(f"[sample] pairs={args.pairs} sampler={sampler_kind}", flush=True)
        top_i, top_d, wall, stream_meta, vmeta = stream_exact_topk_virtual_depth2(args, ginn, gens, p, q, local_top_k_max, progress_label=sid, perf=perf)
        W = int(vmeta['word_ball_size_unique'])
        gen_alpha, gen_beta, gen_words = generator_arrays_from_gens(ginn, gens)
        winner_i = top_i[:, 0]
        winner_d = top_d[:, 0]
        gap12 = top_d[:, 1] - top_d[:, 0] if local_top_k_max >= 2 else np.full(args.pairs, np.nan, dtype=np.float32)
        _, _, winner_depth, _, _ = virtual_depth2_chunk_from_indices(winner_i.astype(np.int64), gen_alpha, gen_beta)
        identity_wins = (winner_i == 0)
        shortcut = ~identity_wins
        if perf is not None:
            perf.log('exact_topk_stream_done', surface_id=sid, word_ball_size=W, raw_word_ball_size=estimated_raw_W, pairs=args.pairs, stream_wall_seconds=wall, candidate_evaluations=args.pairs * W, raw_candidate_evaluations=args.pairs * estimated_raw_W, evals_per_second=stream_meta.get('evals_per_second'), engine_used=stream_meta.get('engine_used'))
        np.savez_compressed(
            outdir / 'exact_topk_atlas.npz',
            p_real=p.real.astype(np.float32), p_imag=p.imag.astype(np.float32),
            q_real=q.real.astype(np.float32), q_imag=q.imag.astype(np.float32),
            top_indices=top_i, top_distances=top_d,
            winner_depth=winner_depth.astype(np.int16), gap12=gap12.astype(np.float32),
            word_ball_size_raw=np.asarray([estimated_raw_W], dtype=np.int64),
            word_ball_size_unique=np.asarray([W], dtype=np.int64),
            virtual_word_ball=np.asarray([1], dtype=np.int8),
        )
        rng = np.random.default_rng(args.seed + 777 + int(triple_index) * 911)
        pool_sizes = [s for s in parse_int_list(args.pool_sizes) if s > 0]
        pool_files: Dict[str, str] = {}
        for ps in pool_sizes:
            ps = int(ps)
            print(f"[pool] building virtual candidate_pool_{ps} for {sid}", flush=True)
            pool = build_random_pool_indices(rng, top_i[:, :min(local_top_k_max, ps)], W, ps)
            feat = virtual_pool_feature_arrays(pool, gen_alpha, gen_beta, args.depth)
            pool_path = outdir / f'candidate_pool_{ps}.npz'
            np.savez_compressed(pool_path, pool_indices=pool, top_k_included=min(local_top_k_max, ps), word_ball_size=W, word_ball_size_raw=estimated_raw_W, deduped=False, local_topk_deduped=True, virtual_word_ball=np.asarray([1], dtype=np.int8), **feat)
            pool_files[f'candidate_pool_{ps}'] = str(pool_path)
        k_list = ensure_topk(args.top_k_list, local_top_k_max)
        csv_k = min(args.csv_top_k, local_top_k_max)
        pair_rows = []
        for i in range(args.pairs):
            row = {'pair_index': i, 'winner_index': int(winner_i[i]), 'winner_distance': float(winner_d[i]), 'winner_depth': int(winner_depth[i]), 'gap12': float(gap12[i]), 'identity_wins': bool(identity_wins[i])}
            for kk in k_list:
                row[f'top{kk}_max_distance'] = float(top_d[i, kk-1]) if kk <= local_top_k_max else ''
            for j in range(csv_k):
                row[f'top{j+1}_index'] = int(top_i[i, j]); row[f'top{j+1}_distance'] = float(top_d[i, j])
            pair_rows.append(row)
        write_csv(outdir / 'pair_topk_summary.csv', pair_rows)
        uniq, cnt = np.unique(winner_i, return_counts=True)
        write_csv(outdir / 'winner_frequency.csv', [{'winner_index': int(u), 'count': int(c), 'frequency': float(c/args.pairs)} for u,c in zip(uniq,cnt)])
        write_csv(outdir / 'word_ball_alias_summary.csv', [{'note': 'virtual depth-2 stream; local top-k geometric deduplication performed; global dedupe not performed', 'estimated_raw_W': estimated_raw_W, 'virtual_W': W}])
        result = AtlasResult(surface_id=sid, q=int(args.q), triple_index=int(triple_index), genus=int(surface.get('genus', -1)), generator_count=len(gens)//2, word_ball_size=W, word_ball_size_raw=int(estimated_raw_W), word_ball_size_unique=W, word_ball_alias_reduction_fraction=0.0, word_ball_alias_max_count=1, word_ball_alias_mean_count=1.0, n_pairs=int(args.pairs), depth=int(args.depth), top_k_max=int(local_top_k_max), engine=str(stream_meta.get('engine_used')), pair_batch_size=int(stream_meta.get('pair_batch_size_used', args.pairs)), candidate_chunk_size=int(stream_meta.get('candidate_chunk_size_used', args.candidate_chunk_size)), wall_seconds=float(wall), exact_candidate_evaluations=int(args.pairs * W), exact_candidate_evaluations_raw=int(args.pairs * estimated_raw_W), evals_per_second=float(stream_meta.get('evals_per_second', 0.0)), identity_wins_fraction=float(np.mean(identity_wins)), shortcut_fraction=float(np.mean(shortcut)), mean_winner_depth=float(np.mean(winner_depth)), max_winner_depth=int(np.max(winner_depth)), median_gap12=float(np.nanmedian(gap12)), mean_gap12=float(np.nanmean(gap12)), near_seam_0p02=float(np.mean(gap12 < 0.02)), near_seam_0p05=float(np.mean(gap12 < 0.05)), outdir=str(outdir))
        write_json(meta_path, {'atlas_result': result.__dict__, 'pool_files': pool_files, 'stream_meta': stream_meta, 'virtual_word_ball': True, 'mathematical_caveat': 'Virtual depth-2 stream over reduced words; global geometric deduplication was not performed; local top-k geometric deduplication was performed on an oversized raw candidate buffer.'})
        return result

    raw_word_ball = ginn.build_word_ball(gens, args.depth)
    raw_W = len(raw_word_ball)
    if args.max_word_ball > 0 and raw_W > args.max_word_ball:
        raise RuntimeError(f"{sid}: raw word_ball_size={raw_W} exceeds --max-word-ball={args.max_word_ball}")

    print(f"[word-ball] raw generators={len(gens)//2} raw_W={raw_W}", flush=True)
    if perf is not None:
        perf.log("raw_word_ball_built", surface_id=sid, depth=args.depth, generator_count=len(gens)//2, raw_word_ball_size=raw_W)

    wb = build_word_ball_data(
        ginn,
        raw_word_ball,
        dedupe=(not args.no_dedupe),
        tol=args.dedupe_tol,
        alias_sample_limit=args.alias_sample_limit,
    )
    # Free object-heavy raw word-ball before the large streaming pass.
    raw_word_ball = []
    W = wb.unique_size
    words = wb.words
    depths = wb.depths
    traces = wb.traces
    if args.max_unique_word_ball > 0 and W > args.max_unique_word_ball:
        raise RuntimeError(f"{sid}: unique word_ball_size={W} exceeds --max-unique-word-ball={args.max_unique_word_ball}")
    local_top_k_max = min(int(args.top_k_max), max(1, int(W)))
    if local_top_k_max < int(args.top_k_max):
        print(f"[top-k warn] requested --top-k-max={args.top_k_max} but unique_W={W}; using {local_top_k_max}", flush=True)
    print(
        f"[word-ball] unique_W={W} alias_reduction={wb.alias_reduction_fraction:.3f} "
        f"max_alias={int(np.max(wb.alias_counts)) if wb.alias_counts.size else 0} top_k_max={local_top_k_max}",
        flush=True,
    )
    write_alias_summary(outdir / "word_ball_alias_summary.csv", wb, max_rows=args.alias_summary_rows)
    if perf is not None:
        perf.log(
            "word_ball_dedup_done",
            surface_id=sid,
            raw_word_ball_size=wb.raw_size,
            unique_word_ball_size=wb.unique_size,
            alias_reduction_fraction=wb.alias_reduction_fraction,
            alias_max_count=int(np.max(wb.alias_counts)) if wb.alias_counts.size else 0,
        )

    p, q, sampler_kind = sample_pairs(ginn, surface, args.pairs, args.seed + int(triple_index) * 100003 + int(args.q))
    print(f"[sample] pairs={args.pairs} sampler={sampler_kind}", flush=True)
    if perf is not None:
        perf.log("pairs_sampled", surface_id=sid, pairs=args.pairs, sampler_kind=sampler_kind)
    k_list = ensure_topk(args.top_k_list, local_top_k_max)
    top_i, top_d, wall, stream_meta = stream_exact_topk_optimized(
        args, wb, p, q, local_top_k_max, progress_label=sid, perf=perf
    )

    winner_i = top_i[:, 0]
    winner_d = top_d[:, 0]
    gap12 = top_d[:, 1] - top_d[:, 0] if local_top_k_max >= 2 else np.full(args.pairs, np.nan, dtype=np.float32)
    winner_depth = depths[winner_i]
    identity_wins = (winner_i == 0)
    shortcut = ~identity_wins

    # Save compact numeric atlas.
    if perf is not None:
        perf.log(
            "exact_topk_stream_done",
            surface_id=sid,
            word_ball_size=W,
            raw_word_ball_size=wb.raw_size,
            pairs=args.pairs,
            stream_wall_seconds=wall,
            candidate_evaluations=args.pairs * W,
            raw_candidate_evaluations=args.pairs * wb.raw_size,
            evals_per_second=stream_meta.get("evals_per_second"),
            engine_used=stream_meta.get("engine_used"),
        )

    np.savez_compressed(
        outdir / "exact_topk_atlas.npz",
        p_real=p.real.astype(np.float32),
        p_imag=p.imag.astype(np.float32),
        q_real=q.real.astype(np.float32),
        q_imag=q.imag.astype(np.float32),
        top_indices=top_i,
        top_distances=top_d,
        winner_depth=winner_depth.astype(np.int16),
        gap12=gap12.astype(np.float32),
        representative_raw_index=wb.representative_raw_index.astype(np.int32),
        alias_counts=wb.alias_counts.astype(np.int32),
        word_ball_size_raw=np.asarray([wb.raw_size], dtype=np.int64),
        word_ball_size_unique=np.asarray([wb.unique_size], dtype=np.int64),
    )

    # Save optional hard-negative/random candidate pools for future reranker training.
    rng = np.random.default_rng(args.seed + 777 + int(triple_index) * 911)
    pool_sizes = [s for s in parse_int_list(args.pool_sizes) if s > 0]
    pool_files: Dict[str, str] = {}
    for ps in pool_sizes:
        ps = int(ps)
        print(f"[pool] building candidate_pool_{ps} for {sid}", flush=True)
        if perf is not None:
            perf.log("candidate_pool_start", surface_id=sid, pool_size=ps)
        pool = build_random_pool_indices(rng, top_i[:, :min(local_top_k_max, ps)], W, ps)
        pool_path = outdir / f"candidate_pool_{ps}.npz"
        np.savez_compressed(pool_path, pool_indices=pool, top_k_included=min(local_top_k_max, ps), word_ball_size=W, word_ball_size_raw=wb.raw_size, deduped=not args.no_dedupe)
        pool_files[f"candidate_pool_{ps}"] = str(pool_path)
        if perf is not None:
            perf.log("candidate_pool_done", surface_id=sid, pool_size=ps, pool_path=str(pool_path))

    # Pair summary CSV includes all requested top-k words up to --csv-top-k.
    csv_k = min(args.csv_top_k, local_top_k_max)
    pair_rows: List[Dict[str, Any]] = []
    for i in range(args.pairs):
        row: Dict[str, Any] = {
            "pair_id": i,
            "p_x": float(p[i].real), "p_y": float(p[i].imag),
            "q_x": float(q[i].real), "q_y": float(q[i].imag),
            "winner_index": int(winner_i[i]),
            "winner_word": words[int(winner_i[i])],
            "winner_distance": float(winner_d[i]),
            "winner_depth": int(winner_depth[i]),
            "identity_win": int(identity_wins[i]),
            "nontrivial_shortcut": int(shortcut[i]),
            "gap12": float(gap12[i]) if np.isfinite(gap12[i]) else "",
            "near_seam_0p02": int(gap12[i] <= 0.02) if np.isfinite(gap12[i]) else "",
            "near_seam_0p05": int(gap12[i] <= 0.05) if np.isfinite(gap12[i]) else "",
        }
        for kk in range(csv_k):
            idx = int(top_i[i, kk])
            row[f"top{kk+1}_index"] = idx
            row[f"top{kk+1}_word"] = words[idx]
            row[f"top{kk+1}_distance"] = float(top_d[i, kk])
            row[f"top{kk+1}_depth"] = int(depths[idx])
        pair_rows.append(row)
    write_csv(outdir / "pair_topk_summary.csv", pair_rows)

    # Winner frequency table.
    unique, counts = np.unique(winner_i, return_counts=True)
    freq_rows = []
    order = np.argsort(-counts)
    for rank, pos in enumerate(order[:args.frequency_rows], start=1):
        idx = int(unique[pos])
        freq_rows.append({
            "frequency_rank": rank,
            "word_index": idx,
            "word": words[idx],
            "depth": int(depths[idx]),
            "count": int(counts[pos]),
            "fraction": float(counts[pos] / args.pairs),
        })
    write_csv(outdir / "winner_frequency.csv", freq_rows)

    # Optional word-ball summary: can be large for PSL(2,13), so default off.
    if args.write_word_ball_summary:
        wb_rows = []
        for j, w in enumerate(words):
            wb_rows.append({"index": j, "word": w, "depth": int(depths[j]), "trace_real": float(traces[j])})
        write_csv(outdir / "word_ball_summary.csv", wb_rows)

    result = AtlasResult(
        surface_id=sid,
        q=int(args.q),
        triple_index=int(triple_index),
        genus=int(surface.get("genus") or -1),
        generator_count=int(surface.get("generator_count") or (len(gens)//2)),
        word_ball_size=int(W),
        word_ball_size_raw=int(wb.raw_size),
        word_ball_size_unique=int(wb.unique_size),
        word_ball_alias_reduction_fraction=float(wb.alias_reduction_fraction),
        word_ball_alias_max_count=int(np.max(wb.alias_counts)) if wb.alias_counts.size else 0,
        word_ball_alias_mean_count=float(np.mean(wb.alias_counts)) if wb.alias_counts.size else 0.0,
        n_pairs=int(args.pairs),
        depth=int(args.depth),
        top_k_max=int(local_top_k_max),
        engine=str(stream_meta.get("engine_used", resolve_engine(args))),
        pair_batch_size=int(stream_meta.get("pair_batch_size_used", args.pairs)),
        candidate_chunk_size=int(stream_meta.get("candidate_chunk_size_used", args.candidate_chunk_size)),
        wall_seconds=float(wall),
        exact_candidate_evaluations=int(args.pairs * W),
        exact_candidate_evaluations_raw=int(args.pairs * wb.raw_size),
        evals_per_second=float(stream_meta.get("evals_per_second", (args.pairs * W) / max(wall, 1.0e-9))),
        identity_wins_fraction=float(np.mean(identity_wins)),
        shortcut_fraction=float(np.mean(shortcut)),
        mean_winner_depth=float(np.mean(winner_depth)),
        max_winner_depth=int(np.max(winner_depth)),
        median_gap12=float(np.nanmedian(gap12)),
        mean_gap12=float(np.nanmean(gap12)),
        near_seam_0p02=float(np.mean(gap12 <= 0.02)),
        near_seam_0p05=float(np.mean(gap12 <= 0.05)),
        outdir=str(outdir),
    )
    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "surface_id": sid,
        "created": datetime.now().isoformat(timespec="seconds"),
        "sampler_kind": sampler_kind,
        "top_k_list_reported": k_list,
        "top_k_max": local_top_k_max,
        "top_k_max_requested": args.top_k_max,
        "csv_top_k": csv_k,
        "word_ball_manifest": {
            "word_ball_size": W,
            "word_ball_size_raw": wb.raw_size,
            "word_ball_size_unique": wb.unique_size,
            "word_ball_alias_reduction_fraction": wb.alias_reduction_fraction,
            "dedupe_enabled": not args.no_dedupe,
            "dedupe_tol": args.dedupe_tol,
            "word_ball_depth": args.depth,
            "word_hash_sampled": sha1_strings(words, sample=5000),
            "word_hash_note": "Hash uses unique representative words after projective geometric deduplication unless --no-dedupe was used.",
            "alias_summary_csv": str(outdir / "word_ball_alias_summary.csv"),
        },
        "stream_meta": stream_meta,
        "pool_files": pool_files,
        "output_files": {
            "exact_topk_atlas_npz": str(outdir / "exact_topk_atlas.npz"),
            "pair_topk_summary_csv": str(outdir / "pair_topk_summary.csv"),
            "winner_frequency_csv": str(outdir / "winner_frequency.csv"),
            "word_ball_alias_summary_csv": str(outdir / "word_ball_alias_summary.csv"),
        },
        "atlas_result": result.__dict__,
        "interpretation": {
            "winner": "exact finite-word winner in the selected depth word ball",
            "topk": "exact top-k nearest distinct geometric lifts by hyperbolic disk distance among the selected finite word ball after PSU(1,1) deduplication unless --no-dedupe was used",
            "not_global_warning": "This atlas is exact for the selected finite word ball, not a theorem-level all-Gamma global minimum certificate.",
            "future_training_note": "candidate_pool_*.npz files include exact top candidates plus random negatives; future reranker training should recompute exact distances for selected pool entries if needed.",
        },
    }
    write_json(meta_path, manifest)
    print(f"[atlas done] {sid} raw_W={wb.raw_size} unique_W={W} pairs={args.pairs} wall={wall:.1f}s eval_rate={result.evals_per_second:.1f}/s shortcut={result.shortcut_fraction:.3f}", flush=True)
    if perf is not None:
        perf.log("atlas_surface_done", surface_id=sid, word_ball_size=W, raw_word_ball_size=wb.raw_size, pairs=args.pairs, wall_seconds_surface=wall, evals_per_second=result.evals_per_second, shortcut_fraction=result.shortcut_fraction, engine=result.engine)
    return result


def write_report(run_root: Path, args: argparse.Namespace, surface_rows: List[Dict[str, Any]], atlas_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Fuchsian Big Hurwitz Zoo v1.3 Report")
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This run builds large Hurwitz kernel surfaces and generates exact finite-word top-k branch-atlas data. It does not train a GINN and it does not benchmark search methods. The atlas is intended to prepare future candidate-pool / hard-negative reranker experiments.")
    lines.append("")
    lines.append("## Run parameters")
    lines.append("")
    for k in ["q", "triple_equivalence", "max_triples", "depth", "pairs", "top_k_max", "engine", "candidate_chunk_size", "pair_batch_size", "target_ram_mb", "target_vram_mb", "no_dedupe", "dedupe_tol", "pool_sizes"]:
        lines.append(f"- `{k}`: `{getattr(args, k)}`")
    lines.append("")
    lines.append("## Surface summary")
    lines.append("")
    if surface_rows:
        cols = ["surface_id", "q", "triple_index", "genus", "generator_count", "tile_count", "mainline_dataset_eligible"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in surface_rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Atlas summary")
    lines.append("")
    if atlas_rows:
        cols = ["surface_id", "word_ball_size_raw", "word_ball_size_unique", "word_ball_alias_reduction_fraction", "n_pairs", "engine", "pair_batch_size", "candidate_chunk_size", "wall_seconds", "evals_per_second", "shortcut_fraction", "mean_winner_depth", "median_gap12", "near_seam_0p02", "near_seam_0p05"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in atlas_rows:
            lines.append("| " + " | ".join(str(round(r.get(c), 5)) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Mathematical caveat")
    lines.append("")
    lines.append("The winner and top-k candidates are exact within the selected finite word ball after projective geometric deduplication of PSU(1,1) transformations unless `--no-dedupe` was used. They are not claimed to certify the global minimum over the infinite Fuchsian group.")
    lines.append("")
    lines.append("## v1.2/v1.3 performance and larger-q safety note")
    lines.append("")
    lines.append("Version 1.4 can either use the standard deduplicated explicit word-ball engine or, for very large q=29+ depth-2 probes, an experimental streamed virtual reduced-word engine enabled by `--stream-huge-word-ball`. The virtual engine avoids allocating the full raw Python word list but performs local top-k deduplication but not global geometric deduplication. Detailed RAM, VRAM, CPU, and throughput snapshots are written to `tables/performance_log.csv`.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("Important files are in `tables/`, `surfaces/`, and `atlas/<surface_id>/`. Each atlas directory contains `exact_topk_atlas.npz`, `pair_topk_summary.csv`, `winner_frequency.csv`, `word_ball_alias_summary.csv`, and optional `candidate_pool_*.npz` files. Performance snapshots are written to `tables/performance_log.csv`.")
    (run_root / "report").mkdir(parents=True, exist_ok=True)
    (run_root / "report" / "big_hurwitz_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build big Hurwitz exact branch atlases, starting with PSL(2,13).")
    ap.add_argument("--q", type=int, default=13, help="Prime q for PSL(2,q). Default: 13.")
    ap.add_argument("--triple-equivalence", choices=["inner", "pgl"], default="pgl")
    ap.add_argument("--max-triples", type=int, default=3, help="Number of reduced triples/surfaces to build.")
    ap.add_argument("--mode", choices=["smoke", "atlas"], default="atlas", help="smoke builds surfaces and tiny atlas; atlas uses requested pair count.")
    ap.add_argument("--pairs", type=int, default=5000, help="Point pairs per surface for atlas mode.")
    ap.add_argument("--smoke-pairs", type=int, default=20, help="Point pairs per surface for smoke mode.")
    ap.add_argument("--depth", type=int, default=2, help="Finite word-ball depth.")
    ap.add_argument("--top-k-max", type=int, default=100, help="Number of exact top candidates to save per pair.")
    ap.add_argument("--top-k-list", type=parse_int_list, default=[1, 3, 5, 10, 20, 50, 100], help="Comma-separated top-k list for reporting. Default: 1,3,5,10,20,50,100")
    ap.add_argument("--csv-top-k", type=int, default=20, help="How many top candidates to expand into pair_topk_summary.csv.")
    ap.add_argument("--candidate-chunk-size", type=int, default=8192, help="Candidate chunk size for exact streaming search. Larger is faster but uses more RAM/VRAM. v1.2 default: 8192.")
    ap.add_argument("--pair-batch-size", type=int, default=0, help="Pairs per dense vectorized block. 0 chooses automatically from target RAM/VRAM.")
    ap.add_argument("--engine", choices=["auto", "gpu_torch", "cpu_vec", "cpu_loop"], default="auto", help="Exact streaming engine. auto uses CUDA torch if available, otherwise CPU vectorization.")
    ap.add_argument("--no-gpu", action="store_true", help="Force --engine auto away from CUDA even if torch.cuda is available.")
    ap.add_argument("--target-ram-mb", type=float, default=8192.0, help="Approximate RAM budget for one CPU vectorized distance block when --pair-batch-size=0.")
    ap.add_argument("--target-vram-mb", type=float, default=8192.0, help="Approximate VRAM budget for one GPU distance block when --pair-batch-size=0.")
    ap.add_argument("--max-word-ball", type=int, default=1000000, help="Safety cap on raw finite word-ball size; 0 disables.")
    ap.add_argument("--max-unique-word-ball", type=int, default=1000000, help="Safety cap on unique deduplicated finite word-ball size; 0 disables.")
    ap.add_argument("--allow-huge-word-ball", action="store_true", help="Bypass the pre-build raw word-ball estimate guard and allocate the full raw Python word list. Not recommended for q=29 depth 2.")
    ap.add_argument("--stream-huge-word-ball", action="store_true", help="For depth 2 only: stream the reduced word ball virtually in chunks instead of allocating the full raw Python word list. This is experimental and uses local top-k deduplication rather than global geometric dedupe.")
    ap.add_argument("--virtual-topk-buffer", type=int, default=5000, help="For --stream-huge-word-ball: keep this many raw nearest candidates per pair before local PSU(1,1) top-k deduplication. Larger is cleaner but slightly slower/more memory.")
    ap.add_argument("--virtual-topk-dedupe-tol", type=float, default=0.0, help="Tolerance for local virtual top-k deduplication. Default 0 means use --dedupe-tol.")
    ap.add_argument("--no-dedupe", action="store_true", help="Disable projective geometric deduplication of word-ball transformations. Not recommended for PSL(2,13).")
    ap.add_argument("--dedupe-tol", type=float, default=1.0e-10, help="Quantization tolerance for PSU(1,1) geometric deduplication.")
    ap.add_argument("--alias-summary-rows", type=int, default=500, help="Number of rows to write in word_ball_alias_summary.csv.")
    ap.add_argument("--alias-sample-limit", type=int, default=8, help="Number of alias words sampled in each alias-summary row.")
    ap.add_argument("--pool-sizes", type=str, default="128,256,512,1024", help="Comma-separated candidate-pool sizes to save for future reranker training. Empty string disables.")
    ap.add_argument("--frequency-rows", type=int, default=200, help="Rows to write in winner_frequency.csv.")
    ap.add_argument("--write-word-ball-summary", action="store_true", help="Write full word_ball_summary.csv. Can be large for PSL(2,13).")
    ap.add_argument("--outroot", type=str, default="big_hurwitz_runs")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--hurwitz-script", type=str, default=default_hurwitz_script())
    ap.add_argument("--ginn-script", type=str, default=default_ginn_script())
    ap.add_argument("--max-kernel-generators", type=int, default=0, help="Passed to Hurwitz kernel builder; 0 means complete export.")
    ap.add_argument("--max-tiles", type=int, default=0, help="Passed to Hurwitz kernel builder; 0 means full tile scaffold.")
    ap.add_argument("--identity-tol", type=float, default=1.0e-9)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-perf-log", action="store_true", help="Disable terminal performance snapshots. CSV snapshots are still written at major stages when possible.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    print_q_preflight(int(args.q))
    if args.mode == "smoke":
        args.pairs = int(args.smoke_pairs)
        args.top_k_max = min(args.top_k_max, 20)
        args.csv_top_k = min(args.csv_top_k, args.top_k_max)
        if not args.label:
            args.label = "smoke"
    # Make sure report top-k list fits top_k_max after smoke adjustment.
    args.top_k_list = ensure_topk(args.top_k_list, args.top_k_max)

    stamp = now_stamp()
    run_name = f"run_{stamp}" + (f"_{args.label}" if args.label else "")
    run_root = Path(args.outroot) / run_name
    for sub in ["group", "surfaces", "kernel_audits", "atlas", "tables", "report"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    perf = PerfTracker(run_root / "tables" / "performance_log.csv", enabled=(not args.no_perf_log))

    print(f"{PROGRAM} v{VERSION}")
    print(f"run_root={run_root}")
    print(f"q={args.q} triples={args.max_triples} depth={args.depth} pairs={args.pairs}")
    print("-" * 78)

    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "args": vars(args).copy(),
        "python": sys.version,
        "platform": platform.platform(),
        "purpose": "Big Hurwitz exact finite-word branch-atlas generation; no GINN training; no benchmark.",
        "performance_log_csv": str(run_root / "tables" / "performance_log.csv"),
    }
    # argparse objects may contain function-produced list already, safe for JSON.
    write_json(run_root / "manifest.json", manifest)

    t_all = time.perf_counter()
    try:
        perf.log("module_load_start", hurwitz_script=args.hurwitz_script, ginn_script=args.ginn_script)
        hurwitz = load_module(args.hurwitz_script, "hurwitz_v16")
        ginn = load_module(args.ginn_script, "ginn_v24")
        perf.log("module_load_done")
        group, triples, surfaces, audits = build_surfaces(args, hurwitz, run_root, perf=perf)
        surface_rows: List[Dict[str, Any]] = []
        for s, a in zip(surfaces, audits):
            surface_rows.append({
                "surface_id": s.get("surface_id"),
                "q": args.q,
                "triple_index": s.get("finite_group_triple", {}).get("triple_index"),
                "genus": s.get("genus"),
                "quotient_order": s.get("quotient_order"),
                "generator_count": s.get("generator_count"),
                "tile_count": s.get("tile_count"),
                "mainline_dataset_eligible": s.get("mainline_dataset_eligible"),
                "ginn_ready": s.get("ginn_ready"),
                "kernel_generator_export_complete": a.get("kernel_generator_export_complete"),
                "tile_scaffold_complete": a.get("tile_scaffold_complete"),
            })
        write_csv(run_root / "tables" / "big_hurwitz_surface_summary.csv", surface_rows)

        atlas_results: List[AtlasResult] = []
        failure_rows: List[Dict[str, Any]] = []
        for s in surfaces:
            sid = str(s.get("surface_id"))
            tridx = int(s.get("finite_group_triple", {}).get("triple_index", len(atlas_results)))
            try:
                result = atlas_for_surface(args, ginn, s, run_root, tridx, perf=perf)
                atlas_results.append(result)
            except Exception as e:
                print(f"[atlas fail] {sid}: {e}", flush=True)
                failure_rows.append({"surface_id": sid, "error_type": type(e).__name__, "error": str(e)})
                if args.mode == "atlas":
                    # Continue, because a long overnight run should preserve partial successes.
                    continue
                else:
                    raise
        atlas_rows = [r.__dict__ for r in atlas_results]
        write_csv(run_root / "tables" / "big_hurwitz_atlas_summary.csv", atlas_rows)
        write_csv(run_root / "tables" / "failures.csv", failure_rows, ["surface_id", "error_type", "error"])
        write_report(run_root, args, surface_rows, atlas_rows)
        summary = {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": time.perf_counter() - t_all,
            "surfaces_built": len(surfaces),
            "atlases_completed": len(atlas_results),
            "atlas_failures": len(failure_rows),
            "run_root": str(run_root),
            "process_peak_rss_mb": perf.peak_rss_mb,
        }
        perf.log("run_done", surfaces_built=len(surfaces), atlases_completed=len(atlas_results), atlas_failures=len(failure_rows))
        perf.write()
        write_json(run_root / "run_summary.json", summary)
        print("=" * 78)
        print(f"[done] surfaces={len(surfaces)} atlases={len(atlas_results)} failures={len(failure_rows)}")
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
