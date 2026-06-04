#!/usr/bin/env python3
"""
FuchsianPSL2qTieAwareOracleDebug_v1_0.py

Tie-aware oracle-debug program for the PSL(2,q) Hurwitz GINN experiments.

Purpose
-------
A trained shortest-256 candidate-pool GINN can be excellent relative to the
selected finite atlas B_{256,2}.  This audit asks the next question:

    Did the selected B_{256,2} atlas contain the same winner as the complete
    all-Schreier-generator depth-2 atlas B_{all,2}?

For brand-new point pairs, the program computes:

  1. the restricted/selected exact oracle on the trained surface word ball;
  2. the model's top-k predictions scored over that restricted word ball;
  3. the complete depth-2 oracle by streaming the all-generator word ball.

It then reports separate neural-net error and truncation error:

  restricted_model_win_rate     model top-1 agrees with B_selected,2 oracle
  full_oracle_coverage_rate     B_selected,2 winner agrees with B_all,2 winner
  hidden_lift_rate              B_all,2 finds a strictly shorter lift
  absolute_model_win_rate       model top-1 agrees with B_all,2 oracle
  distance/truncation regret    distance gaps relative to B_all,2

Caveat
------
The all-generator oracle is still depth<=2.  It is not an oracle over the full
infinite surface group Gamma.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROGRAM = "FuchsianPSL2qTieAwareOracleDebug_v1_1.py"
VERSION = "1.1"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_slug(label: str) -> str:
    s = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(label))
    return s.strip('_') or 'audit'


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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
    with path.open("w", newline="", encoding="utf-8") as f:
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


def latest_run_under(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def run_contains_matching_surface(run_root: Path, q: Optional[int], triple_filter: Optional[set[int]] = None) -> bool:
    surface_dir = run_root / "surfaces"
    if not surface_dir.exists():
        return False
    for p in surface_dir.glob("*.json"):
        try:
            surf = read_json(p)
        except Exception:
            continue
        if q is not None:
            sq = surf.get("finite_group_triple", {}).get("q", surf.get("q", None))
            ok_q = False
            if sq is not None:
                try:
                    ok_q = int(sq) == int(q)
                except Exception:
                    ok_q = False
            if not ok_q:
                fq = str(surf.get("finite_quotient", ""))
                ok_q = f"PSL(2,{int(q)})" in fq or f"PSL2_{int(q)}" in str(surf.get("surface_id", ""))
            if not ok_q:
                continue
        tridx = int(surf.get("finite_group_triple", {}).get("triple_index", -1))
        if triple_filter is not None and tridx not in triple_filter:
            continue
        sid = str(surf.get("surface_id"))
        ckpt = run_root / "training" / sid / "big_hurwitz_candidate_pool_ginn_v1_0.pt"
        if ckpt.exists():
            return True
    return False


def latest_matching_run_under(root: Path, q: Optional[int], triple_filter: Optional[set[int]] = None) -> Optional[Path]:
    """Return the newest run under root that actually contains matching q surfaces/checkpoints."""
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        if run_contains_matching_surface(run, q, triple_filter):
            return run
    return None



def find_surface_files(trained_run_root: Path, q: Optional[int], triple_filter: Optional[set[int]]) -> List[Path]:
    surface_dir = trained_run_root / "surfaces"
    if not surface_dir.exists():
        raise FileNotFoundError(f"No surfaces directory found in trained run: {surface_dir}")
    out: List[Path] = []
    for p in sorted(surface_dir.glob("*.json")):
        try:
            s = read_json(p)
        except Exception:
            continue
        if q is not None and int(s.get("finite_group_triple", {}).get("q", s.get("q", -999999))) != int(q):
            # Some surface JSONs do not store q in finite_group_triple. Fall back
            # to the finite quotient string if needed.
            fq = str(s.get("finite_quotient", ""))
            if f"PSL(2,{int(q)})" not in fq:
                continue
        tridx = int(s.get("finite_group_triple", {}).get("triple_index", -1))
        if triple_filter is not None and tridx not in triple_filter:
            continue
        if not (trained_run_root / "training" / str(s.get("surface_id")) / "big_hurwitz_candidate_pool_ginn_v1_0.pt").exists():
            print(f"[surface skip] no checkpoint for {s.get('surface_id')}", flush=True)
            continue
        out.append(p)
    if not out:
        raise RuntimeError(f"No matching trained surfaces found in {surface_dir}")
    return out


def disk_distance_np(p: np.ndarray, z: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.complex64)
    z = np.asarray(z, dtype=np.complex64)
    ap = np.abs(p)
    az = np.abs(z)
    p = np.where(ap >= 1.0, p / (ap + 1.0e-7) * (1.0 - 1.0e-7), p)
    z = np.where(az >= 1.0, z / (az + 1.0e-7) * (1.0 - 1.0e-7), z)
    num = 2.0 * np.abs(p - z) ** 2
    den = np.maximum((1.0 - np.abs(p) ** 2) * (1.0 - np.abs(z) ** 2), 1.0e-30)
    return np.arccosh(np.maximum(1.0, 1.0 + num / den)).astype(np.float32)


def apply_pool(alpha: np.ndarray, beta: np.ndarray, q: np.ndarray) -> np.ndarray:
    q2 = q[:, None].astype(np.complex64)
    a = alpha.astype(np.complex64, copy=False)[None, :]
    b = beta.astype(np.complex64, copy=False)[None, :]
    z = (a * q2 + b) / (np.conjugate(b) * q2 + np.conjugate(a))
    az = np.abs(z)
    return np.where(az >= 1.0, z / (az + 1.0e-7) * (1.0 - 1.0e-7), z).astype(np.complex64, copy=False)


def pair_features(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = p.astype(np.complex64)
    q = q.astype(np.complex64)
    identity_dist = disk_distance_np(p, q)
    px = p.real; py = p.imag; qx = q.real; qy = q.imag
    pr = np.abs(p); qr = np.abs(q)
    pa = np.angle(p); qa = np.angle(q)
    return np.column_stack([
        px, py, qx, qy,
        identity_dist,
        np.abs(p - q),
        pr, qr,
        np.cos(pa), np.sin(pa), np.cos(qa), np.sin(qa),
    ]).astype(np.float32)


def candidate_features_and_distances(
    p: np.ndarray,
    q: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    depths: np.ndarray,
    traces: np.ndarray,
    alias_norm: np.ndarray,
    max_depth: int,
) -> Tuple[np.ndarray, np.ndarray]:
    z = apply_pool(alpha, beta, q)
    p2 = p[:, None].astype(np.complex64)
    dx = z.real - p2.real
    dy = z.imag - p2.imag
    euc = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    D = disk_distance_np(p2, z)
    pr = np.abs(p).astype(np.float32)
    qr = np.abs(q).astype(np.float32)
    pa = np.angle(p).astype(np.float32)
    qa = np.angle(q).astype(np.float32)
    gr = np.abs(z).astype(np.float32)
    ga = np.angle(z).astype(np.float32)
    dang = ga - pa[:, None]
    n = p.shape[0]
    c = alpha.shape[0]
    depth_norm = (depths.astype(np.float32) / max(1, max_depth))[None, :]
    trace_feat = (np.clip(traces.astype(np.float32), -4.0, 4.0) / 4.0 + 0.05 * alias_norm.astype(np.float32))[None, :]
    C = np.stack([
        z.real.astype(np.float32),
        z.imag.astype(np.float32),
        dx.astype(np.float32),
        dy.astype(np.float32),
        euc,
        np.broadcast_to(pr[:, None], (n, c)),
        np.broadcast_to(qr[:, None], (n, c)),
        gr,
        np.broadcast_to(np.cos(pa)[:, None].astype(np.float32), (n, c)),
        np.broadcast_to(np.sin(pa)[:, None].astype(np.float32), (n, c)),
        np.broadcast_to(np.cos(qa)[:, None].astype(np.float32), (n, c)),
        np.broadcast_to(np.sin(qa)[:, None].astype(np.float32), (n, c)),
        np.cos(ga).astype(np.float32),
        np.sin(ga).astype(np.float32),
        np.cos(dang).astype(np.float32),
        np.sin(dang).astype(np.float32),
        np.broadcast_to(depth_norm, (n, c)).astype(np.float32),
        np.broadcast_to((np.arange(c) == -1)[None, :], (n, c)).astype(np.float32),  # overwritten below by caller convention if needed
        np.broadcast_to(trace_feat, (n, c)).astype(np.float32),
    ], axis=2)
    return C, D


def update_topk_largest(scores_top: np.ndarray, idx_top: np.ndarray, dist_top: np.ndarray, scores: np.ndarray, idx: np.ndarray, dist: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, c = scores.shape
    cand_idx = np.broadcast_to(idx.astype(np.int32)[None, :], (n, c))
    all_s = np.concatenate([scores_top, scores], axis=1)
    all_i = np.concatenate([idx_top, cand_idx], axis=1)
    all_d = np.concatenate([dist_top, dist], axis=1)
    if all_s.shape[1] > k:
        part = np.argpartition(-all_s, kth=k - 1, axis=1)[:, :k]
    else:
        part = np.arange(all_s.shape[1])[None, :].repeat(n, axis=0)
    ns = np.take_along_axis(all_s, part, axis=1)
    ni = np.take_along_axis(all_i, part, axis=1)
    nd = np.take_along_axis(all_d, part, axis=1)
    order = np.argsort(-ns, axis=1)
    return (
        np.take_along_axis(ns, order, axis=1).astype(np.float32),
        np.take_along_axis(ni, order, axis=1).astype(np.int32),
        np.take_along_axis(nd, order, axis=1).astype(np.float32),
    )


def score_model_over_word_ball(
    bt: Any,
    checkpoint_path: Path,
    wb: Any,
    p: np.ndarray,
    q: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    torch, nn, F = bt.require_torch()
    requested = str(args.model_device).lower()
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() and not args.no_model_gpu else "cpu"
    elif requested == "cuda" and (not torch.cuda.is_available() or args.no_model_gpu):
        raise RuntimeError("--model-device cuda requested but CUDA is unavailable or --no-model-gpu was set")
    else:
        device = requested

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    pair_dim = int(ckpt.get("pair_dim", 12))
    cand_dim = int(ckpt.get("candidate_dim", 19))
    hidden = int(ckpt.get("hidden", 128))
    context_dim = int(ckpt.get("context_dim", 64))
    model = bt.make_model(pair_dim, cand_dim, hidden, context_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm = ckpt.get("normalization") or ckpt.get("metrics", {}).get("normalization")
    if not norm:
        raise RuntimeError(f"Checkpoint has no normalization block: {checkpoint_path}")
    x_mean = np.asarray(norm["x_mean"], dtype=np.float32)
    x_std = np.asarray(norm["x_std"], dtype=np.float32)
    c_mean = np.asarray(norm["candidate_mean"], dtype=np.float32)
    c_std = np.asarray(norm["candidate_std"], dtype=np.float32)

    X = pair_features(p, q)
    Xn = ((X - x_mean) / x_std).astype(np.float32)
    n = p.shape[0]
    W = int(wb.unique_size)
    k = int(args.model_top_k)
    csize = max(1, int(args.model_candidate_chunk_size or args.candidate_chunk_size))
    scores_top = np.full((n, k), -np.inf, dtype=np.float32)
    idx_top = np.full((n, k), -1, dtype=np.int32)
    dist_top = np.full((n, k), np.inf, dtype=np.float32)

    aliases = wb.alias_counts.astype(np.float32)
    alias_norm_all = np.log1p(aliases) / max(1.0, float(np.log1p(np.max(aliases))))
    t0 = time.perf_counter()
    total = 0
    last_log = 0.0
    with torch.no_grad():
        xb = torch.as_tensor(Xn, dtype=torch.float32, device=device)
        for start in range(0, W, csize):
            end = min(W, start + csize)
            idx = np.arange(start, end, dtype=np.int32)
            C, D = candidate_features_and_distances(
                p, q,
                wb.alpha[start:end].astype(np.complex64),
                wb.beta[start:end].astype(np.complex64),
                wb.depths[start:end],
                wb.traces[start:end],
                alias_norm_all[start:end],
                int(args.depth),
            )
            # Correct the is_identity feature: unique index 0 is identity when
            # the standard word-ball builder is used.
            C[:, :, 17] = (idx[None, :] == 0).astype(np.float32)
            Cn = ((C - c_mean) / c_std).astype(np.float32)
            cb = torch.as_tensor(Cn, dtype=torch.float32, device=device)
            scores = model(xb, cb).detach().cpu().numpy().astype(np.float32)
            scores_top, idx_top, dist_top = update_topk_largest(scores_top, idx_top, dist_top, scores, idx, D, k)
            total += n * (end - start)
            elapsed = time.perf_counter() - t0
            if elapsed - last_log >= 20.0 or start == 0 or end == W:
                last_log = elapsed
                rate = total / max(elapsed, 1e-9)
                print(f"[model-score] candidates {end:,}/{W:,} pairs={n} elapsed={elapsed:.1f}s rate={rate:.1f} eval/s", flush=True)
    meta = {
        "device": device,
        "candidate_chunk_size": csize,
        "wall_seconds": time.perf_counter() - t0,
        "scored_candidates": W,
        "model_top_k": k,
        "evals_per_second": float((n * W) / max(time.perf_counter() - t0, 1e-9)),
    }
    return idx_top, scores_top, dist_top, meta


def keys_for_indices(zoo: Any, alpha: np.ndarray, beta: np.ndarray, indices: np.ndarray, tol: float) -> List[List[Tuple[int, int, int, int]]]:
    out: List[List[Tuple[int, int, int, int]]] = []
    for row in indices:
        keys: List[Tuple[int, int, int, int]] = []
        for idx in row.tolist():
            if idx < 0:
                keys.append((0, 0, 0, 0))
            else:
                keys.append(zoo.canonical_psu_key(complex(alpha[int(idx)]), complex(beta[int(idx)]), tol))
        out.append(keys)
    return out


def virtual_keys_for_indices(zoo: Any, indices: np.ndarray, gen_alpha: np.ndarray, gen_beta: np.ndarray, tol: float) -> List[List[Tuple[int, int, int, int]]]:
    out: List[List[Tuple[int, int, int, int]]] = []
    for row in indices:
        aa, bb, _, _, _ = zoo.virtual_depth2_chunk_from_indices(np.asarray(row, dtype=np.int64), gen_alpha, gen_beta)
        out.append([zoo.canonical_psu_key(complex(a), complex(b), tol) for a, b in zip(aa, bb)])
    return out


def contains_at_k(key_rows: List[List[Tuple[int, int, int, int]]], target_keys: List[Tuple[int, int, int, int]], k: int) -> np.ndarray:
    hits = []
    for keys, target in zip(key_rows, target_keys):
        hits.append(target in keys[:min(k, len(keys))])
    return np.asarray(hits, dtype=bool)


def build_all_surface_cached(
    psl2q: Any,
    hurwitz: Any,
    group: Any,
    selected_surface: Dict[str, Any],
    run_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    sid = str(selected_surface.get("surface_id"))
    cache_path = run_root / "all_surfaces" / f"{sid}_ALL_depth_source.json"
    if cache_path.exists() and not args.rebuild_all_surface:
        print(f"[all-surface] loading cached complete-generator surface: {cache_path}", flush=True)
        return read_json(cache_path)
    print(f"[all-surface] building complete Schreier generator surface for {sid}", flush=True)
    tr = selected_surface.get("finite_group_triple")
    if not tr:
        raise RuntimeError(f"Selected surface has no finite_group_triple block: {sid}")
    build_args = SimpleNamespace(
        kernel_generator_mode="all",
        kernel_generator_limit=0,
        identity_tol=float(args.identity_tol),
        kernel_audit_sample_rows=int(args.kernel_audit_sample_rows),
        max_tiles=int(args.max_tiles),
        verbose=bool(args.verbose),
        kernel_progress_every=int(args.kernel_progress_every),
        tile_progress_every=int(args.tile_progress_every),
    )
    all_surface, all_audit = psl2q.build_schreier_kernel_surface_psl2q(group, tr, build_args, run_root.name, hurwitz)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, all_surface)
    write_json(run_root / "all_surfaces" / f"{sid}_ALL_audit.json", all_audit)
    return all_surface


def summarize_surface_audit(
    zoo: Any,
    selected_surface: Dict[str, Any],
    selected_wb: Any,
    selected_top_i: np.ndarray,
    selected_top_d: np.ndarray,
    model_top_i: np.ndarray,
    model_top_d: np.ndarray,
    full_top_i: np.ndarray,
    full_top_d: np.ndarray,
    all_gen_alpha: np.ndarray,
    all_gen_beta: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Summarize exact-key and tie-aware agreement.

    The older audit used exact PSU-key equality for the phrase "oracle coverage".
    This function keeps those exact-key metrics but also asks the geometrically
    more relevant question: is the selected/model lift within tie_epsilon of the
    complete depth-2 oracle distance?

    In particular, if the full oracle chooses a different word but the selected
    winner has distance full_d + O(1e-7), that is classified as a tie/alias-like
    phenomenon rather than a strict hidden-lift failure.
    """
    n = selected_top_i.shape[0]
    tol = float(args.key_tol)
    hidden_eps = float(args.hidden_epsilon)
    tie_eps = float(args.tie_epsilon)
    topks = parse_int_list(args.report_top_k)
    max_model_k = model_top_i.shape[1]
    max_selected_k = selected_top_i.shape[1]
    max_full_k = full_top_i.shape[1]

    selected_keys = keys_for_indices(zoo, selected_wb.alpha, selected_wb.beta, selected_top_i, tol)
    model_keys = keys_for_indices(zoo, selected_wb.alpha, selected_wb.beta, model_top_i, tol)
    full_keys = virtual_keys_for_indices(zoo, full_top_i, all_gen_alpha, all_gen_beta, tol)
    selected_winner_keys = [row[0] for row in selected_keys]
    full_winner_keys = [row[0] for row in full_keys]

    selected_d = selected_top_d[:, 0].astype(np.float64)
    full_d = full_top_d[:, 0].astype(np.float64)
    model_d1 = model_top_d[:, 0].astype(np.float64)

    # Strict hidden-lift: the complete all-generator depth-2 oracle is better by
    # more than hidden_epsilon.  Tie-aware coverage: selected/model distance is
    # within tie_epsilon of the full oracle distance, independent of word key.
    strict_hidden = full_d < selected_d - hidden_eps
    selected_distance_tie = selected_d <= full_d + tie_eps
    model_distance_top1 = model_d1 <= full_d + tie_eps

    exact_coverage = np.asarray([a == b for a, b in zip(selected_winner_keys, full_winner_keys)], dtype=bool)
    exact_model_restricted_top1 = contains_at_k(model_keys, selected_winner_keys, 1)
    exact_model_absolute_top1 = contains_at_k(model_keys, full_winner_keys, 1)

    selected_gap12 = selected_top_d[:, 1].astype(np.float64) - selected_top_d[:, 0].astype(np.float64) if selected_top_d.shape[1] >= 2 else np.full(n, np.nan)
    full_gap12 = full_top_d[:, 1].astype(np.float64) - full_top_d[:, 0].astype(np.float64) if full_top_d.shape[1] >= 2 else np.full(n, np.nan)
    model_dist_gap12 = model_top_d[:, 1].astype(np.float64) - model_top_d[:, 0].astype(np.float64) if model_top_d.shape[1] >= 2 else np.full(n, np.nan)

    # Tie counts within the stored top-k lists. These are lower bounds on the
    # total number of tied full-depth-2 words because we only kept oracle_top_k.
    full_tie_count_stored = np.sum(full_top_d <= (full_d[:, None] + tie_eps), axis=1)
    selected_tie_count_stored = np.sum(selected_top_d <= (selected_d[:, None] + tie_eps), axis=1)
    selected_near_full_count_stored = np.sum(selected_top_d <= (full_d[:, None] + tie_eps), axis=1)
    model_near_full_count_stored = np.sum(model_top_d <= (full_d[:, None] + tie_eps), axis=1)

    # Per-pair bucket labels for quick debugging.
    bucket: List[str] = []
    for i in range(n):
        if exact_coverage[i]:
            bucket.append("EXACT_KEY")
        elif strict_hidden[i]:
            bucket.append("STRICT_HIDDEN_BETTER_OUTSIDE_SELECTED")
        elif selected_distance_tie[i]:
            bucket.append("DIFFERENT_KEY_DISTANCE_TIE")
        else:
            bucket.append("NUMERIC_BORDERLINE")

    row: Dict[str, Any] = {
        "surface_id": selected_surface.get("surface_id"),
        "q": selected_surface.get("finite_group_triple", {}).get("q", ""),
        "triple_index": selected_surface.get("finite_group_triple", {}).get("triple_index", ""),
        "genus": selected_surface.get("genus"),
        "audit_pairs": int(n),
        "tie_epsilon": tie_eps,
        "hidden_epsilon": hidden_eps,
        "selected_unique_word_ball_size": int(selected_wb.unique_size),
        "selected_raw_word_ball_size": int(selected_wb.raw_size),
        "selected_generators": int(selected_surface.get("generator_count", len(selected_surface.get("generators", {})))),
        "full_depth2_raw_oracle_size": int(1 + len(all_gen_alpha) + len(all_gen_alpha) * (len(all_gen_alpha) - 1)),
        "all_oriented_letters": int(len(all_gen_alpha)),
        "all_positive_generators": int(len(all_gen_alpha) // 2),

        # Exact-key metrics from the original audit.
        "restricted_model_win_rate": float(np.mean(exact_model_restricted_top1)),
        "full_oracle_exact_key_coverage_rate": float(np.mean(exact_coverage)),
        "hidden_lift_rate": float(np.mean(strict_hidden)),
        "absolute_model_exact_key_win_rate": float(np.mean(exact_model_absolute_top1)),

        # Tie-aware / geometric-distance metrics.
        "selected_distance_coverage_rate": float(np.mean(selected_distance_tie)),
        "different_key_but_distance_tie_rate": float(np.mean((~exact_coverage) & selected_distance_tie)),
        "strict_hidden_lift_rate": float(np.mean(strict_hidden)),
        "model_top1_distance_success_rate": float(np.mean(model_distance_top1)),
        "ambiguous_full_top1_rate": float(np.mean(full_gap12 <= tie_eps)),
        "ambiguous_selected_top1_rate": float(np.mean(selected_gap12 <= tie_eps)),
        "mean_full_gap12": float(np.nanmean(full_gap12)),
        "median_full_gap12": float(np.nanmedian(full_gap12)),
        "mean_selected_gap12": float(np.nanmean(selected_gap12)),
        "median_selected_gap12": float(np.nanmedian(selected_gap12)),
        "mean_model_distance_gap12": float(np.nanmean(model_dist_gap12)),
        "median_model_distance_gap12": float(np.nanmedian(model_dist_gap12)),
        "mean_full_tie_count_stored_topk": float(np.mean(full_tie_count_stored)),
        "max_full_tie_count_stored_topk": int(np.max(full_tie_count_stored)),
        "mean_selected_tie_count_stored_topk": float(np.mean(selected_tie_count_stored)),
        "mean_selected_near_full_count_stored_topk": float(np.mean(selected_near_full_count_stored)),
        "mean_model_near_full_count_stored_topk": float(np.mean(model_near_full_count_stored)),

        # Regrets relative to the complete depth-2 oracle.
        "mean_truncation_regret": float(np.mean(np.maximum(0.0, selected_d - full_d))),
        "median_truncation_regret": float(np.median(np.maximum(0.0, selected_d - full_d))),
        "max_truncation_regret": float(np.max(np.maximum(0.0, selected_d - full_d))),
        "mean_model_top1_regret": float(np.mean(np.maximum(0.0, model_d1 - full_d))),
        "median_model_top1_regret": float(np.median(np.maximum(0.0, model_d1 - full_d))),
        "max_model_top1_regret": float(np.max(np.maximum(0.0, model_d1 - full_d))),
        "bucket_exact_key_count": int(np.sum(exact_coverage)),
        "bucket_distance_tie_different_key_count": int(np.sum((~exact_coverage) & selected_distance_tie)),
        "bucket_strict_hidden_count": int(np.sum(strict_hidden)),
    }

    for k in topks:
        kk_m = min(k, max_model_k)
        kk_s = min(k, max_selected_k)
        kk_f = min(k, max_full_k)
        row[f"restricted_model_recall_at_{k}"] = float(np.mean(contains_at_k(model_keys, selected_winner_keys, kk_m)))
        row[f"absolute_model_exact_key_recall_at_{k}"] = float(np.mean(contains_at_k(model_keys, full_winner_keys, kk_m)))
        row[f"selected_exact_key_contains_full_at_{k}"] = float(np.mean(contains_at_k(selected_keys, full_winner_keys, kk_s)))

        model_pruned = np.min(model_top_d[:, :kk_m], axis=1).astype(np.float64)
        selected_pruned = np.min(selected_top_d[:, :kk_s], axis=1).astype(np.float64)
        full_tie_count_k = np.sum(full_top_d[:, :kk_f] <= (full_d[:, None] + tie_eps), axis=1)
        row[f"model_distance_success_at_{k}"] = float(np.mean(model_pruned <= full_d + tie_eps))
        row[f"selected_distance_success_at_{k}"] = float(np.mean(selected_pruned <= full_d + tie_eps))
        row[f"mean_model_top{k}_distance_regret"] = float(np.mean(np.maximum(0.0, model_pruned - full_d)))
        row[f"median_model_top{k}_distance_regret"] = float(np.median(np.maximum(0.0, model_pruned - full_d)))
        row[f"mean_selected_top{k}_distance_regret"] = float(np.mean(np.maximum(0.0, selected_pruned - full_d)))
        row[f"full_tie_count_stored_top{k}_mean"] = float(np.mean(full_tie_count_k))
        row[f"full_tie_count_stored_top{k}_max"] = int(np.max(full_tie_count_k))

    pair_rows: List[Dict[str, Any]] = []
    for i in range(n):
        pair_rows.append({
            "pair_id": i,
            "bucket": bucket[i],
            "selected_top1_index": int(selected_top_i[i, 0]),
            "selected_top1_distance": float(selected_top_d[i, 0]),
            "model_top1_index": int(model_top_i[i, 0]),
            "model_top1_distance": float(model_top_d[i, 0]),
            "full_top1_virtual_index": int(full_top_i[i, 0]),
            "full_top1_distance": float(full_top_d[i, 0]),
            "selected_minus_full_distance": float(selected_d[i] - full_d[i]),
            "model_minus_full_distance": float(model_d1[i] - full_d[i]),
            "selected_gap12": float(selected_gap12[i]),
            "full_gap12": float(full_gap12[i]),
            "model_distance_gap12": float(model_dist_gap12[i]),
            "model_equals_selected_top1_key": int(model_keys[i][0] == selected_winner_keys[i]),
            "selected_equals_full_top1_key": int(selected_winner_keys[i] == full_winner_keys[i]),
            "model_equals_full_top1_key": int(model_keys[i][0] == full_winner_keys[i]),
            "selected_distance_tied_to_full": int(bool(selected_distance_tie[i])),
            "model_top1_distance_tied_to_full": int(bool(model_distance_top1[i])),
            "strict_hidden_lift": int(bool(strict_hidden[i])),
            "truncation_regret": float(max(0.0, selected_d[i] - full_d[i])),
            "model_top1_regret": float(max(0.0, model_d1[i] - full_d[i])),
            "full_tie_count_stored_topk": int(full_tie_count_stored[i]),
            "selected_tie_count_stored_topk": int(selected_tie_count_stored[i]),
            "selected_near_full_count_stored_topk": int(selected_near_full_count_stored[i]),
            "model_near_full_count_stored_topk": int(model_near_full_count_stored[i]),
            "model_top5_contains_selected_key": int(selected_winner_keys[i] in model_keys[i][:min(5, max_model_k)]),
            "model_top5_contains_full_key": int(full_winner_keys[i] in model_keys[i][:min(5, max_model_k)]),
            "selected_top20_contains_full_key": int(full_winner_keys[i] in selected_keys[i][:min(20, max_selected_k)]),
            "selected_top5_distances": "|".join(f"{x:.10g}" for x in selected_top_d[i, :min(5, selected_top_d.shape[1])].tolist()),
            "model_top5_indices": "|".join(str(int(x)) for x in model_top_i[i, :min(5, model_top_i.shape[1])].tolist()),
            "model_top5_distances": "|".join(f"{x:.10g}" for x in model_top_d[i, :min(5, model_top_d.shape[1])].tolist()),
            "full_top5_virtual_indices": "|".join(str(int(x)) for x in full_top_i[i, :min(5, full_top_i.shape[1])].tolist()),
            "full_top5_distances": "|".join(f"{x:.10g}" for x in full_top_d[i, :min(5, full_top_d.shape[1])].tolist()),
        })
    return row, pair_rows

def write_report(run_root: Path, args: argparse.Namespace, summary_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(f"# PSL(2,q) Tie-Aware Oracle Debug v{VERSION} Report")
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This debug run compares a selected-generator GINN against both its selected finite atlas and a streamed complete all-Schreier-generator depth-2 oracle, while separating exact word-key misses from distance-level ties.")
    lines.append("")
    lines.append("## Important caveat")
    lines.append("")
    lines.append("The complete oracle here is complete only for the all-Schreier-generator reduced word ball at the requested finite depth, normally depth `<= 2`. It is not an exhaustive search over the infinite surface group `Gamma`.")
    lines.append("")
    lines.append("## Run parameters")
    lines.append("")
    for k in ["trained_run_root", "q", "surface", "audit_pairs", "depth", "oracle_top_k", "model_top_k", "engine", "candidate_chunk_size", "pair_batch_size", "virtual_topk_buffer", "hidden_epsilon", "tie_epsilon"]:
        lines.append(f"- `{k}`: `{getattr(args, k, '')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if summary_rows:
        cols = [
            "surface_id", "triple_index", "audit_pairs",
            "restricted_model_win_rate",
            "full_oracle_exact_key_coverage_rate",
            "selected_distance_coverage_rate",
            "different_key_but_distance_tie_rate",
            "strict_hidden_lift_rate",
            "model_top1_distance_success_rate",
            "absolute_model_exact_key_win_rate",
            "model_distance_success_at_5",
            "model_distance_success_at_20",
            "ambiguous_full_top1_rate",
            "median_full_gap12",
            "mean_truncation_regret", "median_truncation_regret", "max_truncation_regret",
            "mean_model_top1_regret", "median_model_top1_regret",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in summary_rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Tie-aware interpretation guide")
    lines.append("")
    lines.append("- `full_oracle_exact_key_coverage_rate`: selected top-1 word key equals the complete depth-2 oracle top-1 word key.")
    lines.append("- `selected_distance_coverage_rate`: selected top-1 distance is within `tie_epsilon` of the complete depth-2 oracle distance. This is the main geometric coverage metric.")
    lines.append("- `different_key_but_distance_tie_rate`: the exact word key differs, but the distance is tied within `tie_epsilon`.")
    lines.append("- `strict_hidden_lift_rate`: the complete oracle found a strictly shorter lift outside the selected atlas by more than `hidden_epsilon`.")
    lines.append("- `model_top1_distance_success_rate`: model top-1 lift has distance within `tie_epsilon` of the complete depth-2 oracle, even if the word key differs.")
    lines.append("- `ambiguous_full_top1_rate`: the complete oracle's first and second distances differ by at most `tie_epsilon`, indicating a non-unique or near-non-unique winner within the stored oracle top-k.")
    lines.append("")
    lines.append("## Practical reading")
    lines.append("")
    lines.append("If exact-key coverage is low but selected-distance coverage is high and strict-hidden-lift rate is near zero, the issue is mostly tied/equivalent representatives, not a missing shorter lift. If strict-hidden-lift rate or max truncation regret is large, the selected generator pool is genuinely missing better depth-2 lifts.")
    (run_root / "report").mkdir(parents=True, exist_ok=True)
    (run_root / "report" / "psl2q_tie_aware_oracle_debug_report.md").write_text("\n".join(lines), encoding="utf-8")

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tie-aware debug audit of selected PSL(2,q) GINN predictions against complete all-generator depth-2 oracle.")
    ap.add_argument("--trained-run-root", type=str, default="", help="Path to a completed FuchsianPSL2qHurwitzTrainer run. If omitted, use latest under --trained-outroot.")
    ap.add_argument("--trained-outroot", type=str, default="psl2q_hurwitz_training_runs", help="Root used to auto-find latest matching q run when --trained-run-root is omitted.")
    ap.add_argument("--q", type=int, default=29)
    ap.add_argument("--surface", type=str, default="all", help="all, or comma-separated triple indices such as 0,1,2")
    ap.add_argument("--audit-pairs", type=int, default=100)
    ap.add_argument("--depth", type=int, default=2, help="Currently only depth 2 is supported for the complete virtual oracle.")
    ap.add_argument("--oracle-top-k", type=int, default=20)
    ap.add_argument("--model-top-k", type=int, default=20)
    ap.add_argument("--report-top-k", type=str, default="1,5,20")
    ap.add_argument("--seed", type=int, default=920029)
    ap.add_argument("--hidden-epsilon", type=float, default=1.0e-6)
    ap.add_argument("--tie-epsilon", type=float, default=1.0e-5, help="Distance tolerance used to classify two oracle/model lifts as geometrically tied even when their word keys differ.")
    ap.add_argument("--key-tol", type=float, default=1.0e-8)
    ap.add_argument("--dedupe-tol", type=float, default=1.0e-10)
    ap.add_argument("--alias-sample-limit", type=int, default=8)
    ap.add_argument("--engine", choices=["auto", "gpu_torch", "cpu_vec", "cpu_loop"], default="auto")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--target-vram-mb", type=float, default=8192.0)
    ap.add_argument("--target-ram-mb", type=float, default=8192.0)
    ap.add_argument("--candidate-chunk-size", type=int, default=8192)
    ap.add_argument("--pair-batch-size", type=int, default=0)
    ap.add_argument("--virtual-topk-buffer", type=int, default=5000, help="Oversized raw buffer before local dedupe for complete virtual depth-2 oracle.")
    ap.add_argument("--virtual-topk-dedupe-tol", type=float, default=0.0)
    ap.add_argument("--model-device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-model-gpu", action="store_true")
    ap.add_argument("--model-candidate-chunk-size", type=int, default=0)
    ap.add_argument("--outroot", type=str, default="psl2q_tie_debug_runs")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--rebuild-all-surface", action="store_true")
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--identity-tol", type=float, default=1.0e-11)
    ap.add_argument("--kernel-audit-sample-rows", type=int, default=20)
    ap.add_argument("--kernel-progress-every", type=int, default=10000)
    ap.add_argument("--tile-progress-every", type=int, default=20000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--zoo-script", type=str, default=default_local("FuchsianBigHurwitzZoo_v1_8.py"))
    ap.add_argument("--big-trainer-script", type=str, default=default_local("FuchsianBigHurwitzTrainer_v1_7.py"))
    ap.add_argument("--psl2q-trainer-script", type=str, default=default_local("FuchsianPSL2qHurwitzTrainer_v1_0.py"))
    ap.add_argument("--hurwitz-script", type=str, default=default_local("FuchsianHurwitzTester_v1_6.py"))
    ap.add_argument("--ginn-script", type=str, default=default_local("FuchsianDownstairsGINN_v2_4.py"))
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if int(args.depth) != 2:
        raise RuntimeError("Complete all-generator oracle currently supports --depth 2 only.")
    if not args.label:
        args.label = f"PSL2_{args.q}_oracle_audit_pairs{args.audit_pairs}"

    triple_filter: Optional[set[int]] = None
    if str(args.surface).lower() != "all":
        triple_filter = set(parse_int_list(args.surface))

    if args.trained_run_root:
        trained_root = Path(args.trained_run_root).expanduser()
    else:
        trained_root = latest_matching_run_under(Path(args.trained_outroot).expanduser(), int(args.q), triple_filter)
        if trained_root is None:
            # Fall back only to improve the error message below.
            trained_root = latest_run_under(Path(args.trained_outroot).expanduser())
    if trained_root is None:
        raise FileNotFoundError(f"Could not find a trained run under {args.trained_outroot}; pass --trained-run-root explicitly.")
    trained_root = trained_root.resolve()
    args.trained_run_root = str(trained_root)

    run_root = Path(args.outroot) / f"run_{now_stamp()}_{stable_slug(args.label)}"
    for sub in ["tables", "report", "npz", "all_surfaces"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    print(f"{PROGRAM} v{VERSION}")
    print(f"trained_run_root={trained_root}")
    print(f"audit_run_root={run_root}")
    print(f"q={args.q} audit_pairs={args.audit_pairs} depth={args.depth} oracle_top_k={args.oracle_top_k} model_top_k={args.model_top_k}")
    print("-" * 78)

    write_json(run_root / "manifest.json", {
        "program": PROGRAM,
        "version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args).copy(),
        "trained_run_root": str(trained_root),
        "python": sys.version,
        "platform": platform.platform(),
    })

    zoo = load_module(args.zoo_script, "big_hurwitz_zoo_v18")
    bt = load_module(args.big_trainer_script, "big_hurwitz_trainer_v17")
    psl2q = load_module(args.psl2q_trainer_script, "psl2q_trainer_v10")
    hurwitz = load_module(args.hurwitz_script, "hurwitz_v16")
    ginn = load_module(args.ginn_script, "ginn_v24")
    psl2q.install_leq_depth_word_ball_builder(ginn)

    group = hurwitz.PSL2PrimeGroup.build(int(args.q), verbose=bool(args.verbose))
    surface_files = find_surface_files(trained_root, int(args.q), triple_filter)
    print(f"[surfaces] auditing {len(surface_files)} trained surface(s)", flush=True)

    summary_rows: List[Dict[str, Any]] = []
    for sf in surface_files:
        selected_surface = read_json(sf)
        sid = str(selected_surface.get("surface_id"))
        tridx = int(selected_surface.get("finite_group_triple", {}).get("triple_index", len(summary_rows)))
        ckpt_path = trained_root / "training" / sid / "big_hurwitz_candidate_pool_ginn_v1_0.pt"
        print("=" * 78)
        print(f"[surface] {sid} triple={tridx}", flush=True)
        print(f"[checkpoint] {ckpt_path}", flush=True)

        t0 = time.perf_counter()
        p, q, sampler_kind = zoo.sample_pairs(ginn, selected_surface, int(args.audit_pairs), int(args.seed) + tridx * 100003 + int(args.q))
        print(f"[sample] pairs={len(p)} sampler={sampler_kind}", flush=True)

        print("[restricted] building selected word ball", flush=True)
        selected_gens = ginn.parse_generators(selected_surface)
        raw_selected = ginn.build_word_ball(selected_gens, int(args.depth))
        selected_wb = zoo.build_word_ball_data(ginn, raw_selected, dedupe=True, tol=float(args.dedupe_tol), alias_sample_limit=int(args.alias_sample_limit))
        raw_selected = []
        print(f"[restricted] raw_W={selected_wb.raw_size:,} unique_W={selected_wb.unique_size:,}", flush=True)

        selected_top_i, selected_top_d, selected_wall, selected_meta = zoo.stream_exact_topk_optimized(
            args, selected_wb, p, q, int(args.oracle_top_k), progress_label=f"{sid}/selected", perf=None
        )

        print("[model] scoring trained GINN over selected word ball", flush=True)
        model_top_i, model_scores, model_top_d, model_meta = score_model_over_word_ball(bt, ckpt_path, selected_wb, p, q, args)

        all_surface = build_all_surface_cached(psl2q, hurwitz, group, selected_surface, run_root, args)
        all_gens = ginn.parse_generators(all_surface)
        all_gen_alpha, all_gen_beta, all_gen_words = zoo.generator_arrays_from_gens(ginn, all_gens)
        all_W = int(1 + len(all_gen_alpha) + len(all_gen_alpha) * (len(all_gen_alpha) - 1))
        print(f"[full-oracle] complete depth-2 virtual W={all_W:,} oriented_letters={len(all_gen_alpha):,}", flush=True)
        full_top_i, full_top_d, full_wall, full_meta, full_vmeta = zoo.stream_exact_topk_virtual_depth2(
            args, ginn, all_gens, p, q, int(args.oracle_top_k), progress_label=f"{sid}/ALL", perf=None
        )

        row, pair_rows = summarize_surface_audit(
            zoo, selected_surface, selected_wb, selected_top_i, selected_top_d,
            model_top_i, model_top_d, full_top_i, full_top_d,
            all_gen_alpha, all_gen_beta, args,
        )
        row.update({
            "sampler_kind": sampler_kind,
            "selected_oracle_wall_seconds": float(selected_wall),
            "model_score_wall_seconds": float(model_meta.get("wall_seconds", 0.0)),
            "full_oracle_wall_seconds": float(full_wall),
            "selected_oracle_engine": selected_meta.get("engine_used"),
            "full_oracle_engine": full_meta.get("engine_used"),
            "model_device": model_meta.get("device"),
            "surface_audit_wall_seconds": float(time.perf_counter() - t0),
        })
        summary_rows.append(row)
        write_csv(run_root / "tables" / f"oracle_audit_pairs_triple_{tridx:04d}.csv", pair_rows)
        np.savez_compressed(
            run_root / "npz" / f"oracle_audit_triple_{tridx:04d}.npz",
            p_real=p.real.astype(np.float32), p_imag=p.imag.astype(np.float32),
            q_real=q.real.astype(np.float32), q_imag=q.imag.astype(np.float32),
            selected_top_indices=selected_top_i, selected_top_distances=selected_top_d,
            model_top_indices=model_top_i, model_top_scores=model_scores, model_top_distances=model_top_d,
            full_top_indices=full_top_i, full_top_distances=full_top_d,
        )
        print(
            f"[summary {sid}] restricted_win={row['restricted_model_win_rate']:.4f} "
            f"exact_key_coverage={row['full_oracle_exact_key_coverage_rate']:.4f} "
            f"distance_coverage={row['selected_distance_coverage_rate']:.4f} "
            f"strict_hidden={row['strict_hidden_lift_rate']:.4f} "
            f"model_dist_success={row['model_top1_distance_success_rate']:.4f}",
            flush=True,
        )

    write_csv(run_root / "tables" / "psl2q_oracle_audit_summary.csv", summary_rows)
    write_report(run_root, args, summary_rows)
    write_json(run_root / "run_summary.json", {
        "completed": datetime.now().isoformat(timespec="seconds"),
        "surfaces_audited": len(summary_rows),
        "summary_rows": summary_rows,
        "run_root": str(run_root),
        "trained_run_root": str(trained_root),
    })
    print("=" * 78)
    print(f"[done] audited_surfaces={len(summary_rows)}")
    print(f"[done] report={run_root / 'report' / 'psl2q_tie_aware_oracle_debug_report.md'}")
    print(f"[done] summary={run_root / 'tables' / 'psl2q_oracle_audit_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
