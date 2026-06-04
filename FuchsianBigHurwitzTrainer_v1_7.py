#!/usr/bin/env python3
"""
FuchsianBigHurwitzTrainer_v1_7.py

Integrated Big-Hurwitz atlas + candidate-pool GINN trainer.

Default target: the three PGL-reduced PSL(2,13) Hurwitz surfaces, depth 2. Version 1.3 supports an experimental streamed virtual depth-2 backend for larger prime PSL(2,p) smoke/pilot probes such as p=29.
The program first invokes the optimized Big-Hurwitz atlas machinery from
FuchsianBigHurwitzZoo_v1_8.py, then trains a candidate-pool reranker for each
surface.  This is intentionally different from the standard full-word-ball
GINN: for the big PSL(2,13) surfaces it trains on a compact candidate pool
(e.g. exact top-100 plus random hard negatives in a 1024-candidate pool) rather
than on all 200k-300k deduplicated transformations for every pair.

The model does NOT receive exact hyperbolic candidate distances as inputs.
Exact distances are used only as supervised labels/audit values.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROGRAM = "FuchsianBigHurwitzTrainer_v1_7.py"
VERSION = "1.7"


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
        raise FileNotFoundError(
            f"Required module not found: {p}\n"
            "This trainer requires its companion FuchsianBigHurwitzZoo module. "
            "Extract/copy both FuchsianBigHurwitzTrainer_v1_7.py and "
            "FuchsianBigHurwitzZoo_v1_8.py into the same GENN directory, "
            "or pass --zoo-script /path/to/FuchsianBigHurwitzZoo_v1_8.py."
        )
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


def default_zoo_script() -> str:
    """Find the companion Big Hurwitz Zoo module.

    v1.7 searches both the current project directory and the directory
    containing this trainer, and accepts the v1.8 companion module bundled
    with this package. This avoids a confusing FileNotFoundError when only
    one of the two files was copied into the GENN directory.
    """
    candidates = [
        Path.cwd() / "FuchsianBigHurwitzZoo_v1_8.py",
        Path(__file__).resolve().parent / "FuchsianBigHurwitzZoo_v1_8.py",
        Path.cwd() / "FuchsianBigHurwitzZoo_v1_8.py",
        Path(__file__).resolve().parent / "FuchsianBigHurwitzZoo_v1_8.py",
    ]
    for local in candidates:
        if local.exists():
            return str(local)
    return "FuchsianBigHurwitzZoo_v1_8.py"


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


def stable_slug(label: str) -> str:
    s = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(label))
    return s.strip('_') or 'run'


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


def apply_mobius_pool_np(alpha: np.ndarray, beta: np.ndarray, q: np.ndarray) -> np.ndarray:
    # alpha,beta shape [B,P], q shape [B]
    q2 = q[:, None].astype(np.complex64)
    a = alpha.astype(np.complex64, copy=False)
    b = beta.astype(np.complex64, copy=False)
    z = (a * q2 + b) / (np.conjugate(b) * q2 + np.conjugate(a))
    az = np.abs(z)
    return np.where(az >= 1.0, z / (az + 1.0e-7) * (1.0 - 1.0e-7), z).astype(np.complex64, copy=False)


def topk_hit_from_scores(scores: np.ndarray, y_pos: np.ndarray, k: int) -> float:
    k = max(1, min(k, scores.shape[1]))
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(part == y_pos[:, None], axis=1)))


def ordered_topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, scores.shape[1]))
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(part, order, axis=1).astype(np.int32)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


@dataclass
class SurfaceTrainResult:
    surface_id: str
    q: int
    triple_index: int
    genus: int
    unique_word_ball_size: int
    pool_size: int
    train_pairs: int
    val_pairs: int
    test_pairs: int
    device: str
    epochs_ran: int
    best_epoch: int
    best_val_loss: float
    test_recall_at_1: float
    test_recall_at_3: float
    test_recall_at_5: float
    test_recall_at_10: float
    test_recall_at_20: float
    test_top1_distance_rmse: float
    test_top5_pruned_rmse: float
    test_top10_pruned_rmse: float
    winner_pool_coverage: float
    feature_build_seconds: float
    train_seconds: float
    eval_seconds: float
    outdir: str


# PyTorch definitions are wrapped so the file can still --help without torch.
def require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore
        return torch, nn, F
    except Exception as e:
        raise RuntimeError("PyTorch is required for Big-Hurwitz training. Install torch with CUDA if possible.") from e


def build_feature_arrays(
    wb: Any,
    atlas_npz_path: Path,
    pool_npz_path: Path,
    max_depth: int,
    seed: int,
    shuffle_pool: bool = True,
    feature_batch_size: int = 2048,
    perf: Optional[Any] = None,
    surface_id: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build pair features X, candidate features C, exact pool distances D, labels y_pos.

    C has shape [pairs, pool_size, cand_dim].  Exact distances D are labels/audit
    values only; they are not included in C.
    """
    t0 = time.perf_counter()
    atlas = np.load(atlas_npz_path, allow_pickle=False)
    poolz = np.load(pool_npz_path, allow_pickle=False)
    p = (atlas["p_real"].astype(np.float32) + 1j * atlas["p_imag"].astype(np.float32)).astype(np.complex64)
    q = (atlas["q_real"].astype(np.float32) + 1j * atlas["q_imag"].astype(np.float32)).astype(np.complex64)
    true_idx = atlas["top_indices"][:, 0].astype(np.int32)
    true_dist = atlas["top_distances"][:, 0].astype(np.float32)
    pool = poolz["pool_indices"].astype(np.int32)
    n, P = pool.shape
    if len(p) != n:
        raise RuntimeError(f"Atlas/pool pair count mismatch: atlas={len(p)} pool={n}")
    # Ensure true winner is present, then optionally shuffle per pair.
    rng = np.random.default_rng(seed)
    for i in range(n):
        if int(true_idx[i]) not in set(pool[i].tolist()):
            pool[i, 0] = int(true_idx[i])
    if shuffle_pool:
        for i in range(n):
            perm = rng.permutation(P)
            pool[i] = pool[i, perm]
    y_pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        loc = np.where(pool[i] == true_idx[i])[0]
        if len(loc) == 0:
            raise RuntimeError("true winner not present in candidate pool after repair")
        y_pos[i] = int(loc[0])

    # Pair features.
    identity_z = q
    identity_dist = disk_distance_np(p, identity_z)
    px = p.real; py = p.imag; qx = q.real; qy = q.imag
    pr = np.abs(p); qr = np.abs(q)
    pa = np.angle(p); qa = np.angle(q)
    X = np.column_stack([
        px, py, qx, qy,
        identity_dist,
        np.abs(p - q),
        pr, qr,
        np.cos(pa), np.sin(pa), np.cos(qa), np.sin(qa),
    ]).astype(np.float32)
    pair_feature_names = [
        "p_x", "p_y", "q_x", "q_y", "identity_distance", "euclidean_pair_distance",
        "p_radius", "q_radius", "p_angle_cos", "p_angle_sin", "q_angle_cos", "q_angle_sin",
    ]

    cand_dim = 19
    C = np.empty((n, P, cand_dim), dtype=np.float32)
    D = np.empty((n, P), dtype=np.float32)
    pool_has_embedded_features = all(k in poolz.files for k in ["pool_alpha_real", "pool_alpha_imag", "pool_beta_real", "pool_beta_imag", "pool_depths", "pool_traces"])
    if pool_has_embedded_features:
        pool_alpha = (poolz["pool_alpha_real"].astype(np.float32) + 1j * poolz["pool_alpha_imag"].astype(np.float32)).astype(np.complex64)
        pool_beta = (poolz["pool_beta_real"].astype(np.float32) + 1j * poolz["pool_beta_imag"].astype(np.float32)).astype(np.complex64)
        pool_depths = poolz["pool_depths"].astype(np.float32)
        pool_traces = np.nan_to_num(poolz["pool_traces"].astype(np.float32), nan=0.0, posinf=4.0, neginf=-4.0)
        pool_alias = poolz["pool_alias_counts"].astype(np.float32) if "pool_alias_counts" in poolz.files else np.ones_like(pool_depths, dtype=np.float32)
        pool_alias_norm = np.log1p(pool_alias) / max(1.0, float(np.log1p(np.max(pool_alias))))
    else:
        depths = wb.depths.astype(np.float32)
        traces = np.nan_to_num(wb.traces.astype(np.float32), nan=0.0, posinf=4.0, neginf=-4.0)
        aliases = wb.alias_counts.astype(np.float32)
        alias_norm = np.log1p(aliases) / max(1.0, float(np.log1p(np.max(aliases))))
        depth_norm_all = depths / max(1, max_depth)
        alpha_all = wb.alpha.astype(np.complex64)
        beta_all = wb.beta.astype(np.complex64)
    fbs = max(1, int(feature_batch_size))
    for start in range(0, n, fbs):
        end = min(n, start + fbs)
        idx = pool[start:end]
        if pool_has_embedded_features:
            a = pool_alpha[start:end]
            b = pool_beta[start:end]
            depth_norm = pool_depths[start:end] / max(1, max_depth)
            trace_feat = np.clip(pool_traces[start:end], -4.0, 4.0) / 4.0 + 0.05 * pool_alias_norm[start:end]
        else:
            a = alpha_all[idx]
            b = beta_all[idx]
            depth_norm = depth_norm_all[idx].astype(np.float32)
            trace_feat = (np.clip(traces[idx], -4.0, 4.0) / 4.0 + 0.05 * alias_norm[idx]).astype(np.float32)
        z = apply_mobius_pool_np(a, b, q[start:end])
        p2 = p[start:end, None]
        dx = z.real - p2.real
        dy = z.imag - p2.imag
        euc = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        gr = np.abs(z).astype(np.float32)
        ga = np.angle(z).astype(np.float32)
        pa_b = pa[start:end, None].astype(np.float32)
        qa_b = qa[start:end, None].astype(np.float32)
        dang = ga - pa_b
        D[start:end] = disk_distance_np(p2, z)
        C[start:end, :, :] = np.stack([
            z.real.astype(np.float32),
            z.imag.astype(np.float32),
            dx.astype(np.float32),
            dy.astype(np.float32),
            euc,
            np.broadcast_to(pr[start:end, None].astype(np.float32), (end-start, P)),
            np.broadcast_to(qr[start:end, None].astype(np.float32), (end-start, P)),
            gr,
            np.broadcast_to(np.cos(pa[start:end])[:, None].astype(np.float32), (end-start, P)),
            np.broadcast_to(np.sin(pa[start:end])[:, None].astype(np.float32), (end-start, P)),
            np.broadcast_to(np.cos(qa[start:end])[:, None].astype(np.float32), (end-start, P)),
            np.broadcast_to(np.sin(qa[start:end])[:, None].astype(np.float32), (end-start, P)),
            np.cos(ga).astype(np.float32),
            np.sin(ga).astype(np.float32),
            np.cos(dang).astype(np.float32),
            np.sin(dang).astype(np.float32),
            depth_norm.astype(np.float32),
            (idx == 0).astype(np.float32),
            trace_feat.astype(np.float32),
        ], axis=2)
        if perf is not None and (start == 0 or end == n):
            perf.log("trainer_feature_progress", surface_id=surface_id, rows_done=end, rows_total=n, pool_size=P)
    cand_feature_names = [
        "lifted_q_x", "lifted_q_y", "dx_lifted_minus_p", "dy_lifted_minus_p",
        "euclidean_lifted_distance", "p_radius", "q_radius", "lifted_q_radius",
        "p_angle_cos", "p_angle_sin", "q_angle_cos", "q_angle_sin",
        "lifted_angle_cos", "lifted_angle_sin", "angle_delta_cos", "angle_delta_sin",
        "word_depth_norm", "is_identity", "trace_clipped_plus_alias_proxy",
    ]
    meta = {
        "pair_feature_names": pair_feature_names,
        "candidate_feature_names": cand_feature_names,
        "feature_build_seconds": time.perf_counter() - t0,
        "winner_pool_coverage": float(np.mean(pool[np.arange(n), y_pos] == true_idx)),
        "true_distance_mean": float(np.mean(true_dist)),
        "pool_size": int(P),
        "pairs": int(n),
        "pool_has_embedded_features": bool(pool_has_embedded_features),
    }
    return X, C, D, y_pos, true_dist, {**meta, "pool_indices": pool, "true_indices": true_idx}


def normalize_arrays(X: np.ndarray, C: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    x_mean = X[train_idx].mean(axis=0)
    x_std = X[train_idx].std(axis=0)
    x_std[x_std < 1.0e-6] = 1.0
    # Candidate normalization can be exact; pool size 1024, 9000 pairs is fine.
    C_train = C[train_idx].reshape(-1, C.shape[-1])
    c_mean = C_train.mean(axis=0)
    c_std = C_train.std(axis=0)
    c_std[c_std < 1.0e-6] = 1.0
    Xn = ((X - x_mean) / x_std).astype(np.float32)
    Cn = ((C - c_mean) / c_std).astype(np.float32)
    return Xn, Cn, {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "candidate_mean": c_mean.tolist(), "candidate_std": c_std.tolist()}


def make_model(pair_dim: int, cand_dim: int, hidden: int, context_dim: int):
    torch, nn, F = require_torch()

    class CandidatePoolRanker(nn.Module):  # type: ignore[misc]
        def __init__(self):
            super().__init__()
            self.pair_net = nn.Sequential(
                nn.Linear(pair_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, context_dim), nn.SiLU(),
            )
            self.cand_net = nn.Sequential(
                nn.Linear(cand_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, context_dim), nn.SiLU(),
            )
            self.score_net = nn.Sequential(
                nn.Linear(3 * context_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, max(16, hidden // 2)), nn.SiLU(),
                nn.Linear(max(16, hidden // 2), 1),
            )

        def forward(self, x_pair, x_cand):
            B, P, _ = x_cand.shape
            pc = self.pair_net(x_pair)
            ce = self.cand_net(x_cand.reshape(B * P, -1)).reshape(B, P, -1)
            pc_exp = pc[:, None, :].expand(-1, P, -1)
            joint = torch.cat([pc_exp, ce, pc_exp * ce], dim=-1)
            return self.score_net(joint).squeeze(-1)

    return CandidatePoolRanker()


def evaluate_model(model: Any, X: np.ndarray, C: np.ndarray, D: np.ndarray, y_pos: np.ndarray, true_dist: np.ndarray, indices: np.ndarray, device: str, batch_size: int = 256) -> Tuple[Dict[str, Any], np.ndarray]:
    torch, nn, F = require_torch()
    model.eval()
    scores_list: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(indices), batch_size):
            idx = indices[s:s+batch_size]
            xb = torch.as_tensor(X[idx], dtype=torch.float32, device=device)
            cb = torch.as_tensor(C[idx], dtype=torch.float32, device=device)
            sc = model(xb, cb).detach().cpu().numpy().astype(np.float32)
            scores_list.append(sc)
    scores = np.vstack(scores_list) if scores_list else np.empty((0, C.shape[1]), dtype=np.float32)
    yp = y_pos[indices]
    Dsplit = D[indices]
    true = true_dist[indices]
    pred1 = np.argmax(scores, axis=1).astype(np.int32)
    out: Dict[str, Any] = {}
    for k in [1, 3, 5, 10, 20, 50, 100]:
        if k <= scores.shape[1]:
            top = ordered_topk(scores, k)
            hit = np.any(top == yp[:, None], axis=1)
            pruned = np.min(np.take_along_axis(Dsplit, top, axis=1), axis=1)
            out[f"recall_at_{k}"] = float(np.mean(hit))
            out[f"pruned_rmse_at_{k}"] = rmse(true, pruned)
    pred1_d = Dsplit[np.arange(len(indices)), pred1]
    out["top1_distance_rmse"] = rmse(true, pred1_d)
    ranks = []
    order_all = np.argsort(-scores, axis=1)
    for i in range(len(indices)):
        loc = np.where(order_all[i] == yp[i])[0]
        ranks.append(int(loc[0] + 1) if len(loc) else scores.shape[1] + 1)
    out["true_rank_mean"] = float(np.mean(ranks)) if ranks else float('nan')
    out["true_rank_median"] = float(np.median(ranks)) if ranks else float('nan')
    return out, scores


def train_surface_reranker(
    args: argparse.Namespace,
    zoo: Any,
    ginn: Any,
    surface: Dict[str, Any],
    run_root: Path,
    perf: Optional[Any] = None,
) -> SurfaceTrainResult:
    torch, nn, F = require_torch()
    sid = str(surface.get("surface_id"))
    tridx = int(surface.get("finite_group_triple", {}).get("triple_index", 0))
    train_out = run_root / "training" / sid
    train_out.mkdir(parents=True, exist_ok=True)
    atlas_dir = run_root / "atlas" / sid
    atlas_npz = atlas_dir / "exact_topk_atlas.npz"
    pool_npz = atlas_dir / f"candidate_pool_{args.train_pool_size}.npz"
    if not atlas_npz.exists():
        raise FileNotFoundError(f"Missing atlas file for {sid}: {atlas_npz}")
    if not pool_npz.exists():
        raise FileNotFoundError(f"Missing candidate pool for {sid}: {pool_npz}. Make sure --pool-sizes includes {args.train_pool_size}.")

    print("=" * 78)
    print(f"[train] {sid} pool={args.train_pool_size}", flush=True)
    if perf is not None:
        perf.log("train_surface_start", surface_id=sid, pool_size=args.train_pool_size)

    # Rebuild the deduplicated word ball only for standard explicit word-ball pools.
    # Large q=29 depth-2 virtual-stream pools embed candidate alpha/beta/depth/trace
    # arrays directly inside candidate_pool_*.npz, so rebuilding a 50M raw word list
    # would defeat the purpose of the streamed atlas engine.
    pool_probe = np.load(pool_npz, allow_pickle=False)
    pool_has_embedded_features = all(k in pool_probe.files for k in ["pool_alpha_real", "pool_alpha_imag", "pool_beta_real", "pool_beta_imag", "pool_depths", "pool_traces"])
    pool_word_ball_size = int(pool_probe["word_ball_size"]) if "word_ball_size" in pool_probe.files else -1
    pool_probe.close()
    if pool_has_embedded_features:
        from types import SimpleNamespace
        wb = SimpleNamespace(unique_size=pool_word_ball_size, raw_size=pool_word_ball_size)
        print(f"[train-word-ball] {sid} using embedded virtual pool features; no full word-ball rebuild; virtual_W={pool_word_ball_size}", flush=True)
        if perf is not None:
            perf.log("train_word_ball_skipped_embedded_pool", surface_id=sid, virtual_word_ball_size=pool_word_ball_size)
    else:
        t_wb = time.perf_counter()
        gens = ginn.parse_generators(surface)
        raw_word_ball = ginn.build_word_ball(gens, args.depth)
        wb = zoo.build_word_ball_data(
            ginn,
            raw_word_ball,
            dedupe=(not args.no_dedupe),
            tol=args.dedupe_tol,
            alias_sample_limit=args.alias_sample_limit,
        )
        raw_word_ball = []
        print(f"[train-word-ball] {sid} unique_W={wb.unique_size} rebuilt in {time.perf_counter()-t_wb:.2f}s", flush=True)
        if perf is not None:
            perf.log("train_word_ball_rebuilt", surface_id=sid, raw_word_ball_size=wb.raw_size, unique_word_ball_size=wb.unique_size)

    X, C, D, y_pos, true_dist, meta = build_feature_arrays(
        wb, atlas_npz, pool_npz, max_depth=args.depth, seed=args.seed + 1009 * (tridx + 1),
        shuffle_pool=(not args.no_shuffle_pool), feature_batch_size=args.feature_batch_size,
        perf=perf, surface_id=sid,
    )
    n = X.shape[0]
    P = C.shape[1]
    rng = np.random.default_rng(args.seed + 31337 + tridx)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = max(1, int(args.train_fraction * n))
    n_val = max(1, int(args.val_fraction * n))
    if n_train + n_val >= n:
        n_train = max(1, int(0.70 * n))
        n_val = max(1, int(0.15 * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    if len(test_idx) == 0:
        test_idx = val_idx.copy()
    Xn, Cn, norm = normalize_arrays(X, C, train_idx)

    device = args.train_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and not args.no_train_gpu else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[train warn] CUDA requested but unavailable; using CPU", flush=True)
        device = "cpu"
    if device.startswith("cuda"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    model = make_model(Xn.shape[1], Cn.shape[2], args.hidden, args.context_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[train] device={device} pairs={n} split={len(train_idx)}/{len(val_idx)}/{len(test_idx)} pool={P} hidden={args.hidden}", flush=True)

    cache_gpu = bool(args.cache_tensors_gpu and device.startswith("cuda"))
    tensors: Dict[str, Any] = {}
    if cache_gpu:
        try:
            tensors["X"] = torch.as_tensor(Xn, dtype=torch.float32, device=device)
            tensors["C"] = torch.as_tensor(Cn, dtype=torch.float32, device=device)
            tensors["D"] = torch.as_tensor(D, dtype=torch.float32, device=device)
            tensors["y"] = torch.as_tensor(y_pos, dtype=torch.long, device=device)
            tensors["true"] = torch.as_tensor(true_dist, dtype=torch.float32, device=device)
            print(f"[train] cached X/C/D tensors on {device}", flush=True)
        except Exception as e:
            print(f"[train warn] GPU tensor cache failed ({type(e).__name__}: {e}); using per-batch transfers", flush=True)
            tensors.clear()
            cache_gpu = False
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def batch_tensors(batch_idx: np.ndarray):
        if cache_gpu:
            bt = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
            return tensors["X"][bt], tensors["C"][bt], tensors["D"][bt], tensors["y"][bt]
        xb = torch.as_tensor(Xn[batch_idx], dtype=torch.float32, device=device)
        cb = torch.as_tensor(Cn[batch_idx], dtype=torch.float32, device=device)
        db = torch.as_tensor(D[batch_idx], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y_pos[batch_idx], dtype=torch.long, device=device)
        return xb, cb, db, yb

    t_train = time.perf_counter()
    train_log: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_left = int(args.patience)
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        perm = train_idx.copy()
        rng.shuffle(perm)
        loss_sum = 0.0
        ce_sum = 0.0
        kl_sum = 0.0
        nb = 0
        for s in range(0, len(perm), args.batch_size):
            bidx = perm[s:s+args.batch_size]
            xb, cb, db, yb = batch_tensors(bidx)
            opt.zero_grad(set_to_none=True)
            scores = model(xb, cb)
            ce = F.cross_entropy(scores, yb)
            if args.soft_distance_weight > 0:
                # Soft target over the candidate pool based on exact distances; exact
                # distances are labels only, not model inputs.
                target = torch.softmax(-db / max(args.soft_distance_tau, 1.0e-6), dim=1)
                kl = F.kl_div(F.log_softmax(scores, dim=1), target, reduction="batchmean")
                loss = ce + float(args.soft_distance_weight) * kl
            else:
                kl = torch.tensor(0.0, dtype=torch.float32, device=device)
                loss = ce
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_sum += float(loss.detach().cpu())
            ce_sum += float(ce.detach().cpu())
            kl_sum += float(kl.detach().cpu())
            nb += 1
        # Validation loss/recalls.
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            vce = 0.0
            vnb = 0
            val_scores_parts: List[np.ndarray] = []
            for s in range(0, len(val_idx), args.eval_batch_size):
                bidx = val_idx[s:s+args.eval_batch_size]
                xb, cb, db, yb = batch_tensors(bidx)
                scores = model(xb, cb)
                ce = F.cross_entropy(scores, yb)
                if args.soft_distance_weight > 0:
                    target = torch.softmax(-db / max(args.soft_distance_tau, 1.0e-6), dim=1)
                    kl = F.kl_div(F.log_softmax(scores, dim=1), target, reduction="batchmean")
                    loss = ce + float(args.soft_distance_weight) * kl
                else:
                    loss = ce
                vloss += float(loss.detach().cpu())
                vce += float(ce.detach().cpu())
                vnb += 1
                val_scores_parts.append(scores.detach().cpu().numpy().astype(np.float32))
            val_scores = np.vstack(val_scores_parts) if val_scores_parts else np.empty((0, P), dtype=np.float32)
        val_r1 = topk_hit_from_scores(val_scores, y_pos[val_idx], 1) if len(val_idx) else 0.0
        val_r5 = topk_hit_from_scores(val_scores, y_pos[val_idx], min(5, P)) if len(val_idx) else 0.0
        row = {
            "epoch": ep,
            "train_loss": loss_sum / max(nb, 1),
            "train_ce": ce_sum / max(nb, 1),
            "train_soft_kl": kl_sum / max(nb, 1),
            "val_loss": vloss / max(vnb, 1),
            "val_ce": vce / max(vnb, 1),
            "val_recall_at_1": val_r1,
            "val_recall_at_5": val_r5,
            "elapsed_seconds": time.perf_counter() - t_train,
        }
        train_log.append(row)
        if ep == 1 or ep % args.print_every == 0:
            print(f"[train {sid}] epoch {ep:4d}/{args.epochs} loss={row['train_loss']:.5f} val={row['val_loss']:.5f} valR1={val_r1:.3f} valR5={val_r5:.3f}", flush=True)
            if perf is not None:
                perf.log("train_epoch", surface_id=sid, epoch=ep, train_loss=row["train_loss"], val_loss=row["val_loss"], val_recall_at_1=val_r1, val_recall_at_5=val_r5)
        if row["val_loss"] < best_val - args.min_delta:
            best_val = row["val_loss"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(args.patience)
        else:
            patience_left -= 1
            if patience_left <= 0 and ep >= args.min_epochs:
                print(f"[train {sid}] early stopping at epoch {ep}; best_epoch={best_epoch} best_val={best_val:.5f}", flush=True)
                break
    train_seconds = time.perf_counter() - t_train
    if best_state is not None:
        model.load_state_dict(best_state)
    t_eval = time.perf_counter()
    train_metrics, _ = evaluate_model(model, Xn, Cn, D, y_pos, true_dist, train_idx, device, batch_size=args.eval_batch_size)
    val_metrics, _ = evaluate_model(model, Xn, Cn, D, y_pos, true_dist, val_idx, device, batch_size=args.eval_batch_size)
    test_metrics, test_scores = evaluate_model(model, Xn, Cn, D, y_pos, true_dist, test_idx, device, batch_size=args.eval_batch_size)
    eval_seconds = time.perf_counter() - t_eval

    # Predictions CSV for test split.
    pool_indices = meta["pool_indices"]
    true_indices = meta["true_indices"]
    order5 = ordered_topk(test_scores, min(5, P))
    order20 = ordered_topk(test_scores, min(20, P))
    pred_rows: List[Dict[str, Any]] = []
    for local_i, global_i in enumerate(test_idx):
        y = int(y_pos[global_i])
        top5_pos = order5[local_i].tolist()
        top20_pos = order20[local_i].tolist()
        global_top5 = [int(pool_indices[global_i, pos]) for pos in top5_pos]
        global_top20 = [int(pool_indices[global_i, pos]) for pos in top20_pos]
        pred_rows.append({
            "pair_id": int(global_i),
            "true_global_index": int(true_indices[global_i]),
            "true_pool_position": y,
            "pred_top1_pool_position": int(top5_pos[0]),
            "pred_top1_global_index": int(global_top5[0]),
            "winner_correct": int(top5_pos[0] == y),
            "top3_contains_true": int(y in set(order20[local_i, :min(3, P)].tolist())),
            "top5_contains_true": int(y in set(top5_pos)),
            "top10_contains_true": int(y in set(order20[local_i, :min(10, P)].tolist())),
            "top20_contains_true": int(y in set(top20_pos)),
            "true_distance": float(true_dist[global_i]),
            "pred_top1_distance": float(D[global_i, top5_pos[0]]),
            "top5_pruned_distance": float(np.min(D[global_i, top5_pos])),
            "top20_pruned_distance": float(np.min(D[global_i, top20_pos])),
            "top5_global_indices": "|".join(map(str, global_top5)),
            "top20_global_indices": "|".join(map(str, global_top20)),
            "split": "test",
        })
    write_csv(train_out / "predictions_test.csv", pred_rows)
    write_csv(train_out / "train_log.csv", train_log)

    metrics = {
        "program": PROGRAM,
        "version": VERSION,
        "surface_id": sid,
        "training_mode": "candidate_pool_reranker_not_full_word_ball_softmax",
        "interpretation": "The model reranks a compact candidate pool produced from the exact atlas. It does not score the full PSL(2,13) word ball and does not receive exact hyperbolic candidate distances as inputs.",
        "q": int(args.q),
        "triple_index": tridx,
        "genus": int(surface.get("genus") or -1),
        "unique_word_ball_size": int(wb.unique_size),
        "raw_word_ball_size": int(wb.raw_size),
        "pool_size": int(P),
        "pairs": int(n),
        "pool_has_embedded_features": bool(pool_has_embedded_features),
        "split_counts": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "device": device,
        "epochs_ran": int(train_log[-1]["epoch"] if train_log else 0),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "winner_pool_coverage": float(meta["winner_pool_coverage"]),
        "feature_build_seconds": float(meta["feature_build_seconds"]),
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
        "normalization": norm,
        "pair_feature_names": meta["pair_feature_names"],
        "candidate_feature_names": meta["candidate_feature_names"],
        "not_model_inputs": ["exact hyperbolic candidate distances", "full word-ball distance matrix"],
        "loss": {"cross_entropy_weight": 1.0, "soft_distance_kl_weight": float(args.soft_distance_weight), "soft_distance_tau": float(args.soft_distance_tau)},
    }
    write_json(train_out / "metrics.json", metrics)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "pair_dim": int(Xn.shape[1]),
        "candidate_dim": int(Cn.shape[2]),
        "hidden": int(args.hidden),
        "context_dim": int(args.context_dim),
        "normalization": norm,
        "pool_size": int(P),
        "note": "Candidate-pool reranker for Big Hurwitz PSL(2,13); not a full-word-ball GINN.",
    }
    torch.save(checkpoint, train_out / "big_hurwitz_candidate_pool_ginn_v1_0.pt")
    print(f"[train result] {sid} TEST R@1={test_metrics.get('recall_at_1', float('nan')):.3f} R@5={test_metrics.get('recall_at_5', float('nan')):.3f} R@20={test_metrics.get('recall_at_20', float('nan')):.3f} top5_RMSE={test_metrics.get('pruned_rmse_at_5', float('nan')):.5f}", flush=True)
    if perf is not None:
        perf.log("train_surface_done", surface_id=sid, test_recall_at_1=test_metrics.get("recall_at_1"), test_recall_at_5=test_metrics.get("recall_at_5"), train_seconds=train_seconds)
    # Free large arrays before next surface.
    result = SurfaceTrainResult(
        surface_id=sid,
        q=int(args.q),
        triple_index=tridx,
        genus=int(surface.get("genus") or -1),
        unique_word_ball_size=int(wb.unique_size),
        pool_size=int(P),
        train_pairs=int(len(train_idx)),
        val_pairs=int(len(val_idx)),
        test_pairs=int(len(test_idx)),
        device=device,
        epochs_ran=int(metrics["epochs_ran"]),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val),
        test_recall_at_1=float(test_metrics.get("recall_at_1", float('nan'))),
        test_recall_at_3=float(test_metrics.get("recall_at_3", float('nan'))),
        test_recall_at_5=float(test_metrics.get("recall_at_5", float('nan'))),
        test_recall_at_10=float(test_metrics.get("recall_at_10", float('nan'))),
        test_recall_at_20=float(test_metrics.get("recall_at_20", float('nan'))),
        test_top1_distance_rmse=float(test_metrics.get("top1_distance_rmse", float('nan'))),
        test_top5_pruned_rmse=float(test_metrics.get("pruned_rmse_at_5", float('nan'))),
        test_top10_pruned_rmse=float(test_metrics.get("pruned_rmse_at_10", float('nan'))),
        winner_pool_coverage=float(meta["winner_pool_coverage"]),
        feature_build_seconds=float(meta["feature_build_seconds"]),
        train_seconds=float(train_seconds),
        eval_seconds=float(eval_seconds),
        outdir=str(train_out),
    )
    try:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def write_training_report(run_root: Path, args: argparse.Namespace, surface_rows: List[Dict[str, Any]], atlas_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Fuchsian Big Hurwitz Trainer v1.2 Report")
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This run builds PSL(2,q) Hurwitz kernel surfaces, generates exact finite-word top-k atlases, and trains one candidate-pool GINN reranker per surface. The trainer is not a full-word-ball softmax over every deduplicated transformation; it reranks the saved compact candidate pool.")
    lines.append("")
    lines.append("## Run parameters")
    lines.append("")
    for k in ["q", "triple_equivalence", "max_triples", "depth", "pairs", "top_k_max", "train_pool_size", "epochs", "batch_size", "hidden", "context_dim", "engine", "candidate_chunk_size", "pair_batch_size", "target_vram_mb"]:
        lines.append(f"- `{k}`: `{getattr(args, k, '')}`")
    lines.append("")
    lines.append("## Atlas summary")
    lines.append("")
    if atlas_rows:
        cols = ["surface_id", "word_ball_size_raw", "word_ball_size_unique", "n_pairs", "engine", "pair_batch_size", "candidate_chunk_size", "wall_seconds", "evals_per_second", "shortcut_fraction", "median_gap12"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in atlas_rows:
            lines.append("| " + " | ".join(str(round(r.get(c), 5)) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Training summary")
    lines.append("")
    if train_rows:
        cols = ["surface_id", "unique_word_ball_size", "pool_size", "train_pairs", "val_pairs", "test_pairs", "device", "epochs_ran", "test_recall_at_1", "test_recall_at_3", "test_recall_at_5", "test_recall_at_10", "test_recall_at_20", "test_top5_pruned_rmse", "train_seconds"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in train_rows:
            lines.append("| " + " | ".join(str(round(r.get(c), 5)) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Top-k success here means: among the candidate pool supplied to the reranker, does the learned top-k set contain the exact finite-word winner from the atlas? The exact atlas is computed over the deduplicated finite word ball, but the neural model is trained only on the candidate pool, not the full word ball.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("Each surface has `training/<surface_id>/metrics.json`, `train_log.csv`, `predictions_test.csv`, and `big_hurwitz_candidate_pool_ginn_v1_0.pt`. Aggregate summaries are in `tables/`.")
    (run_root / "report").mkdir(parents=True, exist_ok=True)
    (run_root / "report" / "big_hurwitz_training_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build PSL(2,13) Big-Hurwitz atlases and train candidate-pool GINN rerankers.")
    # Atlas/Zoo options, intentionally aligned with FuchsianBigHurwitzZoo_v1_2.
    ap.add_argument("--q", type=int, default=13)
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
    ap.add_argument("--allow-huge-word-ball", action="store_true", help="Bypass the pre-build word-ball estimate guard and allocate full raw word list. Not recommended for q=29 depth 2.")
    ap.add_argument("--stream-huge-word-ball", action="store_true", help="For q=29+ depth 2: stream reduced depth-2 words virtually instead of allocating the full raw Python word list. Experimental; no global geometric deduplication.")
    ap.add_argument("--virtual-topk-buffer", type=int, default=5000, help="For --stream-huge-word-ball: keep this many raw nearest candidates per pair before local PSU(1,1) top-k deduplication. Larger is cleaner but uses more memory/time.")
    ap.add_argument("--virtual-topk-dedupe-tol", type=float, default=0.0, help="For --stream-huge-word-ball: tolerance for local top-k geometric deduplication. Default 0 means use --dedupe-tol.")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--dedupe-tol", type=float, default=1.0e-10)
    ap.add_argument("--alias-summary-rows", type=int, default=500)
    ap.add_argument("--alias-sample-limit", type=int, default=8)
    ap.add_argument("--pool-sizes", type=str, default="128,256,512,1024")
    ap.add_argument("--frequency-rows", type=int, default=200)
    ap.add_argument("--write-word-ball-summary", action="store_true")
    ap.add_argument("--outroot", type=str, default="big_hurwitz_training_runs")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--zoo-script", type=str, default=default_zoo_script())
    ap.add_argument("--hurwitz-script", type=str, default=default_hurwitz_script())
    ap.add_argument("--ginn-script", type=str, default=default_ginn_script())
    ap.add_argument("--max-kernel-generators", type=int, default=0)
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--identity-tol", type=float, default=1.0e-9)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-perf-log", action="store_true")
    ap.add_argument("--skip-atlas", action="store_true", help="Reuse existing atlas files under run_root. Mainly for debugging; default is to build the atlas first.")
    ap.add_argument("--no-train", action="store_true", help="Build atlas only, then stop.")
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
    if args.mode == "smoke":
        args.pairs = int(args.smoke_pairs)
        args.top_k_max = min(int(args.top_k_max), 20)
        args.csv_top_k = min(int(args.csv_top_k), int(args.top_k_max))
        args.epochs = min(int(args.epochs), 4)
        args.min_epochs = min(int(args.min_epochs), 1)
        args.patience = min(int(args.patience), 3)
        args.train_pool_size = min(int(args.train_pool_size), 64)
        if not args.label:
            args.label = "smoke_train"
    # Ensure pool-sizes includes the training pool.
    pool_sizes = sorted(set([x for x in parse_int_list(args.pool_sizes) if x > 0] + [int(args.train_pool_size)]))
    args.pool_sizes = ",".join(str(x) for x in pool_sizes)
    if not args.label:
        args.label = f"psl{args.q}_depth{args.depth}_pairs{args.pairs}_trainer_v11"

    stamp = now_stamp()
    run_name = f"run_{stamp}_{stable_slug(args.label)}"
    run_root = Path(args.outroot) / run_name
    for sub in ["group", "surfaces", "kernel_audits", "atlas", "training", "tables", "report"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    zoo = load_module(args.zoo_script, "big_hurwitz_zoo_v17")
    perf = zoo.PerfTracker(run_root / "tables" / "performance_log.csv", enabled=(not args.no_perf_log))
    t_all = time.perf_counter()
    print(f"{PROGRAM} v{VERSION}")
    print(f"run_root={run_root}")
    print(f"q={args.q} triples={args.max_triples} depth={args.depth} pairs={args.pairs} train_pool={args.train_pool_size}")
    print("-" * 78)
    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "args": vars(args).copy(),
        "python": sys.version,
        "platform": platform.platform(),
        "purpose": "Integrated Big-Hurwitz exact atlas generation plus candidate-pool GINN reranker training.",
    }
    write_json(run_root / "manifest.json", manifest)

    try:
        require_torch()
        perf.log("module_load_start", hurwitz_script=args.hurwitz_script, ginn_script=args.ginn_script, zoo_script=args.zoo_script)
        hurwitz = zoo.load_module(args.hurwitz_script, "hurwitz_v16")
        ginn = zoo.load_module(args.ginn_script, "ginn_v24")
        perf.log("module_load_done")
        group, triples, surfaces, audits = zoo.build_surfaces(args, hurwitz, run_root, perf=perf)
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

        atlas_results: List[Any] = []
        failure_rows: List[Dict[str, Any]] = []
        if not args.skip_atlas:
            # The imported v1.2 atlas builder expects mode atlas/smoke; it is harmless
            # for this trainer to pass mode=train except on failure behavior.  Use atlas.
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
        write_csv(run_root / "tables" / "big_hurwitz_atlas_summary.csv", atlas_rows)
        if args.no_train:
            write_csv(run_root / "tables" / "failures.csv", failure_rows, ["stage", "surface_id", "error_type", "error"])
            summary = {"completed": datetime.now().isoformat(timespec="seconds"), "wall_seconds": time.perf_counter() - t_all, "surfaces_built": len(surfaces), "atlases_completed": len(atlas_results), "trained_surfaces": 0, "failures": len(failure_rows), "run_root": str(run_root), "process_peak_rss_mb": perf.peak_rss_mb}
            write_json(run_root / "run_summary.json", summary)
            perf.log("run_done", **summary)
            return 0 if not failure_rows else 1

        train_results: List[SurfaceTrainResult] = []
        for s in surfaces:
            sid = str(s.get("surface_id"))
            if failure_rows and any(r.get("surface_id") == sid and r.get("stage") == "atlas" for r in failure_rows):
                continue
            try:
                train_results.append(train_surface_reranker(args, zoo, ginn, s, run_root, perf=perf))
            except Exception as e:
                print(f"[train fail] {sid}: {type(e).__name__}: {e}", flush=True)
                failure_rows.append({"stage": "train", "surface_id": sid, "error_type": type(e).__name__, "error": str(e)})
                continue
        train_rows = [r.__dict__ for r in train_results]
        write_csv(run_root / "tables" / "big_hurwitz_training_summary.csv", train_rows)
        write_csv(run_root / "tables" / "failures.csv", failure_rows, ["stage", "surface_id", "error_type", "error"])
        write_training_report(run_root, args, surface_rows, atlas_rows, train_rows)
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
