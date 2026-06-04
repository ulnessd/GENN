#!/usr/bin/env python3
"""FuchsianSchottkyGridBatchTrainer_v2_0.py

Batch wrapper for the Schottky/free-Fuchsian part of the GENN zoo.

This script deliberately stays within the requested scope:
  * all records are smooth Riemann-surface quotients D/Gamma;
  * no cone-point/orbifold records are generated;
  * each record has an explicit ideal-geodesic Schottky domain in the disk plus
    the bounded sampling scaffold already used by FuchsianSchottkyTester_v1_1.

It is a thin, robust launcher around FuchsianSchottkyTester_v1_1.py, adding
curated grid profiles and a batch-level report.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROGRAM = "FuchsianSchottkyGridBatchTrainer_v2_0.py"
VERSION = "v2.0"


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_float_list(s: str) -> List[float]:
    if not s.strip():
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def profile_values(profile: str) -> Dict[str, Any]:
    if profile == "smoke":
        return {"ranks": [2], "gaps": [0.18], "rotations": [0.0], "pairs": 600, "epochs": 5}
    if profile == "small":
        return {"ranks": [2, 3, 4], "gaps": [0.10, 0.18, 0.26], "rotations": [0.0], "pairs": 4000, "epochs": 60}
    if profile == "balanced":
        return {"ranks": [2, 3, 4, 5, 6], "gaps": [0.05, 0.10, 0.18, 0.26, 0.34], "rotations": [0.0, 11.0, 23.0], "pairs": 9000, "epochs": 80}
    if profile == "wide":
        return {"ranks": [2, 3, 4, 5, 6, 7, 8], "gaps": [0.04, 0.08, 0.12, 0.18, 0.26, 0.34, 0.42], "rotations": [0.0, 7.5, 15.0, 30.0], "pairs": 9000, "epochs": 100}
    raise ValueError(f"Unknown profile {profile!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch Schottky/free-Fuchsian grid trainer")
    ap.add_argument("--profile", default="balanced", choices=["smoke", "small", "balanced", "wide", "manual"])
    ap.add_argument("--ranks", default="", help="comma-list override, e.g. 2,3,4,5")
    ap.add_argument("--gaps", default="", help="comma-list override, e.g. 0.05,0.10,0.18")
    ap.add_argument("--rotations", default="", help="comma-list override in degrees, e.g. 0,11,23")
    ap.add_argument("--tester-script", default="FuchsianSchottkyTester_v1_1.py")
    ap.add_argument("--maker", default="FuchsianDomainMaker_v13.py")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--outroot", default="schottky_grid_batch_runs")
    ap.add_argument("--tester-outroot", default="schottky_grid_tester_runs")
    ap.add_argument("--label", default="")
    ap.add_argument("--sample-radius", type=float, default=0.82)
    ap.add_argument("--endpoint-tol", type=float, default=5e-5)
    ap.add_argument("--pairs", type=int, default=0, help="override profile pairs if >0")
    ap.add_argument("--epochs", type=int, default=0, help="override profile epochs if >0")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-word-ball", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--candidate-chunk-size", type=int, default=8192)
    ap.add_argument("--pair-hidden", type=int, default=192)
    ap.add_argument("--score-hidden", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = profile_values(args.profile) if args.profile != "manual" else {"ranks": [2], "gaps": [0.18], "rotations": [0.0], "pairs": 9000, "epochs": 80}
    ranks = parse_int_list(args.ranks) or cfg["ranks"]
    gaps = parse_float_list(args.gaps) or cfg["gaps"]
    rotations = parse_float_list(args.rotations) or cfg["rotations"]
    pairs = args.pairs if args.pairs > 0 else int(cfg["pairs"])
    epochs = args.epochs if args.epochs > 0 else int(cfg["epochs"])

    label = args.label or f"schottky_grid_{args.profile}"
    run_root = Path(args.outroot) / f"run_{now_stamp()}_{label}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "schottky_grid_tester.log"

    cmd = [
        sys.executable, args.tester_script,
        "--rank", *[str(x) for x in ranks],
        "--gaps", *[str(x) for x in gaps],
        "--rotations", *[str(x) for x in rotations],
        "--sample-radius", str(args.sample_radius),
        "--endpoint-tol", str(args.endpoint_tol),
        "--maker", args.maker,
        "--ginn-script", args.ginn_script,
        "--outroot", args.tester_outroot,
        "--label", label,
        "--run-ginn",
        "--ginn-pairs", str(pairs),
        "--ginn-depth", str(args.depth),
        "--ginn-max-word-ball", str(args.max_word_ball),
        "--ginn-epochs", str(epochs),
        "--ginn-batch-size", str(args.batch_size),
        "--ginn-device", args.device,
        "--ginn-pair-hidden", str(args.pair_hidden),
        "--ginn-score-hidden", str(args.score_hidden),
        "--ginn-lr", str(args.lr),
        "--ginn-patience", str(args.patience),
        "--ginn-candidate-chunk-size", str(args.candidate_chunk_size),
        "--seed", str(args.seed),
    ]

    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "label": label,
        "profile": args.profile,
        "ranks": ranks,
        "gaps": gaps,
        "rotations": rotations,
        "surface_count": len(ranks) * len(gaps) * len(rotations),
        "pairs": pairs,
        "epochs": epochs,
        "depth": args.depth,
        "max_word_ball": args.max_word_ball,
        "dry_run": args.dry_run,
        "cmd": cmd,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        rc = None
        stdout_tail = ""
        stderr_tail = ""
        wall = 0.0
    else:
        t0 = time.time()
        p = subprocess.run(cmd, text=True, capture_output=True)
        wall = time.time() - t0
        rc = p.returncode
        log_path.write_text("COMMAND:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + p.stdout + "\n\nSTDERR:\n" + p.stderr)
        stdout_tail = p.stdout[-4000:]
        stderr_tail = p.stderr[-4000:]

    report = run_root / "schottky_grid_batch_report.md"
    report.write_text(
        f"# Schottky Grid Batch Trainer Report\n\n"
        f"Program: `{PROGRAM}` {VERSION}\n\n"
        f"Profile: `{args.profile}`\n\n"
        f"Surface count: **{len(ranks) * len(gaps) * len(rotations)}**\n\n"
        f"Ranks: `{ranks}`\n\nGaps: `{gaps}`\n\nRotations: `{rotations}`\n\n"
        f"Pairs: `{pairs}`  Epochs: `{epochs}`  Depth: `{args.depth}`\n\n"
        f"Dry run: `{args.dry_run}`\n\nReturn code: `{rc}`\n\nWall seconds: `{wall:.2f}`\n\n"
        f"## Command\n\n```bash\n{' '.join(cmd)}\n```\n\n"
        f"## Stdout tail\n\n```text\n{stdout_tail}\n```\n\n"
        f"## Stderr tail\n\n```text\n{stderr_tail}\n```\n"
    )
    print(f"[done] report={report}")
    return 0 if (args.dry_run or rc == 0) else int(rc or 1)


if __name__ == "__main__":
    raise SystemExit(main())
