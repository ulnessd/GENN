#!/usr/bin/env python3
"""FuchsianHeckeGridBatchTrainer_v1_0.py

Batch wrapper for torsion-free Hecke cover trainers in the GENN zoo.

This script uses FuchsianHeckeTester_v1_2.py.  It generates only the existing
smooth torsion-free Hecke cover families used by that tester: abelian and
nonabelian dihedral covers of Hecke groups H_q = Delta(2,q,infinity).  The
parent Hecke group is an orbifold, but the exported cover records are audited
as torsion-free finite-area Riemann surfaces with explicit disk/Ford domains.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROGRAM = "FuchsianHeckeGridBatchTrainer_v1_0.py"
VERSION = "v1.0"


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_int_list(s: str) -> List[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def profile_values(profile: str) -> Dict[str, Any]:
    if profile == "smoke":
        return {"q": [5], "pairs": 600, "epochs": 5}
    if profile == "small":
        return {"q": [3,4,5,6,7,8,9,10,12], "pairs": 4000, "epochs": 60}
    if profile == "balanced":
        return {"q": list(range(3, 21)), "pairs": 9000, "epochs": 80}
    if profile == "wide":
        return {"q": list(range(3, 31)), "pairs": 9000, "epochs": 100}
    raise ValueError(profile)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch Hecke torsion-free cover grid trainer")
    ap.add_argument("--profile", default="balanced", choices=["smoke", "small", "balanced", "wide", "manual"])
    ap.add_argument("--q", default="", help="comma-list override, e.g. 3,4,5,6")
    ap.add_argument("--families", default="ab,d", help="HeckeTester families: ab,d")
    ap.add_argument("--tester-script", default="FuchsianHeckeTester_v1_2.py")
    ap.add_argument("--maker", default="FuchsianDomainMaker_v13.py")
    ap.add_argument("--ginn-script", default="FuchsianDownstairsGINN_v2_4.py")
    ap.add_argument("--outroot", default="hecke_grid_batch_runs")
    ap.add_argument("--tester-outroot", default="hecke_grid_tester_runs")
    ap.add_argument("--label", default="")
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-word-ball", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--profile-ginn", default="balanced", choices=["balanced", "accurate", "fast", "manual"])
    ap.add_argument("--max-generators", type=int, default=0)
    ap.add_argument("--max-cosets", type=int, default=20000)
    ap.add_argument("--candidate-chunk-size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = profile_values(args.profile) if args.profile != "manual" else {"q": [5], "pairs": 9000, "epochs": 80}
    q_vals = parse_int_list(args.q) or cfg["q"]
    pairs = args.pairs if args.pairs > 0 else int(cfg["pairs"])
    epochs = args.epochs if args.epochs > 0 else int(cfg["epochs"])
    fams = [x.strip() for x in args.families.split(",") if x.strip()]

    label = args.label or f"hecke_grid_{args.profile}"
    run_root = Path(args.outroot) / f"run_{now_stamp()}_{label}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "hecke_grid_tester.log"

    cmd = [
        sys.executable, args.tester_script,
        "--q", *[str(x) for x in q_vals],
        "--families", args.families,
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
        "--profile", args.profile_ginn,
        "--max-generators", str(args.max_generators),
        "--max-cosets", str(args.max_cosets),
        "--candidate-chunk-size", str(args.candidate_chunk_size),
        "--seed", str(args.seed),
    ]

    manifest = {
        "program": PROGRAM,
        "version": VERSION,
        "label": label,
        "profile": args.profile,
        "q": q_vals,
        "families": fams,
        "surface_count_expected": len(q_vals) * len(fams),
        "pairs": pairs,
        "epochs": epochs,
        "depth": args.depth,
        "max_word_ball": args.max_word_ball,
        "dry_run": args.dry_run,
        "cmd": cmd,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        rc = None; stdout_tail = ""; stderr_tail = ""; wall = 0.0
    else:
        t0 = time.time()
        p = subprocess.run(cmd, text=True, capture_output=True)
        wall = time.time() - t0
        rc = p.returncode
        log_path.write_text("COMMAND:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + p.stdout + "\n\nSTDERR:\n" + p.stderr)
        stdout_tail = p.stdout[-4000:]
        stderr_tail = p.stderr[-4000:]

    report = run_root / "hecke_grid_batch_report.md"
    report.write_text(
        f"# Hecke Grid Batch Trainer Report\n\n"
        f"Program: `{PROGRAM}` {VERSION}\n\n"
        f"Profile: `{args.profile}`\n\n"
        f"Expected surface count: **{len(q_vals) * len(fams)}**\n\n"
        f"q values: `{q_vals}`\n\nFamilies: `{fams}`\n\n"
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
