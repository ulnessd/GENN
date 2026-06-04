#!/usr/bin/env python3
"""FuchsianNonTriangleZooOrchestrator_v1_0.py

Sequential orchestrator for the six non-triangle zoo expansion trainers:

1. FuchsianSchottkyGridBatchTrainer_v2_0.py
2. FuchsianHeckeGridBatchTrainer_v1_0.py
3. FuchsianModularPermutationSubgroupTrainer_v1_1.py
4. FuchsianIdealCuspedSurfaceBatchTrainer_v1_0.py
5. FuchsianCompactCoverBatchTrainer_v1_1.py
6. FuchsianSidePairingPolygonBatchTrainer_v1_1.py

The goal is convenience and rigor rather than clever scheduling: run each family
in succession, collect logs, continue after failures by default, and write a
single master report linking to the individual family reports.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

VERSION = "1.0"

DEFAULT_RUN_ROOT = "nontriangle_zoo_runs"
DEFAULT_PROFILE = "overnight"

TRAINER_FILES = {
    "schottky": "FuchsianSchottkyGridBatchTrainer_v2_0.py",
    "hecke": "FuchsianHeckeGridBatchTrainer_v1_0.py",
    "modperm": "FuchsianModularPermutationSubgroupTrainer_v1_1.py",
    "ideal": "FuchsianIdealCuspedSurfaceBatchTrainer_v1_0.py",
    "compact_cover": "FuchsianCompactCoverBatchTrainer_v1_1.py",
    "sidepair": "FuchsianSidePairingPolygonBatchTrainer_v1_1.py",
}

FAMILY_ORDER = [
    "schottky",
    "hecke",
    "modperm",
    "ideal",
    "compact_cover",
    "sidepair",
]

EXPECTED_COUNTS = {
    "smoke": {
        "schottky": None,
        "hecke": None,
        "modperm": 2,
        "ideal": 2,
        "compact_cover": None,
        "sidepair": None,
    },
    "overnight": {
        "schottky": 75,
        "hecke": 36,
        "modperm": 12,
        "ideal": 16,
        "compact_cover": 36,
        "sidepair": 28,
    },
}


@dataclasses.dataclass
class FamilyResult:
    name: str
    script_path: str
    command: List[str]
    log_path: str
    return_code: Optional[int] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    status: str = "PENDING"
    report_path: Optional[str] = None
    expected_animals: Optional[int] = None
    notes: Optional[str] = None


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def ts_compact() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_scripts_exist(script_dir: Path, family_names: List[str]) -> List[str]:
    missing = []
    for fam in family_names:
        path = script_dir / TRAINER_FILES[fam]
        if not path.exists():
            missing.append(str(path))
    return missing


def build_family_command(
    family: str,
    args: argparse.Namespace,
    family_label: str,
) -> List[str]:
    py = args.python_bin
    script = str(Path(args.script_dir) / TRAINER_FILES[family])

    # Shared defaults for real runs.
    common_real = [
        "--pairs", str(args.pairs),
        "--epochs", str(args.epochs),
        "--depth", str(args.depth),
        "--max-word-ball", str(args.max_word_ball),
        "--batch-size", str(args.batch_size),
        "--candidate-chunk-size", str(args.candidate_chunk_size),
    ]
    # Some trainers use --device, earlier ones may ignore/accept it.
    common_real += ["--device", args.device]

    if family == "schottky":
        cmd = [py, script, "--profile", "smoke" if args.profile == "smoke" else "balanced"]
        if args.dry_run:
            cmd += ["--dry-run"]
        else:
            cmd += common_real
        cmd += ["--label", family_label]
        return cmd

    if family == "hecke":
        cmd = [py, script, "--profile", "smoke" if args.profile == "smoke" else "balanced"]
        if args.dry_run:
            cmd += ["--dry-run"]
        else:
            cmd += common_real
        cmd += ["--label", family_label]
        return cmd

    if family == "modperm":
        if args.profile == "smoke":
            indices = args.modperm_indices_smoke
            samples = 1
            max_surfaces = 2
        else:
            indices = args.modperm_indices_overnight
            samples = args.modperm_samples_per_index
            max_surfaces = None
        cmd = [
            py, script,
            "--indices", indices,
            "--samples-per-index", str(samples),
        ]
        if args.profile != "smoke":
            cmd += ["--dedupe-signature"]
        if max_surfaces is not None:
            cmd += ["--max-surfaces", str(max_surfaces)]
        if args.dry_run:
            cmd += ["--dry-run"]
        else:
            cmd += ["--run-ginn"] + common_real
        cmd += ["--label", family_label]
        return cmd

    if family == "ideal":
        targets = args.ideal_targets_smoke if args.profile == "smoke" else args.ideal_targets_overnight
        samples = 1 if args.profile == "smoke" else args.ideal_samples_per_target
        max_surfaces = 2 if args.profile == "smoke" else None
        cmd = [
            py, script,
            "--targets", targets,
            "--samples-per-target", str(samples),
        ]
        if args.profile != "smoke":
            cmd += ["--dedupe-cusp-widths"]
        if max_surfaces is not None:
            cmd += ["--max-surfaces", str(max_surfaces)]
        if args.dry_run:
            cmd += ["--dry-run"]
        else:
            cmd += ["--run-ginn"] + common_real
        cmd += ["--label", family_label]
        return cmd

    if family == "compact_cover":
        cmd = [py, script, "--profile", "smoke" if args.profile == "smoke" else "balanced"]
        if args.profile == "smoke" and not args.dry_run:
            cmd += ["--max-surfaces", "1"]
        if args.dry_run:
            cmd += ["--dry-run"]
        else:
            cmd += ["--run-ginn"] + common_real
        cmd += ["--label", family_label]
        return cmd

    if family == "sidepair":
        cmd = [py, script, "--profile", "smoke" if args.profile == "smoke" else "balanced"]
        if args.profile == "smoke" and not args.dry_run:
            cmd += ["--max-surfaces", "1"]
        if args.dry_run:
            cmd += ["--dry-run"]
        else:
            cmd += ["--run-ginn"] + common_real
        cmd += ["--label", family_label]
        return cmd

    raise ValueError(f"Unknown family: {family}")


def parse_report_from_text(text: str) -> Optional[str]:
    patterns = [
        r"report=([^\s]+)",
        r"\[done\]\s+report=([^\s]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def run_one_family(family: str, command: List[str], log_path: Path, expected_animals: Optional[int]) -> FamilyResult:
    result = FamilyResult(
        name=family,
        script_path=command[1] if len(command) > 1 else "",
        command=command,
        log_path=str(log_path),
        expected_animals=expected_animals,
    )
    result.started_at = now_iso()
    start = _dt.datetime.now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# Command\n{shlex.join(command)}\n\n")
        logf.flush()
        proc = subprocess.run(command, stdout=logf, stderr=subprocess.STDOUT, text=True)
        result.return_code = proc.returncode
    end = _dt.datetime.now()
    result.ended_at = now_iso()
    result.elapsed_seconds = (end - start).total_seconds()
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        text = f"[log read failed: {exc}]"
    result.report_path = parse_report_from_text(text)
    if result.return_code == 0:
        result.status = "OK"
    else:
        result.status = "FAILED"
        tail = "\n".join(text.splitlines()[-20:])
        result.notes = tail
    return result


def make_master_report(out_path: Path, run_root: Path, args: argparse.Namespace, results: List[FamilyResult]) -> None:
    ok = sum(1 for r in results if r.status == "OK")
    failed = sum(1 for r in results if r.status == "FAILED")
    expected_total = sum(r.expected_animals or 0 for r in results)

    lines: List[str] = []
    lines.append(f"# Non-triangle zoo orchestrator report\n")
    lines.append(f"- Version: {VERSION}")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Profile: `{args.profile}`")
    lines.append(f"- Dry run: `{args.dry_run}`")
    lines.append(f"- Started: `{args._started_at}`")
    lines.append(f"- Finished: `{now_iso()}`")
    lines.append(f"- Families requested: `{', '.join(args.families)}`")
    lines.append(f"- Succeeded: **{ok}**")
    lines.append(f"- Failed: **{failed}**")
    if expected_total:
        lines.append(f"- Approximate expected animals from selected profiles: **{expected_total}**")
    lines.append("")

    lines.append("## Global settings\n")
    settings = {
        "pairs": args.pairs,
        "epochs": args.epochs,
        "depth": args.depth,
        "max_word_ball": args.max_word_ball,
        "batch_size": args.batch_size,
        "candidate_chunk_size": args.candidate_chunk_size,
        "device": args.device,
        "continue_on_error": args.continue_on_error,
    }
    for k, v in settings.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Family summary\n")
    lines.append("| Family | Status | Approx. animals | Elapsed (s) | Report | Log |")
    lines.append("|---|---:|---:|---:|---|---|")
    for r in results:
        rep = f"`{r.report_path}`" if r.report_path else ""
        log = f"`{r.log_path}`"
        elapsed = f"{r.elapsed_seconds:.1f}" if r.elapsed_seconds is not None else ""
        approx = "" if r.expected_animals is None else str(r.expected_animals)
        lines.append(f"| {r.name} | {r.status} | {approx} | {elapsed} | {rep} | {log} |")
    lines.append("")

    lines.append("## Commands\n")
    for r in results:
        lines.append(f"### {r.name}\n")
        lines.append("```bash")
        lines.append(shlex.join(r.command))
        lines.append("```")
        if r.notes:
            lines.append("")
            lines.append("Last lines from failure log:")
            lines.append("```text")
            lines.append(r.notes)
            lines.append("```")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_manifest(out_path: Path, run_root: Path, args: argparse.Namespace, results: List[FamilyResult]) -> None:
    payload = {
        "program": "FuchsianNonTriangleZooOrchestrator_v1_0.py",
        "version": VERSION,
        "run_root": str(run_root),
        "profile": args.profile,
        "dry_run": args.dry_run,
        "started_at": args._started_at,
        "written_at": now_iso(),
        "families": args.families,
        "settings": {
            "pairs": args.pairs,
            "epochs": args.epochs,
            "depth": args.depth,
            "max_word_ball": args.max_word_ball,
            "batch_size": args.batch_size,
            "candidate_chunk_size": args.candidate_chunk_size,
            "device": args.device,
            "continue_on_error": args.continue_on_error,
        },
        "results": [dataclasses.asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequential orchestrator for the six non-triangle zoo expansion trainers."
    )
    p.add_argument("--profile", choices=["smoke", "overnight"], default=DEFAULT_PROFILE)
    p.add_argument("--dry-run", action="store_true", help="Ask each sub-trainer for a dry run instead of real training.")
    p.add_argument("--families", default=",".join(FAMILY_ORDER),
                   help="Comma-separated subset of families to run. Choices: " + ",".join(FAMILY_ORDER))
    p.add_argument("--script-dir", default=".", help="Directory containing the six trainer scripts.")
    p.add_argument("--python-bin", default=sys.executable, help="Python interpreter to use for sub-runs.")
    p.add_argument("--outroot", default=DEFAULT_RUN_ROOT, help="Root directory for orchestrator outputs.")
    p.add_argument("--label", default="nontriangle_overnight", help="Run label suffix.")
    p.add_argument("--continue-on-error", action="store_true", default=True,
                   help="Continue after a family failure (default true).")
    p.add_argument("--stop-on-error", action="store_true", help="Stop after first failing family.")

    p.add_argument("--pairs", type=int, default=9000)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--max-word-ball", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--candidate-chunk-size", type=int, default=8192)
    p.add_argument("--device", default="auto")

    p.add_argument("--modperm-indices-smoke", default="6,12")
    p.add_argument("--modperm-indices-overnight", default="6,12,18,24,30,36")
    p.add_argument("--modperm-samples-per-index", type=int, default=2)

    p.add_argument("--ideal-targets-smoke", default="0:3,1:1")
    p.add_argument("--ideal-targets-overight", dest="ideal_targets_overnight",
                   default="0:3,1:1,0:4,1:2,2:1,0:5,1:3,2:2")
    p.add_argument("--ideal-samples-per-target", type=int, default=2)

    ns = p.parse_args(argv)
    ns._started_at = now_iso()
    ns.families = [x.strip() for x in ns.families.split(",") if x.strip()]
    bad = [x for x in ns.families if x not in FAMILY_ORDER]
    if bad:
        raise SystemExit(f"Unknown family names: {bad}")
    if ns.stop_on_error:
        ns.continue_on_error = False
    return ns


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    script_dir = Path(args.script_dir).resolve()
    missing = ensure_scripts_exist(script_dir, args.families)
    if missing:
        print("Missing required trainer scripts:")
        for m in missing:
            print("  ", m)
        return 2

    run_root = Path(args.outroot) / f"run_{ts_compact()}_{args.label}"
    logs_dir = run_root / "logs"
    reports_dir = run_root / "reports"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: List[FamilyResult] = []
    print(f"FuchsianNonTriangleZooOrchestrator_v1_0.py v{VERSION}")
    print(f"run_root={run_root}")
    print(f"profile={args.profile} dry_run={args.dry_run}")
    print("families=", ", ".join(args.families))
    print("-" * 78)

    for family in args.families:
        fam_label = f"{family}_{args.profile}"
        cmd = build_family_command(family, args, fam_label)
        log_path = logs_dir / f"{family}.log"
        expected = EXPECTED_COUNTS.get(args.profile, {}).get(family)
        print(f"[run] {family} expected~{expected if expected is not None else '?'}")
        print(f"[cmd] {shlex.join(cmd)}")
        fam_res = run_one_family(family, cmd, log_path, expected)
        results.append(fam_res)
        print(f"[done] {family} status={fam_res.status} rc={fam_res.return_code} elapsed={fam_res.elapsed_seconds:.1f}s")
        if fam_res.report_path:
            print(f"       report={fam_res.report_path}")
        print()
        if fam_res.status != "OK" and not args.continue_on_error:
            print("Stopping on first error as requested.")
            break

    report_path = reports_dir / "nontriangle_zoo_orchestrator_report.md"
    manifest_path = run_root / "nontriangle_zoo_manifest.json"
    make_master_report(report_path, run_root, args, results)
    make_manifest(manifest_path, run_root, args, results)

    n_fail = sum(1 for r in results if r.status != "OK")
    print("=" * 78)
    print(f"[done] families_run={len(results)} failures={n_fail}")
    print(f"[done] report={report_path}")
    print(f"[done] manifest={manifest_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
