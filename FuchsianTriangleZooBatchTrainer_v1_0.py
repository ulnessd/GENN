#!/usr/bin/env python3
"""
FuchsianTriangleZooBatchTrainer_v1_0.py

Manifest-driven batch trainer for the triangle-quotient zoo.

Reads the JSON candidate manifest produced by FuchsianTriangleQuotientSearcher_v1_2.py
and sequentially launches FuchsianTriangleQuotientTrainer_v1_2.py for each candidate.

Design intent:
  COMPLETE_BALL_READY    -> train with --kernel-generator-mode all
  SELECTED_ATLAS_READY   -> train with --kernel-generator-mode shortest --kernel-generator-limit N
  TOO_LARGE...           -> skipped by default

This is deliberately conservative: one subprocess per candidate, a log per candidate,
and a batch summary report at the end. It can also produce a dry-run shell script.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

VERSION = "v1.0"

COMPLETE = "COMPLETE_BALL_READY"
SELECTED = "SELECTED_ATLAS_READY"
TOO_LARGE = "TOO_LARGE_FOR_CURRENT_PIPELINE"


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(s: str, max_len: int = 120) -> str:
    s = str(s).strip()
    s = s.replace("(", "").replace(")", "").replace(",", "_")
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def find_latest_candidates(search_outroot: Path) -> Path:
    runs = sorted(search_outroot.glob("run_*/candidates/triangle_quotient_candidates.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError(f"No triangle_quotient_candidates.json found under {search_outroot}/run_*/candidates")
    return runs[0]


def parse_sig(c: Dict[str, Any]) -> Tuple[int, int, int]:
    if "signature_tuple" in c:
        return tuple(int(x) for x in c["signature_tuple"])
    s = str(c.get("signature", "")).strip()
    m = re.findall(r"\d+", s)
    if len(m) != 3:
        raise RuntimeError(f"Cannot parse signature from candidate: {s!r}")
    return tuple(int(x) for x in m)


def candidate_group(c: Dict[str, Any]) -> str:
    return str(c.get("group") or c.get("group_name") or c.get("quotient") or "UNKNOWN")


def candidate_category(c: Dict[str, Any]) -> str:
    return str(c.get("category") or "")


def candidate_label(c: Dict[str, Any], mode_tag: str, pairs: int, depth: int) -> str:
    p, q, r = parse_sig(c)
    g = candidate_group(c)
    return slugify(f"tri_{p}_{q}_{r}_{g}_{mode_tag}_depth{depth}_pairs{pairs}")


def estimate_raw_complete(c: Dict[str, Any]) -> int:
    for key in ("complete_depth2_raw_word_ball_est", "complete_depth2_raw_word_ball"):
        if key in c and str(c[key]) not in {"", "None"}:
            try:
                return int(float(c[key]))
            except Exception:
                pass
    return -1


def sort_key(c: Dict[str, Any]) -> Tuple[int, int, int, str]:
    cat_rank = {COMPLETE: 0, SELECTED: 1, TOO_LARGE: 2}.get(candidate_category(c), 9)
    raw = estimate_raw_complete(c)
    order = int(float(c.get("order") or 0))
    return (cat_rank, raw if raw >= 0 else 10**18, order, candidate_group(c))


def filter_candidates(cands: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    cats = {x.strip() for x in args.categories.split(",") if x.strip()}
    out: List[Dict[str, Any]] = []
    for c in cands:
        cat = candidate_category(c)
        if cat not in cats:
            continue
        if args.include_regex and not re.search(args.include_regex, json.dumps(c)):
            continue
        if args.exclude_regex and re.search(args.exclude_regex, json.dumps(c)):
            continue
        if args.complete_only and cat != COMPLETE:
            continue
        if args.selected_only and cat != SELECTED:
            continue
        if args.max_order and int(float(c.get("order") or 0)) > args.max_order:
            continue
        if args.max_candidates and len(out) >= args.max_candidates:
            break
        out.append(c)
    out.sort(key=sort_key)
    return out


def command_for_candidate(c: Dict[str, Any], desc_path: Path, trainer_path: Path, args: argparse.Namespace) -> Tuple[List[str], str, str]:
    sig = parse_sig(c)
    cat = candidate_category(c)
    if cat == COMPLETE:
        kernel_mode = "all"
        kernel_limit = args.complete_kernel_limit
        train_pool = args.complete_train_pool_size
        pools = args.complete_pool_sizes or default_pool_sizes(train_pool)
        mode_tag = "ALL"
        max_word = args.complete_max_word_ball
        max_unique = args.complete_max_unique_word_ball
    elif cat == SELECTED:
        kernel_mode = "shortest"
        kernel_limit = args.selected_kernel_limit
        train_pool = args.selected_train_pool_size or args.selected_kernel_limit
        pools = args.selected_pool_sizes or default_pool_sizes(train_pool)
        mode_tag = f"shortest{kernel_limit}"
        max_word = args.selected_max_word_ball
        max_unique = args.selected_max_unique_word_ball
    else:
        raise RuntimeError(f"Unsupported category for training: {cat}")

    label = args.label_prefix + candidate_label(c, mode_tag, args.pairs, args.depth)
    cmd = [
        sys.executable,
        str(trainer_path),
        "--signature", f"{sig[0]},{sig[1]},{sig[2]}",
        "--quotient", candidate_group(c),
        "--quotient-json", str(desc_path),
        "--mode", "train",
        "--kernel-generator-mode", kernel_mode,
        "--kernel-generator-limit", str(kernel_limit),
        "--depth", str(args.depth),
        "--pairs", str(args.pairs),
        "--top-k-max", str(args.top_k_max),
        "--csv-top-k", str(args.csv_top_k),
        "--train-pool-size", str(train_pool),
        "--pool-sizes", pools,
        "--epochs", str(args.epochs),
        "--min-epochs", str(args.min_epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--candidate-chunk-size", str(args.candidate_chunk_size),
        "--pair-batch-size", str(args.pair_batch_size),
        "--engine", args.engine,
        "--target-vram-mb", str(args.target_vram_mb),
        "--max-word-ball", str(max_word),
        "--max-unique-word-ball", str(max_unique),
        "--outroot", args.trainer_outroot,
        "--label", label,
    ]
    if args.stream_huge_word_ball:
        cmd.append("--stream-huge-word-ball")
    if args.write_word_ball_summary:
        cmd.append("--write-word-ball-summary")
    return cmd, label, mode_tag


def default_pool_sizes(train_pool: int) -> str:
    vals = []
    for v in (64, 128, 256, 512, 1024):
        if v <= train_pool:
            vals.append(v)
    if train_pool not in vals:
        vals.append(train_pool)
    return ",".join(str(v) for v in vals)


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run_one(cmd: Sequence[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> Tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("# COMMAND\n")
        f.write(shell_join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(list(cmd), stdout=f, stderr=subprocess.STDOUT, env=env)
    return int(proc.returncode), time.perf_counter() - t0


def write_report(run_root: Path, rows: List[Dict[str, Any]], args: argparse.Namespace, candidates_json: Path) -> None:
    lines: List[str] = []
    lines.append(f"# Triangle Zoo Batch Trainer {VERSION} Report")
    lines.append("")
    lines.append(f"Created: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Manifest-driven overnight trainer for finite quotient surfaces of hyperbolic triangle groups.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- candidates_json: `{candidates_json}`")
    lines.append(f"- trainer_script: `{args.trainer_script}`")
    lines.append(f"- categories: `{args.categories}`")
    lines.append("")
    lines.append("## Training policy")
    lines.append("")
    lines.append(f"- COMPLETE_BALL_READY -> `all`, train_pool={args.complete_train_pool_size}")
    lines.append(f"- SELECTED_ATLAS_READY -> `shortest{args.selected_kernel_limit}`, train_pool={args.selected_train_pool_size or args.selected_kernel_limit}")
    lines.append(f"- pairs={args.pairs}, depth={args.depth}, epochs={args.epochs}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if rows:
        cols = ["index", "status", "returncode", "category", "signature", "group", "order", "genus", "mode_tag", "seconds", "label", "log"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    else:
        lines.append("No candidates selected.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("A successful batch row means the trainer subprocess returned zero. Inspect the individual trainer reports for recall metrics.")
    (run_root / "report").mkdir(parents=True, exist_ok=True)
    (run_root / "report" / "triangle_zoo_batch_training_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Batch train triangle-quotient zoo candidates from searcher manifest.")
    ap.add_argument("--candidates-json", type=str, default="", help="Path to triangle_quotient_candidates.json. If omitted, uses latest search run.")
    ap.add_argument("--search-outroot", type=str, default="triangle_quotient_search_runs")
    ap.add_argument("--trainer-script", type=str, default="FuchsianTriangleQuotientTrainer_v1_2.py")
    ap.add_argument("--outroot", type=str, default="triangle_zoo_batch_training_runs")
    ap.add_argument("--trainer-outroot", type=str, default="triangle_zoo_training_runs")
    ap.add_argument("--label", type=str, default="triangle_zoo_batch")
    ap.add_argument("--label-prefix", type=str, default="zoo_")

    ap.add_argument("--categories", type=str, default=f"{COMPLETE},{SELECTED}")
    ap.add_argument("--complete-only", action="store_true")
    ap.add_argument("--selected-only", action="store_true")
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--max-order", type=int, default=0)
    ap.add_argument("--include-regex", type=str, default="")
    ap.add_argument("--exclude-regex", type=str, default="")

    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=9000)
    ap.add_argument("--top-k-max", type=int, default=100)
    ap.add_argument("--csv-top-k", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--candidate-chunk-size", type=int, default=8192)
    ap.add_argument("--pair-batch-size", type=int, default=0)
    ap.add_argument("--engine", type=str, default="auto")
    ap.add_argument("--target-vram-mb", type=float, default=8192.0)

    ap.add_argument("--complete-kernel-limit", type=int, default=256, help="Passed through but ignored when mode=all by trainer.")
    ap.add_argument("--complete-train-pool-size", type=int, default=256)
    ap.add_argument("--complete-pool-sizes", type=str, default="")
    ap.add_argument("--complete-max-word-ball", type=int, default=2000000)
    ap.add_argument("--complete-max-unique-word-ball", type=int, default=1500000)

    ap.add_argument("--selected-kernel-limit", type=int, default=512)
    ap.add_argument("--selected-train-pool-size", type=int, default=0, help="Default: same as selected-kernel-limit")
    ap.add_argument("--selected-pool-sizes", type=str, default="")
    ap.add_argument("--selected-max-word-ball", type=int, default=2000000)
    ap.add_argument("--selected-max-unique-word-ball", type=int, default=1500000)

    ap.add_argument("--stream-huge-word-ball", action="store_true")
    ap.add_argument("--write-word-ball-summary", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--make-shell-script-only", action="store_true")
    args = ap.parse_args(argv)
    if args.stop_on_error:
        args.continue_on_error = False

    candidates_json = Path(args.candidates_json) if args.candidates_json else find_latest_candidates(Path(args.search_outroot))
    trainer_path = Path(args.trainer_script)
    if not trainer_path.exists():
        # If user runs from project dir but script was not copied yet, fail clearly.
        raise RuntimeError(f"Trainer script not found: {trainer_path}. Copy FuchsianTriangleQuotientTrainer_v1_2.py into this directory.")

    all_cands = read_json(candidates_json)
    if not isinstance(all_cands, list):
        raise RuntimeError("candidates JSON must contain a list")
    cands = filter_candidates(all_cands, args)

    run_root = Path(args.outroot) / f"run_{now_stamp()}_{slugify(args.label)}"
    desc_dir = run_root / "candidate_descriptors"
    log_dir = run_root / "logs"
    cmd_dir = run_root / "commands"
    for d in (desc_dir, log_dir, cmd_dir, run_root / "report", run_root / "tables"):
        d.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    shell_lines = ["#!/usr/bin/env bash", "set -u", "mkdir -p " + shlex.quote(str(log_dir))]

    print(f"FuchsianTriangleZooBatchTrainer_v1_0.py {VERSION}")
    print(f"candidates_json={candidates_json}")
    print(f"trainer_script={trainer_path}")
    print(f"run_root={run_root}")
    print(f"selected_candidates={len(cands)}")
    print("-" * 78, flush=True)

    for idx, c in enumerate(cands):
        sig = parse_sig(c)
        group = candidate_group(c)
        cat = candidate_category(c)
        desc_path = desc_dir / f"cand_{idx:04d}_{slugify(str(sig))}_{slugify(group)}.json"
        write_json(desc_path, c)
        cmd, label, mode_tag = command_for_candidate(c, desc_path, trainer_path, args)
        log_path = log_dir / f"cand_{idx:04d}_{label}.log"
        cmd_path = cmd_dir / f"cand_{idx:04d}_{label}.sh"
        cmd_path.write_text("#!/usr/bin/env bash\nset -e\n" + shell_join(cmd) + "\n", encoding="utf-8")
        try:
            os.chmod(cmd_path, 0o755)
        except Exception:
            pass
        shell_lines.append(f"echo '[batch] {idx}: {label}'")
        shell_lines.append(shell_join(cmd) + " > " + shlex.quote(str(log_path)) + " 2>&1")
        row = {
            "index": idx,
            "status": "DRY_RUN" if (args.dry_run or args.make_shell_script_only) else "PENDING",
            "returncode": "",
            "category": cat,
            "signature": f"({sig[0]},{sig[1]},{sig[2]})",
            "group": group,
            "order": c.get("order", ""),
            "genus": c.get("genus", ""),
            "mode_tag": mode_tag,
            "seconds": "",
            "label": label,
            "log": str(log_path),
            "command": shell_join(cmd),
        }
        if args.dry_run or args.make_shell_script_only:
            print(f"[dry-run] {idx:04d} {cat} {sig}->{group} mode={mode_tag}")
        else:
            print(f"[run] {idx:04d} {cat} {sig}->{group} mode={mode_tag}", flush=True)
            rc, sec = run_one(cmd, log_path)
            row["returncode"] = rc
            row["seconds"] = round(sec, 3)
            row["status"] = "OK" if rc == 0 else "FAIL"
            print(f"[done] {idx:04d} status={row['status']} rc={rc} seconds={sec:.1f} log={log_path}", flush=True)
            if rc != 0 and not args.continue_on_error:
                rows.append(row)
                break
        rows.append(row)
        write_csv(run_root / "tables" / "triangle_zoo_batch_training_rows.csv", rows)
        write_report(run_root, rows, args, candidates_json)

    shell_path = run_root / "run_batch_commands.sh"
    shell_path.write_text("\n".join(shell_lines) + "\n", encoding="utf-8")
    try:
        os.chmod(shell_path, 0o755)
    except Exception:
        pass
    write_csv(run_root / "tables" / "triangle_zoo_batch_training_rows.csv", rows)
    write_json(run_root / "manifest" / "selected_candidates.json", cands)
    write_report(run_root, rows, args, candidates_json)
    print("-" * 78)
    print(f"[batch-done] rows={len(rows)} run_root={run_root}")
    print(f"[shell-script] {shell_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
