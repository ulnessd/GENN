#!/usr/bin/env python3
"""FuchsianZooBuilder_v1_3.py

Orchestrates family-level Fuchsian/Riemann-surface zoo generation, then calls
FuchsianMasterDatasetBuilder_v3_1.py to assemble the result.

Design principle:
  * ZooBuilder controls what gets attempted.
  * MasterBuilder v3 classifies and preserves what actually came back.

Profiles:
  smoke      : fast all-family sanity run.
  standard   : moderate, train/smoke-ready zoo foundation.
  catalog500 : broad geometry catalog intended to exceed roughly 500 records;
               default is geometry-only to avoid runaway word balls.
  training   : conservative production-training subset; use with care.

Version 1.2 patch:
  * Caps regular compact-polygon genus at 13 by default, because the current
    compact tester supports at most 26 side-pairing generators.  Regular
    genus g uses 2g side-pairing generators, so g=14 and above fail in the
    current engine.
  * Expands noncompact/catalog-only families in catalog500 enough to retain a
    500+ record target without relying on unsupported compact genera.
  * Writes a lightweight zoo_summary.csv for quick diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROGRAM = "FuchsianZooBuilder_v1_3.py"
VERSION = "1.3"

FAMILY_ORDER = ["elementary", "compact", "modular", "hecke", "hurwitz", "schottky"]

SCRIPT_DEFAULTS = {
    "elementary": "FuchsianElementaryTester_v1_1.py",
    "compact": "FuchsianCompactPolygonTester_v1_2.py",
    "modular": "FuchsianModularCongruenceTester_v1_3.py",
    "hecke": "FuchsianHeckeTester_v1_2.py",
    "hurwitz": "FuchsianHurwitzTester_v1_6.py",
    "schottky": "FuchsianSchottkyTester_v1_1.py",
    "master": "FuchsianMasterDatasetBuilder_v3_1.py",
}


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str), encoding="utf-8")


def which_script(name: str, explicit: str = "") -> Path:
    candidate = Path(explicit or SCRIPT_DEFAULTS[name])
    if candidate.exists():
        return candidate.resolve()
    here = Path(__file__).resolve().parent / candidate.name
    if here.exists():
        return here.resolve()
    cwd = Path.cwd() / candidate.name
    if cwd.exists():
        return cwd.resolve()
    # Return cwd candidate even if missing so dry-run still displays useful command.
    return cwd.resolve()


def int_range(a: int, b: int) -> List[str]:
    return [str(i) for i in range(a, b + 1)]


def compact_genera_range(max_genus: int) -> List[str]:
    """Regular compact genus g currently uses 2g side-pairing generators.

    The compact polygon tester supports at most 26 side-pairing generators,
    hence g <= 13.  Keep this cap in ZooBuilder so long overnight runs do not
    repeatedly attempt known-impossible compact records.
    """
    HARD_CAP = 13
    return int_range(2, min(max_genus, HARD_CAP))


def common_ginn_flags(mode: str, args: argparse.Namespace) -> List[str]:
    flags: List[str] = []
    if mode == "smoke":
        flags += ["--ginn-smoke", "--ginn-pairs", str(args.smoke_pairs), "--ginn-depth", str(args.smoke_depth), "--ginn-max-word-ball", str(args.smoke_word_ball_cap)]
    elif mode == "train":
        flags += ["--run-ginn", "--ginn-pairs", str(args.train_pairs), "--ginn-depth", str(args.train_depth), "--ginn-max-word-ball", str(args.train_word_ball_cap), "--ginn-epochs", str(args.train_epochs), "--ginn-batch-size", str(args.train_batch_size), "--ginn-device", args.device]
    elif mode == "label":
        flags += ["--ginn-smoke", "--ginn-pairs", str(args.label_pairs), "--ginn-depth", str(args.label_depth), "--ginn-max-word-ball", str(args.catalog_word_ball_cap)]
    return flags


def profile_mode(profile: str, args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if profile == "smoke":
        return "smoke"
    if profile == "training":
        return "train"
    # standard uses smoke/preflight by default, catalog500 is geometry-only.
    if profile == "standard":
        return "smoke"
    return "geometry"


def build_family_commands(args: argparse.Namespace, run_root: Path) -> List[Tuple[str, List[str], Path]]:
    profile = args.profile
    mode = profile_mode(profile, args)
    selected = FAMILY_ORDER
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [f for f in selected if f in wanted]
    if args.skip:
        skip = {x.strip() for x in args.skip.split(",") if x.strip()}
        selected = [f for f in selected if f not in skip]

    scripts = {name: which_script(name, getattr(args, f"{name}_script", "")) for name in SCRIPT_DEFAULTS}
    family_root = run_root / "family_runs"
    cmds: List[Tuple[str, List[str], Path]] = []

    def cmd_base(fam: str) -> List[str]:
        return [sys.executable, str(scripts[fam])]

    # Profile parameter sets.  Keep principal Gamma(N) conservative.
    if profile == "smoke":
        compact_genera = compact_genera_range(4)
        gamma_N = int_range(3, 6)
        gamma1_N = int_range(4, 8)
        gamma0_N = int_range(4, 12)
        hecke_q = ["3", "5", "7"]
        hurwitz_q = ["7", "13"]
        schottky_rank = ["2", "3"]
        schottky_gaps = ["0.18"]
        schottky_rots = ["0"]
        elementary_extra = []
    elif profile == "standard":
        compact_genera = compact_genera_range(13)
        gamma_N = int_range(3, 8)
        gamma1_N = int_range(4, 20)
        gamma0_N = int_range(4, 50)
        hecke_q = int_range(3, 30)
        hurwitz_q = ["7", "13", "29", "41", "43"]
        schottky_rank = int_range(2, 10)
        schottky_gaps = ["0.10", "0.18", "0.26"]
        schottky_rots = ["0"]
        elementary_extra = []
    elif profile == "catalog500":
        compact_genera = compact_genera_range(13)
        gamma_N = int_range(3, 8)
        gamma1_N = int_range(4, 60)
        gamma0_N = int_range(4, 140)
        hecke_q = int_range(3, 90)
        hurwitz_q = ["7", "13", "29", "41", "43"]
        schottky_rank = int_range(2, 30)
        schottky_gaps = ["0.06", "0.10", "0.14", "0.18", "0.22", "0.26", "0.30"]
        schottky_rots = ["0"]
        elementary_extra = []
    elif profile == "training":
        compact_genera = compact_genera_range(13)
        gamma_N = int_range(3, 8)
        gamma1_N = int_range(4, 16)
        gamma0_N = int_range(4, 30)
        hecke_q = int_range(3, 20)
        hurwitz_q = ["7"]  # production training keeps Hurwitz to PSL(2,7) by default; q=13 needs special large-word-ball strategy.
        schottky_rank = int_range(2, 8)
        schottky_gaps = ["0.10", "0.18", "0.26"]
        schottky_rots = ["0"]
        elementary_extra = []
    else:
        raise ValueError(f"Unknown profile {profile!r}")

    # Allow caps/ranges to be overridden from the command line.
    if args.compact_max_genus is not None:
        if args.compact_max_genus > 13:
            print(f"[zoo-plan] compact-max-genus={args.compact_max_genus} requested, but current compact engine caps regular compact genus at 13; using 13.")
        compact_genera = compact_genera_range(args.compact_max_genus)
    if args.gamma_max_N is not None:
        gamma_N = int_range(3, args.gamma_max_N)
    if args.gamma1_max_N is not None:
        gamma1_N = int_range(4, args.gamma1_max_N)
    if args.gamma0_max_N is not None:
        gamma0_N = int_range(4, args.gamma0_max_N)
    if args.hecke_max_q is not None:
        hecke_q = int_range(3, args.hecke_max_q)
    if args.schottky_max_rank is not None:
        schottky_rank = int_range(2, args.schottky_max_rank)

    if "elementary" in selected:
        c = cmd_base("elementary") + [
            "--families", "gamma2,commutator,cyclic",
            "--include-elliptic",
            "--outroot", str(family_root / "elementary_tester_runs"),
            "--label", f"zoo_{profile}_elementary",
        ] + common_ginn_flags(mode if mode != "label" else "smoke", args) + elementary_extra
        cmds.append(("elementary", c, family_root / "elementary_tester_runs"))

    if "compact" in selected:
        c = cmd_base("compact") + [
            "--genera", *compact_genera,
            "--include-hurwitz",
            "--outroot", str(family_root / "compact_polygon_tester_runs"),
            "--label", f"zoo_{profile}_compact",
            "--max-cosets", str(args.max_cosets),
            "--max-generators", "0",
        ] + common_ginn_flags(mode, args)
        cmds.append(("compact", c, family_root / "compact_polygon_tester_runs"))

    if "modular" in selected:
        c = cmd_base("modular") + [
            "--principal-N", *gamma_N,
            "--gamma1-N", *gamma1_N,
            "--gamma0-N", *gamma0_N,
            "--outroot", str(family_root / "modular_congruence_tester_runs"),
            "--label", f"zoo_{profile}_modular",
            "--max-cosets", str(args.max_cosets),
            "--max-generators", "0",
        ] + common_ginn_flags(mode, args)
        if args.include_gamma0_orbifolds:
            c.append("--include-gamma0-orbifolds")
        cmds.append(("modular", c, family_root / "modular_congruence_tester_runs"))

    if "hecke" in selected:
        c = cmd_base("hecke") + [
            "--q", *hecke_q,
            "--families", "ab,d",
            "--outroot", str(family_root / "hecke_tester_runs"),
            "--label", f"zoo_{profile}_hecke",
            "--max-cosets", str(args.max_cosets),
            "--max-generators", "0",
        ] + common_ginn_flags(mode, args)
        cmds.append(("hecke", c, family_root / "hecke_tester_runs"))

    if "hurwitz" in selected:
        # Hurwitz v1.6 has a different CLI.  Use depth 1 for catalog500 unless the user overrides mode.
        h_mode = mode
        h_depth = args.smoke_depth if h_mode == "smoke" else args.train_depth
        if profile == "catalog500" and args.mode == "auto":
            h_mode = "geometry"
        c = cmd_base("hurwitz") + [
            "--q", *hurwitz_q,
            "--max-triples", str(args.hurwitz_max_triples),
            "--triple-equivalence", "pgl",
            "--build-kernel",
            "--out", str(family_root / "hurwitz_tester_runs"),
            "--label", f"zoo_{profile}_hurwitz",
        ]
        if h_mode == "smoke":
            c += ["--ginn-smoke", "--ginn-pairs", str(max(20, args.smoke_pairs // 2)), "--ginn-depth", str(min(1, h_depth) if profile != "training" else h_depth), "--ginn-max-word-ball", str(args.catalog_word_ball_cap)]
        elif h_mode == "train":
            c += ["--run-ginn", "--ginn-pairs", str(args.train_pairs), "--ginn-depth", str(args.train_depth), "--ginn-max-word-ball", str(args.train_word_ball_cap), "--ginn-epochs", str(args.train_epochs), "--ginn-batch-size", str(args.train_batch_size), "--ginn-device", args.device]
            # v1.6 supports smaller chunked settings.
            c += ["--ginn-pair-hidden", str(args.hurwitz_pair_hidden), "--ginn-score-hidden", str(args.hurwitz_score_hidden), "--ginn-candidate-chunk-size", str(args.hurwitz_candidate_chunk_size), "--ginn-auto-chunk-threshold-mb", str(args.hurwitz_auto_chunk_mb)]
        cmds.append(("hurwitz", c, family_root / "hurwitz_tester_runs"))

    if "schottky" in selected:
        c = cmd_base("schottky") + [
            "--rank", *schottky_rank,
            "--gaps", *schottky_gaps,
            "--rotations", *schottky_rots,
            "--outroot", str(family_root / "schottky_tester_runs"),
            "--label", f"zoo_{profile}_schottky",
        ] + common_ginn_flags(mode, args)
        cmds.append(("schottky", c, family_root / "schottky_tester_runs"))

    return cmds


def run_streaming(name: str, cmd: List[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("-" * 78)
    print(f"[run] {name}: {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        log.write(f"\nRETURN_CODE: {rc}\n")
    print(f"[done-family] {name} rc={rc} log={log_path}", flush=True)
    return rc


def build_zoo(args: argparse.Namespace) -> int:
    stamp = now_stamp()
    run_root = Path(args.zoo_root) / f"run_{stamp}_{args.profile}{('_' + args.label) if args.label else ''}"
    run_root.mkdir(parents=True, exist_ok=True)
    logs = run_root / "logs"
    commands = build_family_commands(args, run_root)
    manifest: Dict[str, Any] = {
        "program": PROGRAM,
        "version": VERSION,
        "created": stamp,
        "profile": args.profile,
        "args": vars(args),
        "run_root": str(run_root),
        "commands": [{"family": name, "cmd": cmd, "input_root": str(root)} for name, cmd, root in commands],
    }
    write_json(run_root / "zoo_plan.json", manifest)

    print(f"{PROGRAM} v{VERSION}")
    print(f"run_root={run_root}")
    print(f"profile={args.profile} mode={profile_mode(args.profile, args)} families={[name for name, _, _ in commands]}")
    if args.dry_run:
        print("[dry-run] commands written to zoo_plan.json; nothing executed")
        for name, cmd, _ in commands:
            print(f"[{name}] " + " ".join(shlex.quote(x) for x in cmd))
        return 0

    cwd = Path.cwd()
    results: List[Dict[str, Any]] = []
    failed = False
    for name, cmd, root in commands:
        missing = [cmd[1]] if len(cmd) > 1 and not Path(cmd[1]).exists() else []
        if missing:
            rc = 127
            msg = f"script not found: {missing[0]}"
            print(f"[fail] {name}: {msg}")
            (logs / f"{name}.log").write_text(msg + "\n", encoding="utf-8")
        else:
            rc = run_streaming(name, cmd, logs / f"{name}.log", cwd)
        results.append({"family": name, "returncode": rc, "input_root": str(root), "log": str(logs / f"{name}.log")})
        if rc != 0:
            failed = True
            if not args.continue_on_error:
                print("[stop] family failed; use --continue-on-error to keep going")
                break

    # Master build over the family run roots that were attempted/succeeded enough to have directories.
    master_script = which_script("master", args.master_script)
    master_inputs = [str(root) for _, _, root in commands if root.exists()]
    master_cmd = [
        sys.executable, str(master_script),
        "--inputs", *master_inputs,
        "--outroot", str(run_root / "master_dataset_runs"),
        "--label", f"zoo_{args.profile}",
        "--training-word-ball-cap", str(args.train_word_ball_cap),
        "--catalog-word-ball-cap", str(args.catalog_word_ball_cap),
    ]
    if args.master_all_runs:
        master_cmd.append("--all-runs")
    if args.copy_light_artifacts:
        master_cmd.append("--copy-light-artifacts")
    if args.copy_heavy_artifacts:
        master_cmd.append("--copy-heavy-artifacts")
    master_rc = 0
    if args.skip_master:
        print("[skip] master builder not run")
    elif not master_inputs:
        print("[skip] no family run roots to feed to master builder")
        master_rc = 1
    elif not master_script.exists():
        print(f"[fail] master builder script not found: {master_script}")
        master_rc = 127
    else:
        master_rc = run_streaming("master_v3", master_cmd, logs / "master_v3.log", cwd)
    results.append({"family": "master_v3", "returncode": master_rc, "input_roots": master_inputs, "log": str(logs / "master_v3.log")})
    manifest["results"] = results
    manifest["completed"] = time.time()
    write_json(run_root / "zoo_manifest.json", manifest)

    # Lightweight CSV summary for quick diagnostics.
    summary_path = run_root / "zoo_summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("family,returncode,input_root_or_roots,log\n")
        for r in results:
            root_field = r.get("input_root", ";".join(r.get("input_roots", [])))
            f.write(f"{r.get('family','')},{r.get('returncode','')},{root_field},{r.get('log','')}\n")

    print("-" * 78)
    print(f"[zoo-done] run_root={run_root}")
    print(f"[zoo-done] zoo_summary={summary_path}")
    print(f"[zoo-done] family_failures={sum(1 for r in results if r['family'] != 'master_v3' and r['returncode'] != 0)} master_rc={master_rc}")
    return 0 if (not failed and master_rc == 0) else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Consolidated Fuchsian/Riemann-surface zoo builder")
    ap.add_argument("--profile", choices=["smoke", "standard", "catalog500", "training"], default="smoke")
    ap.add_argument("--mode", choices=["auto", "geometry", "smoke", "label", "train"], default="auto", help="Override profile default generation mode")
    ap.add_argument("--zoo-root", default="zoo_runs")
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    ap.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    ap.add_argument("--skip-master", action="store_true")
    ap.add_argument("--master-all-runs", action="store_true")
    ap.add_argument("--only", default="", help="Comma-list subset of families: elementary,compact,modular,hecke,hurwitz,schottky")
    ap.add_argument("--skip", default="", help="Comma-list families to skip")

    # Script overrides.
    ap.add_argument("--elementary-script", default="")
    ap.add_argument("--compact-script", default="")
    ap.add_argument("--modular-script", default="")
    ap.add_argument("--hecke-script", default="")
    ap.add_argument("--hurwitz-script", default="")
    ap.add_argument("--schottky-script", default="")
    ap.add_argument("--master-script", default="")

    # Range overrides.
    ap.add_argument("--compact-max-genus", type=int, default=None)
    ap.add_argument("--gamma-max-N", type=int, default=None)
    ap.add_argument("--gamma1-max-N", type=int, default=None)
    ap.add_argument("--gamma0-max-N", type=int, default=None)
    ap.add_argument("--hecke-max-q", type=int, default=None)
    ap.add_argument("--schottky-max-rank", type=int, default=None)
    ap.add_argument("--include-gamma0-orbifolds", action="store_true")
    ap.add_argument("--hurwitz-max-triples", type=int, default=3)

    # Shared budgets and training/smoke sizes.
    ap.add_argument("--max-cosets", type=int, default=20000)
    ap.add_argument("--smoke-pairs", type=int, default=100)
    ap.add_argument("--smoke-depth", type=int, default=2)
    ap.add_argument("--smoke-word-ball-cap", type=int, default=50000)
    ap.add_argument("--label-pairs", type=int, default=100)
    ap.add_argument("--label-depth", type=int, default=1)
    ap.add_argument("--train-pairs", type=int, default=9000)
    ap.add_argument("--train-depth", type=int, default=2)
    ap.add_argument("--train-word-ball-cap", type=int, default=50000)
    ap.add_argument("--catalog-word-ball-cap", type=int, default=1000000)
    ap.add_argument("--train-epochs", type=int, default=160)
    ap.add_argument("--train-batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")

    # Hurwitz large-word-ball safety knobs.
    ap.add_argument("--hurwitz-pair-hidden", type=int, default=64)
    ap.add_argument("--hurwitz-score-hidden", type=int, default=32)
    ap.add_argument("--hurwitz-candidate-chunk-size", type=int, default=128)
    ap.add_argument("--hurwitz-auto-chunk-mb", type=float, default=512.0)

    # Master artifact copy knobs.
    ap.add_argument("--copy-light-artifacts", action="store_true")
    ap.add_argument("--copy-heavy-artifacts", action="store_true")

    args = ap.parse_args(argv)
    return build_zoo(args)


if __name__ == "__main__":
    raise SystemExit(main())
