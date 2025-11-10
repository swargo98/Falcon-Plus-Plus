#!/usr/bin/env python3
"""
run_experiments.py
Automates experiments across combinations of pipelining (p), parallelism (pp), and concurrency (cc)
for three dataset sizes. It programmatically updates config_sender.py between runs and executes
the appropriate sender script sequentially.

Layout:
- Logs are written to logs/<model_version>/stdout.txt and stderr.txt
- The script backs up config_sender.py to config_sender.py.bak and restores it when done.

Usage:
  python run_experiments.py
  python run_experiments.py --dry-run         # print the plan without running
  python run_experiments.py --datasets 1MB    # limit to one or more dataset tags
  python run_experiments.py --skip 2 3        # skip specific experiment IDs
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config_sender.py"
CONFIG_BACKUP = ROOT / "config_sender.py.bak"
LOGS_DIR = ROOT / "logs"


# ---------------------------- Experiment matrix ---------------------------- #

# Dataset mapping
DATASETS = {
    "1MB": "/mnt/nvme0n1/abl-1MB/",
    "32MB": "/mnt/nvme0n1/abl-32MB/",
    # "1024MB": "/mnt/nvme0n1/abl-1024MB/",
    "2048MB": "/mnt/nvme0n1/abl-2048MB/",
}

# Each experiment: id, p, pp, cc_values, script
# Based on the provided table
EXPERIMENTS = [
    # 1: Only pipelining
    {"id": 1, "p": 1, "pp": 0, "cc_values": [1], "script": "sender_heuristic.py"},
    # 2: Only concurrency
    {"id": 2, "p": 0, "pp": 0, "cc_values": [1, 2, 3, 5, 8], "script": "sender_wo_heuristic.py"},
    # 3: Only parallelism
    {"id": 3, "p": 0, "pp": 1, "cc_values": [1], "script": "sender_pp-cc.py"},
    # 4: Parallelism + Concurrency
    {"id": 4, "p": 0, "pp": 1, "cc_values": [1, 2, 3, 5, 8], "script": "sender_pp-cc.py"},
    # 5: Parallelism + Pipelining
    {"id": 5, "p": 1, "pp": 1, "cc_values": [1], "script": "sender_heuristic.py"},
    # 6: Concurrency + Pipelining
    {"id": 6, "p": 1, "pp": 0, "cc_values": [1, 2, 3, 5, 8], "script": "sender_heuristic.py"},
    # 7: All (p + pp + cc)
    {"id": 7, "p": 1, "pp": 1, "cc_values": [1, 2, 3, 5, 8], "script": "sender_heuristic.py"},
]


# ------------------------------ Helper types ------------------------------ #

@dataclass
class RunSpec:
    exp_id: int
    dataset_tag: str
    dataset_dir: str
    p: int
    pp: int
    cc: int
    script: str

    @property
    def model_version(self) -> str:
        return f"p_{self.p}_pp_{self.pp}_cc_{self.cc}_{self.dataset_tag}"

    @property
    def log_dir(self) -> Path:
        return LOGS_DIR / self.model_version


# ------------------------------ Config helpers ----------------------------- #

def load_base_config() -> Dict:
    """
    Execute config_sender.py to get the 'configurations' dict as our base template.
    """
    ns = runpy.run_path(str(CONFIG_FILE))
    if "configurations" not in ns or not isinstance(ns["configurations"], dict):
        raise RuntimeError("config_sender.py must define a 'configurations' dict")
    return ns["configurations"]


def build_config(base: Dict, *, data_dir: str, p: int, pp: int, cc: int, dataset_tag: str) -> Dict:
    """
    Create a new configurations dict for a specific run, starting from the base.
    Only the required fields are modified.
    """
    cfg = dict(base)  # shallow copy is fine since we rewrite nested below

    # Required fields per instructions
    cfg["data_dir"] = data_dir
    cfg["max_file_chunk_in_MB"] = 256 if pp == 1 else 4096
    # fixed_probing.thread = cc if cc>1 else 1, bsize constant 10
    fixed = dict(cfg.get("fixed_probing", {}))
    fixed["bsize"] = 10
    fixed["thread"] = cc if cc > 1 else 1
    cfg["fixed_probing"] = fixed

    cfg["model_version"] = f"p_{p}_pp_{pp}_cc_{cc}_{dataset_tag}"

    return cfg


def write_config(cfg: Dict) -> None:
    """
    Overwrite config_sender.py with the given configurations dict and tail that preserves
    setdefault logic present in the original file.
    """
    # The original file had some derived defaults based on model_version.
    content = (
        "configurations = "
        + repr(cfg)
        + "\n\n"
        + "mv = configurations['model_version']\n"
        + "configurations.setdefault('inference_value_model', f\"best_models/{mv}_offline_value.pth\")\n"
        + "configurations.setdefault('inference_policy_model', f\"best_models/{mv}_offline_policy.pth\")\n"
        + "configurations.setdefault('max_episodes', 6 if configurations.get('mode') == 'random' else 20000)\n"
        + "configurations.setdefault('multiplier', 20 if configurations.get('mode') == 'random' else 1)\n"
    )

    CONFIG_FILE.write_text(content)


# ------------------------------ Run helpers -------------------------------- #

def ensure_backup():
    if not CONFIG_BACKUP.exists():
        shutil.copy2(CONFIG_FILE, CONFIG_BACKUP)


def restore_backup():
    if CONFIG_BACKUP.exists():
        shutil.copy2(CONFIG_BACKUP, CONFIG_FILE)


def run_process(script: str, log_dir: Path, dry_run: bool = False) -> int:
    """
    Run 'python <script>' with stdout and stderr captured to files in log_dir.
    Return the process return code.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, script]

    if dry_run:
        return 0

    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err)
        return proc.returncode


def iter_runs(selected_datasets: List[str], skip_ids: List[int]) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for ds_tag in selected_datasets:
        ds_dir = DATASETS[ds_tag]
        for ex in EXPERIMENTS:
            if ex["id"] in skip_ids:
                continue
            for cc in ex["cc_values"]:
                runs.append(
                    RunSpec(
                        exp_id=ex["id"],
                        dataset_tag=ds_tag,
                        dataset_dir=ds_dir,
                        p=ex["p"],
                        pp=ex["pp"],
                        cc=cc,
                        script=ex["script"],
                    )
                )
    return runs


def prepare_and_run(runs: List[RunSpec], dry_run: bool = False) -> Tuple[int, int]:
    base_cfg = load_base_config()

    success = 0
    failed = 0

    ensure_backup()

    try:
        for r in runs:
            # Build config for this run
            cfg = build_config(
                base_cfg,
                data_dir=r.dataset_dir,
                p=r.p,
                pp=r.pp,
                cc=r.cc,
                dataset_tag=r.dataset_tag,
            )
            write_config(cfg)

            # Progress line
            print(f"[Exp {r.exp_id} | Dataset={r.dataset_tag} | p={r.p} pp={r.pp} cc={r.cc}] \u2192 {r.script} started...")

            # Run the sender script
            rc = run_process(r.script, r.log_dir, dry_run=dry_run)

            if rc == 0:
                success += 1
                print(f"[Exp {r.exp_id} | Dataset={r.dataset_tag} | p={r.p} pp={r.pp} cc={r.cc}] finished OK. Logs: {r.log_dir}")
            else:
                failed += 1
                print(f"[Exp {r.exp_id} | Dataset={r.dataset_tag} | p={r.p} pp={r.pp} cc={r.cc}] FAILED with code {rc}. Logs: {r.log_dir}")

    finally:
        # Always restore original config
        restore_backup()

    return success, failed


# ---------------------------------- Main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Automate p, pp, cc experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without executing scripts")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Subset of datasets to run by tag",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        type=int,
        default=[],
        help="Experiment IDs to skip (e.g., --skip 2 5)",
    )
    args = parser.parse_args()

    runs = iter_runs(args.datasets, args.skip)

    total = len(runs)
    print(f"Planned runs: {total}")
    for r in runs:
        print(f"  - Exp {r.exp_id} | {r.dataset_tag} | p={r.p} pp={r.pp} cc={r.cc} | script={r.script} | model={r.model_version}")

    success, failed = prepare_and_run(runs, dry_run=args.dry_run)

    print(f"\nSummary: {success} succeeded, {failed} failed, out of {total} total runs.")


if __name__ == "__main__":
    sys.exit(main())
