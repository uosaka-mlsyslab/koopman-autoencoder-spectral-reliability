#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from spectral_reliability.threads import NUM_THREADS, bootstrap_threads

# Set native thread counts before importing numerical libraries.
num_threads = bootstrap_threads(sys.argv[1:])

import yaml  # noqa: E402

from spectral_reliability.config import (  # noqa: E402
    completed,
    configure_torch,
    expand_grid,
    load_run_config,
    run_directory,
)
from spectral_reliability.config_shared import (  # noqa: E402
    REPOSITORY_ROOT,
    atomic_write,
)
from spectral_reliability.training import train_run  # noqa: E402

configure_torch(num_threads)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def train_grid(
    grid: Path,
    *,
    max_workers: int,
    num_threads: int,
    device: str,
) -> int:
    runs = expand_grid(grid)
    environment = dict(os.environ)

    def train_config(config: dict) -> int:
        directory = run_directory(config)
        if completed(config):
            print(f"skip {directory}: config SHA-256 matches", flush=True)
            return 0
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="spectral_reliability_run_"
        ) as scratch:
            path = Path(scratch) / "run.yaml"
            atomic_write(path, yaml.safe_dump(config, sort_keys=False))
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "train.py"),
                "--config",
                str(path),
                "--device",
                device,
                "--num-threads",
                str(num_threads),
            ]
            with (directory / "run.log").open("a") as log:
                returncode = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
        print(
            f"{'done' if returncode == 0 else 'FAILED'} {directory} "
            f"(exit {returncode})",
            flush=True,
        )
        return returncode

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_config, config) for config in runs]
        failed = sum(future.result() != 0 for future in as_completed(futures))
    print(f"{len(runs)} runs, {failed} failed", flush=True)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and save one run or a grid of runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="one run YAML")
    group.add_argument("--grid", type=Path, help="grid YAML")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="training device",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=NUM_THREADS,
        help="native threads per run, including grid subprocesses",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=3,
        help="maximum parallel runs with --grid only",
    )
    args = parser.parse_args()
    if args.config is not None:
        train_run(load_run_config(args.config), device=args.device)
        return 0
    return train_grid(
        args.grid,
        max_workers=args.workers,
        num_threads=args.num_threads,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
