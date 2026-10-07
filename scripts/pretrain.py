#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from spectral_reliability.threads import NUM_THREADS, bootstrap_threads

# Set native thread counts before importing numerical libraries.
num_threads = bootstrap_threads(sys.argv[1:])

from spectral_reliability.config import (  # noqa: E402
    configure_torch,
    effective_latent_dim,
    expand_grid,
    load_run_config,
    resolved_precision,
)
from spectral_reliability.pretraining import (  # noqa: E402
    pretrain_autoencoder,
    pretraining_config_sha256,
    pretraining_settings,
)

configure_torch(num_threads)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain the enabled Koopman autoencoders for one run or a grid."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="one run YAML")
    group.add_argument("--grid", type=Path, help="grid YAML")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="pretraining device",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=NUM_THREADS,
        help="native threads",
    )
    args = parser.parse_args()
    runs = (
        [load_run_config(args.config)]
        if args.config is not None
        else expand_grid(args.grid)
    )
    checkpoints = {}
    for run in runs:
        if (
            run["model"]["name"] != "koopman_autoencoder"
            or not run["pretrained"]["enabled"]
        ):
            continue
        settings = pretraining_settings(run)
        latent_dim, precision = (
            effective_latent_dim(run),
            resolved_precision(run),
        )
        checkpoints.setdefault(
            (
                run["system"],
                latent_dim,
                precision,
                pretraining_config_sha256(settings),
            ),
            settings,
        )
    for (system, latent_dim, precision, _), settings in checkpoints.items():
        path = pretrain_autoencoder(
            system,
            latent_dim,
            precision=precision,
            device=args.device,
            settings=settings,
        )
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
