#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from spectral_reliability.threads import NUM_THREADS, bootstrap_threads

# Set native thread counts before importing numerical libraries.
bootstrap_threads(sys.argv[1:])

from spectral_reliability.config import (  # noqa: E402
    expand_grid,
    load_run_config,
)
from spectral_reliability.data.dataset import (  # noqa: E402
    generate_dataset,
    load_data_config,
)
from spectral_reliability.pretraining import (  # noqa: E402
    pretraining_arrays,
    pretraining_settings,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate datasets and enabled reconstruction-pretraining data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="one run YAML")
    group.add_argument("--grid", type=Path, help="grid YAML")
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
    datasets, pretraining_data = {}, {}
    for run in runs:
        data_config = load_data_config(run)
        key = (run["system"], run["seed"])
        datasets.setdefault(key, data_config)
        if (
            run["model"]["name"] == "koopman_autoencoder"
            and run["pretrained"]["enabled"]
        ):
            settings = pretraining_settings(run)
            if settings["initial_conditions"] == "dataset":
                datasets.setdefault(
                    (settings["system"], settings["seed"]), data_config
                )
            else:
                pretraining_key = (
                    settings["system"],
                    settings["seed"],
                    settings["data_dir"],
                    settings["initial_conditions"],
                    settings.get("initial_condition_min"),
                    settings.get("initial_condition_max"),
                    settings.get("initial_perturbation_min"),
                    settings.get("initial_perturbation_max"),
                )
                pretraining_data.setdefault(pretraining_key, settings)
    for (system, seed), data_config in datasets.items():
        dataset = generate_dataset(system, seed, data_config)
        print(
            f"{system} seed{seed}: train={dataset.train.shape}, "
            f"valid={dataset.valid.shape}, test={dataset.test.shape}",
            flush=True,
        )
    for settings in pretraining_data.values():
        train, valid = pretraining_arrays(settings, build=True)
        print(
            f"{settings['system']} pretraining seed{settings['seed']}: "
            f"train={train.shape}, valid={valid.shape}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
