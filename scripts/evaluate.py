#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spectral_reliability.threads import NUM_THREADS, bootstrap_threads

num_threads = bootstrap_threads(sys.argv[1:])

# Set native thread counts before importing numerical libraries.
from spectral_reliability import attractors, pseudospectra  # noqa: E402
from spectral_reliability.config import configure_torch  # noqa: E402
from spectral_reliability.data.dataset import load_dataset  # noqa: E402
from spectral_reliability.evaluation import (  # noqa: E402
    TrainedModel,
    evaluate_runs,
    load_trained_model,
    read_run_records,
)
from spectral_reliability.figure_config import (  # noqa: E402
    attractor_arrays_path,
    load_figures_config,
    pseudospectra_arrays_path,
)
from spectral_reliability.results import (  # noqa: E402
    RunRecord,
    write_seed_metrics,
)

configure_torch(num_threads)


def _load_models(
    records: list[RunRecord],
    system: str,
    multiplier: int,
    seed: int,
    variants: list[str],
) -> dict[str, TrainedModel]:
    selected = {
        record.model: record
        for record in records
        if record.system == system
        and record.latent_multiplier == multiplier
        and record.seed == seed
        and record.model in variants
    }
    missing = set(variants) - set(selected)
    if missing:
        raise ValueError(
            f"{system} x{multiplier} seed{seed} missing completed runs: "
            f"{sorted(missing)}"
        )
    return {
        name: load_trained_model(selected[name].run_dir) for name in variants
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved runs and compute diagnostic figure arrays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="saved run directory",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("results/seed_metrics.csv"),
        help="destination for per-seed metrics",
    )
    parser.add_argument(
        "--arrays-dir",
        type=Path,
        default=Path("results/arrays"),
        help="destination for diagnostic arrays",
    )
    parser.add_argument(
        "--figures-config",
        type=Path,
        default=Path("configs/figures.yaml"),
        help="figure YAML",
    )
    parser.add_argument(
        "--figure",
        choices=("all", "scaling", "pseudospectra", "attractors"),
        default="all",
        help="diagnostics to compute",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="evaluation device",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=NUM_THREADS,
        help="native threads",
    )
    args = parser.parse_args()
    if args.figure in ("all", "scaling"):
        records = evaluate_runs(args.output_dir, device=args.device)
        if not records:
            print(
                f"No completed runs found in {args.output_dir}; "
                f"leaving {args.metrics_csv} untouched.",
                flush=True,
            )
            return 0
        metric_columns = tuple(
            dict.fromkeys(
                name for record in records for name in record.metrics
            )
        )
        count = write_seed_metrics(
            records,
            args.metrics_csv,
            metric_columns=metric_columns,
        )
        print(f"WROTE {args.metrics_csv}: {count} rows", flush=True)
    else:
        records = read_run_records(args.output_dir)
    if args.figure == "scaling":
        return 0

    config = load_figures_config(args.figures_config)
    pseudospectra_settings, attractor_settings = (
        config[key] for key in ("pseudospectra", "attractors")
    )
    targets = {}
    for system in config["systems"]:
        if args.figure in ("all", "attractors"):
            targets[
                (
                    system,
                    attractor_settings["latent_multiplier"],
                    attractor_settings["seed"],
                )
            ] = ["attractors"]
        if args.figure in ("all", "pseudospectra"):
            targets.setdefault(
                (
                    system,
                    pseudospectra_settings["latent_multiplier"],
                    pseudospectra_settings["seed"],
                ),
                [],
            ).append("pseudospectra")
    for (system, multiplier, seed), figure_types in targets.items():
        variants = list(
            dict.fromkeys(
                name
                for figure_type in figure_types
                for name in config[figure_type]["variants"]
            )
        )
        models = _load_models(records, system, multiplier, seed, variants)
        reference_model = next(iter(models.values()))
        dataset = load_dataset(system, seed, reference_model.data_config)
        if "attractors" in figure_types:
            path = attractor_arrays_path(
                args.arrays_dir,
                system,
                multiplier,
                seed,
            )
            attractors.save_arrays(
                attractors.compute_arrays(
                    models,
                    dataset,
                    settings=attractor_settings,
                    device=args.device,
                ),
                path,
            )
            print(f"WROTE {path}", flush=True)
        if "pseudospectra" in figure_types:
            path = pseudospectra_arrays_path(
                args.arrays_dir,
                system,
                multiplier,
                seed,
            )
            pseudospectra.save_arrays(
                pseudospectra.compute_arrays(
                    models,
                    dataset,
                    settings=pseudospectra_settings,
                    device=args.device,
                ),
                path,
            )
            print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
