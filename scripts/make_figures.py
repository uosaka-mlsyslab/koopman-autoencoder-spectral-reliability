#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spectral_reliability.threads import NUM_THREADS, bootstrap_threads

bootstrap_threads(sys.argv[1:])

# Set native thread counts before importing numerical libraries.
from spectral_reliability.figure_config import (  # noqa: E402
    attractor_arrays_path,
    load_figures_config,
    pseudospectra_arrays_path,
    resolve_scaling_grid,
)
from spectral_reliability.figures import (  # noqa: E402
    attractors,
    pseudospectra,
    scaling,
)

SCALING_MODELS_PDF = "scaling/{metric}.pdf"
SCALING_BASELINES_PDF = "scaling_baselines/{metric}.pdf"
PSEUDOSPECTRA_PDF = "pseudospectra/{system}_x{multiplier}_seed{seed}.pdf"
ATTRACTOR_PDF = "attractors/{system}_x{multiplier}_seed{seed}_{group_name}.pdf"


def _report_missing(path: Path, figure: str, args: argparse.Namespace) -> None:
    destination = (
        f"--metrics-csv {args.metrics_csv}"
        if figure == "scaling"
        else f"--arrays-dir {args.arrays_dir}"
    )
    print(
        f"SKIP missing {path}; run: python scripts/evaluate.py "
        f"--figure {figure} --figures-config {args.figures_config} "
        f"{destination}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw figures from saved metrics and diagnostic arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="figures to draw",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("results/seed_metrics.csv"),
        help="source of per-seed metrics",
    )
    parser.add_argument(
        "--arrays-dir",
        type=Path,
        default=Path("results/arrays"),
        help="source of diagnostic arrays",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="destination for figures",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=NUM_THREADS,
        help="native threads",
    )
    args = parser.parse_args()
    config = load_figures_config(args.figures_config)
    pseudospectra_settings = config["pseudospectra"]
    attractor_settings = config["attractors"]
    if args.figure in ("all", "scaling"):
        if args.figure == "all" and not args.metrics_csv.is_file():
            _report_missing(args.metrics_csv, "scaling", args)
        else:
            settings = config["scaling"]
            scaling_grid = resolve_scaling_grid(settings, config["systems"])
            metrics = scaling.load_scaling_metrics(
                args.metrics_csv,
                scaling_grid,
                settings,
            )
            path = args.figures_dir / SCALING_MODELS_PDF.format(
                metric=scaling_grid["metric_name"],
            )
            scaling.draw_models(metrics, path, scaling_grid, settings)
            print(f"WROTE {path}", flush=True)
            path = args.figures_dir / SCALING_BASELINES_PDF.format(
                metric=scaling_grid["metric_name"],
            )
            scaling.draw_baselines(metrics, path, scaling_grid, settings)
            print(f"WROTE {path}", flush=True)
    for system in config["systems"]:
        if args.figure in ("all", "pseudospectra"):
            multiplier = pseudospectra_settings["latent_multiplier"]
            seed = pseudospectra_settings["seed"]
            arrays_path = pseudospectra_arrays_path(
                args.arrays_dir,
                system,
                multiplier,
                seed,
            )
            if args.figure == "all" and not arrays_path.is_file():
                _report_missing(arrays_path, "pseudospectra", args)
            else:
                path = args.figures_dir / PSEUDOSPECTRA_PDF.format(
                    system=system,
                    multiplier=multiplier,
                    seed=seed,
                )
                arrays = pseudospectra.load_arrays(
                    arrays_path,
                    pseudospectra_settings["variants"],
                )
                pseudospectra.render(
                    arrays,
                    system,
                    path,
                    pseudospectra_settings,
                )
                print(f"WROTE {path}", flush=True)
        if args.figure in ("all", "attractors"):
            multiplier = attractor_settings["latent_multiplier"]
            seed = attractor_settings["seed"]
            arrays_path = attractor_arrays_path(
                args.arrays_dir,
                system,
                multiplier,
                seed,
            )
            if args.figure == "all" and not arrays_path.is_file():
                _report_missing(arrays_path, "attractors", args)
                continue
            computed = attractors.load_arrays(arrays_path)
            for group_name, selection in attractor_settings[
                "model_groups"
            ].items():
                path = args.figures_dir / ATTRACTOR_PDF.format(
                    system=system,
                    multiplier=multiplier,
                    seed=seed,
                    group_name=group_name,
                )
                attractors.render(
                    computed,
                    selection,
                    path,
                    attractor_settings,
                )
                print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
