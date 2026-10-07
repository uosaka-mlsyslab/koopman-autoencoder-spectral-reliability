from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import yaml

from .config import effective_latent_dim, expand_grid
from .config_shared import (
    BASELINES,
    MODEL_VARIANTS,
    REPOSITORY_ROOT,
    SYSTEMS,
    reject_unknown,
    require_keys,
    require_mapping,
    validate_path,
)
from .data.dataset import load_data_config
from .evaluation_config import load_evaluation_config
from .validation import integer_at_least, nonnegative_finite, positive_finite

PSEUDOSPECTRA_FILENAME_TEMPLATE = (
    "pseudospectra_{system}_x{multiplier}_seed{seed}.npz"
)
ATTRACTOR_FILENAME_TEMPLATE = (
    "attractors_{system}_x{multiplier}_seed{seed}.npz"
)
SCALING_QUANTILE_COUNT = 3


def repository_path(path: str | Path) -> Path:
    selected = Path(path)
    return selected if selected.is_absolute() else REPOSITORY_ROOT / selected


def pseudospectra_arrays_path(
    directory: Path, system: str, multiplier: int, seed: int
) -> Path:
    return directory / PSEUDOSPECTRA_FILENAME_TEMPLATE.format(
        system=system, multiplier=multiplier, seed=seed
    )


def attractor_arrays_path(
    directory: Path,
    system: str,
    multiplier: int,
    seed: int,
) -> Path:
    return directory / ATTRACTOR_FILENAME_TEMPLATE.format(
        system=system,
        multiplier=multiplier,
        seed=seed,
    )


def _require_section(config: dict, name: str, keys: set[str]) -> dict:
    section = require_mapping(config.get(name), name)
    reject_unknown(section, keys, name)
    require_keys(section, keys, name)
    return section


def _validate_names(value: object, allowed, location: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(name not in allowed for name in value)
    ):
        raise ValueError(
            f"{location} must be a non-empty list drawn from {tuple(allowed)}"
        )
    if len(set(value)) != len(value):
        raise ValueError(f"{location} must not contain duplicates")


def _validate_scaling(config: dict, variants: tuple[str, ...]) -> None:
    settings = _require_section(
        config,
        "scaling",
        {
            "grid",
            "variants",
            "baseline_reference",
            "baselines",
            "vrmse_window_index",
            "quantiles",
            "linear_y_cap",
            "log_y_cap",
            "linear_padding_fraction",
            "log_padding_factor",
        },
    )
    _validate_names(settings["variants"], variants, "scaling.variants")
    _validate_names(settings["baselines"], BASELINES, "scaling.baselines")
    if settings["baseline_reference"] not in variants:
        raise ValueError(
            "scaling.baseline_reference must name a Koopman variant"
        )
    quantiles = settings["quantiles"]
    if (
        not isinstance(quantiles, list)
        or len(quantiles) != SCALING_QUANTILE_COUNT
        or not all(
            nonnegative_finite(value) and value <= 1 for value in quantiles
        )
        or not quantiles[0] < quantiles[1] < quantiles[2]
    ):
        raise ValueError(
            "scaling.quantiles must contain three increasing fractions "
            "in [0, 1]"
        )
    if not nonnegative_finite(settings["linear_padding_fraction"]):
        raise ValueError(
            "scaling.linear_padding_fraction must be finite and non-negative"
        )
    for key in ("linear_y_cap", "log_y_cap", "log_padding_factor"):
        if not positive_finite(settings[key]):
            raise ValueError(f"scaling.{key} must be finite and positive")
    if settings["log_padding_factor"] <= 1:
        raise ValueError("scaling.log_padding_factor must exceed 1")
    if not integer_at_least(settings["vrmse_window_index"], 0):
        raise ValueError(
            "scaling.vrmse_window_index must be a non-negative integer"
        )
    validate_path(settings["grid"], "scaling.grid")


def _validate_target(
    settings: dict, name: str, variants: tuple[str, ...]
) -> None:
    _validate_names(settings["variants"], variants, f"{name}.variants")
    if not integer_at_least(settings["seed"], 0) or not integer_at_least(
        settings["latent_multiplier"], 1
    ):
        raise ValueError(
            f"{name} requires a non-negative integer seed and positive "
            "latent_multiplier"
        )


def _validate_pseudospectra(config: dict, variants: tuple[str, ...]) -> None:
    settings = _require_section(
        config,
        "pseudospectra",
        {
            "seed",
            "latent_multiplier",
            "variants",
            "splits",
            "half_width",
            "grid_points_per_axis",
            "contour_levels",
            "encoder_batch_size",
        },
    )
    _validate_target(settings, "pseudospectra", variants)
    _validate_names(
        settings["splits"], ("train", "valid", "test"), "pseudospectra.splits"
    )
    if not positive_finite(settings["half_width"]):
        raise ValueError(
            "pseudospectra.half_width must be finite and positive"
        )
    for key, minimum in (
        ("grid_points_per_axis", 2),
        ("contour_levels", 2),
        ("encoder_batch_size", 1),
    ):
        if not integer_at_least(settings[key], minimum):
            raise ValueError(
                f"pseudospectra.{key} must be an integer >= {minimum}"
            )


def _validate_attractors(config: dict, variants: tuple[str, ...]) -> None:
    settings = _require_section(
        config,
        "attractors",
        {
            "seed",
            "latent_multiplier",
            "variants",
            "model_groups",
            "split",
            "prediction_horizons_lt",
            "window_lt",
        },
    )
    _validate_target(settings, "attractors", variants)
    prediction_horizons_lt = settings["prediction_horizons_lt"]
    if (
        not isinstance(prediction_horizons_lt, list)
        or not prediction_horizons_lt
        or not all(positive_finite(value) for value in prediction_horizons_lt)
        or any(
            first >= second
            for first, second in pairwise(prediction_horizons_lt)
        )
    ):
        raise ValueError(
            "attractors.prediction_horizons_lt must contain increasing "
            "positive Lyapunov times"
        )
    if (
        not positive_finite(settings["window_lt"])
        or prediction_horizons_lt[-1] >= settings["window_lt"]
    ):
        raise ValueError(
            "attractors.window_lt must exceed the largest prediction horizon"
        )
    model_groups = require_mapping(
        settings["model_groups"], "attractors.model_groups"
    )
    if not model_groups:
        raise ValueError(
            "attractors.model_groups must contain at least one "
            "named model selection"
        )
    for group_name, selection in model_groups.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise ValueError(
                "attractors.model_groups name must be a non-empty "
                "model-group name"
            )
        _validate_names(
            selection,
            settings["variants"],
            f"attractors.model_groups.{group_name}",
        )
    if settings["split"] not in ("train", "valid", "test"):
        raise ValueError("attractors.split must be train, valid or test")


def load_figures_config(path: Path) -> dict:
    config = require_mapping(
        yaml.safe_load(repository_path(path).read_text()), "figures"
    )
    keys = {"systems", "scaling", "pseudospectra", "attractors"}
    reject_unknown(config, keys, "figures")
    require_keys(config, keys, "figures")
    _validate_names(config["systems"], SYSTEMS, "systems")
    _validate_scaling(config, MODEL_VARIANTS)
    _validate_pseudospectra(config, MODEL_VARIANTS)
    _validate_attractors(config, MODEL_VARIANTS)
    return config


def resolve_scaling_grid(settings: dict, systems: list[str]) -> dict:
    """Resolve scaling dimensions, seeds and metric columns from the
    configured grid.
    """

    runs = expand_grid(repository_path(settings["grid"]))
    selected = [run for run in runs if run["system"] in systems]
    multipliers = tuple(
        dict.fromkeys(run["latent_multiplier"] for run in selected)
    )
    seeds = tuple(dict.fromkeys(run["seed"] for run in selected))
    dimensions, metric_columns = {}, {}
    for system in systems:
        system_runs = [run for run in selected if run["system"] == system]
        if not system_runs:
            raise ValueError(f"scaling grid contains no runs for {system}")
        by_multiplier = {run["latent_multiplier"]: run for run in system_runs}
        if set(by_multiplier) != set(multipliers):
            raise ValueError(
                f"{system}: scaling grid has a different dimension ladder"
            )
        dimensions[system] = tuple(
            effective_latent_dim(by_multiplier[value]) for value in multipliers
        )
        evaluation = load_evaluation_config(load_data_config(system_runs[0]))
        index = settings["vrmse_window_index"]
        if index >= len(evaluation.vrmse_windows_lt):
            raise ValueError(
                f"{system}: scaling vrmse_window_index is outside "
                "evaluation.vrmse_windows_lt"
            )
        metric_columns[system] = evaluation.metric_columns[
            len(evaluation.vpt_epsilons) + index
        ]
    if len(set(metric_columns.values())) != 1:
        raise ValueError(
            "scaling figures require the same selected VRMSE window "
            "for every system"
        )
    return {
        "systems": systems,
        "multipliers": multipliers,
        "seeds": seeds,
        "dimensions": dimensions,
        "metric_columns": metric_columns,
        "metric_name": metric_columns[systems[0]].removesuffix("_mean"),
    }
