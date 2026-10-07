from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from numbers import Real
from pathlib import Path
from string import Formatter

import torch
import yaml

from spectral_reliability.baselines import validate_baseline_config
from spectral_reliability.config_shared import (
    BASELINES,
    REPOSITORY_ROOT,
    SYSTEMS,
    baseline_params,
    construct_config,
    model_id,
    reject_unknown,
    require_keys,
    require_mapping,
    validate_path,
)
from spectral_reliability.data.dataset import input_dimension, load_data_config
from spectral_reliability.model import ModelConfig, NetworkConfig
from spectral_reliability.training_config import (
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from spectral_reliability.validation import (
    integer_at_least,
    nonnegative_finite,
    positive_finite,
)

PRETRAIN_FIELDS = {
    "seed",
    "updates",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "min_lr_fraction",
    "initial_conditions",
    "initial_condition_min",
    "initial_condition_max",
    "initial_perturbation_min",
    "initial_perturbation_max",
    "betas",
    "eps",
}
BETA_COUNT = 2


def configure_torch(num_threads: int) -> None:
    # Preserve the script-level Torch configuration after bootstrap.
    torch.set_num_threads(num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def merge(config: dict, override: dict) -> dict:
    """Deep-merge mappings without mutating the inputs."""
    result = copy.deepcopy(config)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def set_axis(config: dict, key: str, value: object) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        config = config.setdefault(part, {})
    config[parts[-1]] = value


def effective_latent_dim(config: dict, input_dim: int | None = None) -> int:
    latent_dim = config["model"]["latent_dim"]
    if latent_dim == "input":
        if input_dim is None:
            input_dim = input_dimension(config)
        latent_dim = input_dim
    return int(latent_dim) * config["latent_multiplier"]


def build_model_config(config: dict, input_dim: int | None = None):
    model = config["model"]
    return ModelConfig(
        effective_latent_dim(config, input_dim),
        encoder=NetworkConfig(**model["encoder"]),
        decoder=NetworkConfig(**model["decoder"]),
    )


def resolved_precision(config: dict) -> str:
    return TrainingConfig(**config["training"]).resolved_precision(
        LossConfig(**config["loss"])
    )


def validate_pretrained(settings: dict) -> None:
    reject_unknown(
        settings, {"enabled", "dir"} | PRETRAIN_FIELDS, "pretrained"
    )
    require_keys(
        settings,
        {"enabled", "dir"}
        | (
            PRETRAIN_FIELDS
            - {
                "initial_condition_min",
                "initial_condition_max",
                "initial_perturbation_min",
                "initial_perturbation_max",
            }
        ),
        "pretrained",
    )
    if type(settings.get("enabled")) is not bool:
        raise ValueError("pretrained.enabled must be a boolean")
    validate_path(settings.get("dir"), "pretrained.dir")
    for key, minimum in (("seed", 0), ("updates", 1), ("batch_size", 1)):
        if not integer_at_least(settings.get(key), minimum):
            raise ValueError(
                f"pretrained.{key} must be an integer >= {minimum}"
            )
    if not positive_finite(settings.get("learning_rate")):
        raise ValueError(
            "pretrained.learning_rate must be finite and positive"
        )
    if not nonnegative_finite(settings.get("weight_decay")):
        raise ValueError(
            "pretrained.weight_decay must be finite and non-negative"
        )
    betas = settings["betas"]
    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != BETA_COUNT
        or not all(nonnegative_finite(beta) and beta < 1 for beta in betas)
    ):
        raise ValueError("pretrained.betas must contain two values in [0, 1)")
    if not positive_finite(settings["eps"]):
        raise ValueError("pretrained.eps must be finite and positive")
    fraction = settings.get("min_lr_fraction")
    if not nonnegative_finite(fraction) or fraction > 1:
        raise ValueError("pretrained.min_lr_fraction must be in [0, 1]")
    _validate_pretraining_initial_conditions(settings)


def _validate_pretraining_initial_conditions(settings: dict) -> None:
    mode = settings.get("initial_conditions")
    if mode not in ("dataset", "uniform", "uniform_perturbation"):
        raise ValueError(
            "pretrained.initial_conditions must be dataset, uniform "
            "or uniform_perturbation"
        )
    for key in (
        "initial_condition_min",
        "initial_condition_max",
        "initial_perturbation_min",
        "initial_perturbation_max",
    ):
        if key in settings and (
            not isinstance(settings[key], Real)
            or isinstance(settings[key], bool)
            or not math.isfinite(settings[key])
        ):
            raise ValueError(f"pretrained.{key} must be finite")
    if mode != "dataset":
        prefix = (
            "initial_perturbation"
            if mode == "uniform_perturbation"
            else "initial_condition"
        )
        low, high = (
            settings.get(f"{prefix}_min"),
            settings.get(f"{prefix}_max"),
        )
        if low is None or high is None or low >= high:
            raise ValueError(
                f"{mode} pretraining requires {prefix}_min < {prefix}_max"
            )


def validate_run(config: dict) -> dict:
    config = require_mapping(config, "run")
    reject_unknown(
        config,
        {
            "system",
            "model",
            "latent_multiplier",
            "seed",
            "data",
            "output",
            "loss",
            "pretrained",
            "optimizer",
            "scheduler",
            "training",
        },
        "run",
    )
    require_keys(
        config,
        {
            "system",
            "model",
            "latent_multiplier",
            "seed",
            "data",
            "output",
        },
        "run",
    )
    if config.get("system") not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}")
    for key, minimum in (("latent_multiplier", 1), ("seed", 0)):
        if not integer_at_least(config.get(key), minimum):
            raise ValueError(f"{key} must be an integer >= {minimum}")
    validate_path(config["data"], "data")
    output = require_mapping(config.get("output"), "output")
    reject_unknown(output, {"dir"}, "output")
    require_keys(output, {"dir"}, "output")
    validate_path(output["dir"], "output.dir")
    load_data_config(config)
    model = require_mapping(config.get("model"), "model")
    require_keys(model, {"name", "latent_dim"}, "model")
    if model.get("latent_dim") != "input" and not integer_at_least(
        model.get("latent_dim"), 1
    ):
        raise ValueError(
            "model.latent_dim must be input or a positive integer"
        )
    if model.get("name") == "koopman_autoencoder":
        reject_unknown(
            model, {"name", "latent_dim", "encoder", "decoder"}, "model"
        )
        require_keys(
            model, {"name", "latent_dim", "encoder", "decoder"}, "model"
        )
        require_keys(
            config,
            {
                "loss",
                "pretrained",
                "optimizer",
                "scheduler",
                "training",
            },
            "run",
        )
        for side in ("encoder", "decoder"):
            construct_config(
                NetworkConfig,
                require_mapping(model.get(side), f"model.{side}"),
                f"model.{side}",
            )
        validate_pretrained(
            require_mapping(config.get("pretrained"), "pretrained")
        )
        for name, cls in (
            ("loss", LossConfig),
            ("optimizer", OptimizerConfig),
            ("scheduler", SchedulerConfig),
            ("training", TrainingConfig),
        ):
            construct_config(
                cls, require_mapping(config.get(name), name), name
            )
        resolved_precision(config)
    elif model.get("name") in BASELINES:
        extra = set(config) & {
            "loss",
            "pretrained",
            "optimizer",
            "scheduler",
            "training",
        }
        if extra:
            raise ValueError(
                f"baseline settings belong in model, not {sorted(extra)}"
            )
        validate_baseline_config(model["name"], baseline_params(config))
    else:
        raise ValueError(
            f"model.name must be koopman_autoencoder or one of {BASELINES}"
        )
    return config


def load_run_config(path: Path) -> dict:
    return validate_run(
        require_mapping(yaml.safe_load(Path(path).read_text()), "run")
    )


def run_directory(config: dict) -> Path:
    tree = (
        "koopman"
        if config["model"]["name"] == "koopman_autoencoder"
        else "baselines"
    )
    return (
        Path(config["output"]["dir"])
        / tree
        / config["system"]
        / model_id(config)
        / f"x{config['latent_multiplier']}"
        / f"seed{config['seed']}"
    )


def config_sha256(config: dict) -> str:
    effective = {"run": config, "data_config": load_data_config(config)}
    encoded = json.dumps(
        effective,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def completed(config: dict) -> bool:
    path = run_directory(config) / "training.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("config_sha256") == config_sha256(
        config
    )


def expand_grid(path: Path) -> list[dict]:
    """Expand ordered Cartesian axes whose placeholders name dotted
    keys.
    """
    grid = require_mapping(yaml.safe_load(Path(path).read_text()), "grid")
    reject_unknown(grid, {"base", "overrides", "axes"}, "grid")
    require_keys(grid, {"base", "axes"}, "grid")
    validate_path(grid.get("base"), "grid.base")
    axes = require_mapping(grid.get("axes"), "grid.axes")
    keys = tuple(axes)
    if not axes or any(
        not isinstance(axis, list) or not axis for axis in axes.values()
    ):
        raise ValueError("grid axes must be non-empty lists")
    if any(
        not isinstance(key, str) or any(not part for part in key.split("."))
        for key in keys
    ):
        raise ValueError("grid axis keys must be non-empty dotted paths")
    overrides = require_mapping(grid.get("overrides", {}), "grid.overrides")
    runs = []
    formatter = Formatter()
    for values in itertools.product(*(axes[key] for key in keys)):
        selection = dict(zip(keys, values, strict=True))
        pieces = []
        for literal, field, format_spec, conversion in formatter.parse(
            grid["base"]
        ):
            pieces.append(literal)
            if field is not None:
                if field not in selection or format_spec or conversion:
                    raise ValueError(
                        f"base placeholder {{{field}}} must name an axis key "
                        "without formatting"
                    )
                pieces.append(str(selection[field]))
        config_path = Path("".join(pieces))
        if not config_path.is_absolute():
            config_path = REPOSITORY_ROOT / config_path
        config = merge(
            require_mapping(
                yaml.safe_load(config_path.read_text()), "base run"
            ),
            overrides,
        )
        for key, value in selection.items():
            set_axis(config, key, value)
        runs.append(validate_run(config))
    directories = [str(run_directory(config)) for config in runs]
    if len(set(directories)) != len(directories):
        raise ValueError("grid contains runs with the same output directory")
    return runs
