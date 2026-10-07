from __future__ import annotations

import fcntl
import hashlib
import json
import math
from dataclasses import dataclass, fields
from numbers import Real
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from spectral_reliability.config_shared import (
    REPOSITORY_ROOT,
    SYSTEMS,
    atomic_write,
    construct_config,
    data_config_path,
    reject_unknown,
    require_keys,
)
from spectral_reliability.data.systems import SYSTEM_CONFIGS, simulate
from spectral_reliability.evaluation_config import EvaluationConfig
from spectral_reliability.validation import (
    integer_at_least,
    nonnegative_finite,
    positive_finite,
)

ARRAYS = ("train", "valid", "test", "training_mean")
FloatArray = NDArray[np.float64]
MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class Dataset:
    """Store feature-by-time splits with measured state last."""

    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    training_mean: np.ndarray
    measured_dim: int
    num_delay_samples: int
    dt: float
    system: str
    seed: int
    data_dir: Path


def delay_embedding(data: ArrayLike, num_delay_samples: int) -> FloatArray:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != MATRIX_DIMENSIONS:
        raise ValueError("data must be a two-dimensional array")
    if num_delay_samples < 1:
        raise ValueError("num_delay_samples must be at least 1")
    if num_delay_samples > array.shape[1]:
        raise ValueError(
            "num_delay_samples cannot exceed the number of time steps"
        )
    if num_delay_samples == 1:
        return array
    delayed = []
    for delay in range(num_delay_samples):
        delayed.append(
            array[:, delay : array.shape[1] - num_delay_samples + delay + 1]
        )
    return np.vstack(delayed, dtype=np.float64)


def _validate_generation(cls: type, generation_config: dict) -> None:
    hints = get_type_hints(cls)
    for field in fields(cls):
        require_keys(generation_config, {field.name}, "data_generation")
        value = generation_config[field.name]
        annotation = hints[field.name]
        if annotation is float:
            valid = (
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
            if field.name in {
                "dt",
                "duration",
                "internal_dt",
                "rtol",
                "atol",
                "L",
            }:
                valid = positive_finite(value)
            elif field.name == "initial_condition_noise_std":
                valid = nonnegative_finite(value)
        elif annotation is int:
            valid = integer_at_least(
                value, 0 if field.name == "burn_in_steps" else 1
            )
        elif get_origin(annotation) is tuple:
            integer_items = get_args(annotation)[0] is int
            valid = (
                isinstance(value, (list, tuple))
                and bool(value)
                and all(
                    integer_at_least(item, 0)
                    if integer_items
                    else (
                        isinstance(item, Real)
                        and not isinstance(item, bool)
                        and math.isfinite(item)
                    )
                    for item in value
                )
            )
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise ValueError(
                f"invalid data_generation.{field.name}: {value!r}"
            )


def load_data_config(source: str | dict) -> dict:
    system = source if isinstance(source, str) else source["system"]
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}")
    path = (
        REPOSITORY_ROOT / f"configs/data/{system}.yaml"
        if isinstance(source, str)
        else data_config_path(source)
    )
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise TypeError("data config must be a mapping")
    keys = {
        "system",
        "data_generation",
        "preprocessing",
        "split",
        "data_dir",
        "evaluation",
    }
    reject_unknown(config, keys, "data config")
    require_keys(config, keys, "data config")
    if config.get("system") != system:
        raise ValueError("data system must match the run system")
    for name in ("data_generation", "preprocessing", "split", "evaluation"):
        if not isinstance(config.get(name), dict):
            raise TypeError(f"data {name} must be a mapping")
    cls = SYSTEM_CONFIGS[system]
    reject_unknown(
        config["data_generation"],
        {field.name for field in fields(cls)},
        "data_generation",
    )
    _validate_generation(cls, config["data_generation"])
    if system == "ks" and config["data_generation"]["dealiasing"] not in (
        "two_thirds",
        "none",
    ):
        raise ValueError(
            "data_generation.dealiasing must be two_thirds or none"
        )
    if (
        not isinstance(config.get("data_dir"), str)
        or not config["data_dir"].strip()
    ):
        raise ValueError("data data_dir must be a non-empty path string")
    prep = config["preprocessing"]
    reject_unknown(
        prep,
        {
            "num_delay_samples",
            "sampling_stride",
            "spatial_subsample",
        },
        "preprocessing",
    )
    require_keys(
        prep,
        {"num_delay_samples", "sampling_stride"}
        | ({"spatial_subsample"} if system == "ks" else set()),
        "preprocessing",
    )
    for name in ("num_delay_samples", "sampling_stride"):
        if not integer_at_least(prep.get(name), 1):
            raise ValueError(
                f"preprocessing.{name} must be a positive integer"
            )
    if "spatial_subsample" in prep and not integer_at_least(
        prep["spatial_subsample"], 1
    ):
        raise ValueError("spatial_subsample must be a positive integer")
    reject_unknown(
        config["split"], {"train_valid_ratio", "train_ratio"}, "split"
    )
    for name in ("train_valid_ratio", "train_ratio"):
        value = config["split"].get(name)
        if not positive_finite(value) or value >= 1:
            raise ValueError(f"split.{name} must lie in (0, 1)")
    construct_config(EvaluationConfig, config["evaluation"], "evaluation")
    return config


def hashed_data_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "evaluation"}


def data_config_sha256(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(hashed_data_config(config), sort_keys=True).encode()
    ).hexdigest()


def input_dimension(source: str | dict) -> int:
    config = load_data_config(source)
    system = config["system"]
    generation_config, prep = (
        config["data_generation"],
        config["preprocessing"],
    )
    if system == "ks":
        measured = len(
            range(
                0,
                generation_config["num_grid_points"],
                prep["spatial_subsample"],
            )
        )
    else:
        measured = len(generation_config["reference_state"])
    return measured * prep["num_delay_samples"]


def build_arrays(
    system: str,
    seed: int,
    config: dict,
    initial_condition: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Split before embedding, then center with the training-set mean.

    Split the strided trajectory into train/validation and test
    intervals before delay embedding; divide the embedded first
    interval into train and valid, preserving the variance scale.
    """
    prep, split = config["preprocessing"], config["split"]
    series = simulate(system, config, seed, initial_condition)
    if system == "ks":
        series = series[:: prep["spatial_subsample"]]
    series = np.asarray(
        series[:, :: prep["sampling_stride"]], dtype=np.float64
    )
    raw_split = int(split["train_valid_ratio"] * series.shape[1])
    train_valid = delay_embedding(
        series[:, :raw_split], num_delay_samples=prep["num_delay_samples"]
    )
    test = delay_embedding(
        series[:, raw_split:], num_delay_samples=prep["num_delay_samples"]
    )
    train_split = int(split["train_ratio"] * train_valid.shape[1])
    train, valid = train_valid[:, :train_split], train_valid[:, train_split:]
    mean = train.mean(axis=1, keepdims=True, dtype=np.float64)
    return {
        "train": np.asarray(train - mean, dtype=np.float64),
        "valid": np.asarray(valid - mean, dtype=np.float64),
        "test": np.asarray(test - mean, dtype=np.float64),
        "training_mean": np.asarray(mean, dtype=np.float64),
    }


def _dataset_directory(config: dict, system: str, seed: int) -> Path:
    return Path(config["data_dir"]) / system / f"seed{int(seed)}"


def _dataset_complete(directory: Path) -> bool:
    existing = [(directory / f"{name}.npy").exists() for name in ARRAYS]
    if any(existing) and not all(existing):
        raise ValueError(
            f"partial dataset at {directory}; move it aside before rebuilding"
        )
    return all(existing)


def generate_dataset(system: str, seed: int, config: dict) -> Dataset:
    dataset_dir = _dataset_directory(config, system, seed)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / ".dataset.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not _dataset_complete(dataset_dir):
            arrays = build_arrays(system, seed, config)
            for name, array in arrays.items():
                np.save(dataset_dir / f"{name}.npy", array, allow_pickle=False)
            config_hash = data_config_sha256(config)
            atomic_write(
                dataset_dir / "meta.json",
                json.dumps(
                    {
                        "system": system,
                        "seed": seed,
                        "data_config": hashed_data_config(config),
                        "config_sha256": config_hash,
                    },
                    indent=2,
                ),
            )
    return load_dataset(system, seed, config)


def load_dataset(system: str, seed: int, config: dict) -> Dataset:
    if config["system"] != system:
        raise ValueError("data system must match the run system")
    dataset_dir = _dataset_directory(config, system, seed)
    if not _dataset_complete(dataset_dir):
        raise FileNotFoundError(
            f"missing dataset at {dataset_dir}; run "
            "scripts/generate_data.py --config "
            f"configs/koopman/{system}.yaml after setting its seed "
            f"to {int(seed)} and data_dir in its data YAML "
            f"to {config['data_dir']}, or use --grid <grid.yaml>"
        )
    config_hash = data_config_sha256(config)
    meta = json.loads((dataset_dir / "meta.json").read_text())
    if meta["config_sha256"] != config_hash:
        raise ValueError(
            f"dataset configuration differs at {dataset_dir}; move it aside "
            "then run scripts/generate_data.py --config <run.yaml> "
            "or --grid <grid.yaml>"
        )
    arrays = {
        name: np.load(dataset_dir / f"{name}.npy", allow_pickle=False)
        for name in ARRAYS
    }
    delay = config["preprocessing"]["num_delay_samples"]
    generation = config["data_generation"]
    dt = generation["dt"] * config["preprocessing"]["sampling_stride"]
    return Dataset(
        **arrays,
        measured_dim=arrays["train"].shape[0] // delay,
        num_delay_samples=delay,
        dt=float(dt),
        system=system,
        seed=int(seed),
        data_dir=Path(config["data_dir"]),
    )
