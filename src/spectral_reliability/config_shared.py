from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import torch

from spectral_reliability.training_config import LossConfig, TrainingConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SYSTEMS: tuple[str, ...] = (
    "rossler",
    "lorenz63",
    "duffing",
    "mackeyglass",
    "lorenz96",
    "ks",
)
DISPLAY_NAMES: dict[str, str] = {
    "rossler": "Rössler",
    "lorenz63": "Lorenz-63",
    "duffing": "Duffing",
    "mackeyglass": "Mackey-Glass",
    "lorenz96": "Lorenz-96",
    "ks": "Kuramoto-Sivashinsky",
}
BASELINES = ("esn", "kernel_dmd", "neural_ode", "consistent_kae")
MODEL_LABELS = {
    f"{loss}{'_penalty' if penalty else ''}_{procedure}": (
        f"{label}{'+penalty' if penalty else ''} "
        f"({procedure.replace('_', '-')})"
    )
    for procedure in ("two_stage", "joint")
    for loss, label in (
        ("latent_prediction", "Latent-prediction"),
        ("spectral_residual", "Spectral-residual"),
    )
    for penalty in (False, True)
}
MODEL_VARIANTS = tuple(MODEL_LABELS)
MODEL_LABELS.update(
    {
        "esn": "ESN",
        "kernel_dmd": "Kernel DMD",
        "neural_ode": "Neural ODE",
        "consistent_kae": "Consistent KAE",
    }
)


def reject_unknown(config: dict, allowed: set[str], location: str) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {location}: {sorted(unknown)}")


def require_keys(config: dict, required: set[str], location: str) -> None:
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing keys in {location}: {sorted(missing)}")


def construct_config(cls: type, config: dict, location: str):
    names = {field.name for field in fields(cls)}
    reject_unknown(config, names, location)
    require_keys(config, names, location)
    return cls(**config)


def require_mapping(config: object, location: str) -> dict:
    if not isinstance(config, dict):
        raise TypeError(f"{location} must be a mapping")
    return config


def validate_path(value: object, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty path string")


def data_config_path(config: dict) -> Path:
    validate_path(config["data"], "data")
    try:
        path = Path(config["data"].format(system=config["system"]))
    except (KeyError, ValueError) as error:
        raise ValueError(
            "data path may contain only the {system} placeholder"
        ) from error
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def model_id(config: dict) -> str:
    name = config["model"]["name"]
    if name != "koopman_autoencoder":
        return name
    loss = LossConfig(**config["loss"])
    training = TrainingConfig(**config["training"])
    penalty = "_penalty" if loss.operator_norm_weight > 0 else ""
    return f"{loss.type}{penalty}_{training.procedure}"


def baseline_params(config: dict) -> dict:
    return {
        key: value
        for key, value in config["model"].items()
        if key not in {"name", "latent_dim"}
    }


@contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def atomic_write(path: Path, content: str | bytes) -> None:
    """Flush and fsync a same-directory temporary file before
    replacement.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = isinstance(content, bytes)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb" if binary else "w",
            encoding=None if binary else "utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, state: dict) -> None:

    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(state, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
