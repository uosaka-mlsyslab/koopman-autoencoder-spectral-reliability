from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from spectral_reliability.config import build_model_config
from spectral_reliability.config_shared import (
    atomic_torch_save,
    atomic_write,
    file_sha256,
    run_lock,
)
from spectral_reliability.data.dataset import (
    ARRAYS,
    build_arrays,
    data_config_sha256,
    generate_dataset,
    load_data_config,
    load_dataset,
)
from spectral_reliability.model import (
    KoopmanAutoencoder,
    ModelConfig,
    NetworkConfig,
)


def pretraining_settings(config: dict) -> dict:
    settings = {
        key: value
        for key, value in config["pretrained"].items()
        if key not in {"enabled", "dir"}
    }
    settings.update(
        system=config["system"],
        data_dir=load_data_config(config)["data_dir"],
        data=config["data"],
        output_dir=config["pretrained"]["dir"],
        latent_multiplier=config["latent_multiplier"],
    )
    model_config = build_model_config(config)
    settings["networks"] = {
        "encoder": asdict(model_config.encoder),
        "decoder": asdict(model_config.decoder),
    }
    return settings


def pretraining_config_sha256(settings: dict) -> str:
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()


def pretraining_data_sha256(settings: dict) -> str:
    prefix = (
        "initial_perturbation"
        if settings["initial_conditions"] == "uniform_perturbation"
        else "initial_condition"
    )
    hashed_fields = {
        key: settings[key]
        for key in (
            "system",
            "seed",
            "initial_conditions",
            f"{prefix}_min",
            f"{prefix}_max",
        )
    }
    hashed_fields["data_config_sha256"] = data_config_sha256(
        load_data_config(settings),
    )
    return hashlib.sha256(
        json.dumps(hashed_fields, sort_keys=True).encode()
    ).hexdigest()


def _uniform_initial_condition_arrays(settings: dict) -> dict[str, np.ndarray]:
    """Draw absolute states or reference-state offsets, in published
    order.
    """
    system, seed = settings["system"], settings["seed"]
    config = load_data_config(settings)
    generation_config = config["data_generation"]
    rng = np.random.default_rng(seed)
    if system == "ks":
        raise ValueError(
            "KS requires initial_conditions: dataset; its spectral "
            "generator has no uniform-state pretraining data"
        )
    if settings["initial_conditions"] == "uniform":
        low = settings["initial_condition_min"]
        high = settings["initial_condition_max"]
        dimension = (
            generation_config["K"]
            if system == "lorenz96"
            else len(generation_config["reference_state"])
        )
        initial = rng.uniform(low, high, size=dimension)
    else:
        low = settings["initial_perturbation_min"]
        high = settings["initial_perturbation_max"]
        reference_state = generation_config["reference_state"]
        initial = np.asarray(
            [
                float(value) + float(rng.uniform(low, high))
                for value in reference_state
            ]
        )
    return build_arrays(system, seed, config, initial_condition=initial)


def pretraining_arrays(
    settings: dict,
    *,
    build: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse dataset generation or cache a separate uniform
    trajectory.
    """
    system, seed = settings["system"], settings["seed"]
    data_dir = Path(settings["data_dir"])
    if settings["initial_conditions"] == "dataset":
        dataset = (
            generate_dataset(system, seed, load_data_config(settings))
            if build
            else load_dataset(system, seed, load_data_config(settings))
        )
        return dataset.train, dataset.valid
    config_hash = pretraining_data_sha256(settings)
    pretraining_data_dir = (
        data_dir / "pretraining_data" / system / f"seed{seed}"
    )
    with run_lock(pretraining_data_dir):
        existing = [
            (pretraining_data_dir / f"{name}.npy").exists() for name in ARRAYS
        ]
        rebuild_instructions = (
            "move it aside and run scripts/generate_data.py "
            "--config <run.yaml> or --grid <grid.yaml>"
        )
        if any(existing) and not all(existing):
            raise ValueError(
                f"partial pretraining data at {pretraining_data_dir}; "
                f"{rebuild_instructions}"
            )
        if not all(existing):
            if not build:
                raise FileNotFoundError(
                    f"missing pretraining data at {pretraining_data_dir}; "
                    "run scripts/generate_data.py --config <run.yaml> "
                    "or --grid <grid.yaml>"
                )
            arrays = _uniform_initial_condition_arrays(settings)
            for name, array in arrays.items():
                np.save(
                    pretraining_data_dir / f"{name}.npy",
                    array,
                    allow_pickle=False,
                )
            atomic_write(
                pretraining_data_dir / "meta.json",
                json.dumps(
                    {
                        "system": system,
                        "seed": seed,
                        "initial_conditions": settings["initial_conditions"],
                        "config_sha256": config_hash,
                    },
                    indent=2,
                )
                + "\n",
            )
        else:
            metadata_path = pretraining_data_dir / "meta.json"
            if not metadata_path.exists():
                raise ValueError(
                    f"pretraining data lacks metadata at "
                    f"{pretraining_data_dir}; {rebuild_instructions}"
                )
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("config_sha256") != config_hash:
                raise ValueError(
                    f"different pretraining data at {pretraining_data_dir}; "
                    f"{rebuild_instructions}"
                )
        return tuple(
            np.load(pretraining_data_dir / f"{name}.npy", allow_pickle=False)
            for name in ("train", "valid")
        )


def initialize_autoencoder(
    input_dim: int,
    config: ModelConfig,
    dtype: torch.dtype,
) -> KoopmanAutoencoder:
    """Advance initialization RNG to match published pretrained
    checkpoints.

    Six discarded linear layers and one discarded autoencoder advance
    the RNG before constructing the returned model. Keep these draws.
    """
    latent_dim = config.latent_dim
    for in_features, out_features in (
        (input_dim, latent_dim),
        (latent_dim, latent_dim),
        (input_dim, latent_dim),
        (latent_dim, latent_dim),
        (latent_dim, latent_dim),
        (latent_dim, input_dim),
    ):
        nn.Linear(in_features, out_features, bias=True)
    KoopmanAutoencoder(input_dim, config, dtype=torch.float64)
    model = KoopmanAutoencoder(input_dim, config, dtype=torch.float64)
    return model.to(dtype=dtype)


def pretrained_path(
    directory: Path,
    system: str,
    multiplier: int,
    precision: str,
) -> Path:
    return Path(directory) / system / f"x{multiplier}" / precision / "model.pt"


def validate_checkpoint(path: Path, settings: dict, precision: str) -> None:
    rebuild_instructions = (
        "move it aside and run scripts/pretrain.py "
        "--config <run.yaml> or --grid <grid.yaml>"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing pretrained checkpoint {path}; "
            "run scripts/pretrain.py --config <run.yaml> or --grid <grid.yaml>"
        )
    metadata_path = path.with_name("meta.json")
    if not metadata_path.exists():
        raise ValueError(
            f"pretrained checkpoint lacks metadata at {path}; "
            f"{rebuild_instructions}"
        )
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("config_sha256") != pretraining_config_sha256(settings)
        or metadata.get("network_dtype") != precision
    ):
        raise ValueError(
            f"different pretraining checkpoint at {path}; "
            f"{rebuild_instructions}"
        )
    if metadata.get("data_config_sha256") != data_config_sha256(
        load_data_config(settings),
    ):
        raise ValueError(
            f"pretrained data configuration differs at {path}; "
            f"{rebuild_instructions}"
        )
    if file_sha256(path) != metadata.get("sha256"):
        raise ValueError(
            f"pretrained weight SHA-256 mismatch at {path}; "
            f"{rebuild_instructions}"
        )


@dataclass(frozen=True)
class ReconstructionContext:
    system: str
    latent_dim: int
    precision: str
    device: str | torch.device


def _train_reconstruction(
    model: KoopmanAutoencoder,
    train: torch.Tensor,
    valid: torch.Tensor,
    *,
    settings: dict,
    context: ReconstructionContext,
):
    system, latent_dim = context.system, context.latent_dim
    precision, device = context.precision, context.device
    params = list(model.encoder.parameters()) + list(
        model.decoder.parameters()
    )
    learning_rate = settings["learning_rate"]
    updates = int(settings["updates"])
    if updates < 1:
        raise ValueError("pretraining updates must be positive")
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=settings["weight_decay"],
        betas=settings["betas"],
        eps=settings["eps"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=updates,
        eta_min=learning_rate * settings["min_lr_fraction"],
    )
    valid_relative_l2_reconstruction_error_curve = {}
    for update in range(1, updates + 1):
        indices = torch.randint(
            0,
            train.shape[0],
            (min(settings["batch_size"], train.shape[0]),),
            device=device,
        )
        x = train[indices]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - x) ** 2)
        train_reconstruction_mse = float(loss.item())
        if not np.isfinite(train_reconstruction_mse):
            raise RuntimeError(
                f"{system} latent_dim{latent_dim}: non-finite "
                f"reconstruction at update {update}"
            )
        loss.backward()
        optimizer.step()
        scheduler.step()
        if update % max(1, updates // 10) == 0:
            with torch.no_grad():
                valid_relative_l2_reconstruction_error_curve[update] = float(
                    (
                        torch.norm(model(valid) - valid, dim=1)
                        / (torch.norm(valid, dim=1) + 1e-8)
                    )
                    .mean()
                    .item()
                )
            print(
                f"{system} latent_dim{latent_dim} {precision}: "
                f"{update}/{updates}, valid "
                "rel="
                f"{valid_relative_l2_reconstruction_error_curve[update]:.4g}",
                flush=True,
            )
    with torch.no_grad():
        reconstruction_error = model(valid) - valid
        valid_reconstruction_mse = float(
            torch.mean(reconstruction_error**2).item()
        )
        valid_relative_l2_reconstruction_error = float(
            (
                torch.norm(reconstruction_error, dim=1)
                / (torch.norm(valid, dim=1) + 1e-8)
            )
            .mean()
            .item()
        )
    return (
        train_reconstruction_mse,
        valid_reconstruction_mse,
        valid_relative_l2_reconstruction_error,
        valid_relative_l2_reconstruction_error_curve,
    )


def pretrain_autoencoder(
    system: str,
    latent_dim: int,
    *,
    precision: str,
    device: str | torch.device,
    settings: dict,
) -> Path:
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be float32 or float64")
    if system != settings["system"]:
        raise ValueError("pretraining settings and system differ")
    path = pretrained_path(
        Path(settings["output_dir"]),
        system,
        settings["latent_multiplier"],
        precision,
    )
    config_hash = pretraining_config_sha256(settings)
    with run_lock(path.parent):
        if path.exists():
            validate_checkpoint(path, settings, precision)
            return path
        train_array, valid_array = pretraining_arrays(settings)
        dtype = getattr(torch, precision)
        train = torch.tensor(
            np.asarray(train_array).T, device=device, dtype=dtype
        )
        valid = torch.tensor(
            np.asarray(valid_array).T, device=device, dtype=dtype
        )
        torch.manual_seed(settings["seed"])
        networks = settings["networks"]
        config = ModelConfig(
            latent_dim,
            encoder=NetworkConfig(**networks["encoder"]),
            decoder=NetworkConfig(**networks["decoder"]),
        )
        model = initialize_autoencoder(train.shape[1], config, dtype).to(
            device=device
        )
        (
            train_reconstruction_mse,
            valid_reconstruction_mse,
            valid_relative_l2_reconstruction_error,
            valid_relative_l2_reconstruction_error_curve,
        ) = _train_reconstruction(
            model,
            train,
            valid,
            settings=settings,
            context=ReconstructionContext(
                system, latent_dim, precision, device
            ),
        )
        state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        atomic_torch_save(path, state)
        metadata = {
            "system": system,
            "latent_dim": latent_dim,
            "network_dtype": precision,
            "seed": settings["seed"],
            "updates": int(settings["updates"]),
            "model_config": asdict(config),
            "config_sha256": config_hash,
            "data_config_sha256": data_config_sha256(
                load_data_config(settings)
            ),
            "sha256": file_sha256(path),
            "valid_relative_l2_reconstruction_error_curve": (
                valid_relative_l2_reconstruction_error_curve
            ),
            "train_reconstruction_mse": train_reconstruction_mse,
            "valid_reconstruction_mse": valid_reconstruction_mse,
            "valid_relative_l2_reconstruction_error": (
                valid_relative_l2_reconstruction_error
            ),
            "learning_rate": settings["learning_rate"],
            "batch_size": settings["batch_size"],
            "initial_conditions": settings["initial_conditions"],
        }
        atomic_write(
            path.with_name("meta.json"), json.dumps(metadata, indent=2) + "\n"
        )
    return path
