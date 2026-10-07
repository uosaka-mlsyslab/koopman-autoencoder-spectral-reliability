from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from spectral_reliability.baselines import load_baseline
from spectral_reliability.config_shared import atomic_write, model_id, run_lock
from spectral_reliability.data.dataset import Dataset, load_dataset
from spectral_reliability.evaluation_config import load_evaluation_config
from spectral_reliability.koopman import (
    KoopmanFit,
    estimate_latent_operator,
    rank_constrained_operator,
)
from spectral_reliability.metrics import evaluate_rollouts
from spectral_reliability.model import (
    KoopmanAutoencoder,
    ModelConfig,
    NetworkConfig,
)
from spectral_reliability.results import RunRecord


@dataclass
class TrainedModel:
    encoder: nn.Module
    decoder: nn.Module
    rollout_operator: torch.Tensor
    full_rank_operator: torch.Tensor
    dtype: torch.dtype
    system: str
    seed: int
    data_config: dict
    operator_regularization: float

    def to(self, device: str | torch.device) -> TrainedModel:
        self.encoder.to(device=device, dtype=self.dtype)
        self.decoder.to(device=device, dtype=self.dtype)
        self.rollout_operator = self.rollout_operator.to(device=device)
        self.full_rank_operator = self.full_rank_operator.to(device=device)
        return self

    def iter_rollout_tensor(
        self,
        initial: np.ndarray,
        steps: int,
    ) -> Iterator[torch.Tensor]:
        operator = self.rollout_operator
        with torch.inference_mode():
            observations = torch.as_tensor(
                initial,
                dtype=torch.float64,
                device=operator.device,
            )
            latent = (
                self.encoder(observations.to(dtype=self.dtype))
                .to(dtype=operator.dtype)
                .T
            )
        for _ in range(steps):
            with torch.inference_mode():
                latent = operator @ latent
                prediction = self.decoder(
                    latent.T.to(dtype=self.dtype),
                ).to(dtype=torch.float64)
            yield prediction


def model_from_checkpoint(checkpoint: dict) -> KoopmanAutoencoder:
    config = checkpoint["model_config"]
    model_config = ModelConfig(
        latent_dim=int(config["latent_dim"]),
        encoder=NetworkConfig(**config["encoder"]),
        decoder=NetworkConfig(**config["decoder"]),
    )
    dtype = (
        torch.float32
        if checkpoint["network_dtype"] == "float32"
        else torch.float64
    )
    model = KoopmanAutoencoder(
        int(checkpoint["input_dim"]),
        model_config,
        dtype=dtype,
    )
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    return model.eval()


@dataclass(frozen=True)
class OperatorConfig:
    loss_type: str
    regularization: float
    svd_rank: float | None = None


def fit_operators(
    encoder,
    train,
    config: OperatorConfig,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
) -> tuple[KoopmanFit, KoopmanFit]:
    loss_type, regularization = config.loss_type, config.regularization
    svd_rank = config.svd_rank
    encoder.to(device=device, dtype=dtype).eval()
    observations = torch.as_tensor(train.T, dtype=dtype, device=device)
    with torch.inference_mode():
        latent = encoder(observations).T
        full_rank = estimate_latent_operator(
            latent[:, :-1],
            latent[:, 1:],
            regularization=regularization,
            dtype=dtype,
        )
        rollout = (
            rank_constrained_operator(
                latent[:, :-1],
                latent[:, 1:],
                regularization=regularization,
                dtype=dtype,
                svd_rank=svd_rank,
            )
            if loss_type == "latent_prediction"
            else full_rank
        )
    return rollout, full_rank


def fit_rollout_operators(
    model: KoopmanAutoencoder,
    dataset: Dataset,
    config: OperatorConfig,
    *,
    device: str | torch.device,
    data_config: dict,
) -> TrainedModel:
    """Refit rollout and full-rank operators on all adjacent training
    pairs.
    """
    dtype = next(model.parameters()).dtype
    model.to(device=device)
    rollout, full_rank = fit_operators(
        model.encoder,
        dataset.train,
        replace(
            config,
            svd_rank=load_evaluation_config(data_config).rollout_svd_rank,
        ),
        dtype=dtype,
        device=device,
    )
    return TrainedModel(
        model.encoder,
        model.decoder,
        rollout.a_hat,
        full_rank.a_hat,
        dtype,
        dataset.system,
        dataset.seed,
        data_config,
        config.regularization,
    )


def load_trained_model(run_dir: Path) -> TrainedModel:
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    training = json.loads((run_dir / "training.json").read_text())
    checkpoint = torch.load(
        run_dir / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = model_from_checkpoint(checkpoint)
    return TrainedModel(
        model.encoder,
        model.decoder,
        checkpoint["rollout_operator"],
        checkpoint["full_rank_operator"],
        next(model.parameters()).dtype,
        config["system"],
        int(config["seed"]),
        training["data_config"],
        float(config["training"]["operator_regularization"]),
    )


def _read_run_metrics(path: Path, training: dict) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", {})
    keys = load_evaluation_config(training["data_config"]).metric_columns
    if payload.get("config_sha256") != training["config_sha256"] or not all(
        key in metrics for key in keys
    ):
        return {}
    return {key: float(metrics[key]) for key in keys}


def read_run_records(outputs_dir: Path) -> list[RunRecord]:
    records = []
    for tree in ("koopman", "baselines"):
        for training_path in sorted(
            (Path(outputs_dir) / tree).glob("*/*/x*/seed*/training.json"),
        ):
            config_path = training_path.with_name("config.yaml")
            if not config_path.exists():
                continue
            config = yaml.safe_load(config_path.read_text())
            training = json.loads(training_path.read_text())
            metrics_path = training_path.with_name("metrics.json")
            metrics = _read_run_metrics(metrics_path, training)
            records.append(
                RunRecord(
                    system=config["system"],
                    model=model_id(config),
                    latent_multiplier=config["latent_multiplier"],
                    latent_dim=training["latent_dim"],
                    seed=config["seed"],
                    metrics=metrics,
                    run_dir=training_path.parent,
                )
            )
    return records


def evaluate_runs(outputs_dir: Path, *, device: str) -> list[RunRecord]:

    records = read_run_records(outputs_dir)
    for record in records:
        with run_lock(record.run_dir):
            training = json.loads(
                (record.run_dir / "training.json").read_text(),
            )
            record.metrics.clear()
            record.metrics.update(
                _read_run_metrics(
                    record.run_dir / "metrics.json",
                    training,
                )
            )
            if record.metrics:
                print(
                    f"skip {record.run_dir}: metrics config SHA-256 "
                    "and configured keys match",
                    flush=True,
                )
                continue
            config = yaml.safe_load(
                (record.run_dir / "config.yaml").read_text(),
            )
            data_config = training["data_config"]
            dataset = load_dataset(record.system, record.seed, data_config)
            is_koopman = config["model"]["name"] == "koopman_autoencoder"
            if is_koopman:
                predictor = load_trained_model(record.run_dir).to(device)
            else:
                checkpoint = torch.load(
                    record.run_dir / "model.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                predictor = load_baseline(
                    checkpoint,
                    config["model"],
                    device=device,
                )
            metrics = evaluate_rollouts(
                predictor,
                dataset,
                load_evaluation_config(data_config),
            )
            payload = {
                key: value
                for key, value in training.items()
                if key not in {"data_config", "latent_dim"}
            }
            payload["metrics"] = metrics
            atomic_write(
                record.run_dir / "metrics.json",
                json.dumps(payload, indent=2, allow_nan=not is_koopman) + "\n",
            )
            record.metrics.update(metrics)
            print(
                f"WROTE {record.run_dir / 'metrics.json'}: {metrics}",
                flush=True,
            )
    return records
