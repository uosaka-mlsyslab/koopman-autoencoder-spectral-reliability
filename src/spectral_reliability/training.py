from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from spectral_reliability.baselines import build_baseline
from spectral_reliability.config import (
    build_model_config,
    completed,
    config_sha256,
    effective_latent_dim,
    resolved_precision,
    run_directory,
)
from spectral_reliability.config_shared import (
    atomic_torch_save,
    atomic_write,
    baseline_params,
    run_lock,
)
from spectral_reliability.data.dataset import (
    Dataset,
    load_data_config,
    load_dataset,
)
from spectral_reliability.evaluation import (
    OperatorConfig,
    fit_rollout_operators,
)
from spectral_reliability.evaluation_config import (
    EvaluationConfig,
    load_evaluation_config,
)
from spectral_reliability.koopman import (
    KoopmanFit,
    estimate_latent_operator,
    evolution_loss,
    rank_constrained_operator,
)
from spectral_reliability.metrics import validation_vrmse
from spectral_reliability.model import (
    KoopmanAutoencoder,
    encode_snapshot_pairs,
)
from spectral_reliability.pretraining import (
    pretrained_path,
    pretraining_settings,
    validate_checkpoint,
)
from spectral_reliability.training_config import (
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)

EpochRecord = dict[str, float | int]
MATRIX_DIMENSIONS = 2
MINIMUM_PAIR_STATES = 2


@dataclass(frozen=True)
class TrainSettings:
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


@dataclass(frozen=True)
class TrainRuntime:
    seed: int
    device: str | torch.device


@dataclass(frozen=True)
class EpochData:
    train_data: torch.Tensor
    generator: torch.Generator
    pools: tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class TrainResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int


class WarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        after_scheduler: torch.optim.lr_scheduler.LRScheduler,
        warmup_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.after_scheduler = after_scheduler
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.base_lrs = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        self.warmup_finished = self.warmup_epochs <= 0
        self._warmup_step_count = 0
        if self.warmup_epochs > 0:
            initial_scale = 1.0 / float(self.warmup_epochs)
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs, strict=True
            ):
                param_group["lr"] = base_lr * initial_scale

    def step(self) -> None:
        if not self.warmup_finished:
            self._warmup_step_count += 1
            if self._warmup_step_count < self.warmup_epochs:
                scale = float(self._warmup_step_count + 1) / float(
                    self.warmup_epochs
                )
                for param_group, base_lr in zip(
                    self.optimizer.param_groups, self.base_lrs, strict=True
                ):
                    param_group["lr"] = base_lr * scale
                return
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs, strict=True
            ):
                param_group["lr"] = base_lr
            self.warmup_finished = True
            return
        self.after_scheduler.step()


Scheduler = WarmupScheduler | torch.optim.lr_scheduler.LRScheduler


@dataclass
class Optimizers:
    reconstruction: torch.optim.Optimizer | None = None
    evolution: torch.optim.Optimizer | None = None
    joint: torch.optim.Optimizer | None = None
    reconstruction_scheduler: Scheduler | None = None
    evolution_scheduler: Scheduler | None = None
    joint_scheduler: Scheduler | None = None

    def step_schedulers(self) -> None:
        for scheduler in (
            self.reconstruction_scheduler,
            self.evolution_scheduler,
            self.joint_scheduler,
        ):
            if scheduler is not None:
                scheduler.step()


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
) -> Scheduler:
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.t_max, eta_min=config.min_lr
    )
    if config.warmup_epochs > 0:
        return WarmupScheduler(optimizer, cosine, config.warmup_epochs)
    return cosine


def _make_optimizer(
    parameters: Iterable[torch.Tensor],
    learning_rate: float,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    optimizer_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[
        config.name
    ]
    return optimizer_class(
        parameters,
        lr=learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def build_optimizers(
    model: KoopmanAutoencoder,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    training_config: TrainingConfig,
) -> Optimizers:
    if training_config.procedure == "joint":
        joint = _make_optimizer(
            model.parameters(),
            optimizer_config.learning_rate_joint,
            optimizer_config,
        )
        return Optimizers(
            joint=joint,
            joint_scheduler=_make_scheduler(joint, scheduler_config),
        )
    if training_config.procedure != "two_stage":
        raise ValueError(
            f"unknown training procedure {training_config.procedure!r}"
        )
    reconstruction = _make_optimizer(
        model.parameters(),
        optimizer_config.learning_rate_reconstruction,
        optimizer_config,
    )
    evolution = _make_optimizer(
        model.encoder.parameters(),
        optimizer_config.learning_rate_evolution,
        optimizer_config,
    )
    return Optimizers(
        reconstruction=reconstruction,
        evolution=evolution,
        reconstruction_scheduler=_make_scheduler(
            reconstruction,
            scheduler_config,
        ),
        evolution_scheduler=_make_scheduler(evolution, scheduler_config),
    )


class InnerLoopPlateauDetector:
    """Stop on a small relative decrease in the half-window loss means.

    Compare the two halves of the trailing window, expressing the
    relative decrease per 100 updates. Stop when this rate is at most
    improvement_rate_tolerance.
    """

    RATE_NORMALIZATION_UPDATES = 100

    def __init__(
        self,
        *,
        window: int,
        improvement_rate_tolerance: float,
    ) -> None:
        self.window = max(2, int(window))
        self.improvement_rate_tolerance = float(improvement_rate_tolerance)
        self._history: list[float] = []

    def update(self, loss_value: float | torch.Tensor) -> bool:
        try:
            value = float(loss_value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        self._history.append(value)
        if len(self._history) < self.window:
            return False
        trailing = self._history[-self.window :]
        half = self.window // 2
        first = statistics.fmean(trailing[:half])
        second = statistics.fmean(trailing[half:])
        if first == 0.0:
            return True
        improvement_rate = (
            (first - second)
            / abs(first)
            * (self.RATE_NORMALIZATION_UPDATES / half)
        )
        return improvement_rate <= self.improvement_rate_tolerance


def interleaved_pools(
    total_steps: int,
    block_steps: float,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternate adjacent-pair blocks and exclude transitions across
    block boundaries.
    """
    total_steps = int(total_steps)
    rounded_block_steps = round(float(block_steps))
    if rounded_block_steps <= 1:
        raise ValueError(
            f"block_steps={rounded_block_steps} cannot hold a 1-step window; "
            "every start time in it would be dropped"
        )
    if total_steps <= 1:
        raise ValueError(
            f"a 1-step window is too large for sequence length T={total_steps}"
        )
    starts = torch.arange(total_steps - 1, dtype=torch.long, device=device)
    block_offset = starts % rounded_block_steps
    is_estimation = (starts // rounded_block_steps) % 2 == 0
    has_in_block_successor = block_offset < rounded_block_steps - 1
    return (
        starts[is_estimation & has_in_block_successor],
        starts[(~is_estimation) & has_in_block_successor],
    )


def _sample_pair_starts(
    trajectory: torch.Tensor,
    batch_size: int,
    start_pool: torch.Tensor | None,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if (
        trajectory.ndim != MATRIX_DIMENSIONS
        or trajectory.shape[1] < MINIMUM_PAIR_STATES
    ):
        raise ValueError(
            "trajectory must have feature-by-time shape "
            "with at least two times"
        )
    pool = (
        torch.arange(
            trajectory.shape[1] - 1, dtype=torch.long, device=trajectory.device
        )
        if start_pool is None
        else start_pool.to(device=trajectory.device, dtype=torch.long)
    )
    if pool.numel() == 0:
        raise ValueError(
            "the pair-start pool passed to get_pair_batch is empty; "
            "it is built from valid adjacent-pair start indices "
            "in the trajectory"
        )
    if batch_size >= pool.numel():
        starts = pool
    else:
        choices = torch.randint(
            0,
            pool.numel(),
            (batch_size,),
            dtype=torch.long,
            device=generator.device,
            generator=generator,
        )
        starts = pool.index_select(0, choices.to(pool.device))
    return starts


def get_pair_batch(
    trajectory: torch.Tensor,
    batch_size: int,
    start_pool: torch.Tensor | None,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return [x; y] rows of shape (2*m, d_x) for m adjacent pairs."""
    starts = _sample_pair_starts(trajectory, batch_size, start_pool, generator)
    x = trajectory.index_select(1, starts).T.contiguous()
    y = trajectory.index_select(1, starts + 1).T.contiguous()
    return torch.cat([x, y], dim=0)


class NonFiniteGradientError(RuntimeError):
    pass


def clip_and_step(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
    *,
    max_norm: float,
    phase: str = "training",
) -> None:
    parameter_list = list(parameters)
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameter_list, max_norm=float(max_norm)
    )
    if not torch.isfinite(total_norm):
        optimizer.zero_grad(set_to_none=True)
        raise NonFiniteGradientError(
            f"{phase}: gradient norm was {float(total_norm)}; refusing to step"
        )
    optimizer.step()


class BestCheckpointSelector:
    def __init__(self) -> None:
        self.best_metric = float("inf")
        self.best_epoch = -1
        self.best_state: dict[str, torch.Tensor] | None = None

    def update(self, *, epoch: int, metric: float, model: nn.Module) -> None:
        metric = float(metric)
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_epoch = int(epoch)
            self.best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }


def joint_loss(
    evolution_loss: torch.Tensor,
    reconstruction_loss: torch.Tensor,
    *,
    weights: tuple[float, float],
) -> torch.Tensor:
    weight_evolution, weight_reconstruction = weights
    return (
        float(weight_evolution) * evolution_loss
        + float(weight_reconstruction) * reconstruction_loss
    )


def propagate_latent(
    operator: torch.Tensor,
    latent: torch.Tensor,
) -> torch.Tensor:
    aligned_latent = latent.to(device=operator.device, dtype=operator.dtype)
    return operator @ aligned_latent


def _evolution_training_loss(
    model: KoopmanAutoencoder,
    estimation_batch: torch.Tensor,
    held_out_batch: torch.Tensor,
    loss: LossConfig,
    operator_regularization: float,
) -> tuple[torch.Tensor, KoopmanFit, torch.Tensor]:
    z_x, z_y = encode_snapshot_pairs(model, estimation_batch)
    z_x_prime, z_y_prime = encode_snapshot_pairs(model, held_out_batch)
    fit_dtype = next(model.parameters()).dtype
    fit = estimate_latent_operator(
        z_x,
        z_y,
        regularization=operator_regularization,
        dtype=fit_dtype,
    )
    value = evolution_loss(
        fit,
        z_x_prime,
        z_y_prime,
        loss,
    )
    return value, fit, z_x_prime


def _pair_mse(
    model: KoopmanAutoencoder,
    fit: KoopmanFit,
    z_x_prime: torch.Tensor,
    held_out_batch: torch.Tensor,
) -> torch.Tensor:
    """Decode [z_x_prime, a_hat z_x_prime] against held-out [x; y].

    This MSE combines source reconstruction and one-step prediction.
    """
    propagated = propagate_latent(fit.a_hat, z_x_prime)
    pair_latent = torch.cat(
        [z_x_prime.to(propagated.dtype), propagated], dim=1
    )
    prediction = model.decode(
        pair_latent.T.to(
            device=held_out_batch.device,
            dtype=held_out_batch.dtype,
        ),
    )
    return torch.mean((prediction - held_out_batch) ** 2)


def _estimation_held_out_pools(
    train_data: torch.Tensor,
    config: TrainingConfig,
    evaluation: EvaluationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.disjoint_pools:
        block_steps = config.pool_block_lt * evaluation.steps_per_lt
        estimation_pool, held_out_pool = interleaved_pools(
            total_steps=train_data.shape[1],
            block_steps=block_steps,
            device=train_data.device,
        )
        available_pairs = min(estimation_pool.numel(), held_out_pool.numel())
        pool_description = "an estimation/held-out pool"
    else:
        pair_start_count = max(int(train_data.shape[1]) - 1, 0)
        estimation_pool = torch.arange(
            pair_start_count,
            dtype=torch.long,
            device=train_data.device,
        )
        held_out_pool = estimation_pool
        available_pairs = estimation_pool.numel()
        pool_description = "the estimation pool"
    required_pairs = config.batch_size
    if available_pairs < required_pairs:
        raise ValueError(
            f"{pool_description} holds {available_pairs} start times, "
            f"which cannot fill {required_pairs} pairs for "
            f"batch_size={config.batch_size}"
        )
    return estimation_pool, held_out_pool


def _sample_batches(
    train_data: torch.Tensor,
    config: TrainingConfig,
    pools: tuple[torch.Tensor, torch.Tensor],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    estimation_batch = get_pair_batch(
        train_data, config.batch_size, pools[0], generator
    )
    held_out_batch = (
        get_pair_batch(train_data, config.batch_size, pools[1], generator)
        if config.disjoint_pools
        else estimation_batch
    )
    return estimation_batch, held_out_batch


def _mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _reconstruction_phase(
    model: KoopmanAutoencoder,
    train_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    generator: torch.Generator,
) -> list[float]:
    reconstruction_losses = []
    detector = InnerLoopPlateauDetector(
        window=config.plateau_window,
        improvement_rate_tolerance=(
            config.plateau_relative_improvement_rate_tolerance
        ),
    )
    for _ in range(config.max_updates_reconstruction):
        starts = _sample_pair_starts(
            train_data,
            config.batch_size,
            None,
            generator,
        )
        source_batch = train_data.index_select(1, starts).T.contiguous()
        optimizer.zero_grad(set_to_none=True)
        reconstruction_loss = torch.mean(
            (model(source_batch) - source_batch) ** 2
        )
        reconstruction_loss.backward()
        clip_and_step(
            optimizer,
            model.parameters(),
            max_norm=config.grad_clip_max_norm,
            phase="reconstruction",
        )
        value = float(reconstruction_loss.detach())
        reconstruction_losses.append(value)
        if detector.update(value):
            break
    return reconstruction_losses


def _two_stage_epoch(
    model: KoopmanAutoencoder,
    optimizers: Optimizers,
    settings: TrainSettings,
    data: EpochData,
) -> dict[str, float]:
    train_data, generator, pools = data.train_data, data.generator, data.pools
    config, loss = settings.training, settings.loss
    reconstruction_optimizer = optimizers.reconstruction
    evolution_optimizer = optimizers.evolution
    if reconstruction_optimizer is None or evolution_optimizer is None:
        raise RuntimeError(
            "two_stage training requires reconstruction and evolution "
            "optimizers"
        )
    model.train()
    reconstruction_losses = _reconstruction_phase(
        model, train_data, reconstruction_optimizer, config, generator
    )
    evolution_losses, pair_mses = [], []
    detector = InnerLoopPlateauDetector(
        window=config.plateau_window,
        improvement_rate_tolerance=(
            config.plateau_relative_improvement_rate_tolerance
        ),
    )
    model.eval()
    model.decoder.requires_grad_(False)
    try:
        for _ in range(config.max_updates_evolution):
            estimation_batch, held_out_batch = _sample_batches(
                train_data, config, pools, generator
            )
            evolution_optimizer.zero_grad(set_to_none=True)
            value, fit, z_x_prime = _evolution_training_loss(
                model,
                estimation_batch,
                held_out_batch,
                loss,
                config.operator_regularization,
            )
            value.backward()
            clip_and_step(
                evolution_optimizer,
                model.encoder.parameters(),
                max_norm=config.grad_clip_max_norm,
                phase="evolution",
            )
            scalar = float(value.detach())
            evolution_losses.append(scalar)
            with torch.no_grad():
                pair_mses.append(
                    float(
                        _pair_mse(
                            model, fit, z_x_prime, held_out_batch
                        ).detach()
                    )
                )
            if detector.update(scalar):
                break
    finally:
        model.decoder.requires_grad_(True)
        model.train()
    reconstruction_mean = _mean_or_zero(reconstruction_losses)
    evolution_mean = _mean_or_zero(evolution_losses)
    return {
        "train_evolution_loss": evolution_mean,
        "train_pair_mse": _mean_or_zero(pair_mses),
        "train_reconstruction_mse": reconstruction_mean,
        "train_loss": reconstruction_mean + evolution_mean,
    }


def _joint_epoch(
    model: KoopmanAutoencoder,
    optimizers: Optimizers,
    settings: TrainSettings,
    data: EpochData,
) -> dict[str, float]:
    train_data, generator, pools = data.train_data, data.generator, data.pools
    config, loss = settings.training, settings.loss
    optimizer = optimizers.joint
    if optimizer is None:
        raise RuntimeError("joint training requires a joint optimizer")
    model.train()
    evolution_losses, pair_mses = [], []
    reconstruction_losses, joint_losses = [], []
    for _ in range(config.max_updates_joint):
        estimation_batch, held_out_batch = _sample_batches(
            train_data, config, pools, generator
        )
        optimizer.zero_grad(set_to_none=True)
        value, fit, z_x_prime = _evolution_training_loss(
            model,
            estimation_batch,
            held_out_batch,
            loss,
            config.operator_regularization,
        )
        with torch.no_grad():
            pair_mse = _pair_mse(model, fit, z_x_prime, held_out_batch)
        reconstruction_loss = torch.mean(
            (model(held_out_batch) - held_out_batch) ** 2
        )
        combined_loss = joint_loss(
            value.to(reconstruction_loss.device),
            reconstruction_loss,
            weights=config.joint_loss_weights,
        )
        combined_loss.backward()
        clip_and_step(
            optimizer,
            model.parameters(),
            max_norm=config.grad_clip_max_norm,
            phase="joint",
        )
        evolution_losses.append(float(value.detach()))
        pair_mses.append(float(pair_mse.detach()))
        reconstruction_losses.append(float(reconstruction_loss.detach()))
        joint_losses.append(float(combined_loss.detach()))
    return {
        "train_evolution_loss": _mean_or_zero(evolution_losses),
        "train_pair_mse": _mean_or_zero(pair_mses),
        "train_reconstruction_mse": _mean_or_zero(reconstruction_losses),
        "train_loss": _mean_or_zero(joint_losses),
    }


def validation_metrics(
    model: KoopmanAutoencoder,
    train_data: torch.Tensor,
    valid_data: torch.Tensor,
    *,
    settings: TrainSettings,
) -> dict[str, float | int]:
    loss, evaluation = settings.loss, settings.evaluation
    operator_regularization = settings.training.operator_regularization
    if (
        train_data.shape[1] < MINIMUM_PAIR_STATES
        or valid_data.shape[1] < MINIMUM_PAIR_STATES
    ):
        raise ValueError(
            "training and validation data must each contain at least two times"
        )
    model.eval()
    with torch.inference_mode():
        train_latent = model.encode(train_data.T).T
        valid_latent = model.encode(valid_data.T).T
        fit = rank_constrained_operator(
            train_latent[:, :-1],
            train_latent[:, 1:],
            regularization=operator_regularization,
            dtype=train_latent.dtype,
        )
        value = evolution_loss(
            fit,
            valid_latent[:, :-1],
            valid_latent[:, 1:],
            loss,
        )
        valid_samples = valid_data.T
        reconstruction = model(valid_samples)
        reconstruction_mse = torch.mean((reconstruction - valid_samples) ** 2)
        propagated = propagate_latent(fit.a_hat, valid_latent[:, :-1])
        prediction = model.decode(
            propagated.T.to(
                device=valid_samples.device,
                dtype=valid_samples.dtype,
            ),
        )
        prediction = prediction.to(dtype=torch.float64)
        reference = valid_data[:, 1:].T.to(dtype=torch.float64)
        vrmse = validation_vrmse(
            prediction,
            reference,
            eps=evaluation.vrmse_eps,
        )
    return {
        "valid_reconstruction_mse": float(reconstruction_mse),
        "valid_evolution_loss": float(value),
        "valid_vrmse": float(vrmse),
        "retained_rank": fit.rank,
    }


def train(
    model: KoopmanAutoencoder,
    dataset: Dataset,
    settings: TrainSettings,
    *,
    runtime: TrainRuntime,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> TrainResult:
    """Select the strictly best one-step validation VRMSE checkpoint."""
    loss, evaluation = settings.loss, settings.evaluation
    optimizer_config, scheduler_config = settings.optimizer, settings.scheduler
    training_config = settings.training
    seed, device = runtime.seed, runtime.device
    network_device = torch.device(device)
    torch.manual_seed(seed)
    network_dtype = (
        torch.float32
        if training_config.resolved_precision(loss) == "float32"
        else torch.float64
    )
    model.to(device=network_device, dtype=network_dtype)
    train_data = torch.as_tensor(
        dataset.train, dtype=network_dtype, device=network_device
    )
    valid_data = torch.as_tensor(
        dataset.valid, dtype=network_dtype, device=network_device
    )
    if (
        train_data.shape[0] != model.input_dim
        or valid_data.shape[0] != model.input_dim
    ):
        raise ValueError(
            "dataset feature dimension does not match "
            "the model input dimension"
        )
    optimizers = build_optimizers(
        model,
        optimizer_config,
        scheduler_config,
        training_config,
    )
    selector = BestCheckpointSelector()
    pools = _estimation_held_out_pools(
        train_data,
        training_config,
        evaluation,
    )
    for epoch_index in range(training_config.epochs):
        generator = torch.Generator(device=network_device).manual_seed(
            seed + epoch_index,
        )
        run_epoch = (
            _two_stage_epoch
            if training_config.procedure == "two_stage"
            else _joint_epoch
        )
        training_record = run_epoch(
            model,
            optimizers,
            settings,
            EpochData(train_data, generator, pools),
        )
        validation_record = validation_metrics(
            model,
            train_data,
            valid_data,
            settings=settings,
        )
        epoch = epoch_index + 1
        record: EpochRecord = {
            "epoch": epoch,
            **training_record,
            **validation_record,
        }
        if on_epoch is not None:
            on_epoch(record)
        selector.update(
            epoch=epoch,
            metric=float(validation_record["valid_vrmse"]),
            model=model,
        )
        optimizers.step_schedulers()
    if selector.best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    return TrainResult(
        best_state=selector.best_state, best_epoch=selector.best_epoch
    )


def load_pretrained_weights(
    model: KoopmanAutoencoder,
    path: Path,
    *,
    settings: dict,
    precision: str,
) -> None:
    validate_checkpoint(path, settings, precision)
    state = torch.load(path, map_location="cpu", weights_only=False)
    dtype = next(model.parameters()).dtype
    if any(value.dtype != dtype for value in state.values()):
        raise TypeError(
            f"pretrained checkpoint at {path} has the wrong precision"
        )
    model.load_state_dict(state, strict=True)


def train_run(config: dict, *, device: str) -> Path:
    directory = run_directory(config)
    with run_lock(directory):
        if completed(config):
            print(f"skip {directory}: config SHA-256 matches", flush=True)
            return directory
        atomic_write(
            directory / "config.yaml", yaml.safe_dump(config, sort_keys=False)
        )
        data_config = load_data_config(config)
        if config["model"]["name"] != "koopman_autoencoder":
            return _train_baseline(
                config, data_config, directory, device=device
            )
        evaluation = load_evaluation_config(data_config)
        dataset = load_dataset(config["system"], config["seed"], data_config)
        loss = LossConfig(**config["loss"])
        precision = resolved_precision(config)
        dtype = getattr(torch, precision)
        torch.manual_seed(config["seed"])
        model_config = build_model_config(config, dataset.train.shape[0])
        model = KoopmanAutoencoder(
            dataset.train.shape[0], model_config, dtype=dtype
        )
        if config["pretrained"]["enabled"]:
            settings = pretraining_settings(config)
            path = pretrained_path(
                Path(settings["output_dir"]),
                config["system"],
                config["latent_multiplier"],
                precision,
            )
            load_pretrained_weights(
                model, path, settings=settings, precision=precision
            )
        with (directory / "epochs.jsonl").open("w") as epoch_file:

            def on_epoch(record: dict) -> None:
                epoch_file.write(json.dumps(record, allow_nan=False) + "\n")
                epoch_file.flush()
                print(
                    f"epoch {record['epoch']}: validation "
                    f"VRMSE={record['valid_vrmse']:.8g}",
                    flush=True,
                )

            training_config = TrainingConfig(**config["training"])
            result = train(
                model,
                dataset,
                TrainSettings(
                    loss,
                    OptimizerConfig(**config["optimizer"]),
                    SchedulerConfig(**config["scheduler"]),
                    training_config,
                    evaluation,
                ),
                runtime=TrainRuntime(config["seed"], device),
                on_epoch=on_epoch,
            )
        model.load_state_dict(result.best_state)
        trained = fit_rollout_operators(
            model,
            dataset,
            OperatorConfig(loss.type, training_config.operator_regularization),
            device=device,
            data_config=data_config,
        )
        checkpoint = {
            "model_config": asdict(model_config),
            "input_dim": dataset.train.shape[0],
            "network_dtype": precision,
            "best_epoch": result.best_epoch,
            "encoder_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.encoder.state_dict().items()
            },
            "decoder_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.decoder.state_dict().items()
            },
            "rollout_operator": (
                trained.rollout_operator.detach().cpu().clone()
            ),
            "full_rank_operator": (
                trained.full_rank_operator.detach().cpu().clone()
            ),
            "measured_dim": dataset.measured_dim,
            "num_delay_samples": dataset.num_delay_samples,
        }
        atomic_torch_save(directory / "model.pt", checkpoint)
        payload = {
            "config_sha256": config_sha256(config),
            "best_epoch": result.best_epoch,
            "latent_dim": model_config.latent_dim,
            "data_config": data_config,
        }
        atomic_write(
            directory / "training.json",
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
        )
        return directory


def _train_baseline(
    config: dict,
    data_config: dict,
    directory: Path,
    *,
    device: str,
) -> Path:

    name = config["model"]["name"]
    if (
        name in ("neural_ode", "consistent_kae")
        and device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "--device cuda was requested but CUDA is unavailable"
        )
    evaluation = load_evaluation_config(data_config)
    dataset = load_dataset(config["system"], int(config["seed"]), data_config)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    baseline = build_baseline(
        name,
        baseline_params(config),
        effective_latent_dim(config, dataset.train.shape[0]),
        int(config["seed"]),
        device,
    )
    baseline.fit(dataset, evaluation)
    rank = baseline.prepare_rollout(dataset, evaluation.rollout_svd_rank)
    with (directory / "epochs.jsonl").open("w", encoding="utf-8") as stream:
        for row in baseline.baseline.training_logs:
            stream.write(json.dumps(row) + "\n")
    baseline.save(directory / "model.pt")
    payload = {
        "config_sha256": config_sha256(config),
        "method_params": baseline.fit_params,
        "device": str(baseline.baseline.device),
        "rollout_rank": rank,
        "best_epoch": baseline.baseline.best_epoch,
        "latent_dim": baseline.baseline.latent_dim,
        "data_config": data_config,
    }
    atomic_write(
        directory / "training.json", json.dumps(payload, indent=2) + "\n"
    )
    print(f"Wrote {directory}", flush=True)
    return directory
