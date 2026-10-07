import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from spectral_reliability.validation import (
    integer_at_least,
    nonnegative_finite,
    positive_finite,
)

BETA_COUNT = 2
JOINT_LOSS_WEIGHT_COUNT = 2


@dataclass(frozen=True)
class LossConfig:
    type: Literal["latent_prediction", "spectral_residual"]
    operator_norm_weight: float
    residual_backward: Literal["custom", "autograd"]

    def __post_init__(self) -> None:
        if self.type not in ("latent_prediction", "spectral_residual"):
            raise ValueError(
                "loss type must be latent_prediction or spectral_residual"
            )
        if not nonnegative_finite(self.operator_norm_weight):
            raise ValueError(
                "operator_norm_weight must be finite and non-negative"
            )
        if self.residual_backward not in ("custom", "autograd"):
            raise ValueError("residual_backward must be custom or autograd")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    learning_rate_reconstruction: float
    learning_rate_evolution: float
    learning_rate_joint: float

    def __post_init__(self) -> None:
        if self.name not in ("adamw", "adam"):
            raise ValueError("optimizer name must be adamw or adam")
        if (
            not isinstance(self.betas, (tuple, list))
            or len(self.betas) != BETA_COUNT
            or not all(
                nonnegative_finite(beta) and beta < 1 for beta in self.betas
            )
        ):
            raise ValueError(
                "optimizer betas must contain two values in [0, 1)"
            )
        if not positive_finite(self.eps) or not nonnegative_finite(
            self.weight_decay
        ):
            raise ValueError(
                "optimizer eps must be positive and weight_decay "
                "non-negative, both finite"
            )
        if not all(
            positive_finite(rate)
            for rate in (
                self.learning_rate_reconstruction,
                self.learning_rate_evolution,
                self.learning_rate_joint,
            )
        ):
            raise ValueError("learning rates must be finite and positive")


@dataclass(frozen=True)
class SchedulerConfig:
    t_max: int
    warmup_epochs: int
    min_lr: float

    def __post_init__(self) -> None:
        if not integer_at_least(self.t_max, 1) or not integer_at_least(
            self.warmup_epochs, 0
        ):
            raise ValueError(
                "scheduler t_max must be positive and warmup_epochs "
                "non-negative integers"
            )
        if not nonnegative_finite(self.min_lr):
            raise ValueError(
                "scheduler min_lr must be finite and non-negative"
            )


@dataclass(frozen=True)
class TrainingConfig:
    procedure: Literal["two_stage", "joint"]
    precision: Literal["auto", "float32", "float64"]
    epochs: int
    batch_size: int
    max_updates_reconstruction: int
    max_updates_evolution: int
    max_updates_joint: int
    plateau_window: int
    plateau_relative_improvement_rate_tolerance: float
    disjoint_pools: bool
    pool_block_lt: float
    grad_clip_max_norm: float
    joint_loss_weights: tuple[float, float]
    operator_regularization: float

    def __post_init__(self) -> None:
        if self.procedure not in ("two_stage", "joint"):
            raise ValueError("training procedure must be two_stage or joint")
        if self.precision not in ("auto", "float32", "float64"):
            raise ValueError(
                "training precision must be auto, float32 or float64"
            )
        for name in (
            "epochs",
            "batch_size",
            "max_updates_evolution",
            "max_updates_joint",
            "plateau_window",
        ):
            if not integer_at_least(getattr(self, name), 1):
                raise ValueError(f"training {name} must be a positive integer")
        if not integer_at_least(self.max_updates_reconstruction, 0):
            raise ValueError(
                "max_updates_reconstruction must be a non-negative integer"
            )
        if not nonnegative_finite(
            self.plateau_relative_improvement_rate_tolerance
        ) or not nonnegative_finite(self.operator_regularization):
            raise ValueError(
                "plateau_relative_improvement_rate_tolerance and "
                "operator_regularization must be finite and non-negative"
            )
        if not positive_finite(self.pool_block_lt):
            raise ValueError("pool_block_lt must be finite and positive")
        if type(self.disjoint_pools) is not bool:
            raise ValueError("disjoint_pools must be a boolean")
        if (
            not isinstance(self.grad_clip_max_norm, Real)
            or isinstance(self.grad_clip_max_norm, bool)
            or (
                math.isnan(self.grad_clip_max_norm)
                or self.grad_clip_max_norm <= 0
            )
        ):
            raise ValueError(
                "grad_clip_max_norm must be positive "
                "(infinity disables clipping)"
            )
        if (
            not isinstance(self.joint_loss_weights, (tuple, list))
            or len(self.joint_loss_weights) != JOINT_LOSS_WEIGHT_COUNT
            or not all(
                nonnegative_finite(weight)
                for weight in self.joint_loss_weights
            )
            or not any(weight > 0 for weight in self.joint_loss_weights)
        ):
            raise ValueError(
                "joint_loss_weights must contain two finite non-negative "
                "values, at least one positive"
            )

    def resolved_precision(self, loss: LossConfig) -> str:
        precision = self.precision
        if precision == "auto":
            precision = (
                "float32" if loss.type == "latent_prediction" else "float64"
            )
        if loss.type == "spectral_residual" and precision != "float64":
            raise ValueError("spectral_residual requires float64 precision")
        return precision
