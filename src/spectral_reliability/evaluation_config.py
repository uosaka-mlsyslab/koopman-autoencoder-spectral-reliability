from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from spectral_reliability.config_shared import construct_config
from spectral_reliability.validation import (
    integer_at_least,
    nonnegative_finite,
    positive_finite,
)

WINDOW_ENDPOINTS = 2


def _format_number(value: float) -> str:
    return format(Decimal(str(value)).normalize(), "f")


@dataclass(frozen=True)
class EvaluationConfig:
    steps_per_lt: float
    horizon_lt: float
    num_rollout_starts: int
    vpt_epsilons: tuple[float, ...]
    vrmse_windows_lt: tuple[tuple[float, float], ...]
    vrmse_eps: float
    vpt_sigma_eps: float
    rollout_seed: int
    rollout_svd_rank: int | float

    def __post_init__(self) -> None:
        if not positive_finite(self.steps_per_lt) or not positive_finite(
            self.horizon_lt
        ):
            raise ValueError(
                "evaluation steps_per_lt and horizon_lt must be finite "
                "and positive"
            )
        if not integer_at_least(
            self.num_rollout_starts, 1
        ) or not integer_at_least(self.rollout_seed, 0):
            raise ValueError(
                "num_rollout_starts must be positive and rollout_seed "
                "non-negative integers"
            )
        if not positive_finite(self.vrmse_eps) or not positive_finite(
            self.vpt_sigma_eps
        ):
            raise ValueError(
                "evaluation vrmse_eps and vpt_sigma_eps must be finite "
                "and positive"
            )
        if not (
            (
                integer_at_least(self.rollout_svd_rank, -1)
                and self.rollout_svd_rank != 0
            )
            or (
                isinstance(self.rollout_svd_rank, float)
                and positive_finite(self.rollout_svd_rank)
                and self.rollout_svd_rank <= 1
            )
        ):
            raise ValueError(
                "evaluation.rollout_svd_rank must be -1, a positive integer, "
                "or a float in (0, 1]",
            )
        if (
            not isinstance(self.vpt_epsilons, (list, tuple))
            or not self.vpt_epsilons
            or not all(positive_finite(value) for value in self.vpt_epsilons)
            or len(set(self.vpt_epsilons)) != len(self.vpt_epsilons)
        ):
            raise ValueError(
                "vpt_epsilons must contain unique finite positive thresholds"
            )
        if (
            not isinstance(self.vrmse_windows_lt, (list, tuple))
            or not self.vrmse_windows_lt
        ):
            raise ValueError(
                "vrmse_windows_lt must contain at least one "
                "[start, stop] window"
            )
        windows = []
        for window in self.vrmse_windows_lt:
            if (
                not isinstance(window, (list, tuple))
                or len(window) != WINDOW_ENDPOINTS
            ):
                raise ValueError(
                    "each vrmse window must be a [start, stop] window"
                )
            low, high = window
            if (
                not nonnegative_finite(low)
                or not positive_finite(high)
                or not low < high <= self.horizon_lt
            ):
                raise ValueError(
                    "vrmse windows must satisfy "
                    "0 <= start < stop <= horizon_lt"
                )
            first, last = self.window_steps(window)
            if first > last:
                raise ValueError("vrmse window contains no output steps")
            windows.append(tuple(window))
        if len(set(windows)) != len(windows):
            raise ValueError("vrmse_windows_lt must be unique")

    def window_steps(self, window: tuple[float, float]) -> tuple[int, int]:
        """Steps t with start < t/nu <= stop, the window (start, stop]
        LT.
        """
        steps_per_lt = Decimal(str(self.steps_per_lt))
        first = (
            int(
                (Decimal(str(window[0])) * steps_per_lt).to_integral_value(
                    rounding=ROUND_FLOOR,
                )
            )
            + 1
        )
        last = int(
            (Decimal(str(window[1])) * steps_per_lt).to_integral_value(
                rounding=ROUND_FLOOR,
            )
        )
        return first, last

    @property
    def metric_columns(self) -> tuple[str, ...]:
        return (
            *(f"vpt_{_format_number(eps)}_mean" for eps in self.vpt_epsilons),
            *(
                f"vrmse_{_format_number(low)}-{_format_number(high)}_mean"
                for low, high in self.vrmse_windows_lt
            ),
        )


def load_evaluation_config(config: dict) -> EvaluationConfig:
    return construct_config(
        EvaluationConfig,
        config["evaluation"],
        "evaluation",
    )
