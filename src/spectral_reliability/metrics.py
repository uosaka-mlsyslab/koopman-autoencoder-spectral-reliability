from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from spectral_reliability.data.dataset import Dataset
from spectral_reliability.evaluation_config import EvaluationConfig


def validation_vrmse(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Average the variance-normalized RMSE (VRMSE) over samples.
    Each sample's mean squared error is divided by the variance of its
    reference components before the square root.
    """
    squared_error = torch.mean((prediction - reference) ** 2, dim=1)
    variance = torch.var(reference, dim=1, unbiased=False)
    return torch.mean(torch.sqrt(squared_error / (variance + eps)))


def sample_start_indices(
    length: int,
    horizon: int,
    config: EvaluationConfig,
) -> np.ndarray:
    available = length - horizon
    if available < config.num_rollout_starts:
        raise ValueError(
            f"{available} starts cannot fill the required "
            f"{config.num_rollout_starts} rollouts"
        )
    return np.asarray(
        np.random.default_rng(config.rollout_seed).choice(
            available, config.num_rollout_starts, replace=False
        ),
        dtype=np.int64,
    )


@dataclass(frozen=True)
class RolloutScoring:
    horizon: int
    windows: tuple[tuple[int, int], ...]
    physical_offset: int
    sigma: torch.Tensor
    config: EvaluationConfig


def _score_rollout_batch(
    predictions: Iterator,
    trajectory: np.ndarray,
    starts: np.ndarray,
    scoring: RolloutScoring,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return windowed VRMSE means and VPT prefix counts for one batch.

    VRMSE uses the component variance at each reference step. VPT counts
    the initial consecutive steps whose sigma-normalized RMSE is below
    each epsilon; the caller converts the mean count to Lyapunov times.
    """
    horizon, windows = scoring.horizon, scoring.windows
    physical_offset, sigma = scoring.physical_offset, scoring.sigma
    device, config = sigma.device, scoring.config
    window_vrmse_sums = [
        torch.zeros(starts.size, dtype=torch.float64, device=device)
        for _ in windows
    ]
    vpt_valid_prefix = [
        torch.ones(starts.size, dtype=torch.bool, device=device)
        for _ in config.vpt_epsilons
    ]
    vpt_step_counts = [
        torch.zeros(starts.size, dtype=torch.int64, device=device)
        for _ in config.vpt_epsilons
    ]
    produced_steps = 0
    for index, array in enumerate(predictions, start=1):
        if index > horizon:
            raise ValueError(
                "predictor yielded more than the requested horizon"
            )
        produced_steps = index
        prediction = torch.as_tensor(array, dtype=torch.float64, device=device)
        reference = torch.as_tensor(
            trajectory[:, starts + index].T,
            dtype=torch.float64,
            device=device,
        )
        if prediction.shape != reference.shape:
            raise ValueError(
                f"prediction {prediction.shape} differs from reference "
                f"{reference.shape}",
            )
        active_windows = [
            slot
            for slot, (first, last) in enumerate(windows)
            if first <= index <= last
        ]
        if active_windows:
            squared_error = torch.mean((prediction - reference) ** 2, dim=1)
            component_variance = torch.var(reference, dim=1, unbiased=False)
            vrmse_per_start = torch.sqrt(
                squared_error / (component_variance + config.vrmse_eps)
            )
            for slot in active_windows:
                window_vrmse_sums[slot] += vrmse_per_start
        normalized_error = torch.sqrt(
            torch.mean(
                (
                    (
                        prediction[:, physical_offset:]
                        - reference[:, physical_offset:]
                    )
                    / sigma
                )
                ** 2,
                dim=1,
            )
        )
        for slot, epsilon in enumerate(config.vpt_epsilons):
            still_valid = (
                vpt_valid_prefix[slot]
                & torch.isfinite(normalized_error)
                & (normalized_error < epsilon)
            )
            vpt_step_counts[slot] += still_valid
            vpt_valid_prefix[slot] = still_valid
    if produced_steps != horizon:
        raise ValueError("predictor yielded fewer than the requested horizon")
    return (
        [
            total / (last - first + 1)
            for total, (first, last) in zip(
                window_vrmse_sums,
                windows,
                strict=True,
            )
        ],
        [count.to(torch.float64) for count in vpt_step_counts],
    )


def evaluate_rollouts(
    predictor,
    dataset: Dataset,
    config: EvaluationConfig,
) -> dict[str, float]:
    """Reduce test-rollout VRMSE and VPT in float64 with
    predictor-specific batching.
    """
    trajectory = np.asarray(dataset.test, dtype=np.float64)
    horizon = math.ceil(config.horizon_lt * config.steps_per_lt)
    starts = sample_start_indices(trajectory.shape[1], horizon, config)
    windows = tuple(
        config.window_steps(window) for window in config.vrmse_windows_lt
    )
    physical_offset = trajectory.shape[0] - dataset.measured_dim
    sigma_array = np.maximum(
        trajectory[physical_offset:].std(axis=1, keepdims=True),
        config.vpt_sigma_eps,
    ).T
    if not np.isfinite(sigma_array).all():
        raise ValueError("test trajectory contains non-finite physical values")
    iter_rollout_tensor = getattr(predictor, "iter_rollout_tensor", None)
    if iter_rollout_tensor is not None:
        device = predictor.rollout_operator.device
        chunk_size = config.num_rollout_starts
    else:
        device = torch.device("cpu")
        chunk_size = predictor.rollout_batch_size
    sigma = torch.as_tensor(sigma_array, dtype=torch.float64, device=device)
    scoring = RolloutScoring(horizon, windows, physical_offset, sigma, config)
    vrmse_scores = [[] for _ in windows]
    vpt_step_counts = [[] for _ in config.vpt_epsilons]
    for offset in range(0, config.num_rollout_starts, chunk_size):
        chunk = starts[offset : offset + chunk_size]
        initial = trajectory[:, chunk].T
        if iter_rollout_tensor is not None:
            predictions = iter_rollout_tensor(initial, horizon)
        else:
            history = int(getattr(predictor, "history_steps", 0))
            if history:
                histories = [
                    trajectory[
                        :, max(0, int(start) - history) : int(start) + 1
                    ].T
                    for start in chunk
                ]
                predictions = predictor.iter_rollout_array(
                    initial,
                    horizon,
                    histories=histories,
                )
            else:
                predictions = predictor.iter_rollout_array(initial, horizon)
        chunk_scores, chunk_counts = _score_rollout_batch(
            predictions,
            trajectory,
            chunk,
            scoring,
        )
        for values, chunk_values in zip(
            vrmse_scores,
            chunk_scores,
            strict=True,
        ):
            values.append(chunk_values)
        for values, chunk_values in zip(
            vpt_step_counts,
            chunk_counts,
            strict=True,
        ):
            values.append(chunk_values)
    vrmse = [float(torch.cat(values).mean().item()) for values in vrmse_scores]
    vpt = [
        float(torch.cat(values).mean().item()) / config.steps_per_lt
        for values in vpt_step_counts
    ]
    return dict(zip(config.metric_columns, (*vpt, *vrmse), strict=True))
