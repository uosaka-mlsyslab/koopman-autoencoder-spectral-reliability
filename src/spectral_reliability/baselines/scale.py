from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from spectral_reliability.config_shared import atomic_torch_save
from spectral_reliability.data.dataset import Dataset

from .base import Baseline


def feature_std(train: np.ndarray) -> np.ndarray:
    sigma = np.asarray(train, dtype=np.float64).std(axis=1)
    if not np.isfinite(sigma).all():
        raise ValueError("Training split contains non-finite values")
    return np.maximum(sigma, 1e-8)


def scaled_dataset(dataset: Dataset, sigma: np.ndarray) -> Dataset:
    column = np.asarray(sigma, dtype=np.float64)[:, None]
    return replace(
        dataset,
        train=np.asarray(dataset.train, dtype=np.float64) / column,
        valid=np.asarray(dataset.valid, dtype=np.float64) / column,
        test=np.asarray(dataset.test, dtype=np.float64) / column,
    )


class ScaledBaseline:
    """Scale fitted predictors and their rollout histories to original
    coordinates.
    """

    def __init__(self, baseline: Baseline) -> None:
        self.baseline = baseline
        self.sigma: np.ndarray | None = None

    @property
    def history_steps(self) -> int:
        return int(self.baseline.history_steps)

    @property
    def rollout_batch_size(self) -> int:
        return int(self.baseline.params["rollout_batch_size"])

    @property
    def fit_params(self) -> dict[str, Any]:
        return {
            **self.baseline.fit_params,
            "scale_inputs": self.baseline.params["scale_inputs"],
            "feature_std": None if self.sigma is None else self.sigma.tolist(),
        }

    def _require_sigma(self) -> np.ndarray:
        if self.sigma is None:
            raise RuntimeError("Fit the baseline before predicting")
        return self.sigma

    def fit(self, dataset: Dataset, evaluation) -> None:
        self.sigma = (
            feature_std(dataset.train)
            if self.baseline.params["scale_inputs"]
            else np.ones(dataset.train.shape[0], dtype=np.float64)
        )
        self.baseline.fit(scaled_dataset(dataset, self.sigma), evaluation)

    def iter_rollout_array(
        self,
        initial_states: np.ndarray,
        steps: int,
        *,
        histories: Sequence[np.ndarray] | None = None,
    ) -> Iterator[np.ndarray]:
        sigma = self._require_sigma()
        inputs = np.asarray(initial_states, dtype=np.float64) / sigma
        kwargs = (
            {}
            if histories is None
            else {
                "histories": [
                    np.asarray(history, dtype=np.float64) / sigma
                    for history in histories
                ],
            }
        )
        for prediction in self.baseline.iter_rollout_array(
            inputs, int(steps), **kwargs
        ):
            yield np.asarray(prediction, dtype=np.float64) * sigma

    def prepare_rollout(
        self,
        dataset: Dataset,
        svd_rank: float,
    ) -> int | None:
        """Finalize persisted rollout state, including configured linear
        projections.
        """
        return self.baseline.prepare_rollout(
            scaled_dataset(dataset, self._require_sigma()), svd_rank
        )

    def save(self, path: Path) -> None:
        atomic_torch_save(
            path,
            {
                "method": self.baseline.method,
                "latent_dim": self.baseline.latent_dim,
                "seed": self.baseline.seed,
                "method_params": self.fit_params,
                "state": self.baseline.state(),
                "history_steps": self.history_steps,
                "best_epoch": self.baseline.best_epoch,
                "best_validation_vrmse": self.baseline.best_validation_vrmse,
            },
        )

    def load_state(self, checkpoint: dict) -> None:
        params = dict(checkpoint["method_params"])
        self.sigma = np.asarray(params.pop("feature_std"), dtype=np.float64)
        self.baseline.load_state(checkpoint["state"], params)
        self.baseline.history_steps = int(checkpoint["history_steps"])
        self.baseline.best_epoch = checkpoint["best_epoch"]
        self.baseline.best_validation_vrmse = checkpoint[
            "best_validation_vrmse"
        ]
