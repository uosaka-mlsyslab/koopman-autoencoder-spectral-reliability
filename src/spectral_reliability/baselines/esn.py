from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import torch
from reservoirpy.activationsfunc import get_function
from reservoirpy.nodes import Reservoir, Ridge

from spectral_reliability.data.dataset import Dataset
from spectral_reliability.metrics import validation_vrmse
from spectral_reliability.validation import (
    integer_at_least,
    nonnegative_finite,
)

from .base import Baseline, cpu_tensor

MATRIX_DIMENSIONS = 2
MINIMUM_PAIR_STATES = 2


class ESN(Baseline):
    """Fit reservoirpy's Reservoir and Ridge, then use dense saved
    arrays.
    """

    method = "esn"
    parameter_types: ClassVar[dict[str, Any]] = {
        **Baseline.parameter_types,
        "sr": float,
        "lr": float,
        "input_scaling": float,
        "input_connectivity": float,
        "rc_connectivity": float,
        "ridge": float,
        "fit_bias": bool,
        "activation": str,
        "warmup": int,
        "history_steps": (int, str),
    }
    parameter_checks: ClassVar[dict[str, Any]] = {
        "lr": lambda value: nonnegative_finite(value) and value <= 1,
        "input_connectivity": (
            lambda value: nonnegative_finite(value) and value <= 1
        ),
        "rc_connectivity": (
            lambda value: nonnegative_finite(value) and value <= 1
        ),
        "warmup": lambda value: integer_at_least(value, 0),
        "history_steps": (
            lambda value: (
                value == "num_delay_samples" or integer_at_least(value, 0)
            )
        ),
    }

    def fit(self, dataset: Dataset, evaluation) -> None:

        params = self.params
        self.history_steps = (
            max(0, dataset.num_delay_samples)
            if params["history_steps"] == "num_delay_samples"
            else int(params["history_steps"])
        )
        train = dataset.train.T
        valid = dataset.valid.T
        if min(len(train), len(valid)) < MINIMUM_PAIR_STATES:
            raise ValueError(
                "ESN needs at least two train and validation states"
            )
        self._reservoir = Reservoir(
            units=self.latent_dim,
            lr=params["lr"],
            sr=params["sr"],
            input_scaling=params["input_scaling"],
            input_connectivity=params["input_connectivity"],
            rc_connectivity=params["rc_connectivity"],
            activation=params["activation"],
            input_dim=train.shape[1],
            seed=self.seed,
        )
        self._readout = Ridge(
            ridge=params["ridge"],
            fit_bias=params["fit_bias"],
            output_dim=train.shape[1],
        )
        model = self._reservoir >> self._readout
        warmup = min(int(params["warmup"]), max(0, len(train) - 2))
        model.fit(train[:-1], train[1:], warmup=warmup)
        model.reset()
        prediction = np.asarray(model.run(valid[:-1]), dtype=float)
        criterion = float(
            validation_vrmse(
                torch.tensor(prediction, dtype=torch.float32),
                torch.tensor(valid[1:], dtype=torch.float32),
                evaluation.vrmse_eps,
            ).item()
        )
        self.best_epoch = 1
        self.best_validation_vrmse = criterion
        self.fit_params = {**params, "warmup": warmup}
        self.training_logs.append(
            {
                "epoch": 1,
                "train_loss": None,
                "valid_vrmse": criterion,
                "is_best": True,
            }
        )

    def state(self) -> dict[str, Any]:
        reservoir, readout = self._reservoir, self._readout
        return {
            name: cpu_tensor(value)
            for name, value in {
                "W": reservoir.W,
                "Win": reservoir.Win,
                "bias": reservoir.bias,
                "Wout": readout.Wout,
                "readout_bias": readout.bias,
                "lr": reservoir.lr,
            }.items()
        }

    def prepare_rollout(self, dataset: Dataset, svd_rank: float) -> None:
        """Use the dense saved arrays and checkpoint-rounded leak rate
        for rollouts.
        """
        state = self.state()
        reservoir, readout = self._reservoir, self._readout
        reservoir.W = np.asarray(state["W"], dtype=np.float64)
        reservoir.Win = np.asarray(state["Win"], dtype=np.float64)
        reservoir.bias = np.asarray(state["bias"], dtype=np.float64)
        # Published scores use the leak rate stored as float32 in the
        # saved state.
        reservoir.lr = float(np.asarray(state["lr"], dtype=np.float64))
        readout.Wout = np.asarray(state["Wout"], dtype=np.float64)
        readout.bias = np.asarray(state["readout_bias"], dtype=np.float64)

    def load_state(
        self,
        state: dict[str, Any],
        fit_params: dict[str, Any],
    ) -> None:

        self._reservoir = SimpleNamespace(
            units=self.latent_dim,
            activation=get_function(self.params["activation"]),
            W=np.asarray(state["W"], dtype=np.float64),
            Win=np.asarray(state["Win"], dtype=np.float64),
            bias=np.asarray(state["bias"], dtype=np.float64),
            lr=float(np.asarray(state["lr"], dtype=np.float64)),
        )
        self._readout = SimpleNamespace(
            Wout=np.asarray(state["Wout"], dtype=np.float64),
            bias=np.asarray(state["readout_bias"], dtype=np.float64),
        )
        self.fit_params = dict(fit_params)

    def _reservoir_step(
        self,
        state: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        reservoir = self._reservoir
        activated = reservoir.activation(
            (reservoir.W @ state.T).T
            + (reservoir.Win @ inputs.T).T
            + reservoir.bias,
        )
        return (1 - reservoir.lr) * state + reservoir.lr * activated

    @torch.inference_mode()
    def iter_rollout_array(
        self,
        initial_states: np.ndarray,
        steps: int,
        *,
        histories: Sequence[np.ndarray] | None = None,
    ) -> Iterator[np.ndarray]:
        current = np.asarray(initial_states, dtype=np.float64)
        if int(steps) <= 0:
            return
        state = np.zeros(
            (len(current), self._reservoir.units), dtype=np.float64
        )
        if self.history_steps:
            if histories is None or len(histories) != len(current):
                raise ValueError(
                    "ESN needs one observed history per rollout start"
                )
            histories = [
                np.asarray(history, dtype=np.float64) for history in histories
            ]
            for row, history in zip(current, histories, strict=True):
                if (
                    history.ndim != MATRIX_DIMENSIONS
                    or history.shape[1] != current.shape[1]
                    or not 1 <= len(history) <= self.history_steps + 1
                    or not np.array_equal(history[-1], row)
                ):
                    raise ValueError(
                        "Each history must end at its start and fit within "
                        "the delay depth"
                    )
            lengths = np.asarray([len(history) for history in histories])
            for remaining in range(int(lengths.max()), 0, -1):
                active = np.flatnonzero(lengths >= remaining)
                inputs = np.stack(
                    [histories[row][-remaining] for row in active]
                )
                state[active] = self._reservoir_step(state[active], inputs)
        else:
            state = self._reservoir_step(state, current)
        for step in range(int(steps)):
            current = np.asarray(
                state @ self._readout.Wout + self._readout.bias,
                dtype=np.float64,
            )
            yield current
            if step + 1 < int(steps):
                state = self._reservoir_step(state, current)
