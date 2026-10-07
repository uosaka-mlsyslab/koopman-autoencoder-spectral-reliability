from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import pykoopman as pk
import torch
from pykoopman.observables import RandomFourierFeatures
from pykoopman.regression import EDMD
from sklearn.pipeline import Pipeline

from spectral_reliability.data.dataset import Dataset
from spectral_reliability.metrics import validation_vrmse
from spectral_reliability.validation import integer_at_least, positive_finite

from .base import Baseline, cpu_tensor, svd_rank_and_basis

RFF_PAIR_COMPONENTS = 2
MINIMUM_PAIR_STATES = 2


class KernelDMD(Baseline):
    """Fit N/2 cosine/sine RFF pairs with pykoopman's float64 EDMD."""

    method = "kernel_dmd"
    parameter_types: ClassVar[dict[str, Any]] = {
        **Baseline.parameter_types,
        "gamma": float,
        "include_state": bool,
        "svd_rank": int,
    }
    parameter_checks: ClassVar[dict[str, Any]] = {
        "gamma": positive_finite,
        "svd_rank": lambda value: integer_at_least(value, -1) and value != 0,
    }

    def _build_model(self, rank: int):

        if (
            self.latent_dim < RFF_PAIR_COMPONENTS
            or self.latent_dim % RFF_PAIR_COMPONENTS
        ):
            raise ValueError(
                "Kernel DMD representation size must be positive and even"
            )
        observable = RandomFourierFeatures(
            include_state=self.params["include_state"],
            gamma=self.params["gamma"],
            D=self.latent_dim // RFF_PAIR_COMPONENTS,
            random_state=self.seed,
        )
        return pk.Koopman(
            observables=observable, regressor=EDMD(svd_rank=int(rank))
        )

    def fit(self, dataset: Dataset, evaluation) -> None:
        train, valid = dataset.train.T, dataset.valid.T
        if min(len(train), len(valid)) < MINIMUM_PAIR_STATES:
            raise ValueError(
                "Kernel DMD needs at least two train and validation states"
            )
        self.model = self._build_model(self.params["svd_rank"])
        self.model.fit(train[:-1], y=train[1:])
        prediction = self.model.predict(valid[:-1])
        criterion = float(
            validation_vrmse(
                torch.tensor(prediction, dtype=torch.float32),
                torch.tensor(valid[1:], dtype=torch.float32),
                evaluation.vrmse_eps,
            ).item()
        )
        self.fit_params = {
            **self.params,
            "approximation": "random_fourier_features",
            "random_state": self.seed,
        }
        self.best_epoch = 1
        self.best_validation_vrmse = criterion
        self.training_logs.append(
            {
                "epoch": 1,
                "train_loss": None,
                "valid_vrmse": criterion,
                "is_best": True,
            }
        )

    def state(self) -> dict[str, Any]:
        regressor = self.model._regressor()
        return {
            "observables": {
                name: cpu_tensor(value)
                for name, value in vars(self.model.observables).items()
                if isinstance(value, np.ndarray)
            },
            "regressor": {
                name: cpu_tensor(value)
                if isinstance(value, np.ndarray)
                else value
                for name, value in vars(regressor).items()
            },
        }

    def prepare_rollout(self, dataset: Dataset, svd_rank: float) -> int:
        """Select the feature rank and refit EDMD on the training
        pairs.
        """
        train = dataset.train.T
        features = self.model.observables.transform(train[:-1])
        rank, _ = svd_rank_and_basis(features, svd_rank)
        if rank == 0:
            raise ValueError(
                "Zero feature energy: pykoopman's svd_rank=0 would request "
                "automatic rank"
            )
        self.model = self._build_model(rank)
        self.model.fit(train[:-1], y=train[1:])
        self.fit_params["svd_rank"] = rank
        return rank

    def load_state(
        self,
        state: dict[str, Any],
        fit_params: dict[str, Any],
    ) -> None:

        self.model = self._build_model(int(fit_params["svd_rank"]))
        observable = self.model.observables
        for name, value in state["observables"].items():
            setattr(observable, name, np.asarray(value))
        observable.n_input_features_ = observable.w.shape[0]
        observable.n_output_features_ = observable.measurement_matrix_.shape[1]
        observable.n_consumed_samples = 0
        regressor = self.model.regressor
        for name, value in state["regressor"].items():
            setattr(
                regressor,
                name,
                np.asarray(value)
                if isinstance(value, torch.Tensor)
                else value,
            )
        self.model._pipeline = Pipeline(
            [
                ("observables", observable),
                ("regressor", regressor),
            ]
        )
        self.model.n_input_features_ = observable.n_input_features_
        self.model.n_output_features_ = observable.n_output_features_
        self.fit_params = dict(fit_params)

    def iter_rollout_array(
        self,
        initial_states: np.ndarray,
        steps: int,
    ) -> Iterator[np.ndarray]:
        current = np.asarray(initial_states, dtype=np.float64)
        # EDMD stores a vector; propagate each eigenfunction by its
        # value.
        eigenvalues = self.model._regressor().eigenvalues_
        eigenfunction_values = self.model.psi(current.T)
        koopman_modes = np.asarray(self.model.W)
        for _ in range(int(steps)):
            eigenfunction_values = eigenvalues[:, None] * eigenfunction_values
            yield np.asarray(
                np.real((koopman_modes @ eigenfunction_values).T),
                dtype=np.float64,
            )
