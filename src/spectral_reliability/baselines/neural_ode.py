from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import suppress
from copy import deepcopy
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint

from spectral_reliability.data.dataset import Dataset
from spectral_reliability.metrics import validation_vrmse
from spectral_reliability.model import (
    ACTIVATIONS,
    NetworkConfig,
    build_mlp,
)
from spectral_reliability.validation import integer_at_least, positive_finite

from .base import Baseline


class LatentODE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        params: Mapping[str, Any],
    ) -> None:
        super().__init__()
        hidden = int(params["hidden_dim"])
        self.encoder = build_mlp(
            input_dim,
            latent_dim,
            hidden,
            NetworkConfig(
                int(params["encoder_depth"]),
                hidden,
                params["activation"],
            ),
        )
        self.dynamics = build_mlp(
            latent_dim,
            latent_dim,
            hidden,
            NetworkConfig(
                int(params["dynamics_depth"]),
                hidden,
                params["activation"],
            ),
        )
        self.decoder = build_mlp(
            latent_dim,
            input_dim,
            hidden,
            NetworkConfig(
                int(params["decoder_depth"]),
                hidden,
                params["activation"],
            ),
        )

    def dynamics_rhs(
        self,
        _time: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return self.dynamics(latent)


class NeuralODE(Baseline):
    """Integrate latent dynamics with adaptive control shared within
    each rollout batch.
    """

    method = "neural_ode"
    parameter_types: ClassVar[dict[str, Any]] = {
        **Baseline.parameter_types,
        "hidden_dim": int,
        "encoder_depth": int,
        "dynamics_depth": int,
        "decoder_depth": int,
        "learning_rate": float,
        "weight_decay": float,
        "epochs": int,
        "batch_size": int,
        "rollout_steps": (int, str),
        "method": str,
        "optimizer": str,
        "gradclip": float,
        "activation": str,
        "precision": str,
        "rtol": float,
        "atol": float,
    }
    parameter_checks: ClassVar[dict[str, Any]] = {
        "encoder_depth": lambda value: integer_at_least(value, 0),
        "dynamics_depth": lambda value: integer_at_least(value, 0),
        "decoder_depth": lambda value: integer_at_least(value, 0),
        "rollout_steps": (
            lambda value: (
                value == "num_delay_samples" or integer_at_least(value, 1)
            )
        ),
        "learning_rate": positive_finite,
        "gradclip": positive_finite,
        "rtol": positive_finite,
        "atol": positive_finite,
        "activation": lambda value: value in ACTIVATIONS,
        "optimizer": lambda value: value in {"adam", "adamw"},
        "precision": lambda value: value in {"float32", "float64"},
    }

    def _build_model(self, input_dim: int) -> None:

        self.odeint = odeint
        self.dtype = getattr(torch, self.params["precision"])
        self.model = LatentODE(input_dim, self.latent_dim, self.params).to(
            device=self.device, dtype=self.dtype
        )

    def _rollout_loss(
        self,
        data: torch.Tensor,
        starts: torch.Tensor,
        times: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Average decoded prediction MSE over the integration
        horizon.
        """
        initial = self.model.encoder(data[starts])
        trajectory = self.odeint(
            self.model.dynamics_rhs,
            initial,
            times,
            method=self.params["method"],
            rtol=self.params["rtol"],
            atol=self.params["atol"],
        )
        loss = torch.zeros((), dtype=self.dtype, device=self.device)
        for step in range(1, horizon + 1):
            prediction = self.model.decoder(trajectory[step])
            loss = loss + torch.mean((prediction - data[starts + step]) ** 2)
        return loss / horizon

    def _validation_vrmse(
        self,
        data: torch.Tensor,
        starts: torch.Tensor,
        times: torch.Tensor,
        horizon: int,
        eps: float,
    ) -> float:
        self.model.eval()
        with torch.inference_mode():
            initial = self.model.encoder(data[starts])
            trajectory = self.odeint(
                self.model.dynamics_rhs,
                initial,
                times,
                method=self.params["method"],
                rtol=self.params["rtol"],
                atol=self.params["atol"],
            )
            predictions = torch.cat(
                [
                    self.model.decoder(trajectory[step])
                    for step in range(1, horizon + 1)
                ]
            )
            reference = torch.cat(
                [data[starts + step] for step in range(1, horizon + 1)]
            )
            return float(validation_vrmse(predictions, reference, eps).item())

    def fit(self, dataset: Dataset, evaluation) -> None:
        params = self.params
        self.dt = float(dataset.dt)
        horizon = params["rollout_steps"]
        if isinstance(horizon, str):
            if horizon != "num_delay_samples":
                raise ValueError(
                    "rollout_steps must be an integer or 'num_delay_samples'"
                )
            horizon = max(1, dataset.num_delay_samples)
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("rollout_steps must be positive")
        self._build_model(dataset.train.shape[0])
        x_train = torch.tensor(
            dataset.train.T, dtype=self.dtype, device=self.device
        )
        x_valid = torch.tensor(
            dataset.valid.T, dtype=self.dtype, device=self.device
        )
        if min(len(x_train), len(x_valid)) <= horizon:
            raise ValueError(
                "Neural ODE needs more train and validation states "
                "than rollout_steps"
            )
        optimizer = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }[params["optimizer"]](
            self.model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        times = (
            torch.arange(horizon + 1, dtype=self.dtype, device=self.device)
            * self.dt
        )
        candidate_count = len(x_train) - horizon
        valid_starts = torch.arange(len(x_valid) - horizon, device=self.device)
        batch_size = int(params["batch_size"])
        best_state = None
        best_valid = float("inf")
        self.fit_params = {
            **params,
            "input_dim": dataset.train.shape[0],
            "latent_dim": self.latent_dim,
            "dt": self.dt,
            "rollout_steps": horizon,
        }
        for epoch in range(int(params["epochs"])):
            self.model.train()
            order = torch.randperm(candidate_count, device=self.device)
            losses: list[float] = []
            for offset in range(0, candidate_count, batch_size):
                starts = order[offset : offset + batch_size]
                loss = self._rollout_loss(x_train, starts, times, horizon)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=params["gradclip"]
                )
                optimizer.step()
                losses.append(float(loss.item()))
            # Preserve the original float32 rounding of the logged loss.
            logged_loss = torch.tensor(float(np.mean(losses)))
            valid_vrmse = self._validation_vrmse(
                x_valid, valid_starts, times, horizon, evaluation.vrmse_eps
            )
            is_best = valid_vrmse < best_valid
            if is_best:
                best_valid = valid_vrmse
                self.best_epoch = epoch + 1
                best_state = deepcopy(self.model.state_dict())
            self.training_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(logged_loss.item()),
                    "valid_vrmse": valid_vrmse,
                    "is_best": is_best,
                }
            )
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_validation_vrmse = (
            None if best_valid == float("inf") else best_valid
        )

    def load_state(
        self,
        state: dict[str, Any],
        fit_params: dict[str, Any],
    ) -> None:
        super().load_state(state, fit_params)
        self.dt = float(fit_params["dt"])

    def _integrate_interval(
        self,
        latent: torch.Tensor,
        interval: torch.Tensor,
    ):
        """Integrate active rows, retrying failed batches row by row."""
        try:
            propagated = self.odeint(
                self.model.dynamics_rhs,
                latent,
                interval,
                method=self.params["method"],
                rtol=self.params["rtol"],
                atol=self.params["atol"],
            )[-1]
        except Exception:  # noqa: BLE001 -- Preserve failed-batch row retries.
            propagated = torch.full_like(latent, float("nan"))
            for row in range(len(latent)):
                # Preserve NaNs for any row whose integration fails.
                with suppress(Exception):
                    propagated[row] = self.odeint(
                        self.model.dynamics_rhs,
                        latent[row : row + 1],
                        interval,
                        method=self.params["method"],
                        rtol=self.params["rtol"],
                        atol=self.params["atol"],
                    )[-1][0]
        finite_mask = torch.isfinite(propagated).all(dim=1)
        propagated = torch.where(
            finite_mask[:, None],
            propagated,
            torch.full_like(propagated, float("nan")),
        )
        return propagated, finite_mask

    @torch.inference_mode()
    def iter_rollout_array(
        self,
        initial_states: np.ndarray,
        steps: int,
    ) -> Iterator[np.ndarray]:
        if int(steps) <= 0:
            return
        inputs = torch.tensor(
            initial_states, dtype=self.dtype, device=self.device
        )
        latent = self.model.encoder(inputs)
        interval = torch.tensor(
            [0.0, self.dt], dtype=self.dtype, device=self.device
        )
        active_mask = torch.ones(
            len(latent), dtype=torch.bool, device=latent.device
        )
        for _ in range(int(steps)):
            if active_mask.any():
                propagated, finite_mask = self._integrate_interval(
                    latent[active_mask], interval
                )
                latent = latent.clone()
                latent[active_mask] = propagated
                active_mask = active_mask.clone()
                active_mask[
                    active_mask.nonzero(as_tuple=True)[0][~finite_mask]
                ] = False
            yield self.model.decoder(latent).cpu().numpy().astype(np.float64)
