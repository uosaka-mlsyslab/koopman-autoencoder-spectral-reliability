from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Iterator
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn

from spectral_reliability.config_shared import atomic_write
from spectral_reliability.data.dataset import Dataset
from spectral_reliability.metrics import validation_vrmse
from spectral_reliability.model import ACTIVATIONS
from spectral_reliability.validation import (
    integer_at_least,
    nonnegative_finite,
    positive_finite,
)

from .base import Baseline, svd_rank_and_basis

KOOPMANAE_COMMIT = "f6b7db790c959f80dd7a0968f12e89e19eedf381"
KOOPMANAE_SOURCE_SHA256 = (
    "cbff22c698ad0f7c9fe8c8b3ee1c82de9f6b0180649f701eb06fe84b626c4a9e"
)
KOOPMANAE_SOURCE_URL = (
    "https://raw.githubusercontent.com/erichson/koopmanAE/"
    f"{KOOPMANAE_COMMIT}/model.py"
)
MINIMUM_PAIR_STATES = 2


@lru_cache(maxsize=1)
def load_koopmanae_module() -> ModuleType:
    """Download and hash-check the pinned GPL-3.0 implementation on
    first use.
    """
    directory = Path(__file__).resolve().parents[3] / "baselines" / "koopmanAE"
    path = directory / "model.py"
    if path.is_file():
        source = path.read_bytes()
    else:
        try:
            with urlopen(KOOPMANAE_SOURCE_URL, timeout=60) as response:
                source = response.read()
        except OSError as error:
            raise RuntimeError(
                "koopmanAE is not shipped (GPL-3.0). First use downloads "
                "it from the pinned commit "
                f"{KOOPMANAE_COMMIT} and needs network access. Alternatively, "
                f"download {KOOPMANAE_SOURCE_URL} "
                f"and place model.py at {path} by hand; expected SHA-256: "
                f"{KOOPMANAE_SOURCE_SHA256}."
            ) from error
    checksum = hashlib.sha256(source).hexdigest()
    if checksum != KOOPMANAE_SOURCE_SHA256:
        raise ValueError(
            f"koopmanAE model.py SHA-256 mismatch: {checksum} != "
            f"{KOOPMANAE_SOURCE_SHA256}"
        )
    if not path.is_file():
        atomic_write(path, source)
    specification = importlib.util.spec_from_file_location(
        "spectral_reliability_baseline_koopmanae", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import hash-checked koopmanAE at {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _HiddenWidthScale:
    # The pinned constructors only use 16 * ALPHA. Adapt that expression
    # rather than copy or patch upstream, preserving all initialization.
    def __init__(self, width: int) -> None:
        self.width = width

    def __rmul__(self, factor: int) -> int:
        return factor * self.width // 16


def linear_output_decoder(
    input_dim: int,
    latent_dim: int,
    hidden_width: int,
    activation: str,
) -> nn.Module:
    """Keep upstream decoder initialization but omit its output
    nonlinearity.
    """
    upstream = load_koopmanae_module()

    class LinearOutputDecoder(upstream.decoderNet):
        def forward(self, states: torch.Tensor) -> torch.Tensor:
            states = states.view(-1, 1, self.b)
            states = self.tanh(self.fc1(states))
            states = self.tanh(self.fc2(states))
            states = self.fc3(states)
            return states.view(-1, 1, self.m, self.n)

    decoder = LinearOutputDecoder(
        input_dim, 1, latent_dim, ALPHA=_HiddenWidthScale(hidden_width)
    )
    decoder.tanh = ACTIVATIONS[activation]()
    return decoder


class ConsistentKAE(Baseline):
    """Train the pinned koopmanAE with a linear-output decoder."""

    method = "consistent_kae"
    parameter_types: ClassVar[dict[str, Any]] = {
        **Baseline.parameter_types,
        "hidden_width": (int, str),
        "steps": int,
        "steps_back": int,
        "lamb": float,
        "nu": float,
        "eta": float,
        "lr": float,
        "wd": float,
        "epochs": int,
        "batch": int,
        "gradclip": float,
        "init_scale": float,
        "evaluation_batch_size": int,
        "lr_update": list,
        "lr_decay": float,
        "optimizer": str,
        "activation": str,
        "precision": str,
    }
    parameter_checks: ClassVar[dict[str, Any]] = {
        "hidden_width": (
            lambda value: value == "latent" or integer_at_least(value, 1)
        ),
        "lr": positive_finite,
        "gradclip": positive_finite,
        "lr_decay": lambda value: nonnegative_finite(value) and value <= 1,
        "optimizer": lambda value: value in {"adam", "adamw"},
        "activation": lambda value: value in ACTIVATIONS,
        "precision": lambda value: value in {"float32", "float64"},
    }

    def _hidden_width(self) -> int:
        value = self.params["hidden_width"]
        return self.latent_dim if value == "latent" else int(value)

    def _build_model(self, input_dim: int) -> None:
        self.input_dim = input_dim
        upstream = load_koopmanae_module()
        hidden_width = self._hidden_width()
        model = upstream.koopmanAE(
            m=input_dim,
            n=1,
            b=self.latent_dim,
            steps=int(self.params["steps"]),
            steps_back=int(self.params["steps_back"]),
            alpha=_HiddenWidthScale(hidden_width),
            init_scale=float(self.params["init_scale"]),
        )
        # Both decoder constructions consume initialization draws.
        model.decoder = linear_output_decoder(
            input_dim, self.latent_dim, hidden_width, self.params["activation"]
        )
        model.encoder.tanh = ACTIVATIONS[self.params["activation"]]()
        self.dtype = getattr(torch, self.params["precision"])
        self.model = model.to(device=self.device, dtype=self.dtype)

    def _consistency_loss(self) -> torch.Tensor:
        """Sum normalized forward/backward defects over leading rank
        blocks.
        """
        forward_matrix = self.model.dynamics.dynamics.weight
        backward_matrix = self.model.backdynamics.dynamics.weight
        latent_dim = forward_matrix.shape[-1]
        eye = torch.eye(
            latent_dim,
            dtype=forward_matrix.dtype,
            device=forward_matrix.device,
        )
        forward_defect = (backward_matrix @ forward_matrix - eye) ** 2
        backward_defect = (forward_matrix @ backward_matrix - eye) ** 2
        block_sums = (
            forward_defect.cumsum(0).cumsum(1).diagonal()
            + backward_defect.cumsum(0).cumsum(1).diagonal()
        )
        ranks = torch.arange(
            1,
            latent_dim + 1,
            dtype=forward_matrix.dtype,
            device=forward_matrix.device,
        )
        return (block_sums / (2.0 * ranks)).sum()

    def _predict_next(self, data: np.ndarray) -> np.ndarray:
        current = torch.tensor(
            data[:, :-1].T, dtype=self.dtype, device=self.device
        ).view(-1, 1, self.input_dim, 1)
        outputs = []
        batch_size = int(self.params["evaluation_batch_size"])
        with torch.inference_mode():
            for offset in range(0, len(current), batch_size):
                latent = self.model.encoder(
                    current[offset : offset + batch_size]
                )
                prediction = (
                    self.model.decoder(self.model.dynamics(latent))
                    .squeeze(1)
                    .squeeze(-1)
                )
                outputs.append(prediction.cpu().numpy())
        return np.concatenate(outputs, axis=0).T

    def _sequence_loss(
        self,
        x_train: torch.Tensor,
        starts: torch.Tensor,
        sequence_length: int,
        criterion: nn.Module,
        steps: tuple[int, int],
    ) -> torch.Tensor:
        steps, steps_back = steps
        snapshots = [
            x_train[:, starts + snapshot_offset]
            .T.contiguous()
            .view(-1, 1, self.input_dim, 1)
            for snapshot_offset in range(sequence_length)
        ]
        forward, _ = self.model(snapshots[0], mode="forward")
        loss_forward = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for step in range(steps):
            loss_forward = loss_forward + criterion(
                forward[step], snapshots[step + 1]
            )
        loss_reconstruction = criterion(forward[-1], snapshots[0]) * steps
        loss_backward = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        _, backward = self.model(snapshots[-1], mode="backward")
        reversed_snapshots = snapshots[::-1]
        for step in range(steps_back):
            loss_backward = loss_backward + criterion(
                backward[step], reversed_snapshots[step + 1]
            )
        loss_consistency = self._consistency_loss()
        params = self.params
        return (
            loss_forward
            + params["lamb"] * loss_reconstruction
            + params["nu"] * loss_backward
            + params["eta"] * loss_consistency
        )

    def _validation_vrmse(self, valid: np.ndarray, eps: float) -> float:
        self.model.eval()
        with torch.inference_mode():
            prediction = self._predict_next(valid)
            return float(
                validation_vrmse(
                    torch.tensor(prediction.T, dtype=self.dtype),
                    torch.tensor(valid[:, 1:].T, dtype=self.dtype),
                    eps,
                ).item()
            )

    def fit(self, dataset: Dataset, evaluation) -> None:
        params = self.params
        steps, steps_back = int(params["steps"]), int(params["steps_back"])
        sequence_length = max(steps, steps_back) + 1
        train, valid = dataset.train, dataset.valid
        if (
            train.shape[1] < sequence_length
            or valid.shape[1] < MINIMUM_PAIR_STATES
        ):
            raise ValueError(
                "Consistent KAE needs enough train states for its "
                "forward/backward sequence"
            )
        self._build_model(train.shape[0])
        optimizer = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
        }[params["optimizer"]](
            self.model.parameters(), lr=params["lr"], weight_decay=params["wd"]
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=params["lr_update"], gamma=params["lr_decay"]
        )
        x_train = torch.tensor(train, dtype=self.dtype, device=self.device)
        candidate_count = train.shape[1] - sequence_length + 1
        batch_size = int(params["batch"])
        best_valid = float("inf")
        best_state = None
        self.fit_params = {
            **params,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_width": self._hidden_width(),
            "backward": 1,
            "checkpoint_selection": "best_validation",
        }
        for epoch in range(int(params["epochs"])):
            self.model.train()
            losses: list[float] = []
            order = torch.randperm(candidate_count, device=self.device)
            for offset in range(0, candidate_count, batch_size):
                starts = order[offset : offset + batch_size]
                loss = self._sequence_loss(
                    x_train,
                    starts,
                    sequence_length,
                    criterion,
                    (steps, steps_back),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), params["gradclip"]
                )
                optimizer.step()
                losses.append(float(loss.item()))
            valid_vrmse = self._validation_vrmse(valid, evaluation.vrmse_eps)
            scheduler.step()
            is_best = valid_vrmse < best_valid
            if is_best:
                best_valid = valid_vrmse
                self.best_epoch = epoch + 1
                best_state = deepcopy(self.model.state_dict())
            self.training_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    "valid_vrmse": valid_vrmse,
                    "is_best": is_best,
                }
            )
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_validation_vrmse = (
            None if best_valid == float("inf") else best_valid
        )

    def prepare_rollout(self, dataset: Dataset, svd_rank: float) -> int:
        """Project the forward weight as P A P using the training latent
        subspace.
        """
        train = dataset.train.T
        batch_size = int(self.params["evaluation_batch_size"])
        latents = []
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(train), batch_size):
                batch = torch.tensor(
                    train[offset : offset + batch_size],
                    dtype=self.dtype,
                    device=self.device,
                )
                latent = self.model.encoder(
                    batch.view(-1, 1, self.input_dim, 1)
                )
                latents.append(latent.reshape(len(batch), -1).cpu().numpy())
            rank, basis = svd_rank_and_basis(
                np.concatenate(latents, axis=0), svd_rank
            )
            weight = self.model.dynamics.dynamics.weight
            basis_tensor = torch.as_tensor(
                basis, device=weight.device, dtype=weight.dtype
            )
            projection = basis_tensor @ basis_tensor.T
            weight.copy_(projection @ weight @ projection)
        return rank

    @torch.inference_mode()
    def iter_rollout_array(
        self,
        initial_states: np.ndarray,
        steps: int,
    ) -> Iterator[np.ndarray]:
        inputs = torch.tensor(
            initial_states, dtype=self.dtype, device=self.device
        ).view(-1, 1, self.input_dim, 1)
        latent = self.model.encoder(inputs)
        for _ in range(int(steps)):
            latent = self.model.dynamics(latent)
            prediction = self.model.decoder(latent).squeeze(1).squeeze(-1)
            yield prediction.cpu().numpy().astype(np.float64)
