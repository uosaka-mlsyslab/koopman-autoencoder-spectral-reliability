from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
import torch

from spectral_reliability.config_shared import reject_unknown, require_keys
from spectral_reliability.validation import (
    integer_at_least,
    nonnegative_finite,
)


def validate_parameters(
    name: str,
    config: Mapping[str, Any],
    types: Mapping[str, Any],
    checks: Mapping[str, Callable[[Any], bool]],
) -> None:
    """Check the shared schema and defer method-specific constraints."""
    reject_unknown(dict(config), set(types), f"{name} model")
    require_keys(dict(config), set(types), f"{name} model")
    for key, value in config.items():
        expected = types[key]
        if key in checks:
            valid = checks[key](value)
        elif expected is bool:
            valid = type(value) is bool
        elif expected is int:
            valid = integer_at_least(value, 1)
        elif expected is float:
            valid = nonnegative_finite(value)
        elif expected is list:
            valid = isinstance(value, (list, tuple)) and all(
                integer_at_least(item, 0) for item in value
            )
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise ValueError(
                f"invalid {name} model parameter {key}: {value!r}",
            )


def svd_rank_and_basis(
    snapshots: np.ndarray,
    svd_rank: float,
) -> tuple[int, np.ndarray]:
    values = np.asarray(snapshots, dtype=np.float64)
    _, singular_values, vh = np.linalg.svd(values, full_matrices=False)
    energy = singular_values**2
    total = energy.sum()
    if total == 0:
        rank = 0
    elif isinstance(svd_rank, float):
        rank = min(
            int(np.searchsorted(np.cumsum(energy / total), svd_rank)) + 1,
            len(singular_values),
        )
    elif svd_rank == -1:
        rank = len(singular_values)
    else:
        rank = min(svd_rank, len(singular_values))
    return rank, vh.T[:, :rank]


def cpu_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return torch.as_tensor(value).detach().cpu().clone()


class Baseline:
    method: str
    parameter_types: ClassVar[dict[str, Any]] = {
        "scale_inputs": bool,
        "rollout_batch_size": int,
    }
    parameter_checks: ClassVar[dict[str, Callable[[Any], bool]]] = {}
    history_steps = 0

    def __init__(
        self,
        params: Mapping[str, Any],
        latent_dim: int,
        seed: int,
        device: torch.device,
    ) -> None:
        self.validate_parameters(params)
        self.params = dict(params)
        self.latent_dim = int(latent_dim)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.fit_params: dict[str, Any] = {}
        self.training_logs: list[dict[str, Any]] = []
        self.best_epoch: int | None = None
        self.best_validation_vrmse: float | None = None

    @classmethod
    def validate_parameters(cls, params: Mapping[str, Any]) -> None:
        validate_parameters(
            cls.method,
            params,
            cls.parameter_types,
            cls.parameter_checks,
        )

    def prepare_rollout(
        self,
        dataset: Any,
        svd_rank: float,
    ) -> int | None:
        return None

    def state(self) -> dict[str, Any]:
        return {
            name: cpu_tensor(value)
            for name, value in self.model.state_dict().items()
        }

    def load_state(
        self,
        state: dict[str, Any],
        fit_params: dict[str, Any],
    ) -> None:
        self._build_model(int(fit_params["input_dim"]))
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.fit_params = dict(fit_params)
