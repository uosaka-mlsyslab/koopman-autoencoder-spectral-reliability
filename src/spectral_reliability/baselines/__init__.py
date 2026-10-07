from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from spectral_reliability.config_shared import BASELINES, baseline_params

from .consistent_kae import ConsistentKAE
from .esn import ESN
from .kernel_dmd import KernelDMD
from .neural_ode import NeuralODE
from .scale import ScaledBaseline

BASELINE_CLASSES = {
    "esn": ESN,
    "kernel_dmd": KernelDMD,
    "neural_ode": NeuralODE,
    "consistent_kae": ConsistentKAE,
}


def validate_baseline_config(name: str, config: Mapping[str, Any]) -> None:
    BASELINE_CLASSES[name].validate_parameters(config)


def build_baseline(
    method: str,
    config: Mapping[str, Any],
    latent_dim: int,
    seed: int,
    device: str | torch.device,
) -> ScaledBaseline:
    """Build a predictor with CPU placement for ESN and Kernel DMD."""
    try:
        cls = BASELINE_CLASSES[method]
    except KeyError as error:
        raise ValueError(
            f"Unknown baseline {method!r}; choose one of {BASELINES}"
        ) from error
    selected_device = torch.device(
        "cpu" if method in ("esn", "kernel_dmd") else device
    )
    return ScaledBaseline(cls(config, latent_dim, seed, selected_device))


def load_baseline(
    checkpoint: dict,
    model_config: dict,
    *,
    device: str | torch.device,
) -> ScaledBaseline:
    params = baseline_params({"model": model_config})
    predictor = build_baseline(
        checkpoint["method"],
        params,
        int(checkpoint["latent_dim"]),
        int(checkpoint["seed"]),
        device,
    )
    predictor.load_state(checkpoint)
    return predictor
