from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from .data.dataset import Dataset
from .evaluation import OperatorConfig, TrainedModel, fit_operators
from .evaluation_config import load_evaluation_config

AttractorArrays = dict[str, np.ndarray]


def prepare_model_for_figures(
    model: TrainedModel,
    dataset: Dataset,
    *,
    config: OperatorConfig,
    device: str,
) -> TrainedModel:
    # Refit on CPU to preserve the published figure-array computation.
    if config.loss_type == "latent_prediction":
        rollout, _ = fit_operators(
            model.encoder,
            dataset.train,
            config,
            dtype=model.dtype,
            device="cpu",
        )
        model.rollout_operator = rollout.a_hat
    return model.to(device)


def _predictions(
    model: TrainedModel,
    observations: np.ndarray,
    prediction_horizon_steps: tuple[int, ...],
    measured_dim: int,
) -> AttractorArrays:
    """Propagate in float64 and decode at checkpoint precision."""
    model.encoder.eval()
    model.decoder.eval()
    time_steps, trajectories, features = observations.shape
    out = {}
    with torch.no_grad():
        parameter = next(model.encoder.parameters())
        z = model.encoder(
            torch.as_tensor(observations)
            .reshape(-1, features)
            .to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )
        z = z.to(torch.float64)
        operator = torch.as_tensor(
            model.rollout_operator,
            dtype=torch.float64,
        ).to(z.device)
        for k in range(1, prediction_horizon_steps[-1] + 1):
            z = z @ operator.T
            if k in prediction_horizon_steps:
                decoded = model.decoder(z.to(parameter.dtype)).to(
                    torch.float64,
                )
                values = (
                    decoded.reshape(time_steps, trajectories, -1).cpu().numpy()
                )
                out[f"horizon_steps_{k}"] = np.asarray(
                    values[: len(observations) - k, 0, -measured_dim:],
                )
    return out


def delay_lag_steps(config: dict) -> int:
    generation_config = config["data_generation"]
    interval = (
        generation_config["dt"] * config["preprocessing"]["sampling_stride"]
    )
    lag = float(generation_config["tau"]) / interval
    if not lag.is_integer() or lag < 1:
        raise ValueError(
            "Mackey-Glass tau/dt must be a positive integer number of "
            "preprocessed steps"
        )
    return int(lag)


def compute_arrays(
    models: Mapping[str, TrainedModel],
    dataset: Dataset,
    *,
    settings: dict,
    device: str,
) -> AttractorArrays:
    system, seed = dataset.system, dataset.seed
    reference_model = next(iter(models.values()))
    evaluation = load_evaluation_config(reference_model.data_config)
    steps_per_lt = evaluation.steps_per_lt
    prediction_horizon_steps = tuple(
        round(horizon_lt * steps_per_lt)
        for horizon_lt in settings["prediction_horizons_lt"]
    )
    window = settings["window_lt"]
    stop = round(window * steps_per_lt) + 1
    series = getattr(dataset, settings["split"])
    if series.shape[1] < stop:
        raise ValueError(
            f"{settings['split']} series is shorter than "
            f"{window} Lyapunov times"
        )
    lag_steps = (
        delay_lag_steps(reference_model.data_config)
        if system == "mackeyglass"
        else 0
    )
    observations = np.asarray(series).T[:stop, None, :]
    computed = {
        "system": np.asarray(system),
        "seed": np.asarray(seed),
        "window_lt": np.asarray(window),
        "steps_per_lt": np.asarray(steps_per_lt),
        "delay_lag_steps": np.asarray(lag_steps),
        "dt": np.asarray(dataset.dt),
        "prediction_horizon_steps": np.asarray(
            prediction_horizon_steps,
        ),
        "reference": observations[:, 0, -dataset.measured_dim :].copy(),
    }
    for name in settings["variants"]:
        model = models[name]
        if model.system != system or model.seed != seed:
            raise ValueError(
                f"{name}: model and test series identify different runs"
            )
        loss_type = (
            "latent_prediction"
            if name.startswith("latent_prediction")
            else "spectral_residual"
        )
        model = prepare_model_for_figures(
            model,
            dataset,
            config=OperatorConfig(
                loss_type,
                model.operator_regularization,
                evaluation.rollout_svd_rank,
            ),
            device=device,
        )
        predictions = _predictions(
            model,
            observations,
            prediction_horizon_steps,
            dataset.measured_dim,
        )
        computed.update(
            {f"{name}__{key}": value for key, value in predictions.items()}
        )
    return computed


def save_arrays(computed: AttractorArrays, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **computed)
