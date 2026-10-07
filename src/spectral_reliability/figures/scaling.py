from __future__ import annotations

import math
from pathlib import Path
from string import ascii_lowercase
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
)

from ..config_shared import DISPLAY_NAMES, MODEL_LABELS
from ..results import read_seed_metrics
from .style import (
    FIGSIZE_MM,
    FIGURE_LABELS,
    FONT_SIZE,
    MODEL_COLORS,
    MODELS_BY_TRAINING,
    PANEL_FONT_SIZE,
    plain_log_tick,
    save_pdf,
)

ScalingMetrics = dict[str, dict[str, np.ndarray]]


def _model_styles() -> dict[str, dict[str, Any]]:
    styles = {
        "latent_prediction_two_stage": {
            "linestyle": (0, (4.0, 1.4, 1.0, 1.4)),
            "marker": "v",
        },
        "latent_prediction_penalty_two_stage": {
            "linestyle": (0, (3.5, 1.8)),
            "marker": "s",
        },
        "spectral_residual_two_stage": {"linestyle": "-", "marker": "o"},
        "spectral_residual_penalty_two_stage": {
            "linestyle": (0, (1.2, 1.4)),
            "marker": "D",
        },
    }
    for model in MODELS_BY_TRAINING["joint"]:
        styles[model] = dict(styles[model.replace("_joint", "_two_stage")])
    for model, values in styles.items():
        values.update(
            color=MODEL_COLORS[model],
            label=FIGURE_LABELS[model].split(" (")[0],
        )
    return styles


MODEL_STYLES = _model_styles()
BASELINE_STYLES = {
    "esn": {"color": MODEL_COLORS["esn"], "linestyle": ":", "marker": "s"},
    "neural_ode": {
        "color": MODEL_COLORS["neural_ode"],
        "linestyle": "--",
        "marker": "^",
    },
    "kernel_dmd": {
        "color": MODEL_COLORS["kernel_dmd"],
        "linestyle": "-.",
        "marker": "D",
    },
    "consistent_kae": {
        "color": MODEL_COLORS["consistent_kae"],
        "linestyle": (0, (3, 1, 1, 1)),
        "marker": "v",
    },
}


def load_scaling_metrics(
    csv_path: Path,
    scaling_grid: dict,
    settings: dict,
) -> ScalingMetrics:
    """Group selected per-seed CSV metrics into multiplier-by-seed
    arrays.
    """
    multipliers, seeds = scaling_grid["multipliers"], scaling_grid["seeds"]
    selected_models = tuple(
        dict.fromkeys(
            (
                *settings["variants"],
                settings["baseline_reference"],
                *settings["baselines"],
            )
        )
    )
    values = {
        system: {
            model: np.full((len(multipliers), len(seeds)), np.nan)
            for model in selected_models
        }
        for system in scaling_grid["systems"]
    }
    systems = {display: system for system, display in DISPLAY_NAMES.items()}
    models = {label: model for model, label in MODEL_LABELS.items()}
    seen = set()
    for row in read_seed_metrics(csv_path):
        system, model = systems[row["system"]], models[row["model"]]
        if system not in values or model not in selected_models:
            continue
        dimensions = scaling_grid["dimensions"][system]
        latent_dim, seed = int(row["latent_dim"]), int(row["seed"])
        if latent_dim not in dimensions:
            raise ValueError(
                f"{system}: CSV latent_dim={latent_dim} is not configured; "
                f"expected {dimensions}"
            )
        index = dimensions.index(latent_dim)
        key = (system, model, index, seed)
        if key in seen or seed not in seeds:
            raise ValueError(f"duplicate or invalid CSV run: {key}")
        seen.add(key)
        values[system][model][index, seeds.index(seed)] = float(
            row[scaling_grid["metric_columns"][system]],
        )
    expected = (
        len(scaling_grid["systems"])
        * len(selected_models)
        * len(multipliers)
        * len(seeds)
    )
    if len(seen) != expected:
        raise ValueError(f"expected {expected} selected runs, got {len(seen)}")
    return values


def _nan_to_inf(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    return np.where(np.isnan(array), np.inf, array)


def _quantile_bounds(
    array: np.ndarray,
    quantiles: list[float],
) -> tuple[np.ndarray, ...]:
    lower, middle, upper = np.quantile(
        _nan_to_inf(array),
        quantiles,
        axis=1,
    )
    return tuple(
        np.where(np.isnan(value), np.inf, value)
        for value in (lower, middle, upper)
    )


def _tick_decimals(ticks: list[float]) -> int:
    for decimals in range(13):
        if all(
            math.isclose(
                round(float(tick), decimals),
                float(tick),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for tick in ticks
        ):
            return decimals
    return 12


def _tick_label(tick: float, decimals: int) -> str:
    rounded = round(float(tick), decimals)
    if not math.isclose(rounded, float(tick), rel_tol=1e-12, abs_tol=1e-15):
        return f"{tick:.12g}"
    return f"{0.0 if rounded == 0.0 else tick:.{decimals}f}"


def _zoom_ticks(ax: Any, y_lower: float, y_upper: float) -> None:
    ticks = [
        tick
        for tick in MaxNLocator(nbins=4).tick_values(y_lower, y_upper)
        if y_lower - 1e-9 <= tick <= y_upper + 1e-9
    ]
    endpoints = [
        tick
        for tick in (0.5, 1.0, 2.0, 5.0, 10.0)
        if y_upper + 1e-9 < tick <= y_upper * 1.08
    ]
    labels = [_tick_label(tick, _tick_decimals(ticks)) for tick in ticks] + [
        _tick_label(tick, _tick_decimals(endpoints)) for tick in endpoints
    ]
    ax.set_yticks(ticks + endpoints)
    ax.set_yticklabels(labels)


def _zoom_range(
    arrays: list[np.ndarray], settings: dict
) -> tuple[float, float]:
    low, high = np.inf, 0.0
    for array in arrays:
        lower, _middle, upper = _quantile_bounds(array, settings["quantiles"])
        finite_low = lower[np.isfinite(lower)]
        finite_high = upper[np.isfinite(upper)]
        if finite_low.size:
            low = min(low, float(finite_low.min()))
        if finite_high.size:
            high = max(high, float(finite_high.max()))
    if not math.isfinite(low) or high <= 0:
        raise ValueError("no finite quantile bounds in the panel")
    margin = (
        settings["linear_padding_fraction"] * (high - low)
        if high > low
        else settings["linear_padding_fraction"] * high
    )
    return max(0.0, low - margin), high + margin


def _style(config: dict[str, Any], **extra: Any) -> dict[str, Any]:
    style = dict(config)
    style.update(extra)
    style.setdefault("markersize", 3.0)
    style.setdefault("linewidth", 1.0)
    style["markerfacecolor"] = (
        "white" if style["marker"] in ("s", "^", "D", "v") else style["color"]
    )
    style["markeredgewidth"] = 0.8
    return style


def _panel_frame(
    ax: Any,
    index: int,
    system: str,
    scaling_grid: dict,
) -> None:
    ax.set_title(DISPLAY_NAMES[system], fontweight="bold", pad=6)
    ax.set_xticks(
        np.arange(len(scaling_grid["multipliers"])),
        [str(latent_dim) for latent_dim in scaling_grid["dimensions"][system]],
    )
    ax.grid(axis="y", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", which="both")
    for side, spine in ax.spines.items():
        spine.set_color("0.35")
        if side in {"top", "right"}:
            spine.set_visible(False)
    ax.text(
        -0.16,
        1.06,
        ascii_lowercase[index],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=PANEL_FONT_SIZE,
        fontweight="bold",
    )


def _new_figure(systems: list[str]) -> tuple[Any, Any]:
    width, height = FIGSIZE_MM["scaling"]["all"]
    columns = min(3, len(systems))
    fig, axes = plt.subplots(
        math.ceil(len(systems) / columns),
        columns,
        figsize=(width / 25.4, height / 25.4),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.085, right=0.99, top=0.86, bottom=0.12, wspace=0.30, hspace=0.42
    )
    return fig, axes


def _label_and_save_figure(
    fig: Any, axes: Any, handles: list[Line2D], path: Path, *, xlabel: str
) -> None:
    center_x = 0.5 * (
        axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1
    )
    fig.supxlabel(
        xlabel, x=center_x, y=0.02, fontsize=FONT_SIZE, fontweight="bold"
    )
    fig.supylabel("VRMSE", x=0.012, fontsize=FONT_SIZE, fontweight="bold")
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(center_x, 0.995),
        ncol=len(handles),
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.5,
        handlelength=2.2,
        borderaxespad=0.0,
    )
    save_pdf(fig, path)


def draw_models(
    values: ScalingMetrics,
    path: Path,
    scaling_grid: dict,
    settings: dict,
) -> None:
    fig, axes = _new_figure(scaling_grid["systems"])
    try:
        x = np.arange(len(scaling_grid["multipliers"]))
        y_cap = settings["linear_y_cap"]
        models = settings["variants"]
        for index, (ax, system) in enumerate(
            zip(
                axes.ravel()[: len(scaling_grid["systems"])],
                scaling_grid["systems"],
                strict=True,
            ),
        ):
            arrays = {
                model: _nan_to_inf(values[system][model]) for model in models
            }
            y_lower, y_upper = _zoom_range(list(arrays.values()), settings)
            y_upper = min(y_upper, y_cap)
            for model in models:
                style = _style(MODEL_STYLES[model])
                lower, middle, upper = (
                    np.minimum(v, y_cap)
                    for v in _quantile_bounds(
                        arrays[model],
                        settings["quantiles"],
                    )
                )
                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=style["color"],
                    alpha=0.16,
                    linewidth=0,
                )
                ax.plot(x, middle, **style, alpha=0.95, zorder=4)
            ax.set_ylim(y_lower, y_upper)
            _zoom_ticks(ax, y_lower, y_upper)
            _panel_frame(ax, index, system, scaling_grid)
        for ax in axes.ravel()[len(scaling_grid["systems"]) :]:
            ax.set_visible(False)
        handles = [
            Line2D([], [], **_style(MODEL_STYLES[model])) for model in models
        ]
        _label_and_save_figure(
            fig, axes, handles, path, xlabel="Latent Dimension"
        )
    finally:
        plt.close(fig)


def draw_baselines(
    values: ScalingMetrics,
    path: Path,
    scaling_grid: dict,
    settings: dict,
) -> None:
    fig, axes = _new_figure(scaling_grid["systems"])
    try:
        x = np.arange(len(scaling_grid["multipliers"]))
        y_cap = settings["log_y_cap"]
        reference_model = settings["baseline_reference"]
        series = [
            (
                reference_model,
                _style(
                    MODEL_STYLES[reference_model],
                    label=FIGURE_LABELS[reference_model],
                ),
            ),
        ] + [
            (
                method,
                _style(
                    BASELINE_STYLES[method],
                    label=FIGURE_LABELS[method],
                ),
            )
            for method in settings["baselines"]
        ]
        for index, (ax, system) in enumerate(
            zip(
                axes.ravel()[: len(scaling_grid["systems"])],
                scaling_grid["systems"],
                strict=True,
            ),
        ):
            arrays = {
                model: _nan_to_inf(values[system][model])
                for model, _style_value in series
            }
            finite = np.concatenate(
                [a[np.isfinite(a) & (a > 0)] for a in arrays.values()]
            )
            if not finite.size:
                raise ValueError(f"{system}: no finite VRMSE")
            margin = settings["log_padding_factor"]
            top, bottom = (
                min(y_cap, float(finite.max()) * margin),
                float(finite.min()) / margin,
            )
            for model, style in series:
                lower, middle, upper = (
                    np.clip(v, bottom, top)
                    for v in _quantile_bounds(
                        arrays[model],
                        settings["quantiles"],
                    )
                )
                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=style["color"],
                    alpha=0.16,
                    linewidth=0,
                )
                ax.plot(
                    x,
                    middle,
                    **style,
                    alpha=0.95,
                    zorder=5 if model == reference_model else 4,
                )
            ax.set_yscale("log")
            ax.set_ylim(bottom, top)
            ax.yaxis.set_major_locator(
                LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
            )
            ax.yaxis.set_major_formatter(FuncFormatter(plain_log_tick))
            ax.yaxis.set_minor_formatter(NullFormatter())
            _panel_frame(ax, index, system, scaling_grid)
        for ax in axes.ravel()[len(scaling_grid["systems"]) :]:
            ax.set_visible(False)
        handles = [Line2D([], [], **style) for _model, style in series]
        _label_and_save_figure(
            fig, axes, handles, path, xlabel=r"Representation size $N$"
        )
    finally:
        plt.close(fig)
