from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .style import (
    FIGSIZE_MM,
    FIGURE_LABELS,
    FONT_SIZE,
    MODEL_COLORS,
    SMALL_FONT_SIZE,
    save_pdf,
)

PHASE_LABELS = {
    "duffing": ("$x$", r"$\dot{x}$"),
    "mackeyglass": ("$x(t)$", r"$x(t-\tau)$"),
}
AttractorArrays = dict[str, np.ndarray]
MATRIX_DIMENSIONS = 2
MINIMUM_TICK_COUNT = 2
MACKEY_TICK_SPACING = 0.5
MACKEY_LOWER_TICK = -0.5


@dataclass(frozen=True)
class PlotAppearance:
    kind: str
    titles: tuple[str, ...]
    norm: Any
    box_limits: Any
    phase_ticks: Any


@dataclass(frozen=True)
class PlotContext:
    computed: AttractorArrays
    models: list[str]
    settings: dict
    reference: np.ndarray
    appearance: PlotAppearance


SUBPLOT_LAYOUT = {
    "phase": {
        "left": 0.15,
        "right": 0.965,
        "bottom": 0.105,
        "top": 0.89,
        "wspace": 0.13,
        "hspace": 0.17,
    },
    "3d": {
        "left": 0.045,
        "right": 0.97,
        "bottom": 0.08,
        "top": 0.90,
        "wspace": 0.10,
        "hspace": 0.12,
    },
    "hovmoller": {
        "left": 0.075,
        "right": 0.925,
        "bottom": 0.08,
        "top": 0.90,
        "wspace": 0.10,
        "hspace": 0.12,
    },
}


def load_arrays(path: Path) -> AttractorArrays:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_arrays(computed: AttractorArrays, settings: dict) -> None:
    steps_per_lt = float(computed["steps_per_lt"])
    expected_horizons_steps = np.asarray(
        [
            round(horizon * steps_per_lt)
            for horizon in settings["prediction_horizons_lt"]
        ]
    )
    stored_horizons_steps = np.asarray(computed["prediction_horizon_steps"])
    if not np.array_equal(stored_horizons_steps, expected_horizons_steps):
        raise ValueError(
            "stale attractor arrays: prediction_horizons_lt="
            f"{settings['prediction_horizons_lt']} requires step horizons "
            f"{expected_horizons_steps.tolist()} at "
            f"steps_per_lt={steps_per_lt:g}, but NPZ stores "
            f"{stored_horizons_steps.tolist()}; "
            "recompute with evaluate.py --figure attractors"
        )
    if float(computed["window_lt"]) != float(settings["window_lt"]):
        raise ValueError(
            f"stale attractor arrays: window_lt={settings['window_lt']:g}, "
            "but NPZ stores "
            f"{float(computed['window_lt']):g}; recompute with "
            "evaluate.py --figure attractors"
        )


def plot_kind(system: str) -> str:
    if system in ("lorenz96", "ks"):
        return "hovmoller"
    if system in ("lorenz63", "rossler"):
        return "3d"
    if system in ("duffing", "mackeyglass"):
        return "phase"
    raise ValueError(f"unknown system: {system}")


def delay_trace(block: np.ndarray, delay_lag_steps: int) -> np.ndarray:
    """Pair the current scalar with its past lag without wrapping the
    window.
    """
    if delay_lag_steps < 1 or len(block) <= delay_lag_steps:
        raise ValueError(
            "scalar window is too short for the requested delay embedding"
        )
    scalar = block[:, -1]
    return np.column_stack(
        (
            scalar[delay_lag_steps:],
            scalar[:-delay_lag_steps],
        )
    )


def _phase_x_ticks(
    system: str, low: float, high: float
) -> tuple[float, float]:
    if system == "mackeyglass":
        if (
            0.35 * (high - low) <= MACKEY_TICK_SPACING
            and low <= MACKEY_LOWER_TICK
            and high >= 0
        ):
            return -0.5, 0.0
    else:
        tick_candidates = MaxNLocator(nbins=4).tick_values(low, high)
        positions = (low + (high - low) * 0.35, low + (high - low) * 0.65)
        preferred = tuple(
            float(tick_candidates[np.abs(tick_candidates - position).argmin()])
            for position in positions
        )
        if low <= preferred[0] < preferred[1] <= high:
            return preferred
    candidates = MaxNLocator(nbins=2).tick_values(low, high)
    inside = candidates[(candidates >= low) & (candidates <= high)]
    if len(inside) >= MINIMUM_TICK_COUNT and inside[0] < inside[-1]:
        return float(inside[0]), float(inside[-1])
    return low, high


def _draw_3d_panel(
    ax: Any,
    trace: np.ndarray,
    color: str,
    box_limits: list[tuple[float, float]],
) -> None:
    ax.plot(*trace[:, :3].T, color=color, lw=0.3, alpha=0.85)
    for axis_index, setter in enumerate(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim)
    ):
        setter(*box_limits[axis_index])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=-58)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((0.93, 0.93, 0.93, 1.0))
        axis.pane.set_edgecolor((0.75, 0.75, 0.75, 1.0))


def _draw_phase_panel(
    ax: Any,
    trace: np.ndarray,
    color: str,
    context: PlotContext,
) -> None:
    box_limits, phase_ticks = (
        context.appearance.box_limits,
        context.appearance.phase_ticks,
    )
    system = str(context.computed["system"])
    ax.plot(*trace[:, :2].T, color=color, lw=0.3, alpha=0.85)
    for axis_index, setter in enumerate((ax.set_xlim, ax.set_ylim)):
        setter(*box_limits[axis_index])
    ax.set_xticks(phase_ticks)
    if system == "mackeyglass" and phase_ticks == (-0.5, 0.0):
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _pos: "\N{MINUS SIGN}0.5" if value < 0 else "0"
            )
        )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, prune="both"))


def _draw_hovmoller_panel(
    ax: Any,
    block: np.ndarray,
    horizon: int,
    context: PlotContext,
) -> Any:
    reference, settings = context.reference, context.settings
    steps_per_lt = float(context.computed["steps_per_lt"])
    norm = context.appearance.norm
    times = np.arange(horizon, len(reference)) / steps_per_lt
    heatmap = ax.pcolormesh(
        times,
        np.arange(reference.shape[1]),
        np.ma.masked_invalid(block.T),
        cmap="RdBu_r",
        norm=norm,
        shading="nearest",
        rasterized=True,
    )
    ax.set(
        xlim=(
            max(settings["prediction_horizons_lt"]),
            float(settings["window_lt"]),
        ),
        ylim=(-0.5, reference.shape[1] - 0.5),
    )
    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=2, integer=True, prune="both"),
    )
    num_variables = reference.shape[1]
    ax.set_yticks((round(num_variables / 8), round(7 * num_variables / 8)))
    return heatmap


def _plot_limits(
    reference: np.ndarray,
    kind: str,
    system: str,
    lag_steps: int,
) -> tuple[Any, Any, Any, Any]:
    if kind == "hovmoller":
        vmax = float(np.max(np.abs(reference)))
        if vmax == 0:
            raise ValueError("reference window has zero color range")
        return Normalize(vmin=-vmax, vmax=vmax), vmax, None, None
    trace = (
        delay_trace(reference, lag_steps)
        if system == "mackeyglass"
        else reference
    )
    box_limits = []
    for axis_index in range(3 if kind == "3d" else 2):
        low = float(trace[:, axis_index].min())
        high = float(trace[:, axis_index].max())
        margin = 0.05 * (high - low) if high > low else 1.0
        box_limits.append((low - margin, high + margin))
    phase_ticks = (
        _phase_x_ticks(system, *box_limits[0]) if kind == "phase" else None
    )
    return None, None, box_limits, phase_ticks


def _draw_row(
    figure: Any,
    axes: Any,
    row: int,
    horizon: int,
    context: PlotContext,
) -> tuple[Any, Any]:
    computed, models, settings = (
        context.computed,
        context.models,
        context.settings,
    )
    reference = context.reference
    kind, titles = context.appearance.kind, context.appearance.titles
    box_limits = context.appearance.box_limits
    system = str(computed["system"])
    last_row = row == len(computed["prediction_horizon_steps"]) - 1
    phase = kind == "phase"
    blocks = [reference[horizon:]] + [
        computed[f"{model}__horizon_steps_{horizon}"] for model in models
    ]
    heatmap = None
    for column, (ax, raw_block) in enumerate(
        zip(axes[row], blocks, strict=True),
    ):
        block = np.asarray(raw_block)
        if block.shape != reference[horizon:].shape or not len(block):
            raise ValueError(
                "prediction and reference windows must have matching shapes",
            )
        if row == 0:
            ax.set_title(
                titles[column].replace(" (", "\n("),
                fontsize=FONT_SIZE,
                pad=3,
                fontweight="bold",
            )
        if kind == "hovmoller":
            heatmap = _draw_hovmoller_panel(
                ax,
                block,
                horizon,
                context,
            )
        else:
            trace = (
                delay_trace(block, int(computed["delay_lag_steps"]))
                if system == "mackeyglass"
                else block
            )
            color = "0.35" if column == 0 else MODEL_COLORS[models[column - 1]]
            if kind == "3d":
                _draw_3d_panel(ax, trace, color, box_limits)
            else:
                _draw_phase_panel(
                    ax,
                    trace,
                    color,
                    context,
                )
        if kind != "3d":
            ax.tick_params(
                labelbottom=last_row,
                labelleft=column == 0,
                labeltop=False,
                labelright=False,
                labelsize=SMALL_FONT_SIZE,
                pad=1,
                length=1.5,
            )
            if phase:
                ax.tick_params(
                    which="both",
                    bottom=last_row,
                    top=False,
                    left=column == 0,
                    right=False,
                )
            if phase and last_row:
                ax.set_xlabel(
                    PHASE_LABELS[system][0],
                    fontsize=FONT_SIZE,
                    labelpad=1,
                )
            if phase and column == 0:
                ax.set_ylabel(
                    PHASE_LABELS[system][1],
                    fontsize=FONT_SIZE,
                    labelpad=1,
                )
    position = axes[row, 0].get_position()
    extent = axes[row, 0].get_tightbbox()
    left = figure.transFigure.inverted().transform((extent.x0, 0.0))[0]
    label = figure.text(
        left - 0.008,
        (position.y0 + position.y1) / 2,
        f"{settings['prediction_horizons_lt'][row]:.1f} LT",
        rotation=90,
        va="center",
        ha="center",
        fontsize=FONT_SIZE,
    )
    return heatmap, label


def _create_layout(
    system: str,
    kind: str,
    rows: int,
    columns: int,
) -> tuple[Any, Any]:
    width, height = FIGSIZE_MM["attractors"][system]
    return plt.subplots(
        rows,
        columns,
        figsize=(width / 25.4, height / 25.4),
        dpi=300,
        squeeze=False,
        subplot_kw={"projection": "3d"} if kind == "3d" else {},
    )


def build_figure(
    computed: AttractorArrays,
    models: list[str],
    settings: dict,
) -> Any:
    validate_arrays(computed, settings)
    system = str(computed["system"])
    kind = plot_kind(system)
    titles = ("Truth", *(FIGURE_LABELS[name] for name in models))
    reference = np.asarray(computed["reference"])
    prediction_horizon_steps = np.asarray(
        computed["prediction_horizon_steps"],
        dtype=int,
    )
    if (
        not len(prediction_horizon_steps)
        or reference.ndim != MATRIX_DIMENSIONS
        or not np.isfinite(reference).all()
    ):
        raise ValueError(
            "expected stored horizons and a finite time-by-measured window",
        )
    figure, axes = _create_layout(
        system,
        kind,
        len(prediction_horizon_steps),
        len(models) + 1,
    )
    try:
        phase = kind == "phase"
        figure.subplots_adjust(**SUBPLOT_LAYOUT[kind])
        row_labels = []
        norm, vmax, box_limits, phase_ticks = _plot_limits(
            reference,
            kind,
            system,
            int(computed["delay_lag_steps"]) if system == "mackeyglass" else 0,
        )
        context = PlotContext(
            computed,
            models,
            settings,
            reference,
            PlotAppearance(kind, titles, norm, box_limits, phase_ticks),
        )
        for row, horizon in enumerate(prediction_horizon_steps):
            heatmap, label = _draw_row(
                figure,
                axes,
                row,
                horizon,
                context,
            )
            row_labels.append(label)
        if kind == "hovmoller":
            bottom = axes[-1, 0].get_position().y0
            top = axes[0, 0].get_position().y1
            cax = figure.add_axes((0.943, bottom, 0.009, top - bottom))
            decade = 10 ** math.floor(math.log10(vmax))
            bound = max(
                value * decade
                for value in (1, 2, 5, 10)
                if value * decade <= vmax
            )
            colorbar = figure.colorbar(
                heatmap,
                cax=cax,
                ticks=(-bound, 0, bound),
                format=FuncFormatter(
                    lambda value, _pos: f"{value:g}".replace(
                        "-", "\N{MINUS SIGN}"
                    ),
                ),
            )
            colorbar.minorticks_off()
            cax.tick_params(labelsize=SMALL_FONT_SIZE, pad=1, length=2)
        if phase:
            # Keep the measured 5.5-point clearance for PDF mathtext.
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for row, label in enumerate(row_labels):
                label_box = label.get_window_extent(renderer)
                axis_box = axes[row, 0].yaxis.label.get_window_extent(renderer)
                shift = max(
                    0,
                    label_box.x1 - axis_box.x0 + 5.5 * figure.dpi / 72,
                )
                label.set_x(
                    label.get_position()[0] - shift / figure.bbox.width,
                )
        return figure
    except Exception:
        plt.close(figure)
        raise


def render(
    computed: AttractorArrays, models: list[str], path: Path, settings: dict
) -> None:
    figure = build_figure(computed, models, settings)
    try:
        save_pdf(figure, path)
    finally:
        plt.close(figure)
