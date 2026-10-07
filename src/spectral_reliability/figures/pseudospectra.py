from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

from ..pseudospectra import PseudospectrumArrays
from .style import (
    FIGSIZE_MM,
    FIGURE_LABELS,
    FONT_SIZE,
    SMALL_FONT_SIZE,
    plain_log_tick,
    save_pdf,
)

MINIMUM_COLORBAR_TICKS = 2


def load_arrays(path: Path, variants: list[str]) -> PseudospectrumArrays:
    with np.load(path, allow_pickle=False) as data:
        x_pts, y_pts = data["x_pts"], data["y_pts"]
        return {
            model: (
                {
                    "x_pts": x_pts,
                    "y_pts": y_pts,
                    "log10_min_residual": data[f"{model}__log10_min_residual"],
                },
                data[f"{model}__eigenvalues"],
            )
            for model in variants
        }


def build_figure(
    computed: PseudospectrumArrays, system: str, settings: dict
) -> Any:
    variants = settings["variants"]
    half_width = settings["half_width"]
    all_values = np.concatenate(
        [
            computed[model][0]["log10_min_residual"].reshape(-1)
            for model in variants
        ]
    )
    finite = all_values[np.isfinite(all_values)]
    if not finite.size:
        raise ValueError("pseudospectra contain no finite residual")
    lower, upper = float(finite.min()), float(finite.max())
    if lower == upper:
        lower -= 1
        upper += 1
    levels = np.linspace(lower, upper, settings["contour_levels"])
    norm = Normalize(vmin=lower, vmax=upper)
    width, height = FIGSIZE_MM["pseudospectra"][system]
    columns = min(4, len(variants))
    rows = math.ceil(len(variants) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width / 25.4, height / 25.4),
        dpi=300,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    try:
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        for index, (ax, model) in enumerate(
            zip(tuple(axes.flat)[: len(variants)], variants, strict=True),
        ):
            pseudospectrum, eigenvalues = computed[model]
            field = np.asarray(
                pseudospectrum["log10_min_residual"], dtype=np.float64
            )
            contours = ax.contourf(
                pseudospectrum["x_pts"],
                pseudospectrum["y_pts"],
                np.ma.masked_invalid(np.clip(field, lower, upper)),
                levels=levels,
                cmap="viridis",
                norm=norm,
                extend="neither",
            )
            contours.set_rasterized(True)
            ax.plot(np.cos(theta), np.sin(theta), color="white", lw=0.5)
            ax.plot(
                eigenvalues.real,
                eigenvalues.imag,
                linestyle="none",
                marker="o",
                markersize=1.5,
                color="red",
                markeredgewidth=0,
            )
            ax.set_title(
                FIGURE_LABELS[model].replace(" (", "\n("),
                fontsize=FONT_SIZE,
                pad=4,
                fontweight="bold",
            )
            ax.set(
                xlim=(-half_width, half_width),
                ylim=(-half_width, half_width),
                aspect="equal",
            )
            row, column = divmod(index, columns)
            if row == rows - 1:
                ax.set_xlabel("Real")
            if column == 0:
                ax.set_ylabel("Imaginary")
            ax.tick_params(
                labelbottom=row == rows - 1,
                labelleft=column == 0,
                labeltop=False,
                labelright=False,
                labelsize=SMALL_FONT_SIZE,
            )
        for ax in tuple(axes.flat)[len(variants) :]:
            ax.set_visible(False)
        figure.tight_layout(
            rect=(0, 0, 0.92, 1), pad=0.8, w_pad=0.6, h_pad=3.0
        )
        bottom = min(ax.get_position().y0 for ax in axes.flat)
        top = max(ax.get_position().y1 for ax in axes.flat)
        cax = figure.add_axes((0.915, bottom, 0.012, top - bottom))
        ticks = np.arange(np.ceil(lower), np.floor(upper) + 1)
        if ticks.size < MINIMUM_COLORBAR_TICKS:
            offsets = (0.0, math.log10(2.0), math.log10(5.0))
            ticks = [
                power + offset
                for power in range(math.floor(lower), math.ceil(upper) + 1)
                for offset in offsets
                if lower <= power + offset <= upper
            ]
        colorbar = figure.colorbar(
            contours,
            cax=cax,
            ticks=ticks,
            format=FuncFormatter(
                lambda value, pos: plain_log_tick(value, pos, log10=True)
            ),
        )
        cax.tick_params(labelsize=SMALL_FONT_SIZE)
        colorbar.minorticks_off()
        return figure
    except Exception:
        plt.close(figure)
        raise


def render(
    computed: PseudospectrumArrays, system: str, path: Path, settings: dict
) -> None:
    figure = build_figure(computed, system, settings)
    try:
        save_pdf(figure, path)
    finally:
        plt.close(figure)
