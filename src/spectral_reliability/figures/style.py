from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 -- Registers the SciencePlots styles.
from matplotlib.figure import Figure

from ..config_shared import BASELINES, MODEL_LABELS, MODEL_VARIANTS

matplotlib.use("Agg")

SMALL_FONT_SIZE = 8
FONT_SIZE = 9
PLAIN_DECIMAL_LIMIT = 100000
PANEL_FONT_SIZE = 10
SANS_FAMILIES = [
    "Arimo",
    "Liberation Sans",
    "Nimbus Sans",
    "Arial",
    "Helvetica",
    "DejaVu Sans",
]

plt.style.use(["science", "ieee"])
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": SANS_FAMILIES,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arimo",
        "mathtext.it": "Arimo:italic",
        "mathtext.bf": "Arimo:bold",
        "mathtext.sf": "Arimo",
        "mathtext.fallback": None,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "figure.labelsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "xtick.labelsize": SMALL_FONT_SIZE,
        "ytick.labelsize": SMALL_FONT_SIZE,
        "legend.fontsize": SMALL_FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

MODELS_BY_TRAINING = {
    procedure: tuple(
        name for name in MODEL_VARIANTS if name.endswith(f"_{procedure}")
    )
    for procedure in ("two_stage", "joint")
}
FIGURE_LABELS = {
    "latent_prediction_two_stage": "Prediction (two stage)",
    "latent_prediction_penalty_two_stage": "Prediction+L2 (two stage)",
    "spectral_residual_two_stage": "Residual (two stage)",
    "spectral_residual_penalty_two_stage": "Residual+L2 (two stage)",
    "latent_prediction_joint": "Prediction (joint)",
    "latent_prediction_penalty_joint": "Prediction+L2 (joint)",
    "spectral_residual_joint": "Residual (joint)",
    "spectral_residual_penalty_joint": "Residual+L2 (joint)",
}
FIGURE_LABELS.update({model: MODEL_LABELS[model] for model in BASELINES})
MODEL_COLORS = {
    "latent_prediction_two_stage": "#2E5FA7",
    "latent_prediction_joint": "#2E5FA7",
    "latent_prediction_penalty_two_stage": "#4FB3D9",
    "latent_prediction_penalty_joint": "#4FB3D9",
    "spectral_residual_two_stage": "#B84252",
    "spectral_residual_joint": "#B84252",
    "spectral_residual_penalty_two_stage": "#E8913A",
    "spectral_residual_penalty_joint": "#E8913A",
    "esn": "#8C8C8C",
    "neural_ode": "#5B9E93",
    "kernel_dmd": "#8B7FD1",
    "consistent_kae": "#A98A5C",
}
FIGSIZE_MM = {
    "attractors": {
        "rossler": (141, 80),
        "lorenz63": (140, 80),
        "duffing": (140, 84),
        "mackeyglass": (140, 84),
        "lorenz96": (126, 84),
        "ks": (126, 84),
    },
    "pseudospectra": {
        "rossler": (160, 114.3),
        "lorenz63": (152, 114.3),
        "duffing": (152, 114.3),
        "mackeyglass": (152, 114.3),
        "lorenz96": (152, 114.3),
        "ks": (152, 114.3),
    },
    "scaling": {"all": (160, 124.46)},
}


def plain_log_tick(
    value: float, _pos: float | None = None, *, log10: bool = False
) -> str:
    if log10:
        value = 10.0**value
    if not math.isfinite(value):
        return ""
    if value == 0:
        return "0"
    decimal = f"{value:.4f}"
    if abs(value) < PLAIN_DECIMAL_LIMIT and math.isclose(
        float(decimal), value, rel_tol=1e-9, abs_tol=1e-12
    ):
        return decimal.rstrip("0").rstrip(".") if "." in decimal else decimal
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}".replace(
        "-", "\N{MINUS SIGN}"
    )


def save_pdf(figure: Figure, path: Path) -> None:
    """Save the full canvas with 300-dpi raster fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    figure.savefig(path, format="pdf", dpi=300, bbox_inches=None)
