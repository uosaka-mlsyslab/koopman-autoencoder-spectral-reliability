from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.dataset import Dataset
from .evaluation import TrainedModel

SVD_RANK_RTOL = 1.0e-10
CPU_BATCH_BYTES = 128 * 1024**2
ACCELERATOR_BATCH_BYTES = 256 * 1024**2
MATRIX_DIMENSIONS = 2
MINIMUM_PAIR_STATES = 2
PseudospectrumArrays = dict[str, tuple[dict[str, np.ndarray], np.ndarray]]


def compute_pseudospectra(
    psi_x: torch.Tensor,
    psi_y: torch.Tensor,
    w: torch.Tensor,
    z_pts: np.ndarray,
    device: str,
) -> np.ndarray:
    """Return the minimal residual at each complex point of z_pts.

    This is the QR form of Colbrook's KoopPseudoSpecQR. psi_x and psi_y
    are M-by-N evaluation matrices, and w contains M quadrature weights.
    l_matrix, a_matrix and identity represent G_YY, G_XY and G_XX.
    A truncated SVD replaces QR when psi_x is rank deficient. Residuals
    are flat and follow z_pts order; chunk budgets are given in bytes.
    """
    torch_device = torch.device(device)
    psi_x = torch.as_tensor(
        psi_x,
        dtype=torch.float64,
        device=torch_device,
    ).detach()
    psi_y = torch.as_tensor(
        psi_y,
        dtype=torch.float64,
        device=torch_device,
    ).detach()
    if (
        psi_x.ndim != MATRIX_DIMENSIONS
        or psi_x.shape != psi_y.shape
        or not bool(torch.isfinite(psi_x).all() & torch.isfinite(psi_y).all())
    ):
        raise ValueError(
            "psi_x and psi_y must be finite matrices of equal shape"
        )
    sample_count, latent_dim = psi_x.shape
    if not latent_dim or sample_count < latent_dim:
        raise ValueError(
            "need at least as many snapshot pairs as latent coordinates",
        )
    w = torch.as_tensor(w, dtype=torch.float64, device=torch_device)
    if (
        w.shape != (sample_count,)
        or not bool(torch.isfinite(w).all())
        or bool((w < 0).any())
    ):
        raise ValueError("w must be a finite nonnegative vector of length M")
    sqrt_w = torch.sqrt(w).reshape(-1, 1)
    sqrt_w_psi_x = sqrt_w * psi_x
    sqrt_w_psi_y = sqrt_w * psi_y
    q, r = torch.linalg.qr(sqrt_w_psi_x, mode="reduced")
    singular_values = torch.linalg.svdvals(r)
    threshold = SVD_RANK_RTOL * singular_values[0]
    full_rank = bool(torch.all(singular_values > threshold).item())
    if full_rank:
        c1 = torch.linalg.solve(r.mT, sqrt_w_psi_y.mT).mT
        subspace_dimension = latent_dim
    else:
        u, s, vh = torch.linalg.svd(sqrt_w_psi_x, full_matrices=False)
        keep = s > (SVD_RANK_RTOL * s[0])
        subspace_dimension = int(keep.sum().item())
        if subspace_dimension == 0:
            raise ValueError("psi_x has no retained nonzero singular value")
        inverse_s = 1.0 / s[:subspace_dimension]
        q = u[:, :subspace_dimension]
        c1 = sqrt_w_psi_y @ (
            vh[:subspace_dimension, :].mH * inverse_s[None, :]
        )
    l_matrix = c1.mH @ c1
    a_matrix = q.mH @ c1
    identity = torch.eye(
        subspace_dimension,
        dtype=torch.float64,
        device=torch_device,
    )
    z_pts = torch.as_tensor(
        z_pts,
        dtype=torch.complex128,
        device=torch_device,
    ).reshape(-1)
    target_bytes = (
        CPU_BATCH_BYTES
        if torch_device.type == "cpu"
        else ACCELERATOR_BATCH_BYTES
    )
    bytes_per_matrix = 16 * latent_dim * latent_dim
    batch_size = max(1, min(z_pts.numel(), target_bytes // bytes_per_matrix))
    min_residual = np.empty(z_pts.numel(), dtype=np.float64)
    a_adjoint = a_matrix.mH
    for start in range(0, z_pts.numel(), batch_size):
        stop = min(start + batch_size, z_pts.numel())
        z = z_pts[start:stop, None, None]
        l_adjusted = (
            l_matrix[None, :, :]
            - z * a_adjoint[None, :, :]
            - z.conj() * a_matrix[None, :, :]
            + z.abs().square() * identity[None, :, :]
        )
        l_adjusted = 0.5 * (l_adjusted + l_adjusted.mH)
        smallest_eigenvalues = torch.linalg.eigvalsh(l_adjusted)[..., 0]
        min_residual[start:stop] = (
            torch.sqrt(torch.clamp(smallest_eigenvalues, min=0.0))
            .detach()
            .cpu()
            .numpy()
        )
    return min_residual


def consecutive_pairs(
    dataset: Dataset,
    splits: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_blocks, y_blocks = [], []
    for name in splits:
        series = np.asarray(getattr(dataset, name))
        if (
            series.ndim != MATRIX_DIMENSIONS
            or series.shape[1] < MINIMUM_PAIR_STATES
        ):
            raise ValueError(f"{name} split holds no consecutive pair")
        x_blocks.append(series[:, :-1].T)
        y_blocks.append(series[:, 1:].T)
    return np.concatenate(x_blocks), np.concatenate(y_blocks)


def _encode(
    encoder: Any,
    matrix: np.ndarray,
    dtype: Any,
    batch_size: int,
) -> Any:
    """Encode observations on CPU, retaining the single-batch path."""
    with torch.no_grad():
        if matrix.shape[0] <= batch_size:
            return encoder(torch.as_tensor(matrix, dtype=dtype))
        blocks = [
            encoder(
                torch.as_tensor(
                    matrix[start : start + batch_size],
                    dtype=dtype,
                )
            )
            for start in range(0, matrix.shape[0], batch_size)
        ]
    return torch.cat(blocks)


def compute_arrays(
    models: Mapping[str, TrainedModel],
    dataset: Dataset,
    *,
    settings: dict,
    device: str,
) -> PseudospectrumArrays:
    """Evaluate pooled split pairs and overlay the stored training
    operator.

    Encoding stays on CPU at checkpoint precision, independently of the
    device used for the residual computation.
    """
    x, y = consecutive_pairs(dataset, settings["splits"])
    half_width = settings["half_width"]
    x_pts = y_pts = np.linspace(
        -half_width,
        half_width,
        settings["grid_points_per_axis"],
        dtype=np.float64,
    )
    z_pts = (x_pts[None, :] + 1j * y_pts[:, None]).reshape(-1)
    w = torch.full((len(x),), 1.0 / len(x), dtype=torch.float64)
    computed = {}
    for name in settings["variants"]:
        model = models[name].to("cpu")
        model.encoder.eval()
        parameter = next(model.encoder.parameters())
        psi_x = _encode(
            model.encoder,
            x,
            parameter.dtype,
            settings["encoder_batch_size"],
        )
        psi_y = _encode(
            model.encoder,
            y,
            parameter.dtype,
            settings["encoder_batch_size"],
        )
        operator = model.full_rank_operator.to(torch.float64)
        eigenvalues = torch.linalg.eigvals(operator).cpu().numpy()
        min_residual = compute_pseudospectra(
            psi_x,
            psi_y,
            w,
            z_pts,
            device,
        ).reshape(len(y_pts), len(x_pts))
        log10_min_residual = np.log10(
            np.maximum(min_residual, np.finfo(np.float64).tiny),
        )
        computed[name] = (
            {
                "x_pts": x_pts,
                "y_pts": y_pts,
                "log10_min_residual": log10_min_residual,
            },
            eigenvalues,
        )
    return computed


def save_arrays(computed: PseudospectrumArrays, path: Path) -> None:
    arrays_dict = {}
    for model, (pseudospectrum, eigenvalues) in computed.items():
        if not arrays_dict:
            arrays_dict["x_pts"] = np.asarray(pseudospectrum["x_pts"])
            arrays_dict["y_pts"] = np.asarray(pseudospectrum["y_pts"])
        arrays_dict[f"{model}__log10_min_residual"] = np.asarray(
            pseudospectrum["log10_min_residual"],
        )
        arrays_dict[f"{model}__eigenvalues"] = np.asarray(eigenvalues)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays_dict)
