from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from spectral_reliability.training_config import LossConfig

EIGENVALUE_SEPARATION_FLOOR = 1e-10
MATRIX_DIMENSIONS = 2


def machine_precision_rank(
    singular_values: torch.Tensor,
    m: int,
    n: int,
) -> int:
    if singular_values.numel() == 0:
        return 0
    float64_epsilon = torch.finfo(torch.float64).eps
    tolerance = max(int(m), int(n)) * float64_epsilon * singular_values.max()
    return int((singular_values > tolerance).sum().item())


def svd_with_fallback(
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a reduced SVD, falling back on CUDA failure."""
    try:
        return torch.linalg.svd(z, full_matrices=False)
    except torch.linalg.LinAlgError as exc:
        if not torch.isfinite(z).all():
            raise ValueError(
                "SVD input is not finite; the latent diverged before "
                "the decomposition"
            ) from exc
        if not z.is_cuda:
            raise
        try:
            return torch.linalg.svd(z, full_matrices=False, driver="gesvd")
        except torch.linalg.LinAlgError:
            u, singular_values, vh = torch.linalg.svd(
                z.cpu(),
                full_matrices=False,
            )
            return (
                u.to(z.device),
                singular_values.to(z.device),
                vh.to(z.device),
            )


def regularization_parameter(
    singular_values: torch.Tensor,
    latent_dim: int,
    regularization: float,
) -> float:
    """Return epsilon = c ||z_x||_F^2 / N from its singular values."""
    if not regularization:
        return 0.0
    spectrum_energy = (singular_values.double().detach() ** 2).sum()
    return float(regularization) * float(spectrum_energy) / int(latent_dim)


def retained_rank(
    singular_values: torch.Tensor,
    m: int,
    n: int,
    regularization: float,
) -> int:
    ceiling = machine_precision_rank(singular_values, m, n)
    epsilon = regularization_parameter(
        singular_values,
        m,
        regularization=regularization,
    )
    requested = max(1, int((singular_values**2 >= epsilon).sum().item()))
    return min(requested, ceiling)


def truncated_svd(
    z: torch.Tensor,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    u, singular_values, vh = svd_with_fallback(z)
    v = vh.mH
    rank = retained_rank(
        singular_values,
        *z.shape[-2:],
        regularization=regularization,
    )
    return u[:, :rank], singular_values[:rank], v[:, :rank], rank


def ridge_singular_inverse(
    singular_values: torch.Tensor,
    z_x: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    """Use regularization_parameter's epsilon in the fitting dtype."""
    if not regularization:
        return 1.0 / singular_values
    data_energy = (z_x.detach() ** 2).sum()
    epsilon = (
        z_x.new_tensor(float(regularization)) * data_energy / z_x.shape[-2]
    )
    return singular_values / (singular_values**2 + epsilon)


def lift_reduced_operator(
    q_r: torch.Tensor,
    a_tilde: torch.Tensor,
) -> torch.Tensor:
    """Lift a_hat_tr,r = q_r a_tilde q_r^dagger."""
    return q_r @ a_tilde @ torch.linalg.pinv(q_r)


@dataclass(frozen=True)
class KoopmanFit:
    u: torch.Tensor
    rank: int
    a_tilde: torch.Tensor
    a_hat: torch.Tensor

    def eigenpairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return eigenvalues of a_tilde^T and a_k = u v_k.

        u is N-by-r, the reduced vectors are r-by-r, and the lifted
        eigenfunction coefficients a_k are columns of an N-by-r matrix.
        """
        eigenvalues, reduced_vectors = torch.linalg.eig(self.a_tilde.T)
        return eigenvalues, self.u.to(reduced_vectors.dtype) @ reduced_vectors


def rank_constrained_operator(
    z_x: torch.Tensor,
    z_y: torch.Tensor,
    *,
    regularization: float,
    dtype: torch.dtype,
    svd_rank: float | None = None,
) -> KoopmanFit:
    """Construct the rank-constrained lift
    a_hat_tr,r = q_r a_tilde q_r^dagger.

    None uses the regularization threshold and machine-precision ceiling
    for validation. A float retains that fraction of singular-value
    energy; -1 retains all singular values. The two paths keep their
    original epsilon arithmetic and share only the construction tail.
    """
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError(
            "Koopman fitting dtype must be torch.float32 or torch.float64",
        )
    z_x = z_x.to(dtype=dtype)
    z_y = z_y.to(dtype=dtype)
    if svd_rank is None:
        u, singular_values, v, rank = truncated_svd(
            z_x,
            regularization=regularization,
        )
        inverse = ridge_singular_inverse(
            singular_values,
            z_x,
            regularization,
        )
    else:
        u, singular_values, vh = svd_with_fallback(z_x)
        energy = singular_values.double().square()
        if energy.sum() == 0:
            rank = 0
        elif isinstance(svd_rank, float):
            indices = torch.nonzero(
                (energy / energy.sum()).cumsum(0) >= svd_rank,
            )
            rank = (
                int(indices[0].item()) + 1
                if indices.numel()
                else len(singular_values)
            )
        elif svd_rank == -1:
            rank = len(singular_values)
        elif isinstance(svd_rank, int) and svd_rank > 0:
            rank = min(svd_rank, len(singular_values))
        else:
            raise ValueError(
                "svd_rank must be -1, a positive integer or a float in (0, 1]",
            )
        u = u[:, :rank]
        singular_values = singular_values[:rank]
        v = vh.T[:, :rank]
        # Evaluation forms regularization_parameter's epsilon in
        # float64.
        epsilon = (
            float(regularization)
            * float(z_x.detach().double().square().sum())
            / z_x.shape[0]
        )
        inverse = (
            singular_values / (singular_values.square() + epsilon)
            if regularization
            else singular_values.reciprocal()
        )
    q_r = z_y @ v @ torch.diag(inverse)
    a_tilde = u.T @ q_r
    return KoopmanFit(
        u=u,
        rank=rank,
        a_tilde=a_tilde,
        a_hat=lift_reduced_operator(q_r, a_tilde),
    )


def estimate_latent_operator(
    z_x: torch.Tensor,
    z_y: torch.Tensor,
    *,
    regularization: float,
    dtype: torch.dtype,
) -> KoopmanFit:
    """Estimate a_hat and its detached retained Gram eigenspace.

    The ridge parameter follows regularization_parameter's definition,
    evaluated directly from z_x in float64 rather than through an SVD.
    """
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError(
            "Koopman fitting dtype must be torch.float32 or torch.float64"
        )
    z_x = z_x.to(dtype=dtype)
    z_y = z_y.to(dtype=dtype)
    if z_x.ndim != MATRIX_DIMENSIONS or z_y.ndim != MATRIX_DIMENSIONS:
        raise ValueError(
            f"z_x and z_y must be matrices, got {tuple(z_x.shape)} "
            f"and {tuple(z_y.shape)}"
        )
    if z_x.shape != z_y.shape:
        raise ValueError(
            f"z_x and z_y must have the same shape, got {tuple(z_x.shape)} "
            f"and {tuple(z_y.shape)}"
        )
    latent_dim = int(z_x.shape[0])
    if latent_dim == 0:
        raise ValueError("the latent dimension must be positive")
    epsilon = (
        float(regularization)
        * float((z_x.detach().double() ** 2).sum())
        / latent_dim
        if float(regularization) != 0.0
        else 0.0
    )
    gram = z_x @ z_x.T
    identity = torch.eye(latent_dim, dtype=dtype, device=z_x.device)
    regularized_gram = gram + epsilon * identity
    cross_correlation = z_y @ z_x.T  # Equals m G_XY^T.
    a_matrix = torch.linalg.solve(regularized_gram.T, cross_correlation.T).T
    with torch.no_grad():
        kappa, u = torch.linalg.eigh(gram.detach())
        kappa = kappa.flip(0)
        u = u.flip(1)
        retained_mask = kappa >= epsilon
        u = u[:, retained_mask]
        rank = int(retained_mask.sum().item())
    q_r = a_matrix @ u
    a_tilde = u.T @ q_r
    return KoopmanFit(u=u, rank=rank, a_tilde=a_tilde, a_hat=a_matrix)


def _operator_norm_penalty(
    fit: KoopmanFit,
    operator_norm_weight: float,
) -> torch.Tensor | None:
    if not operator_norm_weight:
        return None
    return float(operator_norm_weight) * torch.linalg.matrix_norm(
        fit.a_hat, ord=2
    )


def latent_prediction_loss(
    fit: KoopmanFit,
    z_x_prime: torch.Tensor,
    z_y_prime: torch.Tensor,
) -> torch.Tensor:
    dtype = fit.a_hat.dtype
    return (
        (fit.a_hat @ z_x_prime.to(dtype) - z_y_prime.to(dtype)) ** 2
    ).mean()


def _mean_squared_relative_eigenfunction_residual(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    z_x: torch.Tensor,
    z_y: torch.Tensor,
) -> torch.Tensor:
    eigenfunction_values_x = eigenvectors.T @ z_x.to(eigenvectors.dtype)
    eigenfunction_values_y = eigenvectors.T @ z_y.to(eigenvectors.dtype)
    residual = (
        eigenfunction_values_y - eigenvalues[:, None] * eigenfunction_values_x
    )
    numerator = residual.abs().square().sum(dim=1)
    denominator = eigenfunction_values_x.abs().square().sum(dim=1)
    return (numerator / denominator).mean().real


def _bounded_eigenvalue_differences(eigenvalues: torch.Tensor) -> torch.Tensor:
    differences = eigenvalues[:, None] - eigenvalues[None, :]
    magnitudes = differences.abs()
    phase = torch.where(
        magnitudes > 0,
        differences / magnitudes,
        torch.ones_like(differences),
    )
    return torch.where(
        magnitudes < EIGENVALUE_SEPARATION_FLOOR,
        EIGENVALUE_SEPARATION_FLOOR * phase,
        differences,
    )


class SpectralResidualFunction(torch.autograd.Function):
    """Differentiate mean squared relative eigenfunction residuals.

    Backward differentiates eigenvalues, eigenvectors and held-out data
    separately, then applies the eigenpair perturbation formula.
    Pairwise eigenvalue gaps are bounded by
    EIGENVALUE_SEPARATION_FLOOR; the eigenspace selected by
    estimate_latent_operator remains detached.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        a_matrix: torch.Tensor,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
    ) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eig(a_matrix.T)
        ctx.save_for_backward(eigenvalues, eigenvectors, z_x, z_y)
        return _mean_squared_relative_eigenfunction_residual(
            eigenvalues,
            eigenvectors,
            z_x,
            z_y,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eigenvalues, eigenvectors, z_x, z_y = ctx.saved_tensors
        with torch.enable_grad():
            eigenvalues_leaf = eigenvalues.clone().requires_grad_(True)
            eigenvectors_leaf = eigenvectors.clone().requires_grad_(True)
            z_x_leaf = z_x.clone().requires_grad_(True)
            z_y_leaf = z_y.clone().requires_grad_(True)
            residual = _mean_squared_relative_eigenfunction_residual(
                eigenvalues_leaf,
                eigenvectors_leaf,
                z_x_leaf,
                z_y_leaf,
            )
            (
                grad_eigenvalues,
                grad_eigenvectors,
                grad_z_x,
                grad_z_y,
            ) = torch.autograd.grad(
                residual,
                (
                    eigenvalues_leaf,
                    eigenvectors_leaf,
                    z_x_leaf,
                    z_y_leaf,
                ),
                grad_outputs=grad_output,
                create_graph=False,
                allow_unused=False,
            )
        try:
            eigenvectors_inverse = torch.linalg.inv(eigenvectors)
        except torch.linalg.LinAlgError:
            eigenvectors_inverse = torch.linalg.pinv(eigenvectors)
        differences = _bounded_eigenvalue_differences(eigenvalues)
        eigenbasis_gradient = (
            grad_eigenvectors.conj().T @ eigenvectors
        ) / differences
        diagonal = torch.arange(eigenvalues.numel(), device=eigenvalues.device)
        eigenbasis_gradient[diagonal, diagonal] = grad_eigenvalues.conj()
        grad_a = (
            eigenvectors @ eigenbasis_gradient @ eigenvectors_inverse
        ).real
        return grad_a, grad_z_x, grad_z_y


def spectral_residual_loss(
    fit: KoopmanFit,
    z_x_prime: torch.Tensor,
    z_y_prime: torch.Tensor,
    *,
    backward: Literal["custom", "autograd"],
) -> torch.Tensor:
    """Return held-out spectral residuals with the chosen backward
    policy.
    """
    if fit.a_tilde.dtype != torch.float64:
        raise ValueError("residual losses require float64 Koopman fits")
    if backward == "custom":
        u = fit.u
        return SpectralResidualFunction.apply(
            fit.a_tilde,
            u.T @ z_x_prime.to(u.dtype),
            u.T @ z_y_prime.to(u.dtype),
        )
    if backward == "autograd":
        eigenvalues, eigenvectors = fit.eigenpairs()
        return _mean_squared_relative_eigenfunction_residual(
            eigenvalues,
            eigenvectors,
            z_x_prime,
            z_y_prime,
        )
    raise ValueError("residual_backward must be custom or autograd")


def evolution_loss(
    fit: KoopmanFit,
    z_x_prime: torch.Tensor,
    z_y_prime: torch.Tensor,
    config: LossConfig,
) -> torch.Tensor:
    """Dispatch L_evo and add the operator-norm penalty once."""
    if config.type == "latent_prediction":
        loss = latent_prediction_loss(fit, z_x_prime, z_y_prime)
    elif config.type == "spectral_residual":
        loss = spectral_residual_loss(
            fit,
            z_x_prime,
            z_y_prime,
            backward=config.residual_backward,
        )
    else:
        raise ValueError(f"unknown evolution loss {config.type!r}")
    penalty = _operator_norm_penalty(fit, config.operator_norm_weight)
    return loss if penalty is None else loss + penalty.to(loss.dtype)
