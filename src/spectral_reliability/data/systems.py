from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jitcdde import jitcdde, y
from jitcdde import t as jitcdde_t
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RosslerConfig:
    reference_state: tuple[float, float, float]
    dt: float
    duration: float
    burn_in_steps: int
    rtol: float
    atol: float
    a: float
    b: float
    c: float
    initial_condition_noise_std: float
    method: str


@dataclass(frozen=True)
class DuffingConfig:
    reference_state: tuple[float, float]
    dt: float
    duration: float
    burn_in_steps: int
    rtol: float
    atol: float
    delta: float
    alpha: float
    beta: float
    omega: float
    gamma: float
    initial_condition_noise_std: float
    method: str


@dataclass(frozen=True)
class MackeyGlassConfig:
    reference_state: tuple[float]
    gamma: float
    beta: float
    tau: float
    n: float
    dt: float
    duration: float
    burn_in_steps: int
    initial_condition_noise_std: float


@dataclass(frozen=True)
class Lorenz96Config:
    K: int
    reference_state: tuple[float, ...]
    F: float
    dt: float
    duration: float
    burn_in_steps: int
    rtol: float
    atol: float
    initial_condition_noise_std: float
    method: str


@dataclass(frozen=True)
class KSConfig:
    num_grid_points: int
    initial_modes: tuple[int, ...]
    initial_amplitude: float
    L: float
    dt: float
    internal_dt: float
    duration: float
    burn_in_steps: int
    etdrk4_contour_points: int
    dealiasing: str


@dataclass(frozen=True)
class Lorenz63Config:
    sigma: float
    rho: float
    beta: float
    reference_state: tuple[float, float, float]
    dt: float
    duration: float
    burn_in_steps: int
    initial_condition_noise_std: float
    rtol: float
    atol: float
    method: str


SYSTEM_CONFIGS = {
    "rossler": RosslerConfig,
    "lorenz63": Lorenz63Config,
    "duffing": DuffingConfig,
    "mackeyglass": MackeyGlassConfig,
    "lorenz96": Lorenz96Config,
    "ks": KSConfig,
}
SystemConfig = (
    RosslerConfig
    | Lorenz63Config
    | DuffingConfig
    | MackeyGlassConfig
    | Lorenz96Config
    | KSConfig
)


def _finite_vector(value: object, dimension: int, location: str) -> FloatArray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{location} must be a sequence of {dimension} finite numbers"
        ) from error
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{location} must be a sequence of {dimension} finite numbers"
        )
    return vector


def _validate_system(config: SystemConfig) -> None:
    if config.dt <= 0 or config.duration <= 0:
        raise ValueError("dt and duration must be positive")


def _integrate_ode(
    rhs,
    config: SystemConfig,
    initial: FloatArray,
    name: str,
) -> FloatArray:
    times = np.arange(0.0, config.duration, config.dt, dtype=np.float64)
    solution = solve_ivp(
        rhs,
        (0.0, config.duration),
        initial,
        method=config.method,
        t_eval=times,
        rtol=config.rtol,
        atol=config.atol,
    )
    if not solution.success:
        raise RuntimeError(f"{name} integration failed: {solution.message}")
    return np.asarray(solution.y, dtype=np.float64)


def simulate_rossler(
    config: RosslerConfig,
    initial_condition: FloatArray,
) -> FloatArray:
    _validate_system(config)
    initial = _finite_vector(initial_condition, 3, "initial_condition")

    def rossler_ode(_time: float, state: FloatArray) -> list[float]:
        return [
            -state[1] - state[2],
            state[0] + config.a * state[1],
            config.b + state[2] * (state[0] - config.c),
        ]

    return _integrate_ode(rossler_ode, config, initial, "Rössler")


def simulate_duffing(
    config: DuffingConfig,
    initial_condition: FloatArray,
) -> FloatArray:
    _validate_system(config)
    initial = _finite_vector(initial_condition, 2, "initial_condition")

    def duffing_ode(time: float, state: FloatArray) -> list[float]:
        return [
            state[1],
            config.gamma * np.cos(config.omega * time)
            - config.delta * state[1]
            - config.alpha * state[0]
            - config.beta * state[0] ** 3,
        ]

    return _integrate_ode(duffing_ode, config, initial, "Duffing")


def simulate_lorenz63(
    config: Lorenz63Config,
    initial_condition: FloatArray,
) -> FloatArray:
    _validate_system(config)
    if config.initial_condition_noise_std < 0.0:
        raise ValueError("initial_condition_noise_std must be non-negative")
    if config.rtol <= 0.0 or config.atol <= 0.0:
        raise ValueError("rtol and atol must be positive")
    initial = _finite_vector(initial_condition, 3, "initial_condition")

    def lorenz63_ode(_time: float, state: FloatArray) -> list[float]:
        return [
            config.sigma * (state[1] - state[0]),
            state[0] * (config.rho - state[2]) - state[1],
            state[0] * state[1] - config.beta * state[2],
        ]

    return _integrate_ode(lorenz63_ode, config, initial, "Lorenz-63")


def simulate_mackeyglass(
    config: MackeyGlassConfig, initial_condition: FloatArray
) -> FloatArray:
    """Integrate the Mackey-Glass delay equation with jitcdde."""
    _validate_system(config)
    initial = float(
        _finite_vector(initial_condition, 1, "initial_condition")[0]
    )
    equation = [
        config.beta
        * (
            y(0, jitcdde_t - config.tau)
            / (1 + y(0, jitcdde_t - config.tau) ** config.n)
        )
        - config.gamma * y(0)
    ]
    integrator = jitcdde(equation, max_delay=config.tau)
    integrator.constant_past([initial])
    times = np.arange(
        0.0, config.duration + config.dt, config.dt, dtype=np.float64
    )
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory(
        prefix="jitcdde_build_",
    ) as temporary_directory:
        try:
            os.chdir(temporary_directory)
            integrator.compile_C()
            integrator.adjust_diff()
            values = [integrator.integrate(time) for time in times]
        finally:
            os.chdir(original_directory)
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def simulate_lorenz96(
    config: Lorenz96Config,
    initial_condition: FloatArray,
) -> FloatArray:
    _validate_system(config)
    initial = _finite_vector(initial_condition, config.K, "initial_condition")

    def lorenz96_ode(_time: float, state: FloatArray) -> FloatArray:
        state_ip1 = np.roll(state, -1)
        state_im1 = np.roll(state, 1)
        state_im2 = np.roll(state, 2)
        return (state_ip1 - state_im2) * state_im1 - state + config.F

    return _integrate_ode(lorenz96_ode, config, initial, "Lorenz-96")


def _ks_dealias_mask(n_grid: int) -> NDArray[np.bool_]:
    modes = np.fft.fftfreq(n_grid) * n_grid
    return np.asarray(np.abs(modes) < n_grid / 3.0, dtype=np.bool_)


def _ks_initial_condition_hat(
    config: KSConfig, seed: int
) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    state_hat = np.zeros(config.num_grid_points, dtype=np.complex128)
    for mode_value in config.initial_modes:
        mode = int(mode_value)
        coefficient = config.initial_amplitude * (
            rng.normal() + 1j * rng.normal()
        )
        state_hat[mode] = coefficient
        state_hat[-mode] = np.conj(coefficient)
    state_hat[0] = 0.0
    if config.num_grid_points % 2 == 0:
        state_hat[config.num_grid_points // 2] = 0.0
    return state_hat


def _simulate_ks_etdrk4_from_hat(
    config: KSConfig, initial_condition_hat: NDArray[np.complex128]
) -> FloatArray:
    """Integrate KS in Fourier space with the ETDRK4 scheme.

    Contour-integral coefficients follow Kassam & Trefethen (2005),
    SIAM Journal on Scientific Computing 26, 1214-1233, for
    u_t + u_xx + u_xxxx + u u_x = 0, with optional two-thirds
    dealiasing.
    """
    _validate_system(config)
    state_hat = np.array(initial_condition_hat, dtype=np.complex128, copy=True)
    if state_hat.shape != (config.num_grid_points,) or not np.all(
        np.isfinite(state_hat)
    ):
        raise ValueError(
            "initial_condition_hat must be a sequence of "
            f"{config.num_grid_points} finite complex numbers"
        )
    step_count = round(config.duration / config.dt)
    if step_count < 1:
        raise ValueError("duration/dt must produce at least one step")
    save_stride_float = config.dt / config.internal_dt
    save_stride = round(save_stride_float)
    if save_stride < 1 or not np.isclose(
        save_stride_float, save_stride, rtol=1.0e-12, atol=1.0e-12
    ):
        raise ValueError("dt must be an integer multiple of internal_dt")

    dx = config.L / config.num_grid_points
    wavenumbers = 2 * np.pi * np.fft.fftfreq(config.num_grid_points, d=dx)
    if config.num_grid_points % 2 == 0:
        wavenumbers[config.num_grid_points // 2] = 0.0
    linear_eigenvalues = wavenumbers**2 - wavenumbers**4
    exp_full = np.exp(config.internal_dt * linear_eigenvalues)
    exp_half = np.exp(0.5 * config.internal_dt * linear_eigenvalues)

    roots = np.exp(
        1j
        * np.pi
        * (np.arange(1, config.etdrk4_contour_points + 1, dtype=float) - 0.5)
        / config.etdrk4_contour_points
    )
    contour_shifts = (
        config.internal_dt * linear_eigenvalues[:, None] + roots[None, :]
    )
    q_coef = config.internal_dt * np.real(
        np.mean((np.exp(contour_shifts / 2.0) - 1.0) / contour_shifts, axis=1)
    )
    f1_coef = config.internal_dt * np.real(
        np.mean(
            (
                -4.0
                - contour_shifts
                + np.exp(contour_shifts)
                * (4.0 - 3.0 * contour_shifts + contour_shifts**2)
            )
            / contour_shifts**3,
            axis=1,
        )
    )
    f2_coef = config.internal_dt * np.real(
        np.mean(
            (
                2.0
                + contour_shifts
                + np.exp(contour_shifts) * (-2.0 + contour_shifts)
            )
            / contour_shifts**3,
            axis=1,
        )
    )
    f3_coef = config.internal_dt * np.real(
        np.mean(
            (
                -4.0
                - 3.0 * contour_shifts
                - contour_shifts**2
                + np.exp(contour_shifts) * (4.0 - contour_shifts)
            )
            / contour_shifts**3,
            axis=1,
        )
    )

    if config.dealiasing == "two_thirds":
        dealias_mask = _ks_dealias_mask(config.num_grid_points)
    elif config.dealiasing == "none":
        dealias_mask = np.ones(config.num_grid_points, dtype=bool)
    else:
        raise ValueError("dealiasing must be two_thirds or none")

    nonlinear_multiplier = -0.5j * wavenumbers

    def nonlinear_term(
        state_hat: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        state = np.fft.ifft(state_hat * dealias_mask).real
        term = nonlinear_multiplier * np.fft.fft(state * state)
        return np.asarray(term * dealias_mask, dtype=np.complex128)

    data = np.empty((config.num_grid_points, step_count), dtype=np.float64)
    for index in range(step_count):
        for _ in range(save_stride):
            nv = nonlinear_term(state_hat)
            a_hat = exp_half * state_hat + q_coef * nv
            na = nonlinear_term(a_hat)
            b_hat = exp_half * state_hat + q_coef * na
            nb = nonlinear_term(b_hat)
            c_hat = exp_half * a_hat + q_coef * (2.0 * nb - nv)
            nc = nonlinear_term(c_hat)
            state_hat = (
                exp_full * state_hat
                + f1_coef * nv
                + 2.0 * f2_coef * (na + nb)
                + f3_coef * nc
            )
            state_hat *= dealias_mask
            state_hat = np.fft.fft(np.fft.ifft(state_hat).real) * dealias_mask
        data[:, index] = np.fft.ifft(state_hat).real
    return data


def simulate(
    system: str,
    config: dict,
    seed: int,
    initial_condition: np.ndarray | None = None,
) -> np.ndarray:
    system_config = SYSTEM_CONFIGS[system](**config["data_generation"])
    if system == "ks":
        if initial_condition is not None:
            raise ValueError(
                "KS does not accept an initial_condition; configure "
                "its spectral initial modes"
            )
        trajectory = _simulate_ks_etdrk4_from_hat(
            system_config,
            _ks_initial_condition_hat(system_config, seed),
        )
    else:
        reference_state = np.asarray(
            system_config.reference_state,
            dtype=np.float64,
        )
        initial = (
            reference_state
            + system_config.initial_condition_noise_std
            * np.random.default_rng(seed).standard_normal(reference_state.size)
            if initial_condition is None
            else initial_condition
        )
        if system == "mackeyglass":
            initial = np.maximum(initial, 1.0e-6)
        integrators = {
            "rossler": simulate_rossler,
            "lorenz63": simulate_lorenz63,
            "duffing": simulate_duffing,
            "mackeyglass": simulate_mackeyglass,
            "lorenz96": simulate_lorenz96,
        }
        trajectory = integrators[system](system_config, initial)
    return np.asarray(
        trajectory[:, system_config.burn_in_steps :],
        dtype=np.float64,
    )
