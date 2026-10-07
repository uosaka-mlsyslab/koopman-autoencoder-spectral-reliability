from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from spectral_reliability.validation import integer_at_least

Activation = Literal["relu", "tanh", "gelu", "silu"]
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


@dataclass(frozen=True)
class NetworkConfig:
    hidden_layers: int
    hidden_width: int | Literal["latent"]
    activation: Activation

    def __post_init__(self) -> None:
        if not integer_at_least(self.hidden_layers, 0):
            raise ValueError("hidden_layers must be a non-negative integer")
        if self.hidden_width != "latent" and not integer_at_least(
            self.hidden_width, 1
        ):
            raise ValueError(
                "hidden_width must be 'latent' or a positive integer"
            )
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {', '.join(ACTIVATIONS)}"
            )

    def width(self, latent_dim: int) -> int:
        return (
            int(latent_dim)
            if self.hidden_width == "latent"
            else int(self.hidden_width)
        )


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int
    encoder: NetworkConfig
    decoder: NetworkConfig

    def __post_init__(self) -> None:
        if not integer_at_least(self.latent_dim, 1):
            raise ValueError("latent_dim must be a positive integer")


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    config: NetworkConfig,
    dtype: torch.dtype | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width_in = input_dim
    for _ in range(config.hidden_layers):
        layers.append(nn.Linear(width_in, hidden_dim, bias=True, dtype=dtype))
        layers.append(ACTIVATIONS[config.activation]())
        width_in = hidden_dim
    layers.append(nn.Linear(width_in, output_dim, bias=True, dtype=dtype))
    return nn.Sequential(*layers)


class KoopmanAutoencoder(nn.Module):
    """Build an encoder and decoder with linear output layers."""

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(config.latent_dim)
        self.encoder = build_mlp(
            self.input_dim,
            self.latent_dim,
            config.encoder.width(self.latent_dim),
            config.encoder,
            dtype,
        )
        self.decoder = build_mlp(
            self.latent_dim,
            self.input_dim,
            config.decoder.width(self.latent_dim),
            config.decoder,
            dtype,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def encode_snapshot_pairs(
    model: KoopmanAutoencoder,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode [x; y] rows into feature-by-pair z_x and z_y matrices.

    The input has shape (2*m, d_x), with all source snapshots before
    their successors. Both outputs have shape (N, m), in pair order.
    """
    encoded = model.encode(batch).T
    if encoded.shape[1] % 2:
        raise ValueError(
            "encoded snapshot pair block must have an even number of columns"
        )
    midpoint = encoded.shape[1] // 2
    return encoded[:, :midpoint], encoded[:, midpoint:]
