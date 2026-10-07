# Understanding Latent-Dimension Scaling in Dynamical-System Learning through Spectral Reliability

This repository contains the code for the paper. It trains Koopman autoencoders and four baselines (ESN, kernel DMD, neural ODE and Consistent KAE) on six dynamical systems at five latent dimensions, and plots how the prediction error changes with the latent dimension.

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/) and needs an NVIDIA GPU:

```console
uv sync
```

## Data generation

To generate trajectories of the Rössler, Lorenz-63, Duffing, Mackey-Glass, Lorenz-96 and Kuramoto-Sivashinsky systems, run:

```console
uv run python scripts/generate_data.py --grid configs/koopman.yaml
```

The system settings are in `configs/data/`, and the datasets and the data for pretraining the autoencoders are saved to `data/`.

## Training

To pretrain the autoencoders on reconstruction and train all models, run:

```console
uv run python scripts/pretrain.py --grid configs/koopman.yaml
uv run python scripts/train.py --grid configs/koopman.yaml
uv run python scripts/train.py --grid configs/baselines.yaml
```

The runs are listed in `configs/koopman.yaml` and `configs/baselines.yaml`, and each run reads its model settings from `configs/koopman/` or `configs/baselines/`. The pretrained weights are saved to `checkpoints/` and each run to `outputs/`. On first use, the Consistent KAE baseline downloads the [koopmanAE](https://github.com/erichson/koopmanAE) code (GPL-3.0), which needs network access.

The Consistent KAE encoder and decoder each have two hidden layers. Set `hidden_width: latent` to make each hidden layer exactly as wide as the latent dimension, or set `hidden_width` to a positive integer for a fixed width.

## Evaluation

To evaluate the models on the test split and draw the figures, run:

```console
uv run python scripts/evaluate.py
uv run python scripts/make_figures.py
```

The metrics of each run are written to `results/seed_metrics.csv`, the arrays for the pseudospectra and attractor plots to `results/arrays/`, and the figures set in `configs/figures.yaml` to `figures/`. `results/seed_metrics.csv` ships with the metrics of our trained models; to plot them against the latent dimension, run:

```console
uv run python scripts/make_figures.py --figure scaling
```
