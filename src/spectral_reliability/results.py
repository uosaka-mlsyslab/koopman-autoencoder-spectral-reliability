from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from spectral_reliability.config_shared import (
    DISPLAY_NAMES,
    MODEL_LABELS,
    SYSTEMS,
)

CSV_ID_COLUMNS = ("system", "model", "latent_dim", "seed")


@dataclass(frozen=True)
class RunRecord:
    system: str
    model: str
    latent_multiplier: int
    latent_dim: int
    seed: int
    metrics: dict[str, float]
    run_dir: Path


def read_seed_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if (
            columns[: len(CSV_ID_COLUMNS)] != CSV_ID_COLUMNS
            or not columns[len(CSV_ID_COLUMNS) :]
        ):
            raise ValueError(
                f"{path}: expected {CSV_ID_COLUMNS} followed by "
                "evaluation metric columns"
            )
        return list(reader)


def _format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "inf"
    return f"{float(value):.10g}"


def write_seed_metrics(
    records: Iterable[RunRecord],
    path: Path,
    *,
    metric_columns: tuple[str, ...],
) -> int:
    """Write selected rows with ten significant figures and CRLF
    endings.
    """
    models = tuple(MODEL_LABELS)
    ordered = sorted(
        records,
        key=lambda r: (
            SYSTEMS.index(r.system),
            models.index(r.model),
            r.latent_multiplier,
            r.seed,
        ),
    )
    seen = set()
    for record in ordered:
        key = (
            record.system,
            MODEL_LABELS[record.model],
            record.latent_dim,
            record.seed,
        )
        if key in seen:
            raise ValueError(
                f"two runs map to the same CSV row: {key}; "
                "choose distinct models, dimensions or seeds"
            )
        seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((*CSV_ID_COLUMNS, *metric_columns))
        for record in ordered:
            writer.writerow(
                (
                    DISPLAY_NAMES[record.system],
                    MODEL_LABELS[record.model],
                    record.latent_dim,
                    record.seed,
                    *(
                        _format_metric(record.metrics.get(name))
                        for name in metric_columns
                    ),
                )
            )
    return len(ordered)
