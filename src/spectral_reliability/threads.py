from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

NUM_THREADS = 8


def bootstrap_threads(arguments: Sequence[str]) -> int:
    """Set native thread counts before numerical-library imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-threads", type=int, default=NUM_THREADS)
    settings, _ = parser.parse_known_args(arguments)
    if settings.num_threads < 1:
        raise ValueError("num-threads must be at least 1")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(settings.num_threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return settings.num_threads
