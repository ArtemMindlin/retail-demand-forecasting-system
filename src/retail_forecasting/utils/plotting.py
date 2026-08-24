"""Matplotlib helpers with no knowledge of the panel they end up drawing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def make_grid(
    n_items: int,
    n_cols: int,
    width: float,
    row_height: float,
    sharex: bool = False,
) -> tuple[Figure, np.ndarray]:
    """Create a grid of subplots sized for ``n_items`` and hide the unused cells."""
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, row_height * n_rows), sharex=sharex)
    axes_flat = np.atleast_1d(axes).flatten()
    for axis in axes_flat[n_items:]:
        axis.axis("off")
    return fig, axes_flat
