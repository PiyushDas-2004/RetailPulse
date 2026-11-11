"""Consistent matplotlib style for every static figure (validated categorical palette, recessive grid)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# fixed colour per model, so a model keeps its identity across every figure
MODEL_COLORS = {
    "Naive": "#9a9994", "SeasonalNaive": "#52514e", "MovingAverage": "#c3c2b7", "SES": "#e87ba4",
    "ETS": "#1baf7a", "SARIMA": "#2a78d6", "DHR-ARIMA": "#6250d6", "Prophet": "#eda100",
    "LightGBM": "#eb6834", "Ensemble": "#e34948",
}


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9.5,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "lines.linewidth": 1.6,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    })


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
