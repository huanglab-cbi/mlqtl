import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from .nda_typing import MatrixFloat64


def _plot_chr(axs, chr, window_mean, threshold):
    y = window_mean[:, 2]
    x = list(range(len(y)))

    axs.plot(x, y, color="black")
    mask = np.array(y) > threshold
    axs.fill_between(x, y, where=mask, color="green", alpha=1)
    axs.axhline(y=threshold, color="#dc7633", linestyle="dashed")
    axs.set_xlabel("Genes")
    axs.set_ylabel("-log10(P-value)")
    axs.set_title(f"Chr {chr} (Gene number: {len(x)})")

    return axs


def plot_graph(
    sliding_window_result: List[Tuple[str, MatrixFloat64, MatrixFloat64]],
    threshold: np.float64,
    filename: str = "result",
    font_size: int = 20,
    save: bool = False,
) -> None:
    """
    Plot the sliding window result
    """
    plt.rcParams["font.size"] = font_size
    _, axs = plt.subplots(len(sliding_window_result), figsize=(20, 40))
    try:
        sliding_window_result.sort(key=lambda x: int(x[0]))
    except ValueError:
        sliding_window_result.sort(key=lambda x: x[0])
    if len(sliding_window_result) == 1:
        axs = [axs]
    for i, res in enumerate(sliding_window_result):
        chr, window_mean, _ = res
        axs[i] = _plot_chr(axs[i], chr, window_mean, threshold)

    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
