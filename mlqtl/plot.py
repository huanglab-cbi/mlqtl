import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from typing import List, Tuple
from .nda_typing import MatrixFloat64


def _plot_chr(axs, chr, window_mean, threshold):
    y = window_mean[:, 2]
    x = list(range(len(y)))

    axs.plot(x, y, color="black")
    mask = np.array(y) > threshold
    axs.fill_between(x, y, where=mask, color="green", alpha=1)
    axs.axhline(y=threshold, color="#dc7633", linestyle="dashed")
    axs.set_xlabel("Window Index")
    axs.set_ylabel("-log10(P-value)")
    axs.set_title(f"Chr {chr} (Gene number: {int(window_mean[-1][1])})")

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
    chr_num = len(sliding_window_result)
    _, axs = plt.subplots(chr_num, figsize=(20, 40 / 12 * chr_num))
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


def plot_feature_importance(
    feature_importance: DataFrame,
    topn: int = 10,
    save: bool = False,
    filename: str = "feature_importance",
) -> None:
    """
    Plot the feature importance of SNPs in different models
    """
    topn = 10
    for row in feature_importance.iterrows():
        model_name = row[0]
        snps = row[1].index.tolist()
        snp_number = {snp: str(i + 1) for i, snp in enumerate(snps)}
        top_n = row[1].sort_values(ascending=False).head(topn)
        y = top_n.values
        x_labels_snp = top_n.index
        x_labels_snp = [f"{snp} ({snp_number[snp]})" for snp in x_labels_snp]
        x_positions = range(len(x_labels_snp))
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(8, 6))
        bar_width = 0.4
        bars = plt.bar(
            x_labels_snp,
            y,
            width=bar_width,
            color="steelblue",
            edgecolor="black",
            alpha=1,
        )
        plt.xlabel(f"Top {topn} SNP", fontsize=14)
        plt.xticks(x_positions, x_labels_snp, rotation=45, ha="right", fontsize=10)
        plt.ylabel("Importance Score", fontsize=14)
        plt.title(
            f"SNP Feature Importance in Model: {model_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.9)
        ax.xaxis.grid(False)
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig(f"{filename}_{model_name}.png", dpi=300, bbox_inches="tight")
