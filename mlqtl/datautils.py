import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Tuple
from sklearn.base import RegressorMixin

from .data import Dataset
from .nda_typing import MatrixFloat64


def cal_padj(group: DataFrame) -> DataFrame:
    """
    Calculate the padj for a given group of p-values
    """
    n = len(group)
    group = group.sort_values("pval", ascending=True)
    group["padj"] = group["pval"] * n / np.arange(1, n + 1)
    return group


def skip_padj(group: DataFrame) -> DataFrame:
    """
    Skip the padj calculation for a given group of p-values
    """
    group["padj"] = group["pval"]
    return group


def proc_train_res(
    train_res: List[List[MatrixFloat64 | None]],
    models: List[RegressorMixin],
    dataset: Dataset,
    padj: bool = False,
) -> DataFrame:
    """
    Integrate the results from different chunks and calculate padj

    Parameters
    ----------
    result : List[List[MatrixFloat64 | None]]
        The result from the regression models, each Matrix element is a list of (pcc, pval)
    models : List[RegressorMixin]
        The list of regression models
    dataset : Dataset
        The dataset containing the information

    Returns
    -------
    DataFrame
        The integrated DataFrame containing the correlation, p-value, padj, and gene information
    """
    met_matrix, gene_idx = [], []
    for chunk in train_res:
        for result in chunk:
            if result is not None:
                gene_idx.append(True)
                met_matrix.append(result)
            else:
                gene_idx.append(False)
    if met_matrix is None or len(met_matrix) == 0:
        raise ValueError("No valid results found in the training results.")
    res = DataFrame(np.array(met_matrix).reshape(-1, 2))
    res.columns = ["corr", "pval"]
    model_names = [model.__name__ for model in models]
    res["model"] = res.index.map(lambda idx: model_names[idx % len(model_names)])
    not_na_genes = dataset.gene.name[gene_idx]
    res["gene"] = res.index.map(lambda idx: not_na_genes[idx // len(model_names)])
    res = res.astype(
        {
            "corr": np.float64,
            "pval": np.float64,
            "gene": np.str_,
            "model": "category",
        }
    )

    if padj:
        padj_func = cal_padj
    else:
        padj_func = skip_padj

    res = (
        res.groupby("model", observed=False)
        .apply(padj_func, include_groups=False)
        .reset_index(level="model")
        .dropna()
        .groupby("gene", observed=False)
        .apply(
            lambda group: group.loc[group["padj"].idxmin()],
            include_groups=False,
        )
        .reset_index(level="gene")
        .drop(columns=["pval"])
        .reset_index(drop=True)
        .merge(
            dataset.gene.df.groupby("gene")
            .apply(
                lambda group: group.assign(
                    start_min=group["start"].min(),
                    end_max=group["end"].max(),
                    center=(group["start"].min() + group["end"].max()) / 2,
                ),
                include_groups=False,
            )
            .filter(["chr", "start_min", "end_max", "center"])
            .reset_index(level="gene")
            .drop_duplicates(),
            how="left",
            left_on="gene",
            right_on="gene",
        )
        .astype({"chr": "string"})
        .assign(padj_norm=lambda df: -np.log10(df["padj"]))
    )

    res["padj_norm"] = -np.log10(res["padj"])
    finite_mask = np.isfinite(res["padj_norm"].to_numpy())
    if finite_mask.any():
        finite_max = float(res.loc[finite_mask, "padj_norm"].max())
        capped = res["padj_norm"].to_numpy()
        capped[~finite_mask] = finite_max + 1.0
        res["padj_norm"] = capped

    return res


def cal_sliding_window(
    met: DataFrame, chrom: str, window_size: int, step: int
) -> MatrixFloat64:
    """
    Sliding window to calculate the mean of the padj_norm values

    Parameters
    ----------
    met : DataFrame
        The DataFrame containing the padj_norm values
    chrom : str
        The chromosome to calculate the mean
    window_size : int
        The size of the window
    step : int
        The step size for the sliding window

    Returns
    -------
    MatrixFloat64
        The mean of the padj_norm values for each window and the start and end positions
    """
    met_chr = (
        met[met["chr"] == chrom]
        .sort_values(by=["start_min"], ascending=True)
        .reset_index()
    )
    window_mean = []
    gene_num = len(met_chr)
    for i in range(0, gene_num, step):
        start, end = i, i + window_size - 1
        end = end if end < gene_num else gene_num - 1
        if window_mean and window_mean[-1][1] == end:
            break
        window_mean.append(
            np.array([start, end, met_chr.loc[start:end, "padj_norm"].mean()])
        )
    return np.array(window_mean)


def merge_window(
    window_mean: MatrixFloat64, threshold: np.float64
) -> MatrixFloat64 | None:
    """
    Merge genes in the same region

    Parameters
    ----------
    window_mean : MatrixFloat64
        The mean of the padj_norm values for each window and the start and end positions
    threshold : np.float64
        The threshold to filter the mean values

    Returns
    -------
    MatrixFloat64
        The merged windows with start and end positions and the mean padj_norm value
    """
    window_loc = window_mean[window_mean[:, 2] >= threshold][:, 0:2]
    if len(window_loc) == 0:
        return None
    loc_merged, tmp = [], window_loc[0]
    for start, end in window_loc[1:]:
        if start <= tmp[1]:
            tmp = np.array([tmp[0], end])
        else:
            loc_merged.append(tmp)
            tmp = np.array([start, end])
    loc_merged.append(tmp)
    return np.array(loc_merged)


def significance(
    sliding_window_result: List[Tuple[str, MatrixFloat64, MatrixFloat64]],
    result: DataFrame,
) -> DataFrame:
    """
    Get the gene in the peek window of the green region

    Parameters
    ----------
    sliding_window_result : List[Tuple[str, MatrixFloat64, MatrixFloat64]]
        Results of the sliding window calculation
    result : DataFrame
        Integrated training results

    Returns
    -------
    DataFrame
        Gene table of the green region in the graph
    """
    region_gene = pd.DataFrame()
    for chr, window_mean, window_merged in sliding_window_result:
        if window_merged is None or window_merged.size == 0:
            continue
        tmp = result[result["chr"] == chr].reset_index(drop=True)
        for i, region in enumerate(window_merged):
            merged_start, merged_end = region
            tmp_sorted = tmp.sort_values(by=["center"], ascending=True).reset_index(
                drop=True
            )
            tmp_res = tmp_sorted.loc[int(merged_start) : int(merged_end)].copy()
            if not tmp_res.empty:
                tmp_res["region"] = i + 1
                tmp_res = tmp_res.sort_values(by=["padj_norm"], ascending=False)
                region_gene = pd.concat([region_gene, tmp_res], axis=0)
    region_gene = region_gene.reset_index(drop=True)
    return region_gene


def sliding_window(
    result: DataFrame,
    window: int,
    step: int,
    threshold: np.float64,
) -> Tuple[List[Tuple[np.str_, MatrixFloat64, MatrixFloat64]], DataFrame]:
    """
    Convert the training results to dataframe and calculate the sliding window and merge significant regions

    Parameters
    ----------
    result : DataFrame
        The integrated training results containing the correlation, p-value, and gene information
    window_size : int
        The size of the window
    step : int
        The step size for the sliding window
    threshold : np.float64
        The threshold to filter the mean values

    Returns
    -------
    sliding_window_result : List[np.str, MatrixFloat64, MatrixFloat64]
        The sliding window results is a list of tuples containing the chromosome, the mean values for each window, and the merged windows
    significant_genes : DataFrame
        The significant genes in the green region of the graph
    """

    chr = result["chr"].unique()
    threshold_norm = -np.log10(threshold)
    sw_res = []
    for c in chr:
        window_mean = cal_sliding_window(result, c, window, step)
        window_merged = merge_window(window_mean, threshold_norm)
        sw_res.append((c, window_mean, window_merged))
    sig_genes = significance(sw_res, result, threshold_norm)
    return sw_res, sig_genes


def cal_sliding_window_quantile(
    met: DataFrame,
    chrom: str,
    center_window_kb: int = None,
    center_step_genes: int = None,
    q: float = 0.9,
) -> MatrixFloat64:
    """
    Sliding window based on padj_norm quantile (e.g. 90% quantile) instead of mean.

    This is used by the new quantile-based method that does not require
    an explicit p-value threshold.
    """
    window_score = []
    met_chr = (
        met[met["chr"] == chrom]
        .sort_values(by=["center"], ascending=True)
        .reset_index(drop=True)
    )
    if len(met_chr) == 0:
        return np.array([])

    window_radius_bp = center_window_kb * 1000
    step_genes = center_step_genes

    gene_num = len(met_chr)
    gene_centers = met_chr["center"].values

    i = 0
    while i < gene_num:
        center_gene_center = gene_centers[i]
        window_left = center_gene_center - window_radius_bp
        window_right = center_gene_center + window_radius_bp

        mask = (gene_centers >= window_left) & (gene_centers <= window_right)
        window_genes_indices = np.where(mask)[0].tolist()
        if len(window_genes_indices) == 0:
            i += step_genes
            if i >= gene_num:
                break
            continue

        window_genes = met_chr.iloc[window_genes_indices]
        padj_norm = window_genes["padj_norm"].to_numpy()
        finite_vals = padj_norm[np.isfinite(padj_norm)]
        if finite_vals.size == 0:
            i += step_genes
            if i >= gene_num:
                break
            continue
        score_val = float(np.quantile(finite_vals, q))

        start_idx = window_genes_indices[0]
        end_idx = window_genes_indices[-1]
        window_score.append(np.array([start_idx, end_idx, score_val]))

        i += step_genes
        if i >= gene_num:
            break

    return np.array(window_score) if window_score else np.array([])


def sliding_window_newmethod(
    result: DataFrame,
    center_window_kb: int = None,
    center_step_genes: int = None,
    q: float = 0.9,
    top_prop: float = 0.10,
) -> Tuple[List[Tuple[np.str_, MatrixFloat64, MatrixFloat64]], DataFrame, float]:
    """
    New quantile-based method:
    - Gene score: padj_norm = -log10(padj)
    - Window score: q-quantile (e.g. 90%) of padj_norm in the window
    - QTL regions: top `top_prop` (e.g. 10%) windows genome-wide by window score

    Returns sliding window results, significant genes and the window-score threshold
    (in padj_norm scale).
    """
    chr_list = result["chr"].unique()
    sw_res = []
    all_scores = []

    for c in chr_list:
        window_score = cal_sliding_window_quantile(
            result,
            c,
            center_window_kb,
            center_step_genes,
            q,
        )
        if window_score.size > 0:
            all_scores.extend(window_score[:, 2].tolist())
        sw_res.append((c, window_score, None))

    if not all_scores:
        return sw_res, result.iloc[0:0].copy(), 0.0

    all_scores_arr = np.array(all_scores, dtype=float)
    finite_scores = all_scores_arr[np.isfinite(all_scores_arr)]
    if finite_scores.size == 0:
        return sw_res, result.iloc[0:0].copy(), 0.0
    window_threshold = float(np.quantile(finite_scores, 1.0 - top_prop))
    sw_res_new = []
    for c, window_score, _ in sw_res:
        if window_score.size == 0:
            sw_res_new.append((c, window_score, None))
            continue
        window_merged = merge_window(window_score, window_threshold)
        sw_res_new.append((c, window_score, window_merged))
    sig_genes = significance(sw_res_new, result)

    return sw_res_new, sig_genes, window_threshold
