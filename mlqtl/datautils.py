import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Tuple
from sklearn.base import RegressorMixin

from .data import Dataset
from .nda_typing import MatrixFloat64


def calculate_fdr(group: DataFrame) -> DataFrame:
    """
    Calculate the FDR for a given group of p-values
    """
    n = len(group)
    group = group.sort_values("pval", ascending=True)
    group["fdr"] = group["pval"] * n / np.arange(1, n + 1)
    return group


def train_res_to_df(
    train_res: List[List[MatrixFloat64 | None]],
    models: List[RegressorMixin],
    dataset: Dataset,
) -> DataFrame:
    """
    Integrate the results from different chunks and calculate FDR

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
        The integrated DataFrame containing the correlation, p-value, FDR, and gene information
    """
    met_matrix, gene_idx = [], []
    for chunk in train_res:
        for result in chunk:
            if result is not None:
                gene_idx.append(True)
                met_matrix.append(result)
            else:
                gene_idx.append(False)

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

    res = (
        res.groupby("model", observed=False)
        .apply(calculate_fdr, include_groups=False)
        .reset_index(level="model")
        .dropna()
        .groupby("gene", observed=False)
        .apply(
            lambda group: group.loc[group["fdr"].idxmin()],
            include_groups=False,
        )
        .reset_index(level="gene")
        .drop(columns=["pval"])
        .reset_index(drop=True)
        .merge(
            dataset.gene.df.groupby("gene")
            .apply(
                lambda group: group.assign(start_min=group["start"].min()),
                include_groups=False,
            )
            .filter(["chr", "start_min"])
            .reset_index(level="gene")
            .drop_duplicates(),
            how="left",
            left_on="gene",
            right_on="gene",
        )
        .astype({"chr": "string"})
        .assign(fdr_norm=lambda df: -np.log10(df["fdr"]))
    )

    return res


def sliding_window(
    met: DataFrame, chrom: str, window_size: int, step: int
) -> MatrixFloat64:
    """
    Sliding window to calculate the mean of the fdr_norm values

    Parameters
    ----------
    met : DataFrame
        The DataFrame containing the fdr_norm values
    chrom : str
        The chromosome to calculate the mean
    window_size : int
        The size of the window
    step : int
        The step size for the sliding window

    Returns
    -------
    MatrixFloat64
        The mean of the fdr_norm values for each window and the start and end positions
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
            np.array([start, end, met_chr.loc[start:end, "fdr_norm"].mean()])
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
        The mean of the fdr_norm values for each window and the start and end positions
    threshold : np.float64
        The threshold to filter the mean values

    Returns
    -------
    MatrixFloat64
        The merged windows with start and end positions and the mean fdr_norm value
    """
    window_loc = window_mean[window_mean[:, 2] > threshold][:, 0:2]
    if len(window_loc) == 0:
        return None
    loc_merged, tmp = [], window_loc[0]
    for start, end in window_loc[1:]:
        if start <= tmp[1]:
            tmp = np.array([tmp[0], end])
        else:
            loc_merged.append(tmp)
            tmp = (start, end)
    loc_merged.append(tmp)
    return np.array(loc_merged)


def get_region_gene(
    sliding_window_result: List[Tuple[str, MatrixFloat64, MatrixFloat64]],
    result: DataFrame,
    threshold: np.float64,
) -> DataFrame:
    """
    Get the gene in the peek window of the green region

    Parameters
    ----------
    sliding_window_result : List[Tuple[str, MatrixFloat64, MatrixFloat64]]
        Results of the sliding window calculation
    result : DataFrame
        Integrated training results
    threshold : np.float64
        The threshold to filter the candidate genes

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
            start, end = region
            window_mean_green = window_mean[
                (window_mean[:, 0] >= start) & (window_mean[:, 1] <= end)
            ]
            peek_window = window_mean_green[np.argmax(window_mean_green[:, 2])]
            start, end = peek_window[0], peek_window[1]
            tmp_res = tmp.loc[int(start) : int(end)].copy()
            tmp_res["region"] = i + 1
            tmp_res = tmp_res.sort_values(by=["fdr_norm"], ascending=False)
            region_gene = pd.concat([region_gene, tmp_res], axis=0)
    region_gene = region_gene.reset_index(drop=True)
    region_gene = region_gene[region_gene["fdr_norm"] > threshold]
    return region_gene


def calculate_sliding_window(
    train_res: List[List[MatrixFloat64 | None]],
    models: List[RegressorMixin],
    dataset: Dataset,
    window: int,
    step: int,
    threshold: np.float64,
) -> Tuple[List[Tuple[np.str_, MatrixFloat64, MatrixFloat64]], DataFrame]:
    """
    Convert the training results to dataframe and calculate the sliding window and merge significant regions

    Parameters
    ----------
    train_res : List[List[MatrixFloat64 | None]]
        The result from the regression models, each Matrix element is a list of (pcc, pval)
    models : List[RegressorMixin]
        The list of regression models
    dataset : Dataset
        The dataset containing the information
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

    result = train_res_to_df(train_res, models, dataset)
    chr = result["chr"].unique().to_numpy()
    sliding_window_result = []
    for c in chr:
        window_mean = sliding_window(result, c, window, step)
        window_merged = merge_window(window_mean, threshold)
        sliding_window_result.append((c, window_mean, window_merged))

    significant_genes = get_region_gene(sliding_window_result, result, threshold)
    return sliding_window_result, significant_genes
