import numpy as np
import multiprocessing
from typing import List, Tuple
from scipy.stats import pearsonr, NearConstantInputWarning, ConstantInputWarning
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from inspect import signature
from sklearn.base import RegressorMixin
from pandas import DataFrame
import warnings
import click

from .data import Dataset
from .nda_typing import (
    MatrixInt8,
    TensorFloat64,
    VectorFloat64,
    MatrixFloat64,
    VectorStr,
)

np.random.seed(42)


class MLMetrics(object):
    def __init__(self):
        self.corr = None
        self.p_value = None

    def update(self, y, y_hat):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NearConstantInputWarning)
            warnings.simplefilter("ignore", ConstantInputWarning)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr, p_value = pearsonr(y, y_hat)
        self.corr = corr
        self.p_value = p_value


def train_batch(
    X: MatrixInt8 | TensorFloat64,
    y: VectorFloat64,
    onehot: bool,
    models: List[RegressorMixin],
    importance: bool = False,
) -> MatrixFloat64:
    """
    Train a batch of models on the given data

    Parameters
    ----------
    X : MatrixInt8 | TensorFloat64
        The encoded SNP data
    y : VectorFloat64
        The trait values
    onehot : bool
        Whether the SNP data is one-hot encoded
    models : List[RegressorMixin]
        The list of models to train
    importance : bool
        Whether to calculate feature importance

    Returns
    -------
    MatrixFloat64
        - importance == False: a matrix of shape (n_models, 2) containing the correlation and p-value for each model
        - importance == True, a matrix of shape (n_models, n_features) containing the feature importance for each model
    """
    if onehot:
        x_shape_ori: Tuple[int, int, int] = X.shape
        X: MatrixFloat64 = X.reshape(y.shape[0], -1)
    else:
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

    instances = []
    for model in models:
        if not issubclass(model, RegressorMixin):
            raise TypeError(f"Model {model} is not a valid regression model")
        try:
            init_params = signature(model).parameters
            if "random_state" in init_params:
                instances.append(model(random_state=42))
            else:
                instances.append(model())
        except TypeError:
            raise TypeError(f"Model {model} should be a class")

    mets, imp_matrix = [], []
    for model in instances:
        met = MLMetrics()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        met.update(y_test, y_pred)
        mets.append([met.corr, met.p_value])

        if importance:
            imp_matrix.append(
                permutation_importance(
                    model, X_train, y_train, n_repeats=10, random_state=42
                ).importances_mean
            )

    if importance:
        imp_matrix = np.array(imp_matrix)
        if onehot:
            imp_matrix = imp_matrix.reshape(-1, x_shape_ori[1], x_shape_ori[2])
            imp_matrix = np.mean(imp_matrix, axis=2)
        return imp_matrix

    return np.array(mets)


def init_worker(dataset: Dataset) -> None:
    """
    Initialize the worker with the dataset for multiprocessing
    """
    global g_dataset
    g_dataset = dataset


def _task(
    genes: VectorStr, trait: str, onehot: bool, models: RegressorMixin
) -> List[MatrixFloat64 | None]:
    """
    task for each chunk of genes
    """
    result = []
    for gene in genes:
        try:
            snps = g_dataset.get(gene)
            trait_value, not_nan_idx = g_dataset.trait.get(trait)
            X = g_dataset.snp.encode(snps, onehot, filter=not_nan_idx)
            y = trait_value
            m1 = train_batch(X, y, onehot, models, False)
            result.append(m1)
        except ValueError:
            result.append(None)
    return result


def train(
    trait: str,
    models: List[RegressorMixin],
    dataset: Dataset,
    max_workers: int = 8,
    onehot: bool = False,
) -> List[List[MatrixFloat64 | None]]:
    """
    Train models on the given dataset using multiprocessing

    Parameters
    ----------
    trait : str
        The trait to train
    onehot : bool
        Whether the SNP data is one-hot encoded
    models : List[RegressorMixin]
        The list of models to train
    dataset : Dataset
        The dataset to train
    max_workers : int
        The number of workers to use for multiprocessing

    Returns
    -------
    List[List[MatrixFloat64 | None]]
        A list of lists containing the correlation and p-value for each model
        for each gene, shape of MatrixFloat64 is (n_models, 2)
    """
    with multiprocessing.Pool(
        processes=max_workers, initializer=init_worker, initargs=(dataset,)
    ) as pool:
        results = pool.starmap(
            _task,
            [
                (
                    dataset.gene.name[chunk],
                    trait,
                    onehot,
                    models,
                )
                for chunk in dataset.gene.chunks(max_workers)
            ],
        )

    return results


def train_single(
    gene: str,
    trait: str,
    models: List[RegressorMixin],
    dataset: Dataset,
    onehot: bool = False,
) -> MatrixFloat64:
    """
    Train a single gene and trait

    Parameters
    ----------
    gene : str
        The gene to train
    trait : str
        The trait to train
    dataset : Dataset
        The dataset to train
    models : List[RegressorMixin]
        The list of models to train
    onehot : bool
        Whether the SNP data is one-hot encoded

    Returns
    -------
    MatrixFloat64
        a matrix of shape (n_models, 2) containing the correlation and p-value for each model

    """
    snps = dataset.get(gene)
    trait_value, not_nan_idx = dataset.trait.get(trait)
    X = dataset.snp.encode(snps, onehot, filter=not_nan_idx)
    y = trait_value
    res = train_batch(X, y, onehot, models)
    return res


def feature_importance(
    gene: str,
    trait: str,
    models: List[RegressorMixin],
    dataset: Dataset,
    onehot: bool = False,
) -> DataFrame:
    """
    Train a single gene and trait

    Parameters
    ----------
    gene : str
        The gene to train
    trait : str
        The trait to train
    dataset : Dataset
        The dataset to train
    models : List[RegressorMixin]
        The list of models to train
    onehot : bool
        Whether the SNP data is one-hot encoded

    Returns
    -------
    DataFrame
        a pandas DataFrame containing the feature importance for each model
        with the SNP markers as columns and the model names as rows

    """
    snps = dataset.get(gene)
    trait_value, not_nan_idx = dataset.trait.get(trait)
    X = dataset.snp.encode(snps, onehot, filter=not_nan_idx)
    y = trait_value
    importance = train_batch(X, y, onehot, models, True)
    feature = np.array([dataset.snp.idx2marker(i[0]) for i in snps], dtype=np.str_)
    models_name = np.array([model.__name__ for model in models], dtype=np.str_)
    res = DataFrame(
        importance,
        index=models_name,
        columns=feature,
    )
    return res


# The following defines the train function with a progress bar and related helper functions used in the CLI


def _progress_bar_manager(queue: multiprocessing, total_genes: int, trait: str):
    """
    Manages the click.progressbar by listening to a queue.
    Receives 'None' to terminate.
    """
    with click.progressbar(
        length=total_genes,
        label=f"{trait}",
        show_pos=True,
        show_percent=True,
        show_eta=True,
    ) as bar:
        processed_count = 0
        while processed_count < total_genes:
            message = queue.get()
            if message is None:
                break
            if message == "TICK":
                bar.update(1)
                processed_count += 1
        bar.finish()


def _task_progressbar(
    genes: VectorStr,
    trait: str,
    onehot: bool,
    models: RegressorMixin,
    progress_queue: multiprocessing.Queue,
) -> List[MatrixFloat64 | None]:
    """
    task for each chunk of genes
    """
    result = []
    for gene in genes:
        try:
            snps = g_dataset.get(gene)
            trait_value, not_nan_idx = g_dataset.trait.get(trait)
            X = g_dataset.snp.encode(snps, onehot, filter=not_nan_idx)
            y = trait_value
            m1 = train_batch(X, y, onehot, models, False)
            result.append(m1)
        except ValueError:
            result.append(None)
        finally:
            progress_queue.put("TICK")
    return result


def train_with_progressbar(
    trait: str,
    models: List[RegressorMixin],
    dataset: Dataset,
    max_workers: int = 8,
    onehot: bool = False,
) -> List[List[MatrixFloat64 | None]]:
    """
    Train models on the given dataset using multiprocessing

    Parameters
    ----------
    trait : str
        The trait to train
    onehot : bool
        Whether the SNP data is one-hot encoded
    models : List[RegressorMixin]
        The list of models to train
    dataset : Dataset
        The dataset to train
    max_workers : int
        The number of workers to use for multiprocessing

    Returns
    -------
    List[List[MatrixFloat64 | None]]
        A list of lists containing the correlation and p-value for each model
        for each gene, shape of MatrixFloat64 is (n_models, 2)
    """
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        progress_manager_proc = multiprocessing.Process(
            target=_progress_bar_manager,
            args=(progress_queue, len(dataset.gene.name), trait),
        )
        progress_manager_proc.start()

        with multiprocessing.Pool(
            processes=max_workers, initializer=init_worker, initargs=(dataset,)
        ) as pool:
            results = pool.starmap(
                _task_progressbar,
                [
                    (
                        dataset.gene.name[chunk],
                        trait,
                        onehot,
                        models,
                        progress_queue,
                    )
                    for chunk in dataset.gene.chunks(max_workers)
                ],
            )
        progress_queue.put(None)
        progress_manager_proc.join()
    return results
