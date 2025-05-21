import numpy as np
import multiprocessing
from typing import List, Tuple
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from inspect import signature
from sklearn.base import BaseEstimator

from mlqtl.data import DataCollection

np.random.seed(42)


class MLMetrics(object):
    def __init__(self):
        self.corr = None
        self.p_value = None

    def update(self, y, y_hat):
        corr, p_value = pearsonr(y, y_hat)
        self.corr = corr
        self.p_value = p_value


def train_batch(
    X: np.ndarray,
    y: np.ndarray,
    onehot: bool,
    models: List[object],
    importance: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a batch of models on the given data
    """
    if onehot:
        X = X.reshape(y.shape[0], -1)
    else:
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

    instances = []
    for model in models:
        try:
            init_params = signature(model).parameters
            if "random_state" in init_params:
                instances.append(model(random_state=42))
            else:
                instances.append(model())
        except TypeError:
            raise TypeError(f"Model {model} should be a class")

    mets, importance_matrix = [], []
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
            importance_matrix.append(
                permutation_importance(
                    model, X_train, y_train, n_repeats=10, random_state=42
                ).importances_mean
            )

    return np.array(mets) if not importance else np.array(importance_matrix)


def init_worker(dataset):
    global g_dataset
    g_dataset = dataset


def _task(genes, trait, onehot, models) -> List[np.ndarray]:
    """
    task for each chunk of genes
    """
    result = []
    for gene in genes:
        try:
            snps = g_dataset.get_gene_snps(gene)
            trait_value, not_nan_idx = g_dataset.trait.get(trait)
            X = g_dataset.snp.encode(snps, onehot, filter=not_nan_idx).T
            y = trait_value
            m1 = train_batch(X, y, onehot, models, False)
            result.append(m1)
        except ValueError:
            result.append(None)
    return result


def train(
    trait: str,
    onehot: bool,
    models: List[BaseEstimator],
    dataset: DataCollection,
    max_workers: int,
) -> List[List[np.ndarray]]:
    """
    Train models on the given dataset using multiprocessing
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
