import numpy as np
from typing import List, Tuple
from .plink import Plink
from ..nda_typing import VectorBool, VectorInt8, MatrixInt8, TensorFloat64

__all__ = ["SNP", "Plink"]


_ml_base = {"AA": 1, "GG": 2, "CC": 3, "TT": 4}


class SNP(Plink):
    """
    SNP data class for handling binary plink data
    """

    def __init__(self, snp_file: str):
        self.plink_prefix: str = snp_file
        super().__init__(self.plink_prefix)

    def get(self, marker: str) -> Tuple[int, VectorInt8]:
        """
        Returns the snp data for a given marker

        Parameters
        ----------
        marker : str
            the marker name

        Returns
        -------
        Tuple[int, VectorInt8]
            the index of the marker and the genotype
        """
        idx: np.int64 = self.marker2idx(marker)
        geno: VectorInt8 = self[idx][1]
        return int(idx), geno

    def _init_encoding_map(self) -> None:
        """
        Initialize the ml encoding map of the snp data
        """
        self._allele_ml_encoding: MatrixInt8 = np.zeros(
            self._allele_encoding.shape, dtype=np.int8
        )
        for k, v in _ml_base.items():
            self._allele_ml_encoding[np.where(self._allele_encoding == k)] = v

    @property
    def samples(self):
        """
        Return the samples in the snp data
        """
        return self._fam["iid"].to_list()

    def encode(
        self,
        snps: List[Tuple[int, VectorInt8]],
        onehot: bool = False,
        filter: VectorBool = None,
    ) -> MatrixInt8 | TensorFloat64:
        """
        Encode the snp data to ml encoding

        Parameters
        ----------
        snps : List[Tuple[int, VectorInt8]]
            the snp data list to encode, each element is a tuple of (index, genotype)
        onehot : bool
            whether to convert to onehot encoding
        filter : VectorBool
            the index ndarray to filter the snp data

        Returns
        -------
        MatrixInt8 or TensorFloat64
            the encoded snp data
            - use onehot: TensorFloat64, shape (n_samples, n_snps, 4)
            - not use onehot: MatrixInt8, shape (n_samples, n_snps)
        """
        if not hasattr(self, "_allele_ml_encoding"):
            self._init_encoding_map()
        result: MatrixInt8 = np.array(
            [self._allele_ml_encoding[i][snp] for i, snp in snps]
        )
        if filter is not None:
            result: MatrixInt8 = result[:, filter]
        if onehot:
            result: TensorFloat64 = SNP.convert_onehot(result)
            result: TensorFloat64 = np.transpose(result, (1, 0, 2))
        else:
            result: MatrixInt8 = result.T
        return result

    @staticmethod
    def convert_onehot(X: MatrixInt8) -> TensorFloat64:
        result: TensorFloat64 = np.zeros((X.shape[0], X.shape[1], 4), dtype=float)
        # A G C T
        # rows_A, cols_A = np.where(X == 1)
        # rows_G, cols_G = np.where(X == 2)
        # rows_C, cols_C = np.where(X == 3)
        # rows_T, cols_T = np.where(X == 4)
        # result[rows_A, cols_A, 0] = 1
        # result[rows_G, cols_G, 1] = 1
        # result[rows_C, cols_C, 2] = 1
        # result[rows_T, cols_T, 3] = 1
        for i in range(0, 4):
            rows, cols = np.where(X == i + 1)
            result[rows, cols, i] = 1
        return result
