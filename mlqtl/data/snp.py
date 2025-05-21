import numpy as np
from .plink import Plink
from mlqtl.utils import convert_onehot
from typing import List, Tuple

__all__ = ["SNP", "Plink"]


_ml_base = {"AA": 1, "GG": 2, "CC": 3, "TT": 4}


class SNP(Plink):
    def __init__(self, snp_file: str):
        self.plink_prefix = snp_file
        super().__init__(self.plink_prefix)

    def get(self, marker: str):
        """
        Returns the snp data for a given marker
        """
        idx = self.marker2idx(marker)
        geno = self[idx][1]
        return int(idx), geno

    def _init_encoding_map(self):
        """
        Initialize the ml encoding map of the snp data
        """
        self._allele_ml_encoding = np.zeros(self._allele_encoding.shape, dtype=np.int8)
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
        snps: List[Tuple[int, np.ndarray]],
        onehot: bool = False,
        filter: np.ndarray = None,
    ) -> np.ndarray:
        """
        Encode the snp data in ml format
        """
        if not hasattr(self, "_allele_ml_encoding"):
            self._init_encoding_map()
        result = np.array([self._allele_ml_encoding[i][snp] for i, snp in snps])
        if filter is not None:
            result = result[:, filter]
        if onehot:
            result = convert_onehot(result)
        return result
