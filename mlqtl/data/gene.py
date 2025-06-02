import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from pandas import DataFrame
from ..nda_typing import VectorStr, VectorBool

__all__ = ["Gene"]


class Gene:
    """
    Gene class for handling gene position information
    """

    def __init__(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        self.df: DataFrame = pd.read_csv(file, sep="\t", header=None)
        self.df.columns = ["chr", "start", "end", "transcript", "gene"]
        self.name: VectorStr = self.df["gene"].unique().astype(np.str_)
        self.df = self.df.astype(
            {
                "chr": np.str_,
                "start": np.int64,
                "end": np.int64,
                "transcript": np.str_,
                "gene": np.str_,
            }
        )

    def __repr__(self):
        """
        String representation of the Gene class
        """
        return f"Gene: total {len(self.name)} genes"

    def __len__(self):
        """
        Length of the gene list
        """
        return len(self.name)

    def __getitem__(self, key) -> np.str_:
        """
        Get the gene name by index or name
        """
        return self.name[key]

    def __iter__(self):
        """
        Iterate over the gene names
        """
        return iter(self.name)

    def get(self, gene: str) -> DataFrame:
        """
        Retrieve the location information for a specific gene
        """
        mask: VectorBool = self.df["gene"] == gene
        return self.df[mask]

    def filter(self, genes: List[str]) -> None:
        """
        Filter the gene dataframe to only include the specified genes
        """
        self.df: DataFrame = self.df[self.df["gene"].isin(genes)]
        self.name: VectorStr = self.df["gene"].unique().astype(np.str_)

    def filter_by_chr(self, chr: List[str]) -> None:
        """
        Filter the gene dataframe to only include genes on a specific chromosome
        """
        self.df: DataFrame = self.df[self.df["chr"].isin(chr)].reset_index(drop=True)
        self.name: VectorStr = self.df["gene"].unique().astype(np.str_)

    def chunks(self, p: int) -> List[VectorBool]:
        """
        Split the gene list into chunks of size p

        Parameters
        ----------
        p : int
            the number of chunks to split the gene list into

        Returns
        -------
        List[VectorBool]:
            a list of boolean arrays which can use to index the gene name ndarray
        """
        num: int = int(np.ceil(len(self) / p))
        chunks: List[VectorBool] = [
            pd.Series(self.name).isin(self.name[i : i + num]).to_numpy()
            for i in range(0, len(self), num)
        ]
        return chunks

    @property
    def chrom(self):
        """
        Return the chromosome
        """
        return self.df["chr"].unique().astype(np.str_)
