import os
import numpy as np
import pandas as pd
from typing import List

__all__ = ["Gene"]


class Gene:
    def __init__(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        self.df = pd.read_csv(file, sep="\t", header=None)
        self.df.columns = ["chr", "start", "end", "transcript", "gene"]
        self.name = self.df["gene"].unique()

    def __repr__(self):
        return f"Gene: total {len(self.name)} genes"

    def __len__(self):
        return len(self.name)

    def get(self, gene: str):
        mask = self.df["gene"] == gene
        return self.df[mask]

    def filter(self, genes: List[str]):
        self.df = self.df[self.df["gene"].isin(genes)]
        self.name = self.df["gene"].unique()

    def chunks(self, p: int):
        num = int(np.ceil(len(self) / p))
        chunks = [
            pd.Series(self.name).isin(self.name[i : i + num])
            for i in range(0, len(self), num)
        ]
        return chunks
