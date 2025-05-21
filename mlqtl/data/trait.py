import os
import numpy as np
import pandas as pd
from typing import Tuple

__all__ = ["Trait"]


class Trait:
    def __init__(self, traits_file: str):
        if not os.path.exists(traits_file):
            raise FileNotFoundError(f"The file {traits_file} does not exist.")
        self.df = pd.read_csv(traits_file, sep="\t")
        self.name = self.df.columns[1:].to_list()

    def __repr__(self):
        return f"Trait({len(self.df):,d} samples, {len(self.name):,d} traits)"

    def __len__(self):
        """
        Returns the number of traits in the file
        """
        return len(self.df)

    def __getitem__(self, key):
        """
        Returns the trait data for a given key
        """
        return self.df.loc[key].values

    def filter_df(self, fam: pd.DataFrame) -> None:
        self.df = (
            fam.filter(["iid"])
            .merge(self.df, left_on="iid", right_on="sample", how="left", sort=False)
            .drop(columns=["sample"])
        )

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the trait data for a given name
        """
        if name not in self.name:
            raise ValueError(f"The trait {name} does not exist.")
        data = self.df[name].values
        not_na = ~np.isnan(data)
        data = data[not_na]
        return data, not_na
