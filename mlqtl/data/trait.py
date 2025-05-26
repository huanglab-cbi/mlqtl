import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from pandas import DataFrame
from ..nda_typing import VectorFloat64, VectorBool

__all__ = ["Trait"]


class Trait:
    """
    Trait class for handling trait data
    """

    def __init__(self, traits_file: str):
        if not os.path.exists(traits_file):
            raise FileNotFoundError(f"The file {traits_file} does not exist.")
        with open(traits_file, "r") as f:
            header = f.readline().strip().split("\t")
            if len(header) < 2:
                raise ValueError(
                    "The traits file must contain at least one trait column"
                )
            if not header[0] == "sample":
                raise ValueError("The first column of the traits file must be 'sample'")
        self.df: DataFrame = pd.read_csv(traits_file, sep="\t", header=0)
        self.df = self.df.astype(
            {
                "sample": str,
                **{col: np.float64 for col in self.df.columns if col != "sample"},
            }
        )
        self.name: List[str] = self.df.columns[1:].to_list()

    def __repr__(self):
        """
        Returns a string representation of the Trait object
        """
        return f"Trait({len(self.df):,d} samples, {len(self.name):,d} traits)"

    def __len__(self):
        """
        Returns the number of traits in the file
        """
        return len(self.df)

    def __getitem__(self, key) -> str:
        """
        Returns the trait name for a given key
        """
        return self.name[key]

    def __iter__(self):
        """
        Returns an iterator over the trait names
        """
        return iter(self.name)

    def filter_df(self, fam: pd.DataFrame) -> None:
        """
        Filters the trait data to only include samples in the fam file
        """
        self.df = (
            fam.filter(["iid"])
            .merge(self.df, left_on="iid", right_on="sample", how="left", sort=False)
            .drop(columns=["sample"])
        )

    def get(self, name: str) -> Tuple[VectorFloat64, VectorBool]:
        """
        Returns the trait data for a given name

        Parameters
        ----------
        name : str
            The name of the trait

        Returns
        -------
        Tuple[VectorFloat64, VectorBool]
            The trait value data and a boolean mask indicating which samples are not NaN
        """
        if name not in self.name:
            raise ValueError(f"The trait {name} does not exist.")
        data: VectorFloat64 = self.df[name].values
        not_na: VectorBool = ~np.isnan(data)
        data: VectorFloat64 = data[not_na]
        return data, not_na
