from .snp import SNP
from .trait import Trait
from .gene import Gene
from mlqtl.utils import run_plink
import pandas as pd
import numpy as np


class DataCollection:
    def __init__(self, snp_file=None, trait_file=None, gene_file=None):

        self._init_trait(trait_file)
        self._init_gene(gene_file)
        self._init_snp(snp_file)

        self.trait.filter_df(self.snp._fam)

    def _init_trait(self, trait_file):
        if trait_file is None:
            raise ValueError("Trait file is required")
        self.trait = Trait(trait_file)

    def _init_gene(self, gene_file):
        if gene_file is None:
            raise ValueError("Gene file is required")
        self.gene = Gene(gene_file)

    def _init_snp(self, snp_file):
        if snp_file is None:
            raise ValueError("SNP file is required")
        self.snp = SNP(snp_file)

    def get_gene_snps(self, gene: str):
        """
        Return the snps for a given gene
        """
        gene_position = self.gene.df[self.gene.df["gene"] == gene].filter(
            ["chr", "start", "end"]
        )
        min_start = gene_position["start"].min()
        max_end = gene_position["end"].max()
        mask = (
            (self.snp._bim["chrom"] == gene_position["chr"].values[0])
            & (self.snp._bim["pos"] >= min_start)
            & (self.snp._bim["pos"] <= max_end)
        )
        merged = pd.merge(
            self.snp._bim[["chrom", "pos", "i"]][mask],
            gene_position,
            left_on="chrom",
            right_on="chr",
        )
        if merged.empty:
            raise ValueError(f"The gene {gene} does not exist in the snp data")

        mask = (merged["pos"] >= merged["start"]) & (merged["pos"] <= merged["end"])
        markers_idx = merged[mask]["i"].to_list()
        if not markers_idx:
            return np.array([])

        return [(i, self.snp._seek_and_read(i)) for i in markers_idx]
