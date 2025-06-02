from .snp import SNP
from .trait import Trait
from .gene import Gene
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Tuple
from ..nda_typing import VectorInt8, VectorBool


class Dataset:
    """
    The Dataset class is used to load and manage the SNP, trait, and gene data
    """

    def __init__(self, snp_file=None, gene_file=None, trait_file=None):

        self._init_gene(gene_file)
        self._init_snp(snp_file)
        self.check_chrom()
        if trait_file is not None:
            self._init_trait(trait_file)
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

    def check_chrom(self) -> None:
        """
        Check if the chromosome names match
        """
        chrom_snp = set(self.snp.chrom)
        chrom_gene = set(self.gene.chrom)
        if not chrom_gene.issubset(chrom_snp):
            err_chr = []
            for chr_g in chrom_gene:
                if chr_g not in chrom_snp:
                    err_chr.append(chr_g)
            raise ValueError(f"Gene chromosomes {err_chr} are not in SNP chromosomes")

    def get(self, gene: str) -> List[Tuple[int, VectorInt8]]:
        """
        Return the snps for a given gene

        Parameters
        ----------
        gene : str
            The gene name

        Returns
        -------
        List[Tuple[int, VectorInt8]]
            A list of tuples, each tuple contains the snp index and the binary snp data
        """
        gene_position: DataFrame = self.gene.df[self.gene.df["gene"] == gene].filter(
            ["chr", "start", "end"]
        )
        min_start: int = gene_position["start"].min()
        max_end: int = gene_position["end"].max()
        mask: VectorBool = (
            (self.snp._bim["chrom"] == gene_position["chr"].values[0])
            & (self.snp._bim["pos"] >= min_start)
            & (self.snp._bim["pos"] <= max_end)
        )
        merged: DataFrame = pd.merge(
            self.snp._bim[["chrom", "pos", "i"]][mask],
            gene_position,
            left_on="chrom",
            right_on="chr",
        )
        if merged.empty:
            raise ValueError(f"The gene {gene} does not exist in the snp data")

        mask: VectorBool = (merged["pos"] >= merged["start"]) & (
            merged["pos"] <= merged["end"]
        )
        markers_idx: List[int] = merged[mask]["i"].to_list()
        if not markers_idx:
            return np.array([])

        return [(i, self.snp._seek_and_read(i)) for i in markers_idx]

    def get_hap(self, gene: str) -> DataFrame:
        """
        Return the haplotype for a given gene
        """
        gene_snps = self.get(gene)
        gene_snps_double_base = np.array([self.snp.base(*s) for s in gene_snps])
        double_to_single = {
            "AA": "A",
            "TT": "T",
            "CC": "C",
            "GG": "G",
            "AG": "R",
            "TC": "Y",
            "AC": "M",
            "GT": "K",
            "GC": "S",
            "AT": "W",
            "00": "-",
        }
        hap = np.full(
            gene_snps_double_base.shape, "N", dtype=gene_snps_double_base.dtype
        )
        for double, single in double_to_single.items():
            hap[gene_snps_double_base == double] = single
        hap = (
            pd.Series(["".join(h) for h in hap.T])
            .value_counts()
            .to_frame()
            .reset_index()
        )
        hap.columns = ["seq", "count"]
        hap.index.name = "rank"
        hap.index = hap.index + 1
        hap["hap"] = [f"Hap{idx}" for idx in hap.index.values]
        hap = hap.filter(["hap", "seq", "count"])
        return hap
