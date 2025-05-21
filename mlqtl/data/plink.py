"""Module that reads binary Plink files."""

# This file is part of pyplink.
#
# The MIT License (MIT)
#
# Copyright (c) 2014 Louis-Philippe Lemieux Perreault
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__author__ = "Louis-Philippe Lemieux Perreault"
__copyright__ = "Copyright 2014 Louis-Philippe Lemieux Perreault"
__license__ = "MIT"

# Modified by: yang1ong
# Date: 2025-04-23
# Description of changes: Removed unused functions to streamline core functionality

import os
import logging
import numpy as np
import pandas as pd
import multiprocessing
from collections import Counter
from itertools import repeat


# The logger
logger = logging.getLogger(__name__)


# The recoding values
_geno_recode = {
    1: -1,  # Unknown genotype
    2: 1,  # Heterozygous genotype
    0: 2,  # Homozygous A1
    3: 0,
}  # Homozygous A2


class Plink(object):
    """Reads and store a set of binary Plink files.

    Args:
        prefix (str): The prefix of the binary Plink files.

    Reads or write binary Plink files (BED, BIM and FAM).
    """

    # The genotypes values
    _geno_values = np.array(
        [[_geno_recode[(i >> j) & 3] for j in range(0, 7, 2)] for i in range(256)],
        dtype=np.int8,
    )

    def __init__(self, prefix: str):
        """Initializes a new Plink instance."""
        # The mode
        self._mode = "r"

        # The bed format
        self._bed_format = "SNP-major"

        # These are the name of the files
        self.bed_filename = f"{prefix}.bed"
        self.bim_filename = f"{prefix}.bim"
        self.fam_filename = f"{prefix}.fam"

        # Checking that all the files exists (otherwise, error...)
        for filename in (self.bed_filename, self.bim_filename, self.fam_filename):
            if not os.path.isfile(filename):
                raise IOError(f"No such file: '{filename}'")

        # The number of markers and samples
        self._nb_markers = None
        self._nb_samples = None

        # Setting BIM and FAM to None
        self._bim = None
        self._fam = None

        # Reading the input files
        self._read_bim()
        self._read_fam()

        # check the BED file
        self._check_bed()

        # index and lock
        self._index = multiprocessing.Value("i", 0)
        self._lock = multiprocessing.Lock()

    def __repr__(self):
        """The representation of the Plink object."""

        return f"Plink({self.nb_samples:,d} samples; " f"{self.nb_markers:,d} markers)"

    def __len__(self):
        """
        Returns the number of snps in the file
        """
        return self.nb_markers

    def __getitem__(self, key: int):
        """The __getitem__ function.
        Args:
            key (int): The index of the marker to read.
        Returns:
            tuple: The marker name as a string and its genotypes as a
            :py:class:`numpy.ndarray`.
        """
        return self.idx2marker(key), self._seek_and_read(key)

    def __iter__(self):
        """The __iter__ function."""
        return self

    def __next__(self):
        """The __next__ function."""
        return self.next()

    def next(self):
        """Returns the next marker.

        Returns:
            tuple: The marker name as a string and its genotypes as a
            :py:class:`numpy.ndarray`.

        """
        with self._lock:
            current_index = self._index.value
            if current_index > self.nb_markers:
                raise StopIteration()
            self._index.value += 1
            return self[current_index]

    def _get_seek_position(self, n: int):
        """Gets the seek position in the file (including special bytes).

        Args:
            n (int): The index of the marker to seek to.

        """
        return 3 + self._nb_bytes * n

    def _seek_and_read(self, n: int):
        """Reads the genotypes of a idx"""
        with open(self.bed_filename, "rb") as bed:
            bed.read(3)
            if n < 0 or n >= self.nb_markers:
                raise ValueError(f"invalid position in BED: {n}")
            bed.seek(self._get_seek_position(n))
            return self._geno_values[
                np.frombuffer(bed.read(self._nb_bytes), dtype=np.uint8)
            ].flatten(order="C")[: self.nb_samples]

    @property
    def nb_markers(self):
        if self._nb_markers is None:
            self._nb_markers = self._bim.shape[0]
        return self._nb_markers

    @property
    def duplicated_markers(self):
        if self._has_duplicated:
            return self._dup_markers
        else:
            return {}

    @property
    def nb_samples(self):
        if self._nb_samples is None:
            self._nb_samples = self._fam.shape[0]
        return self._nb_samples

    def _read_bim(self):
        """Reads the BIM file."""
        # Reading the BIM file and setting the values
        bim = pd.read_csv(
            self.bim_filename,
            sep=r"\s+",
            names=["chrom", "snp", "cm", "pos", "a1", "a2"],
            dtype=dict(snp=str, a1=str, a2=str),
        )

        # Saving the index as integer
        bim["i"] = bim.index

        # Checking for duplicated markers
        try:
            bim = bim.set_index("snp", verify_integrity=True)
            self._has_duplicated = False

        except ValueError:
            # Setting this flag to true
            self._has_duplicated = True

            # Finding the duplicated markers
            duplicated = bim.snp.duplicated(keep=False)
            duplicated_markers = bim.loc[duplicated, "snp"]
            duplicated_marker_counts = duplicated_markers.value_counts()

            # The dictionary that will contain information about the duplicated
            # markers
            self._dup_markers = {m: [] for m in duplicated_marker_counts.index}

            # Logging a warning
            logger.warning("Duplicated markers found")
            for marker, count in duplicated_marker_counts.items():
                logger.warning("  - %s: %s times", marker, count)
            logger.warning(
                "Appending ':dupX' to the duplicated markers "
                "according to their location in the BIM file"
            )

            # Renaming the markers
            counter = Counter()
            for i, marker in duplicated_markers.items():
                counter[marker] += 1
                new_name = f"{marker}:dup{counter[marker]}"
                bim.loc[i, "snp"] = new_name

                # Updating the dictionary containing the duplicated markers
                self._dup_markers[marker].append(new_name)

            # Resetting the index
            bim = bim.set_index("snp", verify_integrity=True)

        # Encoding the allele
        #   - The original 0 is the actual 2 (a1/a1)
        #   - The original 2 is the actual 1 (a1/a2)
        #   - The original 3 is the actual 0 (a2/a2)
        #   - The original 1 is the actual -1 (no call)
        allele_encoding = np.array(
            [bim.a2 * 2, bim.a1 + bim.a2, bim.a1 * 2, list(repeat("00", bim.shape[0]))],
            dtype="U2",
        )
        self._allele_encoding = allele_encoding.T

        # Saving the data in the object
        self._bim = bim[["chrom", "pos", "cm", "a1", "a2", "i"]]

    def _read_fam(self):
        """Reads the FAM file."""
        # Reading the FAM file and setting the values
        fam = pd.read_csv(
            self.fam_filename,
            sep=r"\s+",
            names=["fid", "iid", "father", "mother", "gender", "status"],
            dtype=dict(fid=str, iid=str, father=str, mother=str),
        )

        # Saving the data in the object
        self._fam = fam

    def _check_bed(self):
        """check the BED file."""
        # Checking if BIM and BAM files were both read
        if (self._bim is None) or (self._fam is None):
            raise RuntimeError("no BIM or FAM file were read")

        # The number of bytes per marker
        self._nb_bytes = int(np.ceil(self.nb_samples / 4.0))

        # Checking the file is valid by looking at the first 3 bytes and the
        # last entry (correct size)
        with open(self.bed_filename, "rb") as bed_file:
            # Checking that the first two bytes are OK
            if (ord(bed_file.read(1)) != 108) or (ord(bed_file.read(1)) != 27):
                raise ValueError(f"not a valid BED file: {self.bed_filename}")

            # Checking that the format is SNP-major
            if ord(bed_file.read(1)) != 1:
                raise ValueError(
                    f"not in SNP-major format (please recode): " f"{self.bed_filename}"
                )

            # Checking the last entry (for BED corruption)
            seek_index = self._get_seek_position(self._bim.iloc[-1, :].i)
            bed_file.seek(seek_index)
            geno = self._geno_values[
                np.frombuffer(bed_file.read(self._nb_bytes), dtype=np.uint8)
            ].flatten(order="C")[: self.nb_samples]
            if geno.shape[0] != self.nb_samples:
                raise ValueError("invalid number of entries: corrupted BED?")

        # Opening the file for the rest of the operations (reading 3 bytes)
        # self._bed = open(self.bed_filename, "rb")
        # self._bed.read(3)

    def marker2idx(self, marker: str):
        """Returns the index of a marker"""
        if marker not in self._bim.index:
            raise ValueError(f"{marker}: marker not in BIM")
        return self._bim.loc[marker, "i"]

    def idx2marker(self, idx: int):
        """Returns the marker name from its index"""
        if idx < 0 or idx >= self.nb_markers:
            raise ValueError(f"{idx}: index out of range")
        return self._bim.index[idx]

    def base(self, key, snp: np.ndarray) -> np.ndarray:
        """Convert binary to nucleobase"""
        match key:
            case int():
                return self._allele_encoding[key][snp]
            case str():
                return self._allele_encoding[self.marker2idx(key)][snp]
            case _:
                raise TypeError(f"Invalid key type: {type(key)}")
