import subprocess
import pandas as pd
import numpy as np
import os
import re
import tempfile


def run_plink(cmd: str) -> str:
    """
    Run a plink command and return the output.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running plink command: {e.stderr}")
        raise e


def gff_to_gtf(gff_file: str, gtf_file: str):
    """
    Convert a GFF file to GTF format using gffread.
    """
    if not os.path.exists(gff_file):
        raise FileNotFoundError(f"The file {gff_file} does not exist")
    try:
        cmd = f"gffread {gff_file} -T -o {gtf_file}"
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error converting GFF to GTF: {e.stderr}")
        raise e


def gtf_to_position(gtf_file: str, region: str = "CDS", output: str = "output.tsv"):
    if not os.path.exists(gtf_file):
        raise FileNotFoundError(f"The file {gtf_file} does not exist")
    if region != "CDS" and region != "exon":
        raise ValueError(f"Invalid region specified: {region}. Must be 'CDS' or 'exon'")
    gtf = pd.read_csv(gtf_file, sep="\t", header=None).drop(columns=[1, 5, 6, 7])
    gtf.columns = ["chr", "region", "start", "end", "note"]
    gtf = gtf[gtf["region"] == region]
    gtf["transcript_id"] = gtf["note"].str.extract(r'transcript_id\s+"([^"]+)"')
    gtf["gene_id"] = gtf["note"].str.extract(r'gene_id\s+"([^"]+)"')
    gtf = gtf[["chr", "start", "end", "transcript_id", "gene_id"]]
    gtf.to_csv(output, sep="\t", index=False, header=False)


def gff3_to_position(gff_file: str, region: str = "CDS", output: str = "output.tsv"):
    try:
        tmp_dir = tempfile.TemporaryDirectory(dir=os.getcwd())
        gtf = os.path.join(tmp_dir.name, "temp.gtf")
        gff_to_gtf(gff_file, gtf)
        gtf_to_position(gtf, region, output)
        tmp_dir.cleanup()
        print(f"Converted {gff_file} to {os.path.abspath(output)} for region {region}")
    except Exception as e:
        print(f"Error converting GFF3 to position: {e}")
        raise e


def convert_onehot(X: np.ndarray) -> np.ndarray:
    result = np.zeros((X.shape[0], X.shape[1], 4), dtype=float)
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
