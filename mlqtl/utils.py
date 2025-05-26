import subprocess
import pandas as pd
import numpy as np
import os
import re
import tempfile
import importlib
import inspect
from sklearn.base import RegressorMixin


def get_class_from_path(class_path_string: str) -> RegressorMixin:
    """
    Given a string representing a class path, import the class and return it

    Parameters
    ----------
    class_path_string : str
        A string representing the class path, e.g. "module.submodule.ClassName" or "ClassName"

    Returns
    -------
    RegressorMixin
        The imported class object, which should be a subclass of RegressorMixin
    """
    if not class_path_string:
        raise ValueError("class_path_string must not be empty")

    if "." in class_path_string:
        module_path, class_name = class_path_string.rsplit(".", 1)
    else:
        module_path = class_name = class_path_string

    try:
        imported_module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_path}' from class path '{class_path_string}': {e}"
        )

    try:
        class_object = getattr(imported_module, class_name)
    except AttributeError:
        raise ImportError(
            f"Attribute or class named '{class_name}' not found in module '{module_path}' "
            f"(from class path '{class_path_string}')."
        )

    # Check if the class is a subclass of RegressorMixin
    if not inspect.isclass(class_object) or not issubclass(
        class_object, RegressorMixin
    ):
        raise TypeError(
            f"The class '{class_name}' in module '{module_path}' is not a subclass of RegressorMixin."
        )

    return class_object


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
