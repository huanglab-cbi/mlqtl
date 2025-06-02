import subprocess
import pandas as pd
import numpy as np
import os
import re
from pandas import DataFrame
import importlib
import inspect
from sklearn.base import RegressorMixin


def get_class_from_path(class_path_string: str) -> RegressorMixin:
    """
    Given a string representing a class path, import the class and return it

    Parameters
    ----------
    class_path_string : str
        A string representing the class path, e.g. "module.submodule.ClassName"

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
        model_name = {
            "DecisionTreeRegressor": "sklearn.tree.DecisionTreeRegressor",
            "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
            "SVR": "sklearn.svm.SVR",
        }
        if class_path_string in model_name:
            module_path, class_name = model_name[class_path_string].rsplit(".", 1)
        else:
            raise ValueError(
                f"Invalid class path string '{class_path_string}'. "
                "It should be in the format 'module.submodule.ClassName'"
            )

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
        raise RuntimeError(f"{e.stderr}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while running plink: {e}")


def gtf_to_range(gtf_file: str, region: str = "CDS") -> DataFrame:
    """
    Convert GTF file to plink gene range format
    """
    if not os.path.exists(gtf_file):
        raise FileNotFoundError(f"The file {gtf_file} does not exist")
    gtf = pd.read_csv(gtf_file, sep="\t", header=None).drop(columns=[1, 5, 6, 7])
    gtf.columns = ["chr", "region", "start", "end", "note"]
    gtf = gtf[gtf["region"] == region]
    if gtf.empty:
        return None
    gtf["transcript_id"] = gtf["note"].str.extract(r'transcript_id\s+"([^"]+)"')
    gtf["gene_id"] = gtf["note"].str.extract(r'gene_id\s+"([^"]+)"')
    gtf = gtf[["chr", "start", "end", "transcript_id", "gene_id"]]
    return gtf


def gff3_to_range(gff_file: str, region: str = "CDS") -> DataFrame:
    """
    Convert GFF3 file to plink gene range format
    """
    gff = pd.read_csv(gff_file, sep="\t", header=None, comment="#")
    gff_filtered = gff[gff[2] == region].filter([0, 3, 4, 8])
    if gff_filtered.empty:
        return None
    gff_filtered.columns = ["chr", "start", "end", "attributes"]
    gff_filtered["id"] = gff_filtered["attributes"].str.extract(
        r"[Ii][Dd]=([^;]+)", expand=False
    )
    gff_filtered["parent_temp"] = gff_filtered["attributes"].str.extract(
        r"[Pp]arent=([^;]+)", expand=False
    )
    gff_filtered["parent"] = gff_filtered["parent_temp"].fillna(gff_filtered["id"])
    gff_filtered = gff_filtered.filter(["chr", "start", "end", "id", "parent"])
    return gff_filtered
