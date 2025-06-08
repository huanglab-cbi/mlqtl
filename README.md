ML-QTL
---

## 1. Overview  

`ML-QTL` is a Python package that leverages machine learning to predict phenotype-associated genes. It evaluates the significance of associations between single nucleotide polymorphisms (SNPs) and phenotypes by assessing the predictive performance of regression models. The tool provides native support for `plink` binary files, enabling efficient processing of large-scale genomic datasets.

## 2. Features  

- Utilizes `plink` binary file formats for genotype data, enabling efficient handling of large-scale genomic datasets
- Supports multiple regression models (default: Decision Tree Regression, Random Forest Regression, and Support Vector Regression)
- Generates sliding window prediction results with output visualization capabilities
- Calculates SNP importance scores within genes
- Supports multiprocessing parallelism
- Provides both command-line interface (CLI) and Python API usage modes

## 3. Installation  

### 1. Install via pip

We recommend creating a virtual environment first to avoid dependency conflicts (optional):

```bash
python -m venv venv
source venv/bin/activate 
```
Install from PyPI:

```bash
pip install mlqtl
```

### 2. Install via uv

```bash
uv add mlqtl
```

### 3. Install from source

1. Download source code
 
    ```bash
    # Clone the code repository from GitHub
    git clone https://github.com/huanglab-cbi/mlqtl.git

    # Download from our website
    wget https://cbi.njau.edu.cn/mlqtl-doc/download/mlqtl-{version}.tar.gz
    ```

2. Navigate to source directory

    ```bash
    cd mlqtl
    ```
3. Install dependencies

    ```bash
    #For pip users:
    pip install -r requirements.txt

    #For uv users:
    uv sync
    ```

4. Build package

    ```bash
    # For pip users:
    pip install build
    python -m build

    # For uv users:
    uv build
    ```

5. Install package

    ```bash
    # For pip users:
    pip install dist/mlqtl-{version}-py3-none-any.whl

    # For uv users:
    uv add dist/mlqtl-{version}-py3-none-any.whl
    ```

## 4. Usage

This program is based on genotype data in the plink binary file format. If your data is in VCF format, please install [plink](https://www.cog-genomics.org/plink) first.

```bash
❯ mlqtl --help
Usage: mlqtl [OPTIONS] COMMAND [ARGS]...

ML-QTL: Machine Learning for QTL Analysis

Options:
--help  Show this message and exit.

Commands:
gff2range   Convert GFF3 file to plink gene range format
gtf2range   Convert GTF file to plink gene range format
importance  Calculate feature importance and plot bar chart
rerun       Re-run sliding window analysis with new parameters
run         Run ML-QTL analysis
vcf2plink   Convert VCF file to plink binary format
```

For detailed usage instructions, please refer to the [documentation](https://cbi.njau.edu.cn/mlqtl-doc/en/index.html)