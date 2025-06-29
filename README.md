ML-QTL
---

## 1. Overview  

`ML-QTL` is a machine learning–based Python tool for QTL mapping. It assesses SNP–trait associations using regression model performance and identifies candidate QTL regions through a sliding window approach. The tool enables efficient gene discovery and supports molecular breeding in crops.

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

Note: Starting with version 2.3.0, NumPy no longer provides binary packages for Linux systems with glibc version below 2.28, [link](https://numpy.org/devdocs/release/2.3.0-notes.html). Therefore, if you are using an older Linux system, you need to install an earlier version of NumPy

If you encounter installation issues, you can try the following approaches:

```bash
# Force installation using NumPy's binary wheel package
pip install mlqtl --only-binary=numpy

# Alternatively, install a specific version of NumPy
pip install numpy==2.2.6 mlqtl
```

### 2. Install from source

1. Download source code
 
    ```bash
    # Clone the code repository from GitHub
    git clone https://github.com/huanglab-cbi/mlqtl.git

    # Download from our website
    wget https://cbi.njau.edu.cn/mlqtl-doc/download/source_code.tar.gz
    ```

2. Navigate to source directory

    ```bash
    cd mlqtl
    ```
3. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

4. Build package

    ```bash
    pip install build
    python -m build
    ```

5. Install package

    ```bash
    pip install dist/mlqtl-{version}-py3-none-any.whl
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
```

For detailed usage instructions, please refer to the [documentation](https://cbi.njau.edu.cn/mlqtl-doc/en/index.html)

## 5 Example  

### 1. download sample data

Visit the [download page](https://cbi.njau.edu.cn/mlqtl-doc/download/) to get `imputed_base_filtered_v0.7.vcf.gz`, `gene_location_range.txt`, and `grain_length.txt`.

Alternatively, use the following commands to download them:
```bash
wget https://cbi.njau.edu.cn/mlqtl-doc/download/imputed_base_filtered_v0.7.vcf.gz

wget https://cbi.njau.edu.cn/mlqtl-doc/download/gene_location_range.txt

wget https://cbi.njau.edu.cn/mlqtl-doc/download/grain_length.txt
```
`gene_location_range.txt` is generated based on the GFF file of the reference genome. For details, please refer to the [documentation](https://cbi.njau.edu.cn/mlqtl-doc/en/index.html)

### 2. data preprocessing

```bash
vcf=imputed_base_filtered_v0.7.vcf.gz

plink --vcf ${vcf} --snps-only --allow-extra-chr --make-bed --double-id --vcf-half-call m --extract range gene_location_range.txt --out imputed
```

### 3. run mlqlt command line

```bash
mlqtl run -g imputed -p grain_length.txt -r gene_location_range.txt -j 64 --padj --threshold 2.74e-5 -o result

mlqtl importance -g imputed -p grain_length.txt -r gene_location_range.txt --trait grain_length --gene Os03g0407400 -m DecisionTreeRegressor -o result
```