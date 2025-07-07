# 🧬 ML-QTL: Machine Learning for Quantitative Trait Loci Mapping

[![PyPI version](https://badge.fury.io/py/mlqtl.svg?icon=si%3Apython)](https://badge.fury.io/py/mlqtl)

`ML-QTL` is a machine learning–based Python tool for QTL mapping. It assesses SNP–trait associations using regression model performance and identifies candidate QTL regions through a sliding window approach. The tool enables efficient gene discovery and supports molecular breeding in crops.

-----

## ⚙️ Features

  * **Efficient Data Handling**: Utilizes `plink` binary file formats for genotype data, enabling efficient handling of large-scale genomic datasets
  * **Flexible Modeling**: Supports multiple regression models, including Decision Tree Regression, Random Forest Regression, and Support Vector Regression
  * **Clear Visualization**: Generates sliding window prediction results with output visualization capabilities
  * **Gene-Level Insights**: Calculates and reports SNP importance scores within specific genes
  * **Parallelism**: Built-in support for multiprocessing to dramatically speed up analysis
  * **Flexibility**: Offers a Command-Line Interface (CLI) for automation and a **Python API for custom scripting

-----

## 📦 Installation

We highly recommend using a virtual environment to prevent dependency conflicts.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate
```

### Install with pip (Recommended)

Install the latest version directly from PyPI:

```bash
pip install mlqtl
```

> **Warning**
> As of version 2.3.0, NumPy no longer supports Linux systems with `glibc` version below 2.28. If you are on an older Linux system, please use one of the following installation methods:

```bash
# Force install using a binary wheel for NumPy
pip install mlqtl --only-binary=numpy

# Or, install a compatible version of NumPy before installing mlqtl
pip install numpy==2.2.6 mlqtl
```

### Install from Source

1.  **Download the Source Code**

    ```bash
    # Clone from GitHub
    git clone https://github.com/huanglab-cbi/mlqtl.git

    # Or download from our website
    wget https://cbi.njau.edu.cn/mlqtl-doc/download/source_code.tar.gz
    ```

2.  **Navigate to the Directory**

    ```bash
    cd mlqtl
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Build the Package**

    ```bash
    pip install build
    python -m build
    ```

5.  **Install the Built Package**

    ```bash
    # Replace {version} with the actual version number
    pip install dist/mlqtl-{version}-py3-none-any.whl
    ```

-----

## 🚀 Usage

`ML-QTL` requires genotype data in the plink binary format (`.bed`, `.bim`, `.fam`). If your data is in VCF format, you must first convert it using [plink](https://www.cog-genomics.org/plink).

The primary CLI tool provides several commands:

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

For detailed instructions and API usage, please see the full [**documentation**](https://cbi.njau.edu.cn/mlqtl-doc/en/index.html).

-----

## 🧪 Example Walkthrough

### Step 1: Download Sample Data

Visit the [download page](https://cbi.njau.edu.cn/mlqtl-doc/download/) to get `imputed_base_filtered_v0.7.vcf.gz`, `gene_location_range.txt`, and `grain_length.txt`.
Alternatively, use the following commands to download them:

```bash
wget https://cbi.njau.edu.cn/mlqtl-doc/download/imputed_base_filtered_v0.7.vcf.gz
wget https://cbi.njau.edu.cn/mlqtl-doc/download/gene_location_range.txt
wget https://cbi.njau.edu.cn/mlqtl-doc/download/grain_length.txt
```

> **Note:** The `gene_location_range.txt` is generated based on the GFF file of the reference genome. For details, please refer to the [documentation](https://cbi.njau.edu.cn/mlqtl-doc/en/index.html)

### Step 2: Preprocess the Data

Convert the VCF file to plink's binary format.

```bash
# Define the VCF file variable
vcf=imputed_base_filtered_v0.7.vcf.gz

# Run plink to convert and filter the data
plink --vcf ${vcf} \
      --snps-only \
      --allow-extra-chr \
      --make-bed \
      --double-id \
      --vcf-half-call m \
      --extract range gene_location_range.txt \
      --out imputed
```

### Step 3: Run ML-QTL Analysis

**1. Run Analysis**

```bash
mlqtl run -g imputed \
          -p grain_length.txt \
          -r gene_location_range.txt \
          -j 8 \
          --padj \
          --threshold 2.74e-5 \
          -o result
```

**2. Calculate SNP Importance**

```bash
mlqtl importance -g imputed \
                 -p grain_length.txt \
                 -r gene_location_range.txt \
                 --trait grain_length \
                 --gene Os03g0407400 \
                 -m DecisionTreeRegressor \
                 -o result
```

### 📊 Performance Benchmark

The `-j` option sets the number of parallel processes. Generally, the more processes you use, the shorter the runtime. The following benchmarks were conducted on an **AMD EPYC 7543 CPU**.

| Processes | Memory | Time |
| :---: | :----: | :----: |
| 1 | 1.76G | 5.5h |
| 2 | 2.22G | 2.5h |
| 4 | 3.15G | 1h |
| **8** | **5G** | **35min**|
| 16 | 8.74G | 19min|
| 32 | 16.18G | 10min|
| 64 | 31.04G | 6min |

Please select an appropriate number of processes based on your system's resources.