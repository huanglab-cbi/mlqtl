import click
import os
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _print_params_panel(rows):
    table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
    table.add_column("Parameter", style="bold")
    table.add_column("Value", style="cyan")
    for key, value in rows:
        table.add_row(key, value)
    panel = Panel.fit(table, title="mlQTL Analysis Parameters", border_style="green")
    console.print("\n", panel, "\n", sep="")


def _info(message):
    console.print(f"[bold]INFO:[/bold] {message}")


def _warn(message):
    console.print(f"[bold yellow]WARNING:[/bold yellow] {message}")


def _error(message):
    console.print(f"[bold red]ERROR:[/bold red] {message}")


def _success(message):
    console.print(f"[bold green]SUCCESS:[/bold green] {message}")


# from .data import Dataset
# from .train import train_with_progressbar, feature_importance
# from .datautils import (
#     sliding_window_newmethod,
#     proc_train_res,
#     build_qtl_regions_from_high_windows,
# )
# from .plot import plot_graph, plot_feature_importance
# from .utils import get_class_from_path, gff3_to_range, gtf_to_range


@click.group()
def main():
    """mlQTL: Machine Learning for QTL Analysis"""
    pass


@main.command()
@click.option(
    "-g",
    "--geno",
    type=str,
    required=True,
    help="Path to genotype file (plink binary format)",
)
@click.option(
    "-p",
    "--pheno",
    type=click.Path(exists=True),
    required=True,
    help="Path to phenotype file",
)
@click.option(
    "-r",
    "--range",
    type=click.Path(exists=True),
    required=True,
    help="Path to plink gene range file",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    required=True,
    help="Path to output directory",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=1,
    help="Number of processes to use",
    show_default=True,
)
@click.option(
    "-m",
    "--model",
    type=str,
    default="DecisionTreeRegressor,RandomForestRegressor,SVR",
    help="Model to use",
    show_default=True,
)
@click.option("-c", "--chrom", type=str, default=None, help="Chromosome to analyze")
@click.option("--trait", type=str, default=None, help="Trait to analyze")
@click.option(
    "--onehot",
    is_flag=True,
    default=False,
    help="Use one-hot encoding for categorical features",
)
@click.option(
    "--padj",
    type=bool,
    default=True,
    show_default=True,
    help="Use adjusted p-value for significance threshold",
)
@click.option(
    "--center-window-kb",
    type=int,
    default=400,
    show_default=True,
    help="Window radius in kilobases (kb) for symmetric neighborhood (e.g., 400 for ±400kb)",
)
@click.option(
    "--center-step-genes",
    type=int,
    default=10,
    show_default=True,
    help="Step size in number of genes for center-based window",
)
@click.option(
    "--q",
    type=float,
    default=0.9,
    show_default=True,
    help="Quantile used as window score for --newmethod (e.g. 0.9 = 90% quantile)",
)
@click.option(
    "--top-prop",
    type=float,
    default=0.10,
    show_default=True,
    help="Top proportion of windows genome-wide selected as QTL for --newmethod (e.g. 0.10 = top 10%)",
)
def run(
    geno,
    pheno,
    range,
    out,
    jobs,
    model,
    chrom,
    trait,
    onehot,
    padj,
    center_window_kb,
    center_step_genes,
    q,
    top_prop,
):
    """Run mlQTL analysis"""
    from .data import Dataset
    from .train import train_with_progressbar
    from .datautils import (
        sliding_window_newmethod,
        proc_train_res,
        build_qtl_regions_from_high_windows,
    )
    from .plot import plot_graph
    from .utils import get_class_from_path

    _print_params_panel(
        [
            ("Genotype file", str(geno)),
            ("Phenotype file", str(pheno)),
            ("Gene range file", str(range)),
            ("Output directory", str(out)),
            ("Number of processes", str(jobs)),
            ("Model(s)", str(model)),
            (
                "Sliding window method",
                f"Symmetric neighborhood (±{center_window_kb} kb around each gene)",
            ),
            ("Sliding window step", f"{center_step_genes} genes"),
            ("Window score quantile (q)", str(q)),
            ("QTL definition (top windows genome-wide)", str(top_prop)),
            ("Chromosome", chrom if chrom else "all chromosomes"),
            ("Trait", trait if trait else "all traits"),
            ("One-hot encoding", "enabled" if onehot else "disabled"),
            ("Use adjusted p-value", "enabled" if padj else "disabled"),
        ]
    )

    try:
        dataset = Dataset(geno, range, pheno)
    except Exception as e:
        _error(str(e))
        return
    max_workers = jobs if jobs > 0 else 1
    analysis_trait = dataset.trait.name
    model = model.split(",")

    try:
        models = [get_class_from_path(m) for m in model]
    except ImportError as e:
        _error(str(e))
        return

    # check option values
    if trait:
        input_trait = set(trait.split(","))
        all_trait = set(dataset.trait.name)
        if not input_trait <= all_trait:
            _error(f"Trait {trait} not found in dataset")
            return
        analysis_trait = trait.split(",")

    if chrom:
        input_chrom = set(chrom.split(","))
        all_chrom = set(dataset.gene.df["chr"].unique())
        if not input_chrom <= all_chrom:
            _error(f"Chromosome {chrom} not found in dataset")
            return
        dataset.gene.filter_by_chr(chrom.split(","))

    # create output directory if not exists
    output_dir = os.path.join(out)
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        _warn(
            f"Output directory {output_dir} already exists. Existing files may be overwritten."
        )
    except OSError as e:
        _error(f"Error creating output directory {output_dir}: {e}")
        return

    # Start the analysis
    _info("Starting analysis")
    for trait in analysis_trait:
        console.print(f"[bold]Trait:[/bold] {trait}")
        _info("Training model")
        train_res = train_with_progressbar(
            trait,
            models,
            dataset,
            max_workers,
            onehot,
        )
        _info("Processing training result")
        try:
            train_res_processed = proc_train_res(train_res, models, dataset, padj)
            sw_res, sig_genes, window_threshold = sliding_window_newmethod(
                train_res_processed,
                center_window_kb,
                center_step_genes,
                q=q,
                top_prop=top_prop,
            )

        except Exception as e:
            _error(str(e))
            return

        if sig_genes.empty:
            _warn("No significant genes found for this trait")

        trait_dir = os.path.join(out, f"{trait}")
        os.mkdir(trait_dir) if not os.path.exists(trait_dir) else None
        # plot and save
        plot_path = os.path.join(trait_dir, "sliding_window")
        plot_graph(sw_res, 10 ** (-window_threshold), plot_path, save=True)
        _info(f"Result graph saved to {plot_path}.png")
        # save the sliding window result
        df_path = os.path.join(trait_dir, "candidate_genes.tsv")
        sig_genes.to_csv(
            df_path,
            sep="\t",
            header=True,
            index=False,
        )
        _info(f"Candidate genes table saved to {df_path}")
        qtl_regions = build_qtl_regions_from_high_windows(
            sw_res,
            train_res_processed,
            window_threshold,
        )
        if not qtl_regions.empty:
            qtl_path = os.path.join(trait_dir, "qtl_regions.tsv")
            qtl_regions.to_csv(qtl_path, sep="\t", header=True, index=False)
            _info(f"QTL regions table saved to {qtl_path}")

        # save the original training result
        # import pickle
        # pkl_path = os.path.join(trait_dir, "train_res.pkl")
        # with open(pkl_path, "wb") as f:
        #     pickle.dump(train_res, f)
        # _info(f"Original training result saved to {pkl_path}")
        # save the training result as dataframe
        train_res_processed.to_csv(
            os.path.join(trait_dir, "train_res.tsv"),
            sep="\t",
            index=False,
            header=True,
        )
        _info(
            f"Training result table saved to {os.path.join(trait_dir, 'train_res.tsv')}"
        )
    _success("Analysis completed")


@main.command()
@click.option(
    "-g",
    "--geno",
    type=str,
    required=True,
    help="Path to genotype file (plink binary format)",
)
@click.option(
    "-p",
    "--pheno",
    type=click.Path(exists=True),
    required=True,
    help="Path to phenotype file",
)
@click.option(
    "-r",
    "--range",
    type=click.Path(exists=True),
    required=True,
    help="Path to plink gene range file",
)
@click.option("-o", "--out", type=click.Path(), required=True, help="Output directory")
@click.option("--gene", type=str, required=True, help="Gene name ( only one gene )")
@click.option(
    "-m",
    "--model",
    type=str,
    default="DecisionTreeRegressor,RandomForestRegressor,SVR",
    help="Model to use",
    show_default=True,
)
@click.option("--trait", type=str, required=True, help="Trait name ( only one trait )")
@click.option(
    "--onehot",
    is_flag=True,
    default=False,
    help="Use one-hot encoding for categorical features",
)
def importance(geno, pheno, range, gene, model, trait, out, onehot):
    """Calculate feature importance and plot bar chart"""
    from .data import Dataset
    from .train import feature_importance
    from .plot import plot_feature_importance
    from .utils import get_class_from_path

    try:
        dataset = Dataset(geno, range, pheno)
    except Exception as e:
        _error(str(e))
        return
    model = model.split(",")
    try:
        models = [get_class_from_path(m) for m in model]
    except ImportError as e:
        _error(str(e))
        return

    # create output directory if not exists
    output_dir = os.path.join(out)
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        _warn(
            f"Output directory {output_dir} already exists. Existing files may be overwritten."
        )
    except OSError as e:
        _error(f"Error creating output directory {output_dir}: {e}")
        return

    if gene not in dataset.gene.name:
        _error(f"Gene {gene} not found in dataset")
        return
    if trait not in dataset.trait.name:
        _error(f"Trait {trait} not found in dataset")
        return
    _info(f"Calculating feature importance for gene {gene} and trait {trait}")
    feature_importance_df = feature_importance(gene, trait, models, dataset, onehot)
    _success("Feature importance calculated successfully")
    gene_dir = os.path.join(output_dir, gene)
    if not os.path.exists(gene_dir):
        os.mkdir(gene_dir)
    # save the feature importance dataframe
    feature_importance_df.to_csv(
        os.path.join(gene_dir, f"{gene}_{trait}_feature_importance.tsv"),
        sep="\t",
        index=True,
    )
    _info("Starting feature importance plot")
    # feature importance plot
    plot_path = os.path.join(gene_dir, f"{gene}_{trait}")
    plot_feature_importance(feature_importance_df, 10, True, plot_path)
    _info(f"Feature importance plot saved to {gene_dir}")


@main.command()
@click.option(
    "-f", "--gff", type=click.Path(exists=True), required=True, help="Path to gff file"
)
@click.option("-r", "--region", type=str, required=True, help="The region to convert")
@click.option("-o", "--out", type=click.Path(), required=True, help="Output directory")
def gff2range(gff, region, out):
    """Convert GFF3 file to plink gene range format"""
    from .utils import gff3_to_range

    try:
        os.makedirs(out)
    except FileExistsError:
        _warn(
            f"Output directory {out} already exists. Existing files may be overwritten."
        )
    df = gff3_to_range(gff, region)
    if df is None:
        _error("The selected area does not exist in the file")
        return
    prefix = os.path.splitext(os.path.basename(gff))[0]
    out_path = os.path.join(out, f"{prefix}_{region}.range")
    df.to_csv(out_path, sep="\t", header=False, index=False)
    _success(f"The range file is saved to {out_path}")


@main.command()
@click.option(
    "-f", "--gtf", type=click.Path(exists=True), required=True, help="Path to gff file"
)
@click.option("-r", "--region", type=str, required=True, help="The region to convert")
@click.option("-o", "--out", type=click.Path(), required=True, help="Output directory")
def gtf2range(gtf, region, out):
    """Convert GTF file to plink gene range format"""

    from .utils import gtf_to_range

    try:
        os.makedirs(out)
    except FileExistsError:
        _warn(
            f"Output directory {out} already exists. Existing files may be overwritten."
        )
    df = gtf_to_range(gtf, region)
    if df is None:
        _error("The selected area does not exist in the file")
        return
    prefix = os.path.splitext(os.path.basename(gtf))[0]
    out_path = os.path.join(out, f"{prefix}_{region}.range")
    df.to_csv(out_path, sep="\t", header=False, index=False)
    _success(f"The range file is saved to {out_path}")


if __name__ == "__main__":
    main()
