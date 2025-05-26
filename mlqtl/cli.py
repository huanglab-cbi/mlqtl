import click
import numpy as np
import os
import pickle

from .data import Dataset
from .training import train_with_progressbar
from .datautils import calculate_sliding_window
from .plot import plot_graph
from .utils import get_class_from_path


@click.group()
def main():
    """ML-QTL: Machine Learning for QTL Analysis"""
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
@click.option("-j", "--jobs", type=int, default=1, help="Number of processes to use")
@click.option(
    "--threshold", type=float, default=2.74e-05, help="Significance threshold"
)
@click.option("-w", "--window", type=int, default=100, help="Sliding window size")
@click.option("--step", type=int, default=10, help="Sliding window step size")
@click.option(
    "-m",
    "--model",
    type=str,
    default="DecisionTreeRegressor,RandomForestRegressor,SVR",
    help="Model to use",
)
@click.option("-c", "--chrom", type=(str), default=None, help="Chromosome to analyze")
@click.option("--trait", type=(str), default=None, help="Trait to analyze")
def run(geno, pheno, range, out, jobs, threshold, window, step, model, chrom, trait):
    """Run ML-QTL analysis"""

    # echo the parameters
    click.secho("#### ML-QTL Analysis Parameters ####", fg="green")
    click.echo(f"Genotype file: {geno}")
    click.echo(f"Phenotype file: {pheno}")
    click.echo(f"Gene range file: {range}")
    click.echo(f"Output directory: {out}")
    click.echo(f"Number of processes: {jobs}")
    click.echo(f"Significance threshold: {threshold}")
    click.echo(f"Sliding window size: {window}")
    click.echo(f"Sliding window step size: {step}")
    click.echo(f"Model(s): {model}")
    click.echo(f"Chromosome: {chrom if chrom else 'All chromosomes'}")
    click.echo(f"Trait: {trait if trait else 'All traits'}")
    click.echo("")

    # define some variable
    threshold_norm = -np.log10(threshold)
    kwargs = {
        "snp_file": geno,
        "trait_file": pheno,
        "gene_file": range,
    }
    dataset = Dataset(**kwargs)
    max_workers = jobs if jobs > 0 else 1
    analysis_trait = dataset.trait.name
    default_model = ["DecisionTreeRegressor", "RandomForestRegressor", "SVR"]
    if set(model.split(",")) == set(default_model):
        click.secho(
            "Using default models: DecisionTreeRegressor, RandomForestRegressor, SVR",
            fg="yellow",
        )
        model = [
            "sklearn.tree.DecisionTreeRegressor",
            "sklearn.ensemble.RandomForestRegressor",
            "sklearn.svm.SVR",
        ]

    models = [get_class_from_path(m) for m in model]

    # check option values
    if trait:
        input_trait = set(dataset.trait.name)
        all_trait = set(dataset.trait.name)
        if input_trait not in all_trait:
            click.secho(
                f"Trait {trait} not found in dataset",
                fg="red",
            )
            return
        analysis_trait = trait

    if chrom:
        input_chrom = set(chrom)
        all_chrom = set(dataset.gene.df["chr"].unique())
        if input_chrom not in all_chrom:
            click.secho(
                f"Chromosome {chrom} not found in dataset",
                fg="red",
            )
            return
        dataset.gene.filter_by_chr(list(chrom))
    # create output directory if not exists
    output_dir = os.path.join(out, "mlqtl_result")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start the analysis
    click.secho("#### Starting ML-QTL Analysis ####", fg="green")
    click.echo()
    for trait in analysis_trait:
        click.secho(f"Analyzing trait: {trait}", fg="green")
        click.echo(f"==> Training trait: {trait}")
        train_res = train_with_progressbar(trait, models, dataset, max_workers)
        click.echo(f"==> Training completed for trait: {trait}")
        sliding_window_result, significant_genes = calculate_sliding_window(
            train_res, models, dataset, window, step, threshold_norm
        )

        trait_dir = os.path.join(out, "mlqtl_result", f"{trait}")
        os.mkdir(trait_dir) if not os.path.exists(trait_dir) else None
        # plot and save
        plot_path = os.path.join(trait_dir, f"{trait}_sliding_window")
        plot_graph(sliding_window_result, threshold_norm, plot_path, save=True)
        click.echo(f"==> Graph plotted and saved to {plot_path}")
        # save the sliding window result
        df_path = os.path.join(trait_dir, f"{trait}_significant_genes.tsv")
        significant_genes.to_csv(
            df_path,
            sep="\t",
            header=True,
            index=False,
        )
        click.echo(f"==> Significant genes saved to {df_path}")
        # save the original training result
        pkl_path = os.path.join(trait_dir, f"{trait}_train_res.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(train_res, f)
        click.echo(f"==> Training results saved to {pkl_path}")
        click.echo()
    click.secho("#### Analysis completed ####", fg="green")

@main.command()
@click.option(
    "--geno",
    type=click.Path(exists=True),
    required=True,
    help="Path to genotype file (plink binary format)",
)
@click.option(
    "--pheno",
    type=click.Path(exists=True),
    required=True,
    help="Path to phenotype file",
)
@click.option(
    "--range",
    type=click.Path(exists=True),
    required=True,
    help="Path to plink gene range file",
)
@click.option("-g", "--gene", type=(str), required=True, help="Gene name")
@click.option("-m", "--model", type=(str), required=True, help="Model name")
@click.option("--trait", type=(str), required=True, help="Trait name")
@click.option("-o", "--out", type=click.Path(), required=True, help="Output directory")
def importance(geno, pheno, range, gene, model, trait, out):
    """Calculate feature importance and plot bar chart"""
    pass


@main.command()
def gff2range():
    """Convert GFF3 file to plink gene range format"""
    pass


@main.command()
def gtf2range():
    """Convert GTF file to plink gene range format"""
    pass


if __name__ == "__main__":
    main()
