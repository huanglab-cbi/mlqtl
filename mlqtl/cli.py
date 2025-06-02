import click
import numpy as np
import os
import pickle

from .data import Dataset
from .training import train_with_progressbar, feature_importance
from .datautils import calculate_sliding_window
from .plot import plot_graph, plot_feature_importance
from .utils import get_class_from_path, run_plink, gff3_to_range, gtf_to_range


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
@click.option("-c", "--chrom", type=str, default=None, help="Chromosome to analyze")
@click.option("--trait", type=str, default=None, help="Trait to analyze")
@click.option(
    "--onehot",
    is_flag=True,
    default=False,
    help="Use one-hot encoding for categorical features",
)
def run(
    geno, pheno, range, out, jobs, threshold, window, step, model, chrom, trait, onehot
):
    """Run ML-QTL analysis"""

    # echo the parameters
    click.echo("")
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
    click.echo(f"One-hot encoding: {'Enabled' if onehot else 'Disabled'}")
    click.echo("")

    # define some variable
    threshold_norm = -np.log10(threshold)
    dataset = Dataset(geno, range, pheno)
    max_workers = jobs if jobs > 0 else 1
    analysis_trait = dataset.trait.name
    model = model.split(",")

    try:
        models = [get_class_from_path(m) for m in model]
    except ImportError as e:
        click.secho(f"{e}", fg="red")
        return

    # check option values
    if trait:
        input_trait = set(trait.split(","))
        all_trait = set(dataset.trait.name)
        if not input_trait <= all_trait:
            click.secho(
                f"Trait {trait} not found in dataset",
                fg="red",
            )
            return
        analysis_trait = trait.split(",")

    if chrom:
        input_chrom = set(chrom.split(","))
        all_chrom = set(dataset.gene.df["chr"].unique())
        if not input_chrom <= all_chrom:
            click.secho(
                f"Chromosome {chrom} not found in dataset",
                fg="red",
            )
            return
        dataset.gene.filter_by_chr(chrom.split(","))

    # create output directory if not exists
    output_dir = os.path.join(out, "mlqtl_result")
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        click.secho(
            f"Output directory {output_dir} already exists. Existing files may be overwritten",
            fg="yellow",
        )
    except OSError as e:
        click.secho(
            f"Error creating output directory {output_dir}: {e}",
            fg="red",
        )
        return

    # Start the analysis
    click.echo()
    click.secho("#### Starting ML-QTL Analysis ####", fg="green")
    click.echo()
    for trait in analysis_trait:
        click.secho(f"Analyzing trait: {trait}", fg="green")
        click.echo(f"==> Training trait: {trait}")
        train_res = train_with_progressbar(trait, models, dataset, max_workers, onehot)
        click.echo(f"==> Training completed for trait: {trait}")
        sliding_window_result, significant_genes = calculate_sliding_window(
            train_res, models, dataset, window, step, threshold_norm
        )

        trait_dir = os.path.join(out, "mlqtl_result", f"{trait}")
        os.mkdir(trait_dir) if not os.path.exists(trait_dir) else None
        # plot and save
        plot_path = os.path.join(trait_dir, f"{trait}_sliding_window")
        plot_graph(sliding_window_result, threshold_norm, plot_path, save=True)
        click.echo(f"==> Graph plotted and saved to {plot_path}.png")
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
    type=str,
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
@click.option("-o", "--out", type=click.Path(), required=True, help="Output directory")
@click.option("--gene", type=str, required=True, help="Gene name ( only one gene )")
@click.option(
    "-m",
    "--model",
    type=str,
    default="DecisionTreeRegressor,RandomForestRegressor,SVR",
    help="Model to use",
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
    # echo the parameters
    click.echo("")
    click.secho("#### Feature Importance Parameters ####", fg="green")
    click.echo(f"Genotype file: {geno}")
    click.echo(f"Phenotype file: {pheno}")
    click.echo(f"Gene range file: {range}")
    click.echo(f"Gene name: {gene}")
    click.echo(f"Model(s): {model}")
    click.echo(f"Trait: {trait}")
    click.echo(f"One-hot encoding: {'Enabled' if onehot else 'Disabled'}")
    click.echo(f"Output directory: {out}")
    click.echo("")

    dataset = Dataset(geno, range, pheno)
    model = model.split(",")
    try:
        models = [get_class_from_path(m) for m in model]
    except ImportError as e:
        click.secho(f"{e}", fg="red")
        return

    # create output directory if not exists
    output_dir = os.path.join(out, "feature_importance")
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        click.secho(
            f"Output directory {output_dir} already exists. Existing files may be overwritten",
            fg="yellow",
        )
    except OSError as e:
        click.secho(
            f"Error creating output directory {output_dir}: {e}",
            fg="red",
        )
        return

    if gene not in dataset.gene.name:
        click.secho(
            f"Gene {gene} not found in dataset",
            fg="red",
        )
        return
    if trait not in dataset.trait.name:
        click.secho(
            f"Trait {trait} not found in dataset",
            fg="red",
        )
        return
    click.echo(
        f"==> Calculating feature importance for gene {gene} and trait {trait} ...",
    )
    feature_importance_df = feature_importance(gene, trait, models, dataset, onehot)
    click.echo("==> Feature importance calculated successfully")
    gene_dir = os.path.join(output_dir, gene)
    if not os.path.exists(gene_dir):
        os.mkdir(gene_dir)
    # save the feature importance dataframe
    feature_importance_df.to_csv(
        os.path.join(gene_dir, f"{gene}_{trait}_feature_importance.tsv"),
        sep="\t",
        index=True,
    )
    click.echo("==> Starting to plot feature importance ...")
    # feature importance plot
    plot_path = os.path.join(gene_dir, f"{gene}_{trait}")
    plot_feature_importance(feature_importance_df, 10, True, plot_path)
    click.echo(f"==> Feature importance plot saved to {gene_dir}")


@main.command()
@click.option(
    "-f", "--vcf", type=click.Path(exists=True), required=True, help="Path to VCF file"
)
@click.option("-o", "--out", type=click.Path(), required=True, help="Output directory")
@click.option("-p", "--prefix", type=str, help="Prefix for output files")
def vcf2plink(vcf, out, prefix):
    """Convert VCF file to plink binary format"""

    try:
        os.makedirs(out)
    except FileExistsError:
        click.secho(
            f"Output directory {out} already exists. Existing files may be overwritten",
            fg="yellow",
        )

    if not prefix:
        prefix = os.path.splitext(os.path.basename(vcf))[0]
    out_path = os.path.join(out, prefix)
    cmd = (
        f"plink --vcf {vcf}  --allow-extra-chr --make-bed --double-id --out {out_path}"
    )
    try:
        run_plink(cmd)
        click.secho(
            f"VCF file converted to plink binary format:",
            fg="green",
        )
        click.echo(f"{out_path}.bed \n{out_path}.bim \n{out_path}.fam")
    except Exception as e:
        click.secho(f"Error converting VCF file: {e}", fg="red")


@main.command()
@click.option(
    "-f", "--gff", type=click.Path(exists=True), required=True, help="Path to gff file"
)
@click.option("-r", "--region", type=str, required=True, help="The region to convert")
@click.option("-o", "--out", type=click.Path(), required=True, help="Output directory")
def gff2range(gff, region, out):
    """Convert GFF3 file to plink gene range format"""
    try:
        os.makedirs(out)
    except FileExistsError:
        click.secho(
            f"Output directory {out} already exists. Existing files may be overwritten",
            fg="yellow",
        )
    df = gff3_to_range(gff, region)
    if df is None:
        click.secho("The selected area does not exist in the file", fg="red")
        return
    prefix = os.path.splitext(os.path.basename(gff))[0]
    out_path = os.path.join(out, f"{prefix}_{region}.range")
    df.to_csv(out_path, sep="\t", header=False, index=False)
    click.secho(f"The range file is saved to {out_path}", fg="green")


@main.command()
@click.option(
    "-f", "--gtf", type=click.Path(exists=True), required=True, help="Path to gff file"
)
@click.option("-r", "--region", type=str, required=True, help="The region to convert")
@click.option("-o", "--out", type=click.Path(), required=True, help="Output directory")
def gtf2range(gtf, region, out):
    """Convert GTF file to plink gene range format"""
    try:
        os.makedirs(out)
    except FileExistsError:
        click.secho(
            f"Output directory {out} already exists. Existing files may be overwritten",
            fg="yellow",
        )
    df = gtf_to_range(gtf, region)
    if df is None:
        click.secho("The selected area does not exist in the file", fg="red")
        return
    prefix = os.path.splitext(os.path.basename(gtf))[0]
    out_path = os.path.join(out, f"{prefix}_{region}.range")
    df.to_csv(out_path, sep="\t", header=False, index=False)
    click.secho(f"The range file is saved to {out_path}", fg="green")


if __name__ == "__main__":
    main()
