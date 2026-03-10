import click
import os

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
    is_flag=True,
    default=True,
    help="Use adjusted p-value for significance threshold (default: True)",
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

    # echo the parameters
    click.echo("\n" + "=" * 40)
    click.secho("     mlQTL Analysis Parameters     ", fg="green", bold=True)
    click.echo("=" * 40)
    click.secho(f"{'Genotype file:':<25} {geno}", fg="cyan")
    click.secho(f"{'Phenotype file:':<25} {pheno}", fg="cyan")
    click.secho(f"{'Gene range file:':<25} {range}", fg="cyan")
    click.secho(f"{'Output directory:':<25} {out}", fg="cyan")
    click.secho(f"{'Number of processes:':<25} {jobs}", fg="cyan")
    click.secho(f"{'Model(s):':<25} {model}", fg="cyan")
    click.secho(
        f"{'Sliding window method:':<25} Symmetric neighborhood (±{center_window_kb}kb around each gene)",
        fg="cyan",
    )
    click.secho(f"{'Sliding window step:':<25} {center_step_genes} genes", fg="cyan")
    click.secho(f"{'Window score quantile (q):':<25} {q}", fg="cyan")
    click.secho(
        f"{'QTL definition: top windows genome-wide':<25} {top_prop}", fg="cyan"
    )
    click.secho(
        f"{'Chromosome:':<25} {chrom if chrom else 'all chromosomes'}", fg="cyan"
    )
    click.secho(f"{'Trait:':<25} {trait if trait else 'all traits'}", fg="cyan")
    click.secho(
        f"{'One-hot encoding:':<25} {'enabled' if onehot else 'disabled'}", fg="cyan"
    )
    click.secho(
        f"{'Use adjusted p-value:':<25} {'enabled' if padj else 'disabled'}", fg="cyan"
    )
    click.echo("=" * 40 + "\n")

    try:
        dataset = Dataset(geno, range, pheno)
    except Exception as e:
        click.secho(f"ERROR: {e}", fg="red")
        return
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
    output_dir = os.path.join(out)
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
    click.echo("==> Starting Analysis ...")
    for trait in analysis_trait:
        click.echo(f"==> Analyzing Trait: {trait}")
        click.echo("==> Training Model ...")
        train_res = train_with_progressbar(trait, models, dataset, max_workers, onehot)
        click.echo("==> Processing Training Result ...")
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
            click.secho(f"ERROR: {e}", fg="red")
            return

        if sig_genes.empty:
            click.secho(
                "==> No significant genes found for this trait",
                fg="yellow",
            )

        trait_dir = os.path.join(out, f"{trait}")
        os.mkdir(trait_dir) if not os.path.exists(trait_dir) else None
        # plot and save
        plot_path = os.path.join(trait_dir, "sliding_window")
        plot_graph(sw_res, 10 ** (-window_threshold), plot_path, save=True)
        click.echo(f"==> Result Graph [{plot_path}.png]")
        # save the sliding window result
        df_path = os.path.join(trait_dir, "significant_genes.tsv")
        sig_genes.to_csv(
            df_path,
            sep="\t",
            header=True,
            index=False,
        )
        click.echo(f"==> Significant Genes Table [{df_path}]")
        qtl_regions = build_qtl_regions_from_high_windows(
            sw_res,
            train_res_processed,
            window_threshold,
        )
        if not qtl_regions.empty:
            qtl_path = os.path.join(trait_dir, "qtl_regions.tsv")
            qtl_regions.to_csv(qtl_path, sep="\t", header=True, index=False)
            click.echo(f"==> QTL Regions Table [{qtl_path}]")

        # save the original training result
        # pkl_path = os.path.join(trait_dir, "train_res.pkl")
        # with open(pkl_path, "wb") as f:
        #     pickle.dump(train_res, f)
        # click.echo(f"==> Training Result Pkl [{pkl_path}]")
        # save the training result as dataframe
        train_res_processed.to_csv(
            os.path.join(trait_dir, "train_res.tsv"),
            sep="\t",
            index=False,
            header=True,
        )
        click.echo(
            f"==> Training Result Table [{os.path.join(trait_dir, 'train_res.tsv')}]"
        )
    click.secho("Analysis completed", fg="green")


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
        click.secho(f"ERROR: {e}", fg="red")
        return
    model = model.split(",")
    try:
        models = [get_class_from_path(m) for m in model]
    except ImportError as e:
        click.secho(f"{e}", fg="red")
        return

    # create output directory if not exists
    output_dir = os.path.join(out)
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

    from .utils import gtf_to_range

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
