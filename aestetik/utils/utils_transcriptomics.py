import scanpy as sc
import numpy as np


def preprocess_adata(
        adata,
        min_cells_filter=10,
        normalize_target_sum=1e4,
        pca_n_comps=15,
        variances_norm_cutoff=1,
        n_top_genes=None,
        use_highly_variable=True):
    """
    Preprocess the anndata object.

    Parameters
    ----------
    adata : anndata
        Raw anndata object to preprocess.
    min_cells_filter : int, optional (default=10)
        Minimum number of cells expressed required for a gene to pass filtering.
    normalize_target_sum : float, optional (default=1e4)
        Target sum for normalization.
    pca_n_comps : int, optional (default=15)
        Number of principal components to compute during PCA.
    variances_norm_cutoff : float, optional (default=1)
        Cutoff for variance normalization.
    n_top_genes : int, optional
        Number of top genes to select based on variability. If None, no selection is applied.
    use_highly_variable : bool, optional (default=True)
        Whether to use highly variable genes during preprocessing.

    Returns
    -------
    adata : anndata
        The preprocessed annotated data object.
    """

    adata = adata.copy()  # make sure not to work on the reference
    adata.raw = adata  # keep raw data

    n_top_genes = n_top_genes if n_top_genes else 1e6

    sc.pp.filter_genes(adata, min_cells=min_cells_filter)
    # @https://github.com/scverse/scanpy/blob/ed3b277b2f498e3cab04c9416aaddf97eec8c3e2/scanpy/preprocessing/_highly_variable_genes.py#L112C9-L112C22
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    # select high variable genes
    adata = adata[:, adata.var.variances_norm >= variances_norm_cutoff]

    sc.pp.normalize_total(adata, target_sum=normalize_target_sum)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=pca_n_comps, use_highly_variable=use_highly_variable)

    return adata
