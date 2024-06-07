from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import plotnine as p9
from tqdm import tqdm
import pandas as pd
import scanpy as sc
import numpy as np


def find_optimal_n_clusters(adata, start=2, end=10, suggested_n=None):
    """
    Find the optimal number of clusters using different methods.

    Parameters
    ----------
    adata : anndata
        Anndata object with transcripromics stored in X_pca_transcriptomics_cluster and morphology in X_pca_morphology
    start : int, optional (default=2)
        The starting number of clusters to consider.
    end : int, optional (default=10)
        The ending number of clusters to consider.
    suggested_n : int or None, optional (default=None)
        A suggested number of clusters to consider, which can be used as a reference.
    """
    scores = []
    for n_clusters in tqdm(range(start, end)):

        km = KMeans(n_clusters=n_clusters, n_init=10)
        adata.obs["X_pca_transcriptomics_cluster"] = km.fit_predict(adata.obsm["X_pca_transcriptomics"])
        km_transcriptomics = km.inertia_
        sc_transcriptomics = silhouette_score(
            adata.obsm["X_pca_transcriptomics"],
            adata.obs.X_pca_transcriptomics_cluster)

        km = KMeans(n_clusters=n_clusters, n_init=10)
        adata.obs["X_pca_morphology_cluster"] = km.fit_predict(adata.obsm["X_pca_morphology"])
        km_morphology = km.inertia_
        sc_morphology = silhouette_score(adata.obsm["X_pca_morphology"], adata.obs.X_pca_morphology_cluster)

        scores.append([n_clusters, sc_transcriptomics, sc_morphology, km_transcriptomics, km_morphology])

    scores = pd.DataFrame(
        scores,
        columns=[
            "n_clusters",
            "sc_transcriptomics",
            "sc_morphology",
            "km_transcriptomics",
            "km_morphology"])

    # normalise SSE
    scores["km_transcriptomics"] = (scores["km_transcriptomics"] - scores["km_transcriptomics"].min()) / \
        (scores["km_transcriptomics"].max() - scores["km_transcriptomics"].min())
    scores["km_morphology"] = (scores["km_morphology"] - scores["km_morphology"].min()) / \
        (scores["km_morphology"].max() - scores["km_morphology"].min())

    tab = scores.melt(id_vars="n_clusters")
    tab["axis"] = tab.variable.apply(lambda x: "silhouette_score" if "sc_" in x else "SSE")
    tab["variable"] = tab.variable.apply(lambda x: "transcriptomics" if "transcriptomics" in x else "morphology")
    tab[""] = "suggested_n"
    p = (p9.ggplot(tab, p9.aes("n_clusters", "value", color="variable"))
         + p9.geom_line()
         + p9.facet_wrap("~axis", scales="free_y", nrow=2)
         + p9.scale_x_continuous(breaks=range(tab.n_clusters.min(), tab.n_clusters.max() + 1))
         + p9.theme_bw()
         )
    if suggested_n:
        p = (p
             + p9.geom_vline(p9.aes(xintercept=suggested_n, linetype=""), color="b", alpha=0.3)
             #             + p9.geom_text(p9.aes(x=suggested_n + 0.1, y=0.5), label = 'small', color = '#117DCF', size=10)
             )
    print(p)


def search_res(
        adata,
        n_clusters,
        method='leiden',
        use_rep='emb',
        start=0.01,
        end=2.0,
        increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=False):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(
                pd.DataFrame(
                    adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(
                pd.DataFrame(
                    adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))

        if count_unique == n_clusters:
            break

    return res


def clustering(
        adata,
        num_cluster,
        used_obsm,
        method,
        refine_cluster=False,
        n_neighbors=15,
        conf_proba=0.9,
        start=0.1,
        end=3,
        increment=0.02):

    if method == 'leiden':
        if isinstance(num_cluster, int):
            res = search_res(
                adata,
                num_cluster,
                use_rep=used_obsm,
                method=method,
                start=start,
                end=end,
                increment=increment)
        else:
            res = num_cluster
        sc.pp.neighbors(adata, n_neighbors=50, use_rep=used_obsm)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[f"{used_obsm}_cluster"] = adata.obs['leiden']
    elif method == 'louvain':
        if isinstance(num_cluster, int):
            res = search_res(
                adata,
                num_cluster,
                use_rep=used_obsm,
                method=method,
                start=start,
                end=end,
                increment=increment)
        else:
            res = num_cluster
        sc.pp.neighbors(adata, n_neighbors=50, use_rep=used_obsm)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs[f"{used_obsm}_cluster"] = adata.obs['louvain']
    elif method == "kmeans":
        clusters = KMeans(
            n_clusters=num_cluster, n_init=100).fit(
            adata.obsm[used_obsm]).labels_
        adata.obs[f"{used_obsm}_cluster"] = clusters.astype(str)
    elif method == "bgm":
        bgm = BayesianGaussianMixture(
            n_components=num_cluster,
            init_params="random",
            covariance_type="tied",
            max_iter=1000,
            n_init=10,
            random_state=0
        ).fit(
            adata.obsm[used_obsm])
        clusters = bgm.predict(adata.obsm[used_obsm])
        adata.obs[f"{used_obsm}_cluster"] = clusters.astype(str)
        adata.obs[f"{used_obsm}_cluster_proba"] = bgm.predict_proba(
            adata.obsm[used_obsm]).max(axis=1)

    if refine_cluster:
        if f"{used_obsm}_cluster_proba" in adata.obs:
            high_conf = adata.obs[f"{used_obsm}_cluster_proba"] > conf_proba
            X = adata.obs[["x_array", "y_array"]].values[high_conf]
            y = adata.obs[f"{used_obsm}_cluster"].values[high_conf]
        else:
            X = adata.obs[["x_array", "y_array"]].values
            y = adata.obs[f"{used_obsm}_cluster"].values

        # make sure there are multiple clusters,
        # otherwise refinement doesn't make sense
        if np.unique(y).size > 1:
            neigh = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights="uniform",
                algorithm="brute")
            neigh.fit(X,
                      y)

            refined_clusters = neigh.predict(
                adata.obs[["x_array", "y_array"]].values)
            adata.obs[f"{used_obsm}_cluster"] = refined_clusters.astype(
                str)
