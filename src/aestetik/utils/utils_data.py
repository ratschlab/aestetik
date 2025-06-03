import logging
import numpy as np
import anndata

from typing import Literal
from typing import Union
from typing import Tuple

from aestetik.utils.utils_clustering import clustering
from aestetik.utils.utils_grid import create_st_grid


def prepare_input_for_model(
        adata: anndata.AnnData,
        nCluster: Union[int, float],
        used_obsm_transcriptomics: str,
        used_obsm_morphology: str,
        used_obsm_combined: str,
        clustering_method: Literal["bgm", "kmeans", "louvain", "leiden"],
        n_neighbors: int,
        window_size: int,
        n_jobs: int,
        total_weight: float,
        morphology_weight: float) -> Tuple[float, float]:
        """
        Prepare the input for training the model.
        1. Clustering raw input
        2. Grid building
        """
        logging.info(f"Clustering raw input with {clustering_method}...")

        clustering(
            adata=adata,
            num_cluster=nCluster,
            used_obsm=used_obsm_transcriptomics,
            method=clustering_method,
            n_neighbors=n_neighbors,
            refine_cluster=0)

        clustering(
            adata=adata,
            num_cluster=nCluster,
            used_obsm=used_obsm_morphology,
            method=clustering_method,
            n_neighbors=n_neighbors,
            refine_cluster=0)

        transcriptomics_weight, morphology_weight = calibrate_transcriptomics_morphology_ratio(
                                                        adata=adata,
                                                        nCluster=nCluster, 
                                                        used_obsm_transcriptomics=used_obsm_transcriptomics,
                                                        used_obsm_morphology=used_obsm_morphology,
                                                        total_weight=total_weight, 
                                                        morphology_weight=morphology_weight
                                                    )
        
        adata.obsm[used_obsm_combined] = np.concatenate(
                (adata.obsm[used_obsm_transcriptomics], adata.obsm[used_obsm_morphology]), axis=1)
        
        logging.info("Computing transcriptomics grid...")
        X_st_grid_transcriptomics = create_st_grid(
            adata, used_obsm=used_obsm_transcriptomics, window_size=window_size, cpu_count=n_jobs)
        
        logging.info("Computing morphology grid...")
        X_st_grid_morphology = create_st_grid(
            adata, used_obsm=used_obsm_morphology, window_size=window_size, cpu_count=n_jobs)

        adata.obsm["X_st_grid"] = np.concatenate(
            (X_st_grid_transcriptomics, X_st_grid_morphology), axis=1)

        return transcriptomics_weight, morphology_weight

def calibrate_transcriptomics_morphology_ratio(
    adata: anndata.AnnData, 
    nCluster: Union[int, float], 
    used_obsm_transcriptomics: str, 
    used_obsm_morphology: str, 
    total_weight: float, 
    morphology_weight: float) -> Tuple[float, float]:
    transcriptomics_weight = total_weight - morphology_weight

    if (transcriptomics_weight > 0 and
        adata.obs[f"{used_obsm_transcriptomics}_cluster"].unique().size == 1):
        logging.info(
            f"obsm {used_obsm_transcriptomics} resulted in 1 cluster instead of {nCluster}. transcriptomics_weight will be set to 0.")
        transcriptomics_weight = 0

    if (morphology_weight > 0 and
        adata.obs[f"{used_obsm_morphology}_cluster"].unique().size == 1):
        logging.info(
            f"obsm {used_obsm_morphology} resulted in 1 cluster instead of {nCluster}. morphology_weight will be set to 0.")
        morphology_weight = 0

    return transcriptomics_weight, morphology_weight