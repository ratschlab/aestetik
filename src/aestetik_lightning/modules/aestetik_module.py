import lightning as L
import torch
import logging
from torch import nn, Tensor

from aestetik.model import AE
from aestetik_lightning.data_modules.data_module import AESTETIKDataModule
from aestetik.loss_function import compute_loss

from typing import Tuple
from typing import List 
from typing import Optional

class LitAESTETIKModel(L.LightningModule):
    def __init__(self,
                 datamodule: AESTETIKDataModule,
                 rec_alpha: float,
                 triplet_alpha: float,
                 kernel_size: int,
                 latent_dim: int,
                 c_hid: int,
                 lr: float,
                 p: float,
                 n_ensemble_encoder: int,
                 n_ensemble_decoder: int,
                 weight_decay: float
                 ):
        super().__init__()
        self.datamodule = datamodule
        self.save_hyperparameters(ignore=["datamodule"])
        self.model_built = False
        self.predict_params = {
            "num_repeats": None
        }

    def setup(self, 
              stage=None):
        self.weights = {
            "transcriptomics_weight": self.datamodule.loss_regularization_params["transcriptomics_weight"],
            "morphology_weight": self.datamodule.loss_regularization_params["morphology_weight"]
        }
        self.loss = {
            "triplet_loss": nn.TripletMarginLoss(),
            "rec_loss": nn.L1Loss()
        }

        self.adata = self.datamodule.adata
        self.multi_triplet_loss = self.datamodule.loss_regularization_params["multi_triplet_loss"]
        self.obsm_transcriptomics_dim = self.datamodule.adata.obsm[self.datamodule.data_cluster_params["used_obsm_transcriptomics"]].shape[1]
        self.hparams.kernel_size = self.hparams.kernel_size if self.hparams.kernel_size < self.datamodule.data_cluster_params["window_size"] else max(1, self.datamodule.data_cluster_params["window_size"] - 1)
        self.n_repeats = self.datamodule.loss_regularization_params["n_repeats"]

    def configure_model(self):
        if self.model_built:
            return 
    
        logging.info("Initializing model ...")
        self.model = AE(num_input_channels = self.adata.obsm["X_st_grid"].shape[1],
                        morphology_dim = self.adata.obsm["X_st_grid"].shape[2],
                        c_hid = self.hparams.c_hid,
                        latent_dim = self.hparams.latent_dim,
                        kernel_size = self.hparams.kernel_size,
                        p = self.hparams.p,
                        n_ensemble_encoder = self.hparams.n_ensemble_encoder,
                        n_ensemble_decoder = self.hparams.n_ensemble_decoder)
        self.model_built = True
    
    def training_step(self,
                      batch: Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
                      batch_idx: int) -> Tensor: 
        anchor, pos_transcriptomics_list, neg_transcriptomics_list, pos_morphology_list, neg_morphology_list = batch
        anchor_encode, anchor_decode = self.model(x=anchor)
        total_loss, rec_loss_transcriptomics, rec_loss_morphology, triplet_loss_transcriptomics, triplet_loss_morphology = compute_loss(model=self.model,
                                                                                                                                        anchor=anchor,
                                                                                                                                        anchor_encode=anchor_encode,
                                                                                                                                        anchor_decode=anchor_decode,
                                                                                                                                        pos_transcriptomics_list=pos_transcriptomics_list,
                                                                                                                                        neg_transcriptomics_list=neg_transcriptomics_list,
                                                                                                                                        pos_morphology_list=pos_morphology_list,
                                                                                                                                        neg_morphology_list=neg_morphology_list,
                                                                                                                                        transcriptomics_weight=self.weights["transcriptomics_weight"],
                                                                                                                                        morphology_weight=self.weights["morphology_weight"],
                                                                                                                                        triplet_loss=self.loss["triplet_loss"],
                                                                                                                                        triplet_alpha=self.hparams.triplet_alpha,
                                                                                                                                        rec_loss=self.loss["rec_loss"],
                                                                                                                                        rec_alpha=self.hparams.rec_alpha,
                                                                                                                                        obsm_transcriptomics_dim=self.obsm_transcriptomics_dim,
                                                                                                                                        device=self.device)
        self.log("L", total_loss)
        self.log("RC", rec_loss_transcriptomics)
        self.log("RI", rec_loss_morphology)
        self.log("TC", triplet_loss_transcriptomics)
        self.log("TI", triplet_loss_morphology)
        return total_loss

    def predict_step(self,
                     batch: Tuple[Tensor,...],
                     batch_idx: int,
                     dataloader_idx: Optional[int] = None,
                     ) -> Tensor:
        if self.predict_params["num_repeats"] is None:
            raise TypeError(f"self.predict_params[\"num_repeats\"] must be an integer, got {type(self.predict)} instead.") 

        self.model.train()
        batch_spots = batch[0]
        batch_latent_space = []

        for _ in range(self.predict_params["num_repeats"]):
            out = self.model.encoder(batch_spots)
            batch_latent_space.append(out)

        batch_latent_space = torch.stack(batch_latent_space, dim=0)
        batch_latent_space = torch.mean(batch_latent_space, dim=0)
        return batch_latent_space 


    def configure_optimizers(self) -> dict:
        logging.info("Configuring optimizer ...")
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.hparams.lr,
                                     amsgrad=True,
                                     weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer}