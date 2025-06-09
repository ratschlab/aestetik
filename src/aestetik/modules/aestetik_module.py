import lightning as L
import torch
import logging
from torch import nn, Tensor

from aestetik.models.model import AE
from aestetik.data_modules.data_module import AESTETIKDataModule
from aestetik.loss_function import compute_loss

from typing import Tuple
from typing import List 
from typing import Optional

class AESTETIKModel(L.LightningModule):
    def __init__(self,
                 datamodule: AESTETIKDataModule,
                 grid_params: dict,
                 model_architecture_params: dict,
                 training_params: dict,
                 optimizer_params: dict, 
                 ):
        """
        Parameters
        ----------
        grid_params: dict 
            Dictionary with grid parameters. Expected keys:
                -'obsm_transcriptomics_dim: int
                    Number of transcriptomics channels in the input grid
        model_architecture_params : dict
            Dictionary with model architecture parameters. Expected keys:
                -'kernel_size': int
                    Size of the CNN kernel.
                -'latent_dim': int
                    Dimensionality of the latent space.
                -'n_ensemble_encoder': int
                    Number of ensemble models.
                -'n_ensemble_decoder': int
                    Number of ensemble decoders.
                -'p': float
                    Dropout probability.
                -'c_hid': int
                    Number of channels produced by the convolution
        training_params : dict
            Dictionary with training parameters. Expected keys:
                -'rec_alpha': float,
                    Alpha value for reconstruction.
                -'triplet_alpha': float,
                    Alpha value for triplet loss.
        optimizer_params : dict
            Dictionary with optimizer parameters. Expected keys:
                -'lr': float,
                    Learning rate.
                -'weight_decay': float,
                    Weight decay for optimizer.
        """
        super().__init__()

        self.datamodule = datamodule
        self.save_hyperparameters(ignore=["datamodule"])
        self.predict_params = {
            "num_repeats": None
        }
        self.model_built = False
        
        self._validate_params()
    

    def setup(self, 
              stage=None) -> None:
        self.weights = {
            "transcriptomics_weight": self.datamodule.loss_regularization_params["transcriptomics_weight"],
            "morphology_weight": self.datamodule.loss_regularization_params["morphology_weight"]
        }
        self.loss = {
            "triplet_loss": nn.TripletMarginLoss(),
            "rec_loss": nn.L1Loss()
        }


    def configure_model(self) -> None:
        if self.model_built:
            return 

        self.adata = self.datamodule.adata
        self.hparams["grid_params"]["morphology_dim"] = self.adata.obsm["X_st_grid"].shape[2]
        self.hparams["grid_params"]["num_input_channels"] = self.adata.obsm["X_st_grid"].shape[1]

        self.model = AE(**self.hparams["grid_params"], 
                        **self.hparams["model_architecture_params"])
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
                                                                                                                                        obsm_transcriptomics_dim=self.hparams["grid_params"]["obsm_transcriptomics_dim"],
                                                                                                                                        device=self.device,
                                                                                                                                        **self.loss,
                                                                                                                                        **self.weights,
                                                                                                                                        **self.hparams["training_params"])
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
            raise TypeError(f"self.predict_params['num_repeats'] must be an integer, got {type(self.predict_params['num_repeats'])} instead.")


        self.model.train()
        batch_spots = batch[0]
        batch_latent_space = []

        for _ in range(self.predict_params["num_repeats"]):
            out = self.model.encoder(batch_spots)
            batch_latent_space.append(out)

        batch_latent_space = torch.stack(batch_latent_space, dim=0)
        batch_latent_space = torch.mean(batch_latent_space, dim=0)
        return batch_latent_space

        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        logging.info("Configuring optimizer ...")
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     amsgrad=True,
                                     **self.hparams["optimizer_params"])
        return optimizer 


    def _validate_params(self) -> None:
        logging.info("Validate parameters in LitAESTETIKModel ...")
        required = {
            'grid_params': ['obsm_transcriptomics_dim'],
            'model_architecture_params': ['kernel_size','latent_dim',
                                          'n_ensemble_encoder','n_ensemble_decoder',
                                          'c_hid','p'],
            'training_params': ['rec_alpha','triplet_alpha'],
            'optimizer_params': ['lr','weight_decay'],
        }
        for group_name, keys in required.items():
            param_dict = getattr(self.hparams, group_name)
            for key in keys:
                if key not in param_dict:
                    raise ValueError(f"Missing required key '{key}' in '{group_name}' dictionary.")