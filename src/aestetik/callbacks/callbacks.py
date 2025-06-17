from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.core.module import LightningModule

from typing import Any, Dict, Optional, Union
class LossHistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_batch_end(self, 
                           trainer: Trainer, 
                           pl_module: LightningModule, 
                           outputs: Optional[Union[Dict[str, Any], Any]],
                           batch: Any, 
                           batch_idx: int) -> None:
        if isinstance(outputs, dict) and "loss" in outputs:
            self.losses.append(outputs["loss"].item())