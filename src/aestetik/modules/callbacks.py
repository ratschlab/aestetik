from lightning.pytorch.callbacks import Callback

class LossHistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and "loss" in outputs:
            self.losses.append(outputs["loss"].item())