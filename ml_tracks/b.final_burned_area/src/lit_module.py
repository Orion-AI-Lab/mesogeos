import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
from typing import Any, List
from torchmetrics import AUROC, AveragePrecision, F1Score
from torch import nn
from typing import Any, List
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUROC, AveragePrecision, F1Score
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class plUNET(pl.LightningModule):
    def __init__(
            self,
            input_vars: list = None,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            loss='dice',
            encoder_name='efficientnet-b1'
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = smp.UnetPlusPlus(encoder_name=encoder_name, in_channels=len(input_vars), classes=2)
        if loss == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        elif loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss()

        self.val_auprc = AveragePrecision(pos_label=1, num_classes=1)
        self.test_auprc = AveragePrecision(pos_label=1, num_classes=1)
        # self.dropout2d = torch.nn.Dropout2d(p=0.5)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        y = y.long()
        # x = self.dropout2d(x)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        # log val metrics
        self.val_auprc.update(preds.flatten(), targets.flatten())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu(),
                "inputs": inputs.detach().cpu()}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        self.test_auprc.update(preds.flatten(), targets.flatten())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "inputs": inputs.detach().cpu(),  "targets": targets.detach().cpu()}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": "train/loss"}
    