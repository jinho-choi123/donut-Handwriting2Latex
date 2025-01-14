import lightning as L
import torch
import numpy as np
import evaluate
import wandb
from .datasets import (
        train_dataset,
        validation_dataset,
        train_collate_fn,
        test_collate_fn,
        )
from torch.utils.data import DataLoader

class Vision_ENC_DEC_LightningModel(L.LightningModule):
    def __init__(self, model, tokenizer, config):
        super().__init__()
        self.model = model
        self.config = config
        self.tokenizer = tokenizer 

        self.lr = self.config.get("INIT_LR", 1e-4)
        self.batch_size = self.config.get("batch_size", 16)

        self.bleu_metric = evaluate.load("bleu", keep_in_memory=True)

        self.train_losses = []
        self.val_bleu_scores = []

    def on_train_epoch_start(self):
        self.train_losses = []

    def on_validation_epoch_start(self):
        self.val_bleu_scores = []

    def training_step(self, batch, batch_idx):
        pixel_values, labels_ids = batch

        print(f"pixel_values: {pixel_values}")
        print(f"labels_ids: {labels_ids}")

        outputs = self.model(pixel_values=pixel_values, labels=labels_ids)

        train_loss = outputs.loss
        self.train_losses.append(train_loss.item())

        self.log("train/loss", train_loss)
        self.log("train/avg_loss", np.mean(self.train_losses))

        return train_loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch

        generated_ids = self.model.generate(pixel_values)

        predictions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        bleu_score: float = self.bleu_metric.compute(
            references=labels, predictions=predictions
        )["bleu"]

        self.val_bleu_scores.append(bleu_score)
        self.log("val/bleu", bleu_score)
        self.log("val/avg_bleu", np.mean(self.val_bleu_scores))

        if self.config.get("verbose", False) and batch_idx % 50 == 0:
            columns = ["global_step", "image", "ground_truth", "prediction"]
            datas = [
                [
                    self.global_step,
                    wandb.Image(pixel_values[i]),
                    labels[i],
                    predictions[i],
                ]
                for i in range(1)
            ]

            self.logger.log_table(key="val/samples", columns=columns, data=datas)

        return predictions

    def on_train_epoch_end(self):
        print(
            f"Average Training Loss in EPOCH #{self.current_epoch}: {np.mean(self.train_losses)}"
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=train_collate_fn,
            pin_memory=True,
            num_workers=self.config.get("num_workers", 2),
        )

    def val_dataloader(self):
        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=test_collate_fn,
            pin_memory=True,
            num_workers=self.config.get("num_workers", 2),
        )
