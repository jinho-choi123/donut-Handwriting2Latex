import lightning as L 
import torch
import numpy as np 
import evaluate
import wandb
from .datasets import train_dataset, validation_dataset, train_collate_fn, test_collate_fn
from torch.utils.data import DataLoader


class DonutLightningModel(L.LightningModule):
    def __init__(self, model, processor, config):
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config

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
        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch 

        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values=pixel_values,
                labels=labels
                )

        train_loss = outputs.loss
        self.train_losses.append(train_loss.item())

        self.log("train/loss", train_loss)
        self.log("train/avg_loss", np.mean(self.train_losses))

        return train_loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, labels = batch

        generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=self.config.get("max_new_tokens", 64)
                )

        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1)+1:], skip_special_tokens=True)
        bleu_score: float = self.bleu_metric.compute(references=labels, predictions=predictions)['bleu']
        self.val_bleu_scores.append(bleu_score)
        self.log("val/bleu", bleu_score)
        self.log("val/avg_bleu", np.mean(self.val_bleu_scores))

        if self.config.get("verbose", False) and batch_idx % 50 == 0:
            columns = ["global_step", "image", "ground_truth", "prediction"]
            datas = [
                    [self.global_step, wandb.Image(pixel_values[i]), labels[i], predictions[i]] for i in range(1)
                    ]

            self.logger.log_table(key="val/samples", columns=columns, data=datas)

        return predictions

    def on_train_epoch_end(self):
        print(f"Average Training Loss in EPOCH #{self.current_epoch}: {np.mean(self.train_losses)}")
    
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
                num_workers=self.config.get("num_workers", 2)
                )
    
    def val_dataloader(self):
        return DataLoader(
                validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=test_collate_fn,
                pin_memory=True,
                num_workers=self.config.get("num_workers", 2)
                )
