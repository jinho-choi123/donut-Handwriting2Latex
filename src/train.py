import torch

torch.manual_seed(42)

from peft.peft_model import PeftModel
from transformers import BitsAndBytesConfig, DonutSwinModel, VisionEncoderDecoderModel
from peft import get_peft_model, LoraConfig
from .config import config
from lightning.pytorch.loggers import WandbLogger
from .callbacks import PushToHubCallback
from .model import DonutLightningModel
from .datasets import processor
import lightning as L
from lightning.pytorch.tuner import Tuner

PRETRAINED_REPO_ID = config.get("PRETRAINED_REPO_ID", "naver-clova-ix/donut-base")
FINETUNED_REPO_ID = config.get("FINTUNED_REPO_ID", "ball1433/Handwriting2Latex")

# define wandb logger
wandb_logger = WandbLogger(project=config.get("wandb_project"))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
lora_config = LoraConfig(
    r=config.get("lora_r", 8),
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = VisionEncoderDecoderModel.from_pretrained(
    PRETRAINED_REPO_ID, quantization_config=bnb_config
)
model.decoder.resize_token_embeddings(len(processor.tokenizer))

if config.get("load_lora", False):
    model = PeftModel.from_pretrained(model, FINETUNED_REPO_ID, is_trainable=True)
else:
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

model_module = DonutLightningModel(model, processor, config)

trainer = L.Trainer(
    accumulate_grad_batches=4,
    max_epochs=100,
    gradient_clip_val=1.0,
    num_sanity_val_steps=5,
    logger=wandb_logger,
    callbacks=[PushToHubCallback()],
)

tuner = Tuner(trainer)

tuner.scale_batch_size(model_module)

trainer.fit(model_module)
