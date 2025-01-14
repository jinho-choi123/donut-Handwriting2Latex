import torch

torch.manual_seed(42)

from peft.peft_model import PeftModel
from transformers import BitsAndBytesConfig, VisionEncoderDecoderModel
from peft import get_peft_model, LoraConfig
from .config import config
from lightning.pytorch.loggers import WandbLogger
from .callbacks import PushToHubCallback
from .model import Vision_ENC_DEC_LightningModel
from .processor import custom_tokenizer
import lightning as L
from lightning.pytorch.tuner import Tuner

# PRETRAINED_DECODER_REPO_ID = config.get("PRETRAINED_DECODER_REPO_ID", "google-bert/bert-base-cased")
PRETRAINED_DECODER_REPO_ID = "google-bert/bert-base-cased"
# PRETRAINED_ENCODER_REPO_ID = config.get("PRETRAINED_ENCODER_REPO_ID", "google/vit-base-patch16-224-in21k")
PRETRAINED_ENCODER_REPO_ID="google/vit-base-patch16-224-in21k"
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
    lora_dropout=0.1,
    target_modules=[
        'query',
        'key',
        'value',
        'intermediate.dense',
        'output.dense',
        'wte',
        'wpe',
        'c_attn',
        'c_proj',
        'q_attn',
        'c_fc'
    ],
)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    PRETRAINED_ENCODER_REPO_ID, PRETRAINED_DECODER_REPO_ID, quantization_config=bnb_config
)


model.config.decoder_start_token_id = custom_tokenizer.cls_token_id
model.config.vocab_size = len(custom_tokenizer)
model.generation_config.decoder_start_token_id = custom_tokenizer.cls_token_id
model.config.pad_token_id = custom_tokenizer.pad_token_id

model.decoder.resize_token_embeddings(len(custom_tokenizer))

if config.get("load_lora", False):
    model = PeftModel.from_pretrained(model, FINETUNED_REPO_ID, is_trainable=True)
else:
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

tmp_pixel_values = torch.rand((3, 3, 224, 224))
tmp_labels_ids = torch.randint(0, 100, (3, 64))

# test forward pass
tmp_outputs = model(pixel_values=tmp_pixel_values, labels=tmp_labels_ids)
