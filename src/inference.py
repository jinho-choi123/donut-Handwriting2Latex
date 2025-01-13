from datasets import PRETRAINED_REPO_ID, TIME_SAMPLING_DELTA
from .config import config

from transformers import DonutSwinModel
from peft import PeftModel
import torch
from PIL import Image
from .inkml_parser import read_inkml_file, get_ink_sequence_token, get_ink_image
from .datasets import processor

PRETRAINED_REPO_ID = config.get("PRETRAINED_REPO_ID", "naver-clova-ix/donut-base")
FINETUNED_REPO_ID = config.get("FINETUNED_REPO_ID", "ball1433/Handwriting2Latex")
TIME_SAMPLING_DELTA = config.get("INK_TIME_SAMPLING_DELTA", 30)
IMG_SIZE = config.get("IMG_SIZE", 224)

# define device
device = torch.device("cuda")

INKML_FILE_PATH = "examples/example.inkml"
ink = read_inkml_file(INKML_FILE_PATH)
# generate ink sequence
text_sequence = [get_ink_sequence_token(ink, TIME_SAMPLING_DELTA)]
image = get_ink_image(ink, IMG_SIZE)
images = [Image.fromarray(image)]

if "normalizedLabel" in ink.annotations:
    label = ink.annotations["normalizedLabel"]
else:
    label = ink.annotations["label"]

model = DonutSwinModel.from_pretrained(PRETRAINED_REPO_ID)
model.decoder.resize_token_embeddings(len(processor.tokenizer))

model = PeftModel.from_pretrained(model, FINETUNED_REPO_ID, is_trainable=False)

print("Loaded peft adapter for DonutSwinModel")

model.to(device)

with torch.no_grad():
    inference_inputs = processor(text=text_sequence, images=images, return_tensors="pt")
    input_ids = inference_inputs["input_ids"].to(device)
    attention_mask = inference_inputs["attention_mask"].to(device)
    pixel_values = inference_inputs["pixel_values"].to(device)

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=config.get("max_new_tokens"),
    )

    predictions = processor.batch_decode(
        generated_ids[:, input_ids.size(1) + 1 :], skip_special_tokens=True
    )

    images[0].show()
    print(f"Prompt: {text_sequence[0]}")
    print(f"Prediction LateX: {predictions[0]}")
    print(f"Ground Truth LateX: {label}")
