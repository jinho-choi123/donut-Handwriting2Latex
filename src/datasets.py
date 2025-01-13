from transformers import AutoProcessor
from .config import config
from .inkml_parser import read_inkml_file, get_ink_sequence_token, get_ink_image
from .processor import custom_tokenizer, custom_image_processor
from torch.utils.data import Dataset
from pathlib import Path


DATA_DIR = Path.cwd() / "data"

PRETRAINED_DECODER_REPO_ID = config.get("PRETRAINED_DECODER_REPO_ID", "google/vit-base-patch16-224-in21k")
PRETRAINED_ENCODER_REPO_ID = config.get("PRETRAINED_ENCODER_REPO_ID", "google-bert/bert-base-cased")
IMG_SIZE = config.get("IMG_SIZE", 224)
TIME_SAMPLING_DELTA = config.get("INK_TIME_SAMPLING_DELTA", 30)


def train_collate_fn(batch):
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]

    labels_ids = custom_tokenizer(labels, return_tensors="pt")["input_ids"]

    pixel_values = custom_image_processor(images, return_tensors="pt")["pixel_values"]

    return pixel_values, labels_ids


def test_collate_fn(batch):
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]

    # labels_ids = custom_tokenizer(labels, return_tensors="pt")["input_ids"]
    pixel_values = custom_image_processor(images, return_tensors="pt")["pixel_values"]

    return pixel_values, labels


class MathWritingDataset(Dataset):
    def __init__(
        self, dataset_dir, data_types=["train", "symbols", "synthetic"], transform=None
    ):
        self.dataset_dir = dataset_dir
        self.types = data_types
        self.filenames = []
        self.transform = transform
        for type_ in self.types:
            filename = [
                f"{type_}/{f.name}" for f in (self.dataset_dir / type_).glob("*.inkml")
            ]
            self.filenames.extend(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # start = time.time()
        assert type(idx) == int
        target_file_path = self.dataset_dir / self.filenames[idx]
        # read inkml file
        ink = read_inkml_file(target_file_path)
        # generate ink sequence
        text_sequence = get_ink_sequence_token(ink, TIME_SAMPLING_DELTA)
        image = get_ink_image(ink, IMG_SIZE)

        if "normalizedLabel" in ink.annotations:
            label = ink.annotations["normalizedLabel"]
        else:
            label = ink.annotations["label"]

        sample = {"image": image, "text": text_sequence, "label": label}

        if self.transform:
            sample = self.transform(sample)
        return sample


train_dataset = MathWritingDataset(
    DATA_DIR, data_types=["train", "symbols", "synthetic"]
)
validation_dataset = MathWritingDataset(DATA_DIR, data_types=["test"])
