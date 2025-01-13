from transformers import AutoProcessor
from .config import config
from .tokenizer import custom_tokenizer
from .inkml_parser import read_inkml_file, get_ink_sequence_token, get_ink_image
from torch.utils.data import Dataset


PRETRAINED_REPO_ID = config.get("PRETRAINED_REPO_ID", "naver-clova-ix/donut-base")
IMG_SIZE = config.get("IMG_SIZE", 224)
TIME_SAMPLING_DELTA = config.get("INK_TIME_SAMPLING_DELTA", 30)

processor = AutoProcessor.from_pretrained(PRETRAINED_REPO_ID)

# change the default tokenizer into custom tokenizer
processor.tokenizer = custom_tokenizer


def train_collate_fn(batch):
    images = [item["image"] for item in batch]
    text_sequences = ["<image>" + item["text"] for item in batch]
    labels = [item["label"] for item in batch]

    inputs = processor(
        text=text_sequences,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding=True,
        truncation="only_second",
        max_length=config.get("max_length"),
    )

    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    labels = inputs["labels"]

    assert input_ids.size(1) == labels.size(1)

    assert pixel_values.size() == (len(images), 3, IMG_SIZE, IMG_SIZE)

    return input_ids, token_type_ids, attention_mask, pixel_values, labels


def test_collate_fn(batch):
    images = [item["image"] for item in batch]
    text_sequences = ["<image>" + item["text"] for item in batch]
    labels = [item["label"] for item in batch]

    inputs = processor(
        text=text_sequences,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation="only_second",
        max_length=config.get("max_length"),
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    return input_ids, attention_mask, pixel_values, labels


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
    "data/", data_types=["train", "symbols", "synthetic"]
)
validation_dataset = MathWritingDataset("data/", data_types=["test"])
