# * load
import datasets
from datasets import load_dataset


#: run `huggingface-cli login` before

#: [[https://huggingface.co/datasets/timm/oxford-iiit-pet][timm/oxford-iiit-pet · Datasets at Hugging Face]]
dataset_name = "oxford-iiit-pet"
dataset_ns = "timm"
dataset_ordered = load_dataset(
    f"{dataset_ns}/{dataset_name}",
    # download_mode="force_redownload",
)

dataset_ordered = dataset_ordered["train"]
#: ~3.68k items

dataset = dataset_ordered.shuffle(seed=43)

dataset = dataset.flatten_indices()
