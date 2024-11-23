# * load
import datasets
from datasets import load_dataset


#: run `huggingface-cli login` before

dataset_name = "MURA"
dataset_ns = "KhalfounMehdi"
dataset_ordered = load_dataset(
    f"{dataset_ns}/{dataset_name}",
    # download_mode="force_redownload",
)

dataset_ordered = dataset_ordered["train"]

dataset = dataset_ordered.shuffle(seed=43)

dataset = dataset.flatten_indices()
