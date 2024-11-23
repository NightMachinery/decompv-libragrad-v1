#### * load
#: @duplicateCode/5048724010ebca014c6a5db721e78921
##
import datasets
from datasets import load_dataset


#: run `huggingface-cli login` before

# dataset = load_dataset("imagenet-1k")

# dataset = load_dataset("frgfm/imagenette")

dataset_name = "ImageNet1K-val"
dataset_ns = "mrm8488"
dataset_ordered = load_dataset(
    f"{dataset_ns}/{dataset_name}",
    # download_mode="force_redownload",
)
#: dataset_ordered has the labels ordered, at least in the first few records I checked.

dataset_ordered = dataset_ordered["train"]

dataset = dataset_ordered.shuffle(seed=43)

dataset = dataset.flatten_indices()
####
