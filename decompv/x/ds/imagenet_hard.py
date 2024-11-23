#### * load
#: @duplicateCode/5048724010ebca014c6a5db721e78921
##
import datasets
from datasets import load_dataset


#: run `huggingface-cli login` before

dataset_name = "imagenet-hard"
dataset_ns = "taesiri"
dataset_ordered = load_dataset(
    f"{dataset_ns}/{dataset_name}",
    # download_mode="force_redownload",
)
#: dataset_ordered has the labels ordered, at least in the first few records I checked.

dataset_ordered = dataset_ordered["validation"]

dataset = dataset_ordered.shuffle(seed=43)


#: Flatten the 'label' and 'english_label' columns
#: Some images have multiple labels in the dataset! We choose only the first.
def flatten_labels(batch):
    flattened_labels = []
    flattened_english_labels = []

    for idx, labels in enumerate(batch["label"]):
        if isinstance(labels, list) and len(labels) > 0:
            flattened_labels.append(labels[0])
        else:
            raise ValueError(f"Invalid label format at index {idx}: {labels}")

    for idx, english_labels in enumerate(batch["english_label"]):
        if isinstance(english_labels, list) and len(english_labels) > 0:
            flattened_english_labels.append(english_labels[0])
        else:
            raise ValueError(
                f"Invalid english_label format at index {idx}: {english_labels}"
            )

    batch["label"] = flattened_labels
    batch["english_label"] = flattened_english_labels
    return batch


dataset = dataset.map(
    flatten_labels,
    batched=True,
)


dataset = dataset.flatten_indices()
####
