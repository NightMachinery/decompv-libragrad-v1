from decompv.x.bootstrap import *
from decompv.x.ds.utils import *
from pynight.common_datasets import (
    mapconcat,
    dataset_index_add,
)


def dataset_indexed_save(dataset, *, dataset_name=None, dest="auto"):
    if dest == "auto":
        assert dataset_name

        name = f"{dataset_name}_indexed"
        dest = f"{DS_ROOT}/{name}"

    dataset_filtered = dataset.filter(dsmap_input_filter, batched=True, num_proc=64)

    dataset_indexed = dataset_index_add(dataset_filtered)
    # ic(dataset_indexed)

    dataset_indexed = dataset_indexed.flatten_indices()
    dataset_indexed.save_to_disk(dest)
    print(f"Dataset saved to: {dest}")

    return dataset_indexed
