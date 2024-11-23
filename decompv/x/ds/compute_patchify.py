#!/usr/bin/env python3
##
import sys
from decompv.x.ds.main import *
from decompv.x.ds.main import batch_size_inference
from pynight.common_icecream import ic


def patchify(batch, batch_transformed):
    # ic(torch_shape_get(batch_transformed), type_only_p=True)

    new_batch = dict()

    new_batch["patches"] = batch_transformed["patches_dv"].cpu()
    # new_batch['label_natural'] = batch_transformed['label_natural']

    return new_batch


tds_patches = tds_patches_lazy_imagenet

if DATASET_NAME == "ImageNetVal":
    tds_patches = tds_patches.select(range(0, 6000))
    #: We don't need the other images.

elif DATASET_NAME == "ImageNet-Hard":
    tds_patches = tds_patches.select(range(0, 2000))
    #: We don't need the other images.

elif DATASET_NAME in [
    "MURA",
    "oxford_pet",
]:
    tds_patches = tds_patches.select(range(0, 2000))

else:
    raise ValueError(f"DATASET_NAME={DATASET_NAME} is not supported.")


if os.path.exists(dataset_patchified_path):
    msg = (
        f"compute_patchify: dataset already exists, aborting: {dataset_patchified_path}"
    )
    print(msg, file=sys.stderr)
    sys.exit(1)

dataset_patchified = mapconcat(
    tds_patches.dataset,
    tds_patches.fn_with_transforms(
        patchify,
    ),
    unchanged_keep_columns=[
        "id",
        "label",
    ],
    batched=True,
    batch_size=ic(batch_size_inference),
    load_from_cache_file=False,
)

dataset_path = f"{DS_PATCHIFIED_PATH}/{model_name}"

save_res = save_and_delete(
    dataset_patchified,
    dataset_path,
    delete_p=True,
    # max_shard_size=999999999999,
)
ic(save_res)
