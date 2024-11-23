## * Start
import os
from os import getenv
import sys
import pprint
import argparse
import re
import decompv

# import decompv.x.imagenet
from decompv.x.bootstrap import *
from decompv.x.ds.utils import *
from decompv.x.ds.main import *
from pynight.common_icecream import ic
from pynight.common_timm import (
    patch_info_from_name,
)


## * Load DecompV Attributions
names = [
    "decompv",
    "IG_s50_raw",
    # "IG_s50",
    # 'IxG',
    # 'Saliency',
    "SaliencySmooth",
]
if len(sys.argv) >= 2:
    names = sys.argv[1:]

print(f"Selected methods:\n{names}")

tds_masked_list = []

if "decompv" in names:
    names.remove("decompv")

    bias_token_p = True
    # signed_p = True
    name = "DecompV"
    sub_names = [
        "attributions_v_logits",
        "logits",
        # 'perf',
    ]

    model_patch_info = patch_info_from_name(model_name, bias_token_p=bias_token_p)
    ic(model_patch_info)

    dataset_scalar = datasets_load1(
        model_name=model_name,
        name=name,
        sub_names=sub_names,
        keep_in_memory=False,
    )
    dataset_scalar = decompv_v0_to_v0_1(dataset_scalar)
    dataset_scalar = dataset_format_set1(dataset_scalar)

    # ic(dataset_scalar)

    tds_scalar = TransformedDataset(dataset_scalar)
    tds_scalar = tds_scalar.transform(
        partial(
            transform_attributions_scalarify,
            mode="identity",
            sum_dim=(-2,),  #: sum over extra attribution sources
        ),
    )

    tds_scalar = tds_scalar.transform(
        partial(
            transform_attributions_select_label,
            dataset_indexed=dataset_indexed,
        ),
    )

    # ic(tds_scalar.preview())

    tds_masked_decompv = masker_entrypoint(
        tds_scalar=tds_scalar,
        model_patch_info=model_patch_info,
        bias_token_p=bias_token_p,
        add_random_p=False,
        # add_random_p=True,
        model=model,
    )
    # tds_masked_decompv.preview()

    tds_masked_list.append(tds_masked_decompv)
## * Load Captum Attributions
tds_masked_captum_list = []

for name in names:
    bias_token_p = False
    # signed_p = True
    sub_names = [""]

    model_patch_info = patch_info_from_name(model_name, bias_token_p=bias_token_p)
    # ic(model_patch_info)

    dataset_scalar = datasets_load1(
        model_name=model_name,
        name=name,
        sub_names=sub_names,
        keep_in_memory=False,
    )
    if not any(col.startswith("attributions_") for col in dataset_scalar.column_names):
        #: @backcompat
        ##
        dataset_scalar = dataset_scalar.rename_columns(
            {name: f"attributions_{name}"},
        )

    dataset_scalar = dataset_scalar.remove_columns(
        [c for c in dataset_scalar.column_names if c.startswith("perf_")]
    )
    #: perf data not used in these experiments

    # ic(dataset_scalar)

    tds_scalar = TransformedDataset(dataset_scalar)
    tds_scalar = tds_scalar.transform(
        partial(
            transform_attributions_scalarify,
        ),
    )

    tds_masked_current = masker_entrypoint(
        tds_scalar=tds_scalar,
        model_patch_info=model_patch_info,
        bias_token_p=bias_token_p,
        model=model,
    )

    tds_masked_captum_list.append(tds_masked_current)
## ** Combine Datasets
tds_masked = pynight.common_datasets.ConcatenatedTransformedDataset(
    tds_masked_list + tds_masked_captum_list
)

ic(len(tds_masked))


## * Compute Logits
def attr2logits_compute_global(
    *,
    name,
    transform_only_p=True,
    dataset_start=None,
    dataset_end=None,
    tds_patches: TransformedDataset = None,
    tds_masked: TransformedDataset = None,
    add_random_p=False,
    output_precision="bfloat16",
    #: @globals
    #: model,
    #: model_name,
):
    if dataset_start is None:
        dataset_start = dataset_start_global

    if dataset_end is None:
        dataset_end = dataset_end_global

    if tds_patches is None:
        tds_patches = tds_patches_imagenet

    tds_masked = tds_masked.select(range(dataset_start, dataset_end))

    bias_token_p = False
    model_patch_info = patch_info_from_name(model_name, bias_token_p=bias_token_p)

    computer = partial(
        masked_predict,
        tds_masked=tds_masked,
        output_precision=output_precision,
        transform_only_p=transform_only_p,
    )

    dataset_compute_gen(
        name=name,
        computer=computer,
        batch_size=batch_size,
        tds_patches=tds_patches,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        model=model,
        # load_from_cache_file=False,
        # **kwargs,
    )


## * Main
if __name__ == "__main__":
    attr2logits_compute_global(
        name="combinedv1",
        tds_masked=tds_masked,
    )
