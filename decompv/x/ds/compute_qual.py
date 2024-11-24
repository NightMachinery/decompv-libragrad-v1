#!/usr/bin/env python3
## * Imports
from os import getenv
from IPython import embed
from decompv.x.ds.compute_attention import (
    attn_transforms_get,
    m6_attn_transforms_get,
    attn_prepare_attribution_columns,
)

##
import os
import pprint
import argparse
import re
from functools import partial
import decompv

from decompv.x.bootstrap import *
import decompv.x.ds.utils
from decompv.x.ds.utils import *
from decompv.x.ds.main import *
from decompv.x.ds.clip_datasets import *
from pynight.common_icecream import ic
from pynight.common_iterable import (
    IndexableList,
)


# raise Exception("Testing exceptions")
##
#: These imports must be last, as we have a faulty simple_obj_update implementation somewhere in the above imports.
from pynight.common_dict import (
    simple_obj,
    simple_obj_update,
)


##
###
def transformed_dataset_constructor_wrapper(ds_dict):
    ds = TransformedDataset(ds_dict)
    return ds


def transform_inputs(
    ds,
    *,
    model,
    device=None,
    grad_p=True,
    prepare_input_p=True,
):
    if prepare_input_p is True:
        tds_torch_cpu = ds.transform(
            partial(
                transform_input_prepare,
                model=model,
                device=device,
                accept_gray_p=True,
            )
        )
    else:
        tds_torch_cpu = ds

    tds_patches = tds_torch_cpu.transform(
        partial(
            transform_pixels2patches,
            model=model,
            grad_p=grad_p,
        )
    )
    return tds_torch_cpu, tds_patches


def compute_attributions(tds_patches, model):
    attn_transforms = m6_attn_transforms_get(
        channel_mixers=[
            "sum",
            "RS",
        ],
    )

    my_attn_prepare_attribution_columns = partial(
        attn_prepare_attribution_columns,
        exclude_images_p=False,
    )

    tds_attn = tds_patches.transform(
        partial(
            dsmap_attn,
            model=model,
            after_transforms=attn_transforms,
            attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
            remove_cols=None,
        )
    )
    return tds_attn


##
def clip_tds_input_get(ds):
    ds_dict = dict(
        id=[],
        image_id=[],
        image=[],
        clip_text=[],
    )

    i = 0
    for img_id, img_obj in ds.items():
        #: For debugging, uncomment the following lines
        # if img_id not in ["giraffe_woman_1"]:
        #     continue

        for text in img_obj["texts"]:
            ds_dict["id"].append(i)
            ds_dict["image_id"].append(img_id)
            ds_dict["image"].append(img_obj["url"])
            ds_dict["clip_text"].append(text)
            i += 1

    ds_dict = BatchedDict(ds_dict)
    return ds_dict


def clip_tds_attn_get(
    *,
    ds,
    model,
):
    ds_dict = clip_tds_input_get(ds)
    ds_transformed = transformed_dataset_constructor_wrapper(ds_dict)
    tds_torch_cpu, tds_patches = transform_inputs(
        ds_transformed,
        model=model,
        device=None,
        grad_p=True,
    )
    tds_attn = compute_attributions(tds_patches, model)

    ds_obj = simple_obj(
        ds_dict=ds_dict,
        tds_indexed=ds_transformed,
        tds_torch_cpu=tds_torch_cpu,
        tds_patches=tds_patches,
        tds_attn=tds_attn,
    )
    return ds_obj


##
def imagenet_s_tds_input_get():
    #: @duplicateCode/66c312bd705e550866f1d4bfe32f590d
    ##

    from decompv.x.imagenet_s import (
        imagenet_s_all_get,
    )

    imagenet_s_all = imagenet_s_all_get()
    ds_dict = imagenet_s_all
    return ds_dict


def imagenet_s_tds_attn_get(
    *,
    model,
    ds=None,  #: dummy, to conform to interface
):
    from decompv.x.ds.seg import (
        imagenet_s_load,
        class_colors,
    )

    ds_dict = imagenet_s_tds_input_get()
    ds_transformed = transformed_dataset_constructor_wrapper(ds_dict)
    ds_transformed = ds_transformed.transform(
        partial(
            imagenet_s_load,
            model_transforms=model_transform_get(model),
            class_colors=class_colors,
            device=None,
            model=model,
        )
    )
    tds_torch_cpu, tds_patches = transform_inputs(
        ds_transformed,
        model=model,
        device=None,
        grad_p=True,
        prepare_input_p=False,
    )
    tds_attn = compute_attributions(tds_patches, model)

    ds_obj = simple_obj(
        ds_dict=ds_dict,
        tds_indexed=ds_transformed,
        tds_torch_cpu=tds_torch_cpu,
        tds_patches=tds_patches,
        tds_attn=tds_attn,
    )
    return ds_obj


##
def zele_tds_input_get():
    #: @duplicateCode/5fd640b06397b1b38b020a1194312f08
    ##
    from decompv.x.ds.coco_utils import coco_image_get_by_class

    images = coco_image_get_by_class(
        class_names=[
            "zebra",
            "elephant",
        ],
        # limit_n=200,
    )
    ic(type(images), len(images))

    # ds_dict = IndexableList(images)
    ds_dict = IndexableList(list(enumerate(images)))
    return ds_dict


def zele_tds_attn_get(
    *,
    model,
    ds=None,  #: dummy, to conform to interface
):
    from decompv.x.ds.coco_utils import (
        transform_input_prepare_coco,
    )

    ds_dict = zele_tds_input_get()
    ds_transformed = transformed_dataset_constructor_wrapper(ds_dict)
    ds_transformed = ds_transformed.transform(
        partial(
            transform_input_prepare_coco,
            model=model,
            mode="Narnia2",
        )
    )
    tds_torch_cpu, tds_patches = transform_inputs(
        ds_transformed,
        model=model,
        device=None,
        grad_p=True,
        prepare_input_p=False,
    )
    tds_attn = compute_attributions(tds_patches, model)

    ds_obj = simple_obj(
        ds_dict=ds_dict,
        tds_indexed=ds_transformed,
        tds_torch_cpu=tds_torch_cpu,
        tds_patches=tds_patches,
        tds_attn=tds_attn,
    )
    return ds_obj


##
def compute_qual(
    ds,
    *,
    dataset_name,
    batch_size=None,
    model,
    model_patch_info=None,
    tds_attn_get_fn,
    # normalize=["relu", "scale_by_max_signed_attr"],
    # normalize="auto",
    normalize=None,
    overlay_alone_p=False,
    batch_from=0,
    batch_to=9999,
    title=None,
    extra_title="",
    extra_tqdm_name="",
    export_dir_suffix="_v7",
    device=None,
    plot_output_p=False,
    attributions_cols=None,
    **kwargs,
):
    model_name = model_name_get(model)

    if batch_size is None:
        batch_size = global_batch_size

    if model_patch_info is None:
        model_patch_info = model.model_patch_info

    my_model_patch_info = model_patch_info

    ic(model_patch_info)

    blocks_len = len(model.blocks)
    ic(blocks_len)

    # Get the ds_obj using the provided tds_attn_get_fn
    ds_obj = tds_attn_get_fn(
        ds=ds,
        model=model,
    )
    my_tds_indexed = ds_obj.tds_indexed
    my_tds_torch_cpu = ds_obj.tds_torch_cpu
    my_tds_patches = ds_obj.tds_patches
    tds_attn = ds_obj.tds_attn
    ##
    attributions_col_patterns = []
    if attributions_cols is None:
        attributions_cols = []

        attributions_col_patterns += ["Image.*_s:(?:sum|RS|L2)"]
        # attributions_col_patterns += ['^attributions_s_.*']

        # attributions_cols += [
        #     f"attributions_s_blocks__{i}__MeanAttn"
        #     for i in range(0, 12)
        # ]
        # attributions_cols += [
        #     # f"attributions_s_blocks__{i}__CAT_s:sum"
        #     f"attributions_s_blocks__{i}__FGrad_s:sum"
        #     for i in range(0, blocks_len)
        # ]

        print(f"Qual Submode: {qual_submode}", file=sys.stderr)
        if qual_submode == "q_full_1":
            attributions_cols += [
                "attributions_s_rnd1",  #: random baseline
                ##
                "attributions_s_TokenTM",  #: TokenTM
                ##
                "attributions_s_MeanReLU__AttnGrad_Attn_ro_str50",  #: GenAtt
                f"attributions_s_CAT_s:sum_AttnFrom_sum_to{blocks_len - 1}",  #: AttCAT
                ##
                f"attributions_s_blocks__{blocks_len - 1}__MeanAttn",  #: RawAtt
                "attributions_s_MeanAttn_ro_str50",  #: AttnRoll
                ##
                # "attributions_s_blocks__7__XACAM_s:sum",
                f"attributions_s_blocks__{blocks_len - 1}__XACAM_s:sum",  #: XGradCAM+
                f"attributions_s_blocks__{blocks_len - 1}__GCAM_s:sum",  #: GradCAM
                # f"attributions_s_blocks__{blocks_len - 1}__GCAM_s:RS",
                "attributions_s_GCAM_s:sum_sum",  #: GradCAM-PLUS
                "attributions_s_XACAM_s:sum_sum",  #: XGradCAM+-PLUS
                ##
                f"attributions_s_blocks__0__FGrad_s:sum",  #: FullGrad
                f"attributions_s_blocks__0__FGrad_s:RS",
                f"attributions_s_blocks__0__CAT_s:sum",  #: IxG (IG, DecompX, AttnLRP, AliLRP)
                f"attributions_s_blocks__0__CAT_s:RS",
                f"attributions_s_blocks__{blocks_len - 1}__CAT_s:sum",  #: HiResCAM
                f"attributions_s_blocks__{blocks_len - 1}__CAT_s:RS",  #: GradCAMElementWise
                f"attributions_s_blocks__{(blocks_len // 2) + 1}__CAT_s:sum",
                f"attributions_s_blocks__{(blocks_len // 2) + 1}__CAT_s:RS",
                ##
                #: CAT
                "attributions_s_CAT_s:sum_sum",
                "attributions_s_CAT_s:RS_sum",
                ##
                #: IxG SkipPLUS (e.g., DecompX+S)
                f"attributions_s_CAT_s:sum_sum_f{blocks_len // 2}",
                f"attributions_s_CAT_s:RS_sum_f{blocks_len // 2}",
                ##
                #: FullGrad+-PLUS
                "attributions_s_CAT_s:sum_sum_FGrad_s:sum",
                "attributions_s_CAT_s:RS_sum_FGrad_s:RS",
                ##
                #: FullGrad+-SkipPLUS
                f"attributions_s_CAT_s:sum_sum_f{blocks_len // 2}_FGrad_s:sum",
                f"attributions_s_CAT_s:RS_sum_f{blocks_len // 2}_FGrad_s:RS",
                ##
                #: FullGrad-PLUS
                "attributions_s_FGrad_s:sum_sum",
                "attributions_s_FGrad_s:RS_sum",
                ##
            ]

            if True:
                #: Filter out any items containing =_s:RS= from attributions_cols:
                attributions_cols = [col for col in attributions_cols if "_s:RS" not in col]

        elif qual_submode == "q_pruned_1":
            attributions_cols += [
                "attributions_s_XACAM_s:sum_sum",  #: XGradCAM+-PLUS
                ##
                f"attributions_s_blocks__0__FGrad_s:sum",  #: FullGrad
                f"attributions_s_blocks__0__CAT_s:sum",  #: IxG (IG, DecompX, AttnLRP, AliLRP)
                ##
                #: CAT
                "attributions_s_CAT_s:sum_sum",
                ##
                #: IxG SkipPLUS (e.g., DecompX+S)
                f"attributions_s_CAT_s:sum_sum_f{blocks_len // 2}",
                ##
                #: FullGrad+-PLUS
                "attributions_s_CAT_s:sum_sum_FGrad_s:sum",
                ##
                #: FullGrad+-SkipPLUS
                f"attributions_s_CAT_s:sum_sum_f{blocks_len // 2}_FGrad_s:sum",
                ##
                #: FullGrad-PLUS
                # "attributions_s_FGrad_s:sum_sum",
                ##
            ]

        elif qual_submode == "q_CAM_1":
            attributions_col_patterns = []

            attributions_cols = [
                f"attributions_s_blocks__{blocks_len - 1}__GCAM_s:sum",  #: GradCAM
                "attributions_s_GCAM_s:sum_sum",  #: GradCAM-PLUS
                f"attributions_s_blocks__{blocks_len - 1}__XACAM_s:sum",  #: XGradCAM+
                "attributions_s_XACAM_s:sum_sum",  #: XGradCAM+-PLUS
                ##
            ]

        else:
            raise ValueError(f"Unknown qual_submode: {qual_submode}")

    outlier_quantile = float(
        getenv(
            "DECOMPV_OUTLIER_QUANTILE",
            default=None,
        )
        or "0.01"
    )
    colormap = (
        getenv(
            "DECOMPV_COLORMAP",
            default=None,
        )
        or "magma"
    )
    if normalize is None:
        normalize = getenv(
            "DECOMPV_QUAL_NORMALIZE",
            default=None,
        )

        if not normalize:
            if qual_submode == "q_CAM_1":
                normalize = "auto"

            else:
                normalize = [
                    "relu",
                    "scale_by_max_signed_attr",
                ]

    ic(outlier_quantile, colormap, normalize)

    tqdm_name = f"{dataset_name}-Qual {all_gbrands.compact_gbrand} {model_name} "
    if extra_tqdm_name:
        tqdm_name += extra_tqdm_name

    ##
    if dataset_name.lower() == "zele":
        coco_p = True

    else:
        coco_p = False

    ic(dataset_name, coco_p)
    ##

    if title is None:
        title = "{all_gbrands.compact_gbrand}\n{extra_title}"

    elif extra_title:
        title += "\n{extra_title}"

    try:
        attributions_show2(
            export_tlg_id=None,
            outlier_quantile=outlier_quantile,
            color_positive=colormap,
            tlg_msg=f"---\n\ngbrand: {all_gbrands.compact_gbrand}",
            title=title,
            tqdm_name=tqdm_name,
            compact_gbrand=all_gbrands.compact_gbrand,
            export_dir=f"{ARTIFACTS_ROOT}/plots{export_dir_suffix}/{dataset_name}/oq{outlier_quantile}/",
            normalize=normalize,
            overlay_alone_p=overlay_alone_p,
            batch=tds_attn,
            attributions_cols=attributions_cols,
            attributions_col_patterns=attributions_col_patterns,
            model=model,
            token_i="auto",
            batch_from=batch_from,
            batch_to=batch_to,
            batch_size=batch_size,
            model_patch_info=my_model_patch_info,
            ##
            tds_torch_cpu=my_tds_torch_cpu,
            coco_p=coco_p,
            ##
            plot_output_p=plot_output_p,
            **kwargs,
            ##
        )

    finally:
        # z("bell-call-remote bell-lm-strawberryjuice ; sleep 3 ; bell-call-remote bella")
        pass

    return None


##
def compute_qual_clip(
    *,
    dataset_name,
    ds,
    batch_size=None,
    **kwargs,
):
    compute_qual(
        ds,
        dataset_name=dataset_name,
        batch_size=batch_size,
        model=model,
        tds_attn_get_fn=clip_tds_attn_get,
        device=None,
        **kwargs,
    )


##
def compute_qual_ImageNetS(
    *,
    batch_size=None,
    batch_from=0,
    batch_to=100,
):
    ds = None  #: Placeholder, as ds is not used directly for ImageNetS

    compute_qual(
        ds,
        dataset_name="ImageNetS",
        batch_size=batch_size,
        model=model,
        tds_attn_get_fn=imagenet_s_tds_attn_get,
        device=None,
        batch_from=batch_from,
        batch_to=batch_to,
    )


##
def compute_qual_zele(
    *,
    batch_size=None,
):
    ds = None  #: Placeholder, as ds is not used directly for zele

    compute_qual(
        ds,
        dataset_name="zele",
        batch_size=batch_size,
        model=model,
        tds_attn_get_fn=zele_tds_attn_get,
        device=None,
    )


def get_compute_function(dataset_name):
    compute_functions = {
        "ImageNetS": compute_qual_ImageNetS,
        "zele": compute_qual_zele,
    }

    #: Check if the dataset_name matches the CLIP pattern
    clip_match = re.match(r"^CLIP(\d+)$", dataset_name)
    if clip_match:
        assert (
            dataset_name not in compute_functions
        ), f"Dataset {dataset_name} already defined in compute_functions but also matches the CLIP-N pattern!"

        clip_number = clip_match.group(1)
        ds_variable_name = f"clip_ds{clip_number}_raw"

        if ds_variable_name in globals():
            return partial(
                compute_qual_clip,
                dataset_name=dataset_name,
                ds=globals()[ds_variable_name],
            )
        else:
            raise ValueError(f"Dataset {ds_variable_name} not found in globals")

    return compute_functions.get(dataset_name, None)


## * Main
def main():
    parser = argparse.ArgumentParser(
        description="Compute attributions for different datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        # choices=["CLIP1", "CLIP2", "ImageNetS", "zele"],
        help="Name of the dataset to compute attributions for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size for processing.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    batch_size = args.batch_size

    compute_fn = get_compute_function(dataset_name)
    if compute_fn is None:
        print(f"Unsupported dataset: {dataset_name}")
        parser.print_help()
        exit(1)

    compute_fn(batch_size=batch_size)


if __name__ == "__main__":
    main()
