from decompv.early_boot import (
    run_check_completeness_mode_p,
    run_compute_completeness_mode_p,
)

# import argparse
import os
from os import getenv
import torch
import datasets
import re
from functools import partial
from pynight.common_icecream import ic
from pynight.common_hosts import (
    mmd1_p,
)
from pynight.common_files import mkdir
from pynight.common_iterable import list_rm
from pynight.common_regex import rget
from pynight.common_sort import version_sort_key
from pynight.common_torch import (
    gpu_memory_get,
)
from pynight.common_benchmark import (
    timed,
    Timed,
)
from pynight.common_dict import (
    simple_obj,
    concatenate_batches,
)

from pynight.common_dynamic import (
    DynamicVariables,
    DynamicObject,
    dynamic_set,
    dynamic_get,
)

dynamic_vars = dict()
dynamic_obj = DynamicObject(dynamic_vars, default_to_none_p=True)


##
def transform_cpu(d):
    """
    Recursively processes a dictionary and detaches and moves any torch tensors to the CPU.

    Args:
    - d (dict): Input dictionary that may contain torch tensors

    Returns:
    - dict: Dictionary with torch tensors detached and moved to CPU
    """
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = value.detach().cpu()

        elif isinstance(value, dict):
            d[key] = transform_cpu(value)

    return d


##
def transform_torch_save(
    batch,
    *,
    columns=None,
    output_dir,
    return_mode=None,
):
    index = batch["id"]
    batch_size = len(index)

    if columns is None:
        columns = list(batch.keys())

    for column in columns:
        if column in [
            "id",
        ]:
            continue

        if (
            run_check_completeness_mode_p
            or run_compute_completeness_mode_p
            and "completeness_error" not in column
        ):
            continue

        column_data = batch[column]

        split_name = f"{index[0]}_{index[-1] + 1}"
        #: The end index is exclusive.

        output_dir_current = os.path.join(output_dir, f"{column}")
        mkdir(output_dir_current)
        output_path = os.path.join(output_dir_current, f"{split_name}.pt")

        torch.save(column_data, output_path)
        print(output_path)

    if return_mode == "dummy":
        return dict(dummy=([0] * batch_size))
    elif return_mode == "same":
        return batch
    elif return_mode is None:
        return None
    else:
        raise ValueError(f"unsupported return_mode: {return_mode}")


def save_tds_torch(
    tds,
    *,
    batch_size,
    output_dir=None,
    return_p=False,
    save_p=True,
    name=None,
    tqdm=None,
    tqdm_name=None,
):
    metadata = dict()

    batched_iterator = tds.batched_iterator(batch_size)
    if tqdm is not None:
        tqdm_name = tqdm_name or name or output_dir
        if tqdm_name:
            tqdm_name = f"{tqdm_name} "

        batched_iterator = tqdm(
            batched_iterator,
            name=tqdm_name,
        )

    bench_dict = dict()
    batches = []
    with Timed(print_p=False, output_dict=bench_dict):
        for i, batch in enumerate(batched_iterator):
            if return_p:
                batch_saved = batch
                if return_p == "cpu":
                    batch_saved = transform_cpu(batch_saved)

                batches.append(batch_saved)

            if save_p:
                transform_torch_save(
                    batch,
                    output_dir=output_dir,
                )

    if name:
        metadata[f"time_{name}"] = bench_dict["time"]

    if return_p:
        batch_all = concatenate_batches(batches)
    else:
        batch_all = None

    return simple_obj(
        metadata=metadata,
        batch_all=batch_all,
    )


##
def h_attr_sort_key(caption):
    # ic(type(caption), caption)

    caption_key = version_sort_key(
        caption,
        pre_key_fn=os.path.basename,
    )

    if True:
        # if mode == 0:
        patterns = [
            "Input",
            lambda c: c.startswith("Decomp"),
            lambda c: c.startswith("Line"),
            lambda c: c.startswith("Raw Attention"),
            lambda c: c.startswith("Attention"),
            # "Attention Sum",
            # "Attention Rollout",
            lambda c: c.startswith("GenAtt"),
            "GradSAM",
            lambda c: c.startswith("ReLUAttnGrad"),
            # "AttnGrad L11",
            "AttCAT",
            "CAT",
        ]

    # Find priority based on pattern
    for index, pattern in enumerate(patterns):
        if isinstance(pattern, str) and caption == pattern:
            return (index, caption_key)
        elif callable(pattern) and pattern(caption):
            return (index, caption_key)

    if caption == "Random":
        return (999999, caption_key)

    # If no pattern matches, return the next priority
    return (len(patterns), caption_key)


def attr_name_official_get(input_str, mode="ALL"):
    if input_str is None:
        return input_str

    # ic(input_str)

    decomp_layer = rget(input_str, r"^logit_DecompV_vector_f(\d+)$")
    if decomp_layer is not None:
        decomp_layer = int(decomp_layer)
        if mode == 0:
            if decomp_layer not in (0, 6, 7, 11):
                return None
        elif mode == 1:
            if decomp_layer not in (7,):
                return None
        # elif mode == 4:
        #     if decomp_layer not in (0, 6, 7, 11):
        #         return None
        # elif mode == 3:
        #     if decomp_layer not in (0, ):
        #         return None

    if mode == "ALL":
        patterns = [
            simple_obj(
                pattern=r"^rnd1$",
                replacement=r"Random",
            ),
            ##
            simple_obj(
                pattern=r"^logit_DecompV_vector$",
                replacement=r"DecompX",
            ),
            simple_obj(
                pattern=r"^logit_GlobEnc_reset$",
                replacement=r"DecompX L11 W/O MLP",
            ),
            simple_obj(
                pattern=r"^logit_DecompV_vector_f(\d+)$",
                replacement=r"DecompX L\1",
            ),
            ##
            simple_obj(
                pattern=r"^blocks__(\d+)__globenc_norm2$",
                replacement=r"GlobEnc L\1",
            ),
            simple_obj(
                pattern=r"^globenc_norm2_ro$",
                replacement=r"GlobEnc Rollout",
            ),
            simple_obj(
                pattern=r"^globenc_norm2_sum$",
                replacement=r"GlobEnc Sum",
            ),
            ##
            simple_obj(
                pattern=r"^blocks__(\d+)__ALTI$",
                replacement=r"ALTI L\1",
            ),
            simple_obj(
                pattern=r"^ALTI_sum$",
                replacement=r"ALTI Sum",
            ),
            simple_obj(
                pattern=r"^ALTI_ro$",
                replacement=r"ALTI",
            ),
            ##
            simple_obj(
                pattern=r"^blocks__(\d+)__MeanAttnGrad$",
                replacement=r"AttnGrad L\1",
            ),
            simple_obj(
                pattern=r"^MeanAttnGrad_sum$",
                replacement=r"AttnGrad Sum",
            ),
            simple_obj(
                pattern=r"^MeanReLUAttnGrad_sum$",
                replacement=r"ReLUAttnGrad Sum",
            ),
            ##
            simple_obj(
                pattern=r"^MeanAttn_ro_str50$",
                replacement=r"Attention Rollout",
            ),
            simple_obj(
                pattern=r"^blocks__(\d+)__MeanAttn$",
                replacement=r"Attention L\1",
            ),
            simple_obj(
                pattern=r"^MeanAttn_sum$",
                replacement=r"Attention Sum",
            ),
            ##
            simple_obj(
                pattern=r"^blocks__(\d+)__MeanReLU__AttnGrad_Attn$",
                replacement=r"GenAtt L\1",
            ),
            simple_obj(
                pattern=r"^MeanReLU__AttnGrad_Attn_ro_str50$",
                replacement=r"GenAtt",
            ),
            simple_obj(
                pattern=r"^MeanReLU__AttnGrad_Attn_relu_to1_ro_str50$",
                replacement=r"GenAtt (RN)",
                # enabled_p=False,
            ),
            simple_obj(
                pattern=r"^MeanReLU__AttnGrad_Attn_sum$",
                replacement=r"GradSAM",
            ),
            ##
            simple_obj(
                pattern=r"^blocks__(\d+)__AttnWHeadGrad$",
                replacement=r"GradCAM L\1",
            ),
            ##
            simple_obj(
                pattern=r"^IxG_s:norm$",
                replacement=r"Gradient×Input-Norm",
            ),
            simple_obj(
                pattern=r"^PatchGrad_s:norm$",
                replacement=r"SimpleGradient-Norm",
            ),
            ##
            simple_obj(
                pattern=r"^CAT_sum$",
                replacement=r"CAT",
            ),
            simple_obj(
                pattern=r"^CAT_AttnFrom_sum$",
                replacement=r"AttCAT",
            ),
            simple_obj(
                pattern=r"^blocks__(\d+)__CAT_AttnFrom$",
                replacement=r"AttCAT L\1",
            ),
            simple_obj(
                pattern=r"^blocks__(\d+)__(.*)$",
                replacement=r"\2 L\1",
            ),
            ##
        ]
    elif mode == 0:
        patterns = [
            simple_obj(
                pattern=r"^rnd1$",
                replacement=r"Random",
            ),
            ##
            simple_obj(
                pattern=r"^logit_DecompV_vector$",
                replacement=r"DecompX",
            ),
            simple_obj(
                pattern=r"^logit_GlobEnc_reset$",
                replacement=r"DecompX L11 W/O MLP",
            ),
            simple_obj(
                pattern=r"^logit_DecompV_vector_f(\d+)$",
                replacement=r"DecompX L\1",
            ),
            ##
            simple_obj(
                pattern=r"^blocks__(\d+)__ALTI$",
                replacement=r"ALTI L\1",
            ),
            simple_obj(
                pattern=r"^ALTI_sum$",
                replacement=r"ALTI Sum",
            ),
            ##
            simple_obj(
                pattern=r"^blocks__9__MeanAttnGrad$",
                replacement=r"AttnGrad L9",
            ),
            # simple_obj(
            #     pattern=r"^blocks__11__MeanAttnGrad$",
            #     replacement=r"AttnGrad L11",
            # ),
            # simple_obj(
            #     pattern=r"^MeanAttnGrad_sum$",
            #     replacement=r"AttnGrad Sum",
            # ),
            simple_obj(
                pattern=r"^MeanReLUAttnGrad_sum$",
                replacement=r"ReLUAttnGrad Sum",
            ),
            ##
            simple_obj(
                pattern=r"^MeanAttn_ro_str50$",
                replacement=r"Attention Rollout",
            ),
            simple_obj(
                pattern=r"^blocks__9__MeanAttn$",
                replacement=r"Attention L9",
            ),
            simple_obj(
                pattern=r"^blocks__11__MeanAttn$",
                replacement=r"Raw Attention (L11)",
            ),
            simple_obj(
                pattern=r"^MeanAttn_sum$",
                replacement=r"Attention Sum",
            ),
            ##
            # simple_obj(
            #     pattern=r"^blocks__10__MeanReLU__AttnGrad_Attn$",
            #     replacement=r"GenAtt L10",
            # ),
            simple_obj(
                pattern=r"^MeanReLU__AttnGrad_Attn_ro_str50$",
                replacement=r"GenAtt",
            ),
            simple_obj(
                pattern=r"^MeanReLU__AttnGrad_Attn_sum$",
                replacement=r"GradSAM",
            ),
            ##
            simple_obj(
                pattern=r"^blocks__11__AttnWHeadGrad$",
                replacement=r"GradCAM",
            ),
            ##
            simple_obj(
                pattern=r"^IxG_s:norm$",
                replacement=r"Gradient×Input-Norm",
            ),
            # simple_obj(
            #     pattern=r"^PatchGrad_s:norm$",
            #     replacement=r"SimpleGradient-Norm",
            # ),
            ##
            simple_obj(
                pattern=r"^CAT_sum$",
                replacement=r"CAT",
            ),
            simple_obj(
                pattern=r"^CAT_AttnFrom_sum$",
                replacement=r"AttCAT",
            ),
            ##
        ]
    elif mode == 1:
        patterns = [
            simple_obj(
                pattern=r"^logit_DecompV_vector$",
                replacement=r"DecompX",
            ),
            simple_obj(
                pattern=r"^logit_DecompV_vector_f(\d+)$",
                replacement=r"DecompX L\1",
            ),
            # simple_obj(
            #     pattern=r"^blocks__11__MeanAttn$",
            #     replacement=r"Raw Attention",
            # ),
            simple_obj(
                pattern=r"^MeanReLU__AttnGrad_Attn_ro_str50$",
                replacement=r"GenAtt",
            ),
            # simple_obj(
            #     pattern=r"^blocks__11__AttnWHeadGrad$",
            #     replacement=r"GradCAM",
            # ),
            # simple_obj(
            #     pattern=r"^IxG_s:norm$",
            #   #  replacement=r"IxG",
            #     replacement=r"Gradient×Input-Norm",
            # ),
            ##
            # simple_obj(
            #     pattern=r"^CAT_sum$",
            #     replacement=r"CAT",
            # ),
            simple_obj(
                pattern=r"^CAT_AttnFrom_sum$",
                replacement=r"AttCAT",
            ),
            simple_obj(
                pattern=r"^MeanAttn_ro_str50$",
                replacement=r"Attention Rollout",
            ),
            ##
        ]
    elif mode == 4:
        patterns = [
            simple_obj(
                pattern=r"^logit_DecompV_vector$",
                replacement=r"DecompX",
            ),
            simple_obj(
                pattern=r"^logit_DecompV_vector_f(\d+)$",
                replacement=r"Linearization L\1",
            ),
            simple_obj(
                pattern=r"^CAT_AttnFrom_sum$",
                replacement=r"AttCAT",
            ),
            simple_obj(
                pattern=r"^CAT_sum$",
                replacement=r"CAT",
            ),
            simple_obj(
                pattern=r"^blocks__11__MeanAttn$",
                replacement=r"Raw Attention L11",
            ),
            simple_obj(
                pattern=r"^MeanReLU__AttnGrad_Attn_ro_str50$",
                replacement=r"GenAtt",
            ),
            simple_obj(
                pattern=r"^blocks__11__AttnWHeadGrad$",
                replacement=r"GradCAM",
            ),
            simple_obj(
                pattern=r"^IxG_s:norm$",
                replacement=r"Gradient×Input-Norm",
            ),
            simple_obj(
                pattern=r"^MeanAttn_ro_str50$",
                replacement=r"Attention Rollout",
            ),
            simple_obj(
                pattern=r"^rnd1$",
                replacement=r"Random",
            ),
        ]

    # Try each pattern separately, if one matches, return the result
    for obj in patterns:
        if re.search(obj.pattern, input_str):
            if "enabled_p" in obj and not obj.enabled_p:
                return None

            input_str = re.sub(obj.pattern, obj.replacement, input_str)

            return input_str
            ##
            # if input_str == "DecompX L0":
            #     return "DecompX"
            # else:
            #     return input_str
            ##

    # If none of the patterns matched
    return None


##
def batch_size_for(
    *,
    model_name,
    all_gbrands,
    override=None,
    gpu_mem_gb=None,
    seg_p=False,
):
    if gpu_mem_gb is None:
        gpu_mem_gb = gpu_memory_get()

    ##
    if any(
        re.search(p, model_name)
        for p in [
            r"huge_patch.*336",
            r"large_patch\d+_(?:clip_)?(?:336|384)",
            r"vit_so400m_patch14_siglip_(?:gap_)?378",
            r"(?:SO400M|so400m).*384",
            #: 'vit_large_patch14_clip_336.openai_ft_in12k_in1k'
            #: @bad 10
        ]
    ):
        model_size = 5

    elif any(re.search(p, model_name) for p in []):
        model_size = 4

    elif any(
        re.search(p, model_name)
        for p in [
            r"large_patch\d+_384",  #: doesn't fit on 12GB?
            r"patch8_",
            r"huge",  #: @untested
        ]
    ):
        model_size = 3

    elif any(
        re.search(p, model_name)
        for p in [
            r"_384",
            r"(?:SO400M|so400m)",
        ]
    ):
        model_size = 2

    elif any(
        re.search(p, model_name)
        for p in [
            r"large",
        ]
    ):
        model_size = 1

    else:
        model_size = 0

    ic(model_size)
    ##
    ds_multiplier_generic = 1
    if override is None:
        if (
            model_name
            in [
                "vit_small_patch16_224.augreg_in21k_ft_in1k",
                "vit_small_patch16_384.augreg_in21k_ft_in1k",
                "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                "eva02_small_patch14_336.mim_in22k_ft_in1k",
                "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
                "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
                "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
                "EVA02-L-14-336.OC.merged2b_s6b_b61k",
                "EVA02-E-14.OC.laion2b_s4b_b115k",
                "EVA02-E-14-plus.OC.laion2b_s9b_b144k",
                "vit_small_patch14_dinov2.lvd142m",
                "vit_base_patch14_dinov2.lvd142m",
                "vit_large_patch14_dinov2.lvd142m",
                "vit_giant_patch14_dinov2.lvd142m",
                "gmixer_24_224.ra3_in1k",
            ]
            or "vit_base_patch16_224" in model_name
        ):
            ds_multiplier_generic = 0.7

            ######
            if model_name == "vit_small_patch14_dinov2.lvd142m":
                if gpu_mem_gb > 20:
                    batch_size = 20
                    #: @good 10

                else:
                    batch_size = 4

            elif model_name == "vit_base_patch14_dinov2.lvd142m":
                if gpu_mem_gb > 20:
                    batch_size = 10
                    #: @good
                    #: @bad 20

                else:
                    batch_size = 4

            elif model_name == "vit_large_patch14_dinov2.lvd142m":
                batch_size = 1
                #: @good 1 (~17GB)
                #: @bad 5, 3, 2

                return batch_size

            elif model_name == "vit_giant_patch14_dinov2.lvd142m":
                batch_size = 1
                #: Even a batch size of 1 does not fit on 24GB (4090).

            elif model_name == "gmixer_24_224.ra3_in1k":
                if gpu_mem_gb > 20:
                    batch_size = 150

                else:
                    batch_size = 75

            elif model_name in [
                "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
                "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
            ]:
                if gpu_mem_gb > 20:
                    batch_size = 2
                    #: @bad 13, 10, 8, 6, 5, 4, 3
                    #: @good

                else:
                    batch_size = 1
                    #: 1: ~7GB

            elif model_name in [
                "EVA02-L-14-336.OC.merged2b_s6b_b61k",
            ]:
                if gpu_mem_gb > 20:
                    batch_size = 10
                    #: @good 10 (22.88GiB / 24.00GiB )
                    #: @bad 11

                else:
                    batch_size = 4

            elif model_name in [
                "EVA02-E-14.OC.laion2b_s4b_b115k",
                "EVA02-E-14-plus.OC.laion2b_s9b_b144k",
            ]:
                batch_size = 1
                #: E-14, E-14-plus: Even a batch size of 1 fails on 24GB

            elif model_name == "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k":
                if gpu_mem_gb > 20:
                    batch_size = 12
                    #: @bad 35, 20, 15
                    #: @good 10

                else:
                    batch_size = 5

            elif model_name == "eva02_small_patch14_336.mim_in22k_ft_in1k":
                if gpu_mem_gb > 20:
                    batch_size = 65
                    #: @bad 90, 80 (failed on t21)
                    #: 70 now fails on NG,h.S1,p.DS100, 1p [jalali:1403/07/20/01:06]

                else:
                    batch_size = 25
                    #: @bad 30

            elif model_name == "vit_small_patch16_384.augreg_in21k_ft_in1k":
                if gpu_mem_gb > 20:
                    batch_size = 100

                else:
                    batch_size = 40
                    #: NG,h.S1,p.DS100: 40: 11918MiB

            elif model_name == "vit_small_patch16_224.augreg_in21k_ft_in1k":
                if gpu_mem_gb > 20:
                    batch_size = 300
                    #: 325 crashes after several batches

                else:
                    batch_size = 120

            elif "vit_base_patch16_224" in model_name:
                #: "vit_base_patch16_224.augreg2_in21k_ft_in1k"
                if gpu_mem_gb > 20:
                    batch_size = 125
                    #: @bad 150 crashes on NG

                else:
                    batch_size = 60
                    #: @untested
            ######

            if not any(
                re.search(pat, all_gbrands.gradient_mode_brand)
                for pat in [
                    r"^NG",
                ]
            ):
                # if seg_p:
                batch_size *= 0.75

        elif model_size == 5:
            if gpu_mem_gb > 20:
                batch_size = 6
            else:
                batch_size = 1

        elif model_size == 4:
            if gpu_mem_gb > 20:
                batch_size = 10
            else:
                batch_size = 4

        elif model_size == 3:
            if gpu_mem_gb > 20:
                batch_size = 20
            else:
                batch_size = 8

        elif model_size == 2:
            if gpu_mem_gb > 20:
                batch_size = 35
            else:
                batch_size = 15

        elif model_size == 1:
            if gpu_mem_gb > 20:
                batch_size = 40
            else:
                batch_size = 20

        else:
            if gpu_mem_gb > 20:
                batch_size = 45
            else:
                batch_size = 30

    else:
        batch_size = override
    ##
    if not all_gbrands.gradient_mode_brand:
        pass

    elif any(
        re.search(pat, all_gbrands.gradient_mode_brand)
        for pat in [
            r"^LX-ADB",  #: The biases take a lot of memory
        ]
    ):
        batch_size = 0.5 * batch_size

    elif any(
        re.search(pat, all_gbrands.gradient_mode_brand)
        for pat in [
            r"^LX-AZR",
            r"^(?:LX|Gate)-XSC",
        ]
    ):
        batch_size = 0.6 * batch_size

    elif any(
        re.search(pat, all_gbrands.gradient_mode_brand)
        for pat in [
            r"^LX-AZO",
            r"^LX-AAG$",
            r"^LX-AD\b",
            r"^LX-GA",
            r"^Gate-D",
            r"^NG-D",
        ]
    ):
        batch_size = 0.75 * batch_size

    ##
    if any(
        re.search(pat, all_gbrands.softmax_mode)
        for pat in [
            r"^XSC",
        ]
    ):
        batch_size = 0.8 * batch_size

    ##
    if gpu_mem_gb > 20:
        ds_multiplier = 0.6
        # ds_multiplier = 0.9
    else:
        ds_multiplier = 0.9

    ds_multiplier *= ds_multiplier_generic

    def ds_update(
        name,
        batch_size,
        ds_multiplier=ds_multiplier,
    ):
        if not all_gbrands.get(name):
            pass
        elif any(
            re.search(pat, all_gbrands.get(name))
            for pat in [
                "^DS",
            ]
        ):
            batch_size = ds_multiplier * batch_size

        return batch_size

    batch_size = ds_update(
        name="mlp_ds_gbrand",
        batch_size=batch_size,
        ds_multiplier=ds_multiplier,
    )
    batch_size = ds_update(
        name="linear_ds_gbrand",
        batch_size=batch_size,
        ds_multiplier=ds_multiplier,
    )
    batch_size = ds_update(
        name="qkv_ds_gbrand",
        batch_size=batch_size,
        ds_multiplier=ds_multiplier,
    )
    ##
    if all_gbrands.ig_steps:
        # batch_size = 0.9 * batch_size
        pass

    if mmd1_p():
        batch_size = 0.7 * batch_size

    manual_batch_size_multiplier = float(
        getenv(
            "DECOMPV_BATCH_SIZE_MUL",
            default=None,
        )
        or 1
    )
    ic(manual_batch_size_multiplier)
    batch_size *= manual_batch_size_multiplier

    batch_size = int(batch_size)
    batch_size = max(batch_size, 1)
    ic(batch_size)
    return batch_size


##
