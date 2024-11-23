#!/usr/bin/env python3

### * Imports
import pynight
import decompv
import decompv.utils
import pprint
import argparse
import re
from decompv.x.bootstrap import *
from decompv.x.bootstrap import (
    dsmap_predict_masked_remove_cols,
)
from decompv.x.ds.main import *
from pynight.common_icecream import ic
from IPython import embed


blocks_len = len(model.blocks)


###
def globalti_after_transforms_get(
    model,
    residual_strengths=[
        0,
        0.5,
    ],
    rollout_scale=[
        1,
        5,
        10,
        100,
    ],
    # sum_from_layers=None,
    sum_from_layers="auto",
    sum_to_layers=None,
    # sum_to_layers="auto",
):
    transforms = []

    blocks_len = len(model.blocks)
    ###
    if sum_from_layers == "auto":
        sum_from_layers = range(0, blocks_len)
    elif sum_from_layers == None:
        sum_from_layers = [0]

    if sum_to_layers == "auto":
        sum_to_layers = range(1, blocks_len + 1)
    elif sum_to_layers == None:
        sum_to_layers = [blocks_len]

    for name in [
        "ALTI",
        "globenc_norm2",
    ]:
        name_fn = lambda i, name=name: f"blocks__{i}__{name}"
        #: `name=name` is necessary to save name, otherwise it would refer to the global scope!
        ###
        transforms += [
            partial(
                transform_aggregate_layers_rollout,
                output_name=name,
                name_fn=name_fn,
                model=model,
                residual_strength=rs,
                scale=scale,
            )
            for rs in residual_strengths
            for scale in rollout_scale
        ]
        ###
        sum_from_layers_current = set(sum_from_layers)
        sum_to_layers_current = set(sum_to_layers)

        for from_layer in sum_from_layers_current:
            for to_layer in sum_to_layers_current:
                fromto_res = fromto_indices_normalize(
                    from_layer=from_layer,
                    to_layer=to_layer,
                    model=model,
                )
                from_layer, to_layer = fromto_res.from_layer, fromto_res.to_layer

                if (to_layer - from_layer) <= 1:
                    continue

                transforms.append(
                    partial(
                        transform_aggregate_layers_sum,
                        output_name=name,
                        name_fn=name_fn,
                        from_layer=from_layer,
                        to_layer=to_layer,
                        model=model,
                    )
                )

    return transforms


def tds_globalti_get(
    *,
    model=model,
    after_transforms_get=None,
    **kwargs,
):
    if after_transforms_get is None:
        after_transforms_get = globalti_after_transforms_get

    decomposition_config = DecompositionConfig(
        device=device,
        attributions_aggregation_strategy="reset",
        # attributions_aggregation_strategy="vector",
        mlp_decompose_p=False,
        save_intermediate_p=True,
        name="GlobEnc",
    )

    return tds_decompv_get(
        model=model,
        decomposition_config=decomposition_config,
        after_transforms_get=after_transforms_get,
        **kwargs,
    )


def globalti_compute_global(
    *,
    name,
    attn_prepare_attribution_columns,
    transform_only_p=True,
    dataset_start=None,
    dataset_end=None,
    tds_patches: TransformedDataset = None,
    add_random_p=False,  #: only True for the first dataset
    early_return_mode=None,
    output_precision="bfloat16",
    #: @globals
    #: model,
    #: model_name,
    **globalti_opts,
):
    if dataset_start is None:
        dataset_start = dataset_start_global

    if dataset_end is None:
        dataset_end = dataset_end_global

    if tds_patches is None:
        tds_patches = tds_patches_imagenet

    bias_token_p = False
    model_patch_info = patch_info_from_name(model_name, bias_token_p=bias_token_p)

    tds_globalti = tds_globalti_get(
        model=model,
        model_name=model_name,
        tds_patches=tds_patches,
        **globalti_opts,
    )

    tds_globalti = tds_globalti.transform(
        partial(
            attn_prepare_attribution_columns,
            model=model,
        ),
    )

    if early_return_mode in ["1"]:
        return tds_globalti
    else:
        tds_masked = masker_entrypoint(
            tds_scalar=tds_globalti,
            model_patch_info=model_patch_info,
            bias_token_p=bias_token_p,
            add_random_p=add_random_p,
            model=model,
        )

        computer = partial(
            masked_predict,
            # tds_masked=tds_masked,
            output_precision=output_precision,
            transform_only_p=transform_only_p,
        )

        dataset_compute_gen(
            name="G",
            # name="tmp",
            # name=name,
            computer=computer,
            batch_size=batch_size,
            tds_patches=tds_masked,
            dataset_start=dataset_start,
            dataset_end=dataset_end,
            model=model,
            # load_from_cache_file=False,
        )


##
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("mode", help="mode of operation")
    parser.add_argument("--submode1", help="description for submode1", default=None)
    args = parser.parse_args()
    mode = args.mode
    submode1 = args.submode1

    if mode == "GlobALTIv1":
        if submode1 == "m3":
            include_patterns = [
                f"_sum_f{blocks_len // 2}($|_)",
            ]
            ic(include_patterns)

        else:
            include_patterns = None

        my_attn_prepare_attribution_columns = partial(
            attn_prepare_attribution_columns,
            include_patterns=include_patterns,
            # filter_fn=my_filter,
            keep_as_is_patterns=[
                # r"^attributions_s_logit(?:_|$)",
            ],
            # exclude_patterns=[],
            exclude_patterns=[
                "^blocks__",
                ##
                "logit_GlobEnc_reset",
                #: DecompX WO MLP Last
            ],
        )

        with DynamicVariables(
            decompv.utils.dynamic_obj,
            dsmap_predict_masked_remove_cols=dsmap_predict_masked_remove_cols,
        ):
            globalti_compute_global(
                name=mode,
                attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
                add_random_p=False,
            )
    elif mode == "embed":
        tds_globalti = tds_globalti_get()

        batch_transformed = tds_globalti[:2]
        ic(torch_shape_get(batch_transformed))

        embed()
        ##
        # transform_torch_save(batch_transformed, output_dir='/tmp/hi1')
    else:
        raise ValueError(f"unsupported mode {mode}")
##
