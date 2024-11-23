#!/usr/bin/env python3
## * Imports
import os
import pprint
import argparse
import re
import decompv

# import decompv.x.imagenet
from decompv.x.bootstrap import *
from decompv.x.bootstrap import (
    dsmap_predict_masked_remove_cols,
)
import decompv.x.ds.utils
from decompv.x.ds.utils import *
from decompv.x.ds.main import *
from decompv.x.ds.main import tds_patches_imagenet
from pynight.common_icecream import ic
from pynight.common_iterable import to_iterable
from pynight.common_torch import (
    rank_tensor,
)

## * Transforms
blocks_len = len(model.blocks)
##
m7_include_patterns = [
    # "rnd1",
    ##
    "^CAT_s:sum_sum$",
    f"^CAT_s:sum_sum_f{blocks_len // 2}$",
    ##
    "^CAT_s:sum_sum_FGrad_s:sum$",
    f"^CAT_s:sum_sum_f{blocks_len // 2}_FGrad_s:sum$",
    ##
    "^FGrad_s:sum_sum$",
    f"^FGrad_s:sum_sum_f{blocks_len // 2}$",
    ##
    "^blocks__0__CAT_s:sum$",
    # f"^blocks__{blocks_len // 2}__CAT_s:sum$",
    f"^blocks__{blocks_len - 1}__CAT_s:sum$",
    ##
    "^blocks__0__FGrad_s:sum$",
    # f"^blocks__{blocks_len // 2}__FGrad_s:sum$",
    # f"^blocks__{blocks_len - 1}__FGrad_s:sum$",
    ##
]

m7_exclude_patterns = [
    "Image",  #: @different from seg
    ##
    "AttnWHeadGrad",
    "PosE",
    "_s:L1",
    "GRCAM",
    "XCAM",
    "XRCAM",
    "XARCAM",
    "PCAM",
    "PACAM",
    "PRCAM",
    "PARCAM",
]


##
m8_include_patterns = [
    # "rnd1",
    ##
    "^blocks__0__CAT_s:sum$",
    ##
]

m8_exclude_patterns = [
    "Image",
    "PosE",
]
##
def m8_attn_transforms_get():
    channel_mixers = [
        "sum",
    ]

    attn_transforms = []

    attn_transforms += [
        partial(
            transform_CAT,
            model=model,
            compute_mode="IxG",
            channel_mixers=channel_mixers,
        ),
    ]

    return attn_transforms


def m6_attn_transforms_get(
    residual_strengths=[
        0.5,
    ],
    rollout_scale=[
        1,
    ],
    sum_from_layers="auto",
    sum_to_layers="auto",
    expensive_lv=100,
    AttCAT_variants=[],
    channel_mixers=[
        "sum",
    ],
    channel_mixers_positive=[
        "sum",
    ],
):
    ##
    channel_mixers = set(channel_mixers)
    channel_mixers_positive = set(channel_mixers_positive)
    decompv.x.ds.utils.channel_mixers = channel_mixers

    decompv.x.ds.utils.channel_mixers_positive = channel_mixers_positive
    ##

    AttCAT_variants = list(AttCAT_variants)  #: copy the list as we'll mutate it

    attn_transforms = []

    attn_transforms += [
        partial(
            transform_add_bamaps,
            model=model,
        ),
    ]

    for mode in channel_mixers:
        attn_transforms += [
            partial(
                transform_scalarify,
                methods=[
                    # "IxG",
                    # "PatchGrad",
                    #: PatchGrad is now blocks__0__PatchGrad_s:raw
                ],
                mode=mode,
            ),
        ]

    attn_transforms += [
        partial(
            transform_MeanAttn,
            model=model,
        ),
    ]

    attn_transforms += [
        partial(
            transform_AttnGrad,
            model=model,
        ),
        partial(
            transform_MeanAttnGrad,
            model=model,
            del_p=False,
        ),
        partial(
            transform_MeanAttnGrad,
            model=model,
            relu_p=True,
            del_p=False,
        ),
        partial(
            transform_MeanAttnGrad_MeanAttn,
            model=model,
        ),
    ]

    attn_transforms += [
        partial(
            transform_Mean__AttnGrad_Attn,
            model=model,
        ),
        partial(
            transform_TokenTM,
            model=model,
        ),
    ]

    attn_transforms += [
        partial(
            transform_CAT,
            model=model,
            channel_mixers=channel_mixers,
        ),
        partial(
            transform_CAT_AttnFrom,
            model=model,
            multiply_with=list(AttCAT_variants),
            #: copy the list as we'll mutate it later!
        ),
    ]

    ##: Rollout
    rollout_fns = [
        transform_rawattn_rollout,
        # transform_AttnGrad_rollout,
        # transform_AbsAttnGrad_rollout,
        # transform_Mean__AttnGrad_Attn_rollout,
        transform_MeanReLU__AttnGrad_Attn_rollout,
        # transform_MeanAbs__AttnGrad_Attn_rollout,
        # transform_MeanAttnGrad_MeanAttn_rollout,
        # transform_MeanReLUAttnGrad_MeanAttn_relu_rollout,
        # transform_MeanAttn_CAT_rollout,
        # transform_MeanAttn_CAT_relu_to1_rollout,
        # transform_MeanReLUAttnGrad_MeanAttn_CAT_relu_to1_rollout,
    ]

    for rollout_fn in rollout_fns:
        attn_transforms += [
            partial(
                rollout_fn,
                model=model,
                residual_strength=rs,
                scale=scale,
            )
            for rs in residual_strengths
            for scale in rollout_scale
        ]

    ###: Sum
    sum_from_beginning = [0]
    sum_to_end = [blocks_len]

    if sum_from_layers == "auto":
        sum_from_layers = [0, blocks_len // 2]

    elif sum_from_layers == None:
        sum_from_layers = list(sum_from_beginning)

    if sum_to_layers == "auto":
        sum_to_layers = list(sum_to_end)

    sum_names = [
        "MeanAttn",
        "MeanAttnGrad",
        "MeanAbsAttnGrad",  #: @new
        # "MeanReLUAttnGrad",  #: @new
        # "MeanAttnGrad_MeanAttn",
        # "MeanReLUAttnGrad_MeanAttn",
        # "Mean__AttnGrad_Attn",
        "MeanReLU__AttnGrad_Attn",
        # "MeanAbs__AttnGrad_Attn",
    ]

    # sum_names += [cat_channel_mixed_name_get(channel_mixer=channel_mixer, name_prefix="FGrad") for channel_mixer in channel_mixers]

    sum_names += [
        cat_channel_mixed_name_get(channel_mixer=channel_mixer, name_prefix="CAT")
        for channel_mixer in channel_mixers
    ]
    # sum_names += [
    #     "CAT",
    # ]
    # sum_names += [f"CAT_s:{name}" for name in channel_mixers_no_sum]

    AttCAT_variants += [
        "AttnFrom",
    ]
    for variant in AttCAT_variants:
        sum_names += [
            f"""{cat_channel_mixed_name_get(channel_mixer=channel_mixer, name_prefix="CAT")}_{variant}"""
            for channel_mixer in channel_mixers
        ]
        # sum_names += [f"CAT_{variant}"]
        # sum_names += [f"CAT_s:{name}_{variant}" for name in channel_mixers_no_sum]

    for name_prefix in [
        "GCAM",
        "XACAM",
        "PatchGrad",
    ]:
        sum_names += [f"{name_prefix}_s:{name}" for name in channel_mixers]

    for name_prefix in [
        # "GRCAM",
        # "XARCAM",
        # "PARCAM",
    ]:
        sum_names += [f"{name_prefix}_s:{name}" for name in channel_mixers_positive]

    def sum_ensemble(name, sum_from_layers, sum_to_layers):
        sum_from_layers_current = set(sum_from_layers)
        sum_to_layers_current = set(sum_to_layers)

        if name == "rawattn":
            raise Exception("Raw Attention is per head and cannot be used directly!")
            # name_fn = lambda i: f"blocks__{i}__attn__rawattn"
        else:
            name_fn = lambda i, name=name: f"blocks__{i}__{name}"
            #: `name=name` is necessary to save name, otherwise it would refer to the global scope!

        if any(
            ("CAT" in name and f"_{variant}" in name) for variant in AttCAT_variants
        ):
            #: This code is needed even for =naming_mode="end"=.
            ##
            if any(i in sum_to_layers_current for i in [0, blocks_len]):
                #: 0 will also become =blocks_len=. The to layer is used with =range=, so it's one higher than what it should be.
                ##
                sum_to_layers_current -= set([0, blocks_len])
                sum_to_layers_current.add(blocks_len - 1)
                #: The last layer is zero, and might be non-existent if     `include_last_p==False`.

        if False and ("CAT" in name and not "+" in name):
            #: @deprecated This code was for =naming_mode="end"=.
            ##
            sum_to_layers_current -= set([0, blocks_len])
            #: 0 will also become =blocks_len=. The to layer is used with =range=, so it's one higher than what it should be.
            #: These methods last layer is zero anyway.

            #: The old code (which I assume is wrong):
            # sum_to_layers_current -= set([-1, blocks_len - 1])

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

                attn_transforms.append(
                    partial(
                        transform_aggregate_layers_sum,
                        output_name=name,
                        name_fn=name_fn,
                        from_layer=from_layer,
                        to_layer=to_layer,
                        model=model,
                    )
                )
                # break
            # break

    for name in sum_names:
        sum_ensemble(name, sum_from_layers, sum_to_layers)
        # sum_ensemble(name, sum_from_beginning, sum_to_end)

    ###

    ###: FullGrad
    #: needs to be after the sum ensembles
    for channel_mixer in channel_mixers:
        # for channel_mixer in channel_mixers_essential:
        attn_transforms += [
            partial(
                transform_fullgrad,
                model=model,
                channel_mixer=channel_mixer,
            )
        ]

    fgrad_sum_names = []
    for name_prefix in [
        "FGrad",
    ]:
        fgrad_sum_names += [f"{name_prefix}_s:{name}" for name in channel_mixers]

    for name in fgrad_sum_names:
        sum_ensemble(name, sum_from_layers, sum_to_layers)

    ###

    return attn_transforms


def attn_transforms_get(
    residual_strengths=[
        # 0,
        # 0.25,
        0.5,
        # 0.85,
        # 0.95,
    ],
    rollout_scale=[
        1,
        # 5,
        # 10,
        # 100,
    ],
    # mean_before_mul_p=True,
    sum_from_layers="auto",
    sum_to_layers="auto",
    expensive_lv=100,
    ensemble_mode=None,
    AttCAT_variants=[],
    # AttCAT_variants=[
    #     "MeanAttn",
    #     # "AttnWHeadGrad",
    #     "MeanAttnGrad",
    #     "MeanReLUAttnGrad",
    #     # "MeanAttnGrad_MeanAttn",
    #     # "MeanReLUAttnGrad_MeanAttn",
    #     "MeanReLU__AttnGrad_Attn",
    # ],
):
    AttCAT_variants = list(AttCAT_variants)  #: copy the list as we'll mutate it

    attn_transforms = []

    attn_transforms += [
        partial(
            transform_add_bamaps,
            model=model,
        ),
    ]

    for mode in channel_mixers:
        attn_transforms += [
            partial(
                transform_scalarify,
                methods=[
                    # "IxG",
                    # "PatchGrad",
                    #: PatchGrad is now blocks__0__PatchGrad_s:raw
                ],
                mode=mode,
            ),
        ]

    attn_transforms += [
        partial(
            transform_MeanAttn,
            model=model,
        ),
    ]

    attn_transforms += [
        partial(
            transform_AttnGrad,
            model=model,
        ),
        partial(
            transform_MeanAttnGrad,
            model=model,
            del_p=False,
        ),
        partial(
            transform_MeanAttnGrad,
            model=model,
            relu_p=True,
            del_p=False,
        ),
        partial(
            transform_MeanAttnGrad_MeanAttn,
            model=model,
        ),
    ]

    attn_transforms += [
        partial(
            transform_Mean__AttnGrad_Attn,
            model=model,
        ),
        partial(
            transform_TokenTM,
            model=model,
        ),
    ]

    attn_transforms += [
        partial(
            transform_AttnWHeadGrad,
            model=model,
        ),
        partial(
            transform_CAT,
            model=model,
        ),
        partial(
            transform_CAT_AttnFrom,
            model=model,
            multiply_with=list(AttCAT_variants),
            #: copy the list as we'll mutate it later!
        ),
    ]

    ##: Rollout
    rollout_fns = [
        transform_rawattn_rollout,
        transform_AttnGrad_rollout,
        transform_AbsAttnGrad_rollout,
        transform_Mean__AttnGrad_Attn_rollout,
        transform_MeanReLU__AttnGrad_Attn_rollout,
        transform_MeanAbs__AttnGrad_Attn_rollout,
        # transform_MeanAttnGrad_MeanAttn_rollout,
        # transform_MeanReLUAttnGrad_MeanAttn_relu_rollout,
        # transform_MeanAttn_CAT_rollout,
        # transform_MeanAttn_CAT_relu_to1_rollout,
        # transform_MeanReLUAttnGrad_MeanAttn_CAT_relu_to1_rollout,
    ]
    if expensive_lv >= 100:
        rollout_fns += [
            # transform_MeanReLU__AttnGrad_Attn_relu_to1_rollout,
            # transform_MeanAttnGrad_MeanAttn_relu_to1_rollout,
            # transform_MeanReLUAttnGrad_MeanAttn_relu_to1_rollout,
        ]

    for rollout_fn in rollout_fns:
        attn_transforms += [
            partial(
                rollout_fn,
                model=model,
                residual_strength=rs,
                scale=scale,
            )
            for rs in residual_strengths
            for scale in rollout_scale
        ]

    ###: Sum
    sum_from_beginning = [0]
    sum_to_end = [blocks_len]

    if sum_from_layers == "auto":
        sum_from_layers = range(0, blocks_len)
    elif sum_from_layers == None:
        sum_from_layers = list(sum_from_beginning)

    if sum_to_layers == "auto":
        sum_to_layers = range(1, blocks_len + 1)
    elif sum_to_layers == None:
        sum_to_layers = list(sum_to_end)

    sum_names = [
        "MeanAttn",
        "MeanAttnGrad",
        "MeanAbsAttnGrad",  #: @new
        "MeanReLUAttnGrad",  #: @new
        "MeanAttnGrad_MeanAttn",
        "MeanReLUAttnGrad_MeanAttn",
        "Mean__AttnGrad_Attn",
        "MeanReLU__AttnGrad_Attn",
        "MeanAbs__AttnGrad_Attn",
        "AttnWHeadGrad",
    ]

    # sum_names += [cat_channel_mixed_name_get(channel_mixer=channel_mixer, name_prefix="FGrad") for channel_mixer in channel_mixers]

    sum_names += [
        cat_channel_mixed_name_get(channel_mixer=channel_mixer, name_prefix="CAT")
        for channel_mixer in channel_mixers
    ]
    # sum_names += [
    #     "CAT",
    # ]
    # sum_names += [f"CAT_s:{name}" for name in channel_mixers_no_sum]

    AttCAT_variants += [
        "AttnFrom",
    ]
    for variant in AttCAT_variants:
        sum_names += [
            f"""{cat_channel_mixed_name_get(channel_mixer=channel_mixer, name_prefix="CAT")}_{variant}"""
            for channel_mixer in channel_mixers
        ]
        # sum_names += [f"CAT_{variant}"]
        # sum_names += [f"CAT_s:{name}_{variant}" for name in channel_mixers_no_sum]

    for name_prefix in [
        "GCAM",
        ##
        "XCAM",
        # "XRCAM",
        "XACAM",
        ##
        # "PCAM",
        "PRCAM",
        # "PACAM",
        ##
        "PatchGrad",
    ]:
        sum_names += [f"{name_prefix}_s:{name}" for name in channel_mixers]

    for name_prefix in [
        "GRCAM",
        "XARCAM",
        "PARCAM",
    ]:
        sum_names += [f"{name_prefix}_s:{name}" for name in channel_mixers_positive]

    def sum_ensemble(name, sum_from_layers, sum_to_layers):
        sum_from_layers_current = set(sum_from_layers)
        sum_to_layers_current = set(sum_to_layers)

        if name == "rawattn":
            raise Exception("Raw Attention is per head and cannot be used directly!")
            # name_fn = lambda i: f"blocks__{i}__attn__rawattn"
        else:
            name_fn = lambda i, name=name: f"blocks__{i}__{name}"
            #: `name=name` is necessary to save name, otherwise it would refer to the global scope!

        if any(
            ("CAT" in name and f"_{variant}" in name) for variant in AttCAT_variants
        ):
            #: This code is needed even for =naming_mode="end"=.
            ##
            if any(i in sum_to_layers_current for i in [0, blocks_len]):
                #: 0 will also become =blocks_len=. The to layer is used with =range=, so it's one higher than what it should be.
                ##
                sum_to_layers_current -= set([0, blocks_len])
                sum_to_layers_current.add(blocks_len - 1)
                #: The last layer is zero, and might be non-existent if     `include_last_p==False`.

        if False and ("CAT" in name and not "+" in name):
            #: @deprecated This code was for =naming_mode="end"=.
            ##
            sum_to_layers_current -= set([0, blocks_len])
            #: 0 will also become =blocks_len=. The to layer is used with =range=, so it's one higher than what it should be.
            #: These methods last layer is zero anyway.

            #: The old code (which I assume is wrong):
            # sum_to_layers_current -= set([-1, blocks_len - 1])

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

                attn_transforms.append(
                    partial(
                        transform_aggregate_layers_sum,
                        output_name=name,
                        name_fn=name_fn,
                        from_layer=from_layer,
                        to_layer=to_layer,
                        model=model,
                    )
                )
                # break
            # break

    for name in sum_names:
        sum_ensemble(name, sum_from_layers, sum_to_layers)
        # sum_ensemble(name, sum_from_beginning, sum_to_end)

    ###

    ###: FullGrad
    #: needs to be after the sum ensembles
    for channel_mixer in channel_mixers:
        # for channel_mixer in channel_mixers_essential:
        attn_transforms += [
            partial(
                transform_fullgrad,
                model=model,
                channel_mixer=channel_mixer,
            )
        ]

    fgrad_sum_names = []
    for name_prefix in [
        "FGrad",
    ]:
        fgrad_sum_names += [f"{name_prefix}_s:{name}" for name in channel_mixers]

    for name in fgrad_sum_names:
        sum_ensemble(name, sum_from_layers, sum_to_layers)

    ###

    ###
    if ensemble_mode == "auto":
        for normalize in [
            None,
            partial(
                rank_tensor,
                reverse_p=True,
                #: We need to give the best token MORE attribution, hence =reverse_p=.
            ),
        ]:
            attn_transforms += [
                partial(
                    transform_attr_combine_sum,
                    methods=[
                        "CAT_sum",
                        "CAT_AttnFrom_sum",
                    ],
                    normalize=normalize,
                ),
            ]

            # for method in [
            #     "CAT_sum+CAT_AttnFrom_sum",
            #     "CAT_sum",
            #     "CAT_AttnFrom_sum",
            #     "MeanReLUAttnGrad_MeanAttn_relu_ro_str50",
            #     # "CAT_AttnFrom_sum_f6",
            #     "blocks__10__MeanReLUAttnGrad",
            # ]:
            #     attn_transforms += [
            #         partial(
            #             transform_attr_combine_sum,
            #             methods=[
            #                 "IxG_s:sum",
            #                 method,
            #             ],
            #             normalize=normalize,
            #         ),
            #     ]
    ###

    return attn_transforms


## * Utils
def attn_compute_global(
    *,
    attn_transforms,
    name,
    attn_prepare_attribution_columns,
    dsmap_attn_opts=None,
    transform_only_p=True,
    dataset_start=None,
    dataset_end=None,
    tds_patches: TransformedDataset = None,
    add_random_p=False,  #: only True for the first dataset
    output_precision="bfloat16",
    early_return_mode=None,
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

    if dsmap_attn_opts is None:
        dsmap_attn_opts = dict()

    bias_token_p = False
    model_patch_info = patch_info_from_name(model_name, bias_token_p=bias_token_p)

    tds_attn = tds_patches.transform(
        partial(
            dsmap_attn,
            model=model,
            after_transforms=attn_transforms,
            attn_prepare_attribution_columns=attn_prepare_attribution_columns,
            **dsmap_attn_opts,
        )
    )

    tds_attn2 = tds_attn
    # tds_attn2 = tds_attn.transform(
    #     partial(
    #         attn_prepare_attribution_columns,
    #         model=model,
    #     ),
    # )

    if run_check_completeness_mode_p or run_compute_completeness_mode_p:
        add_random_p = False  #: @redundant

        ##
        def computer_identity(
            tds_patches=tds_patches,
            **kwargs,
        ):
            return simple_obj(tds_after=tds_patches)

        return dataset_compute_gen(
            name=f"{name}/CE",
            computer=computer_identity,
            batch_size=batch_size,
            tds_patches=tds_attn2,
            dataset_start=dataset_start,
            dataset_end=dataset_end,
            model=model,
        )

    if early_return_mode in ["1"]:
        if add_random_p:
            tds_attn2 = add_random_baseline(
                tds_attn2,
                model_patch_info=model_patch_info,
                model=model,
            )

        return tds_attn2

    else:
        tds_masked_attn = masker_entrypoint(
            tds_scalar=tds_attn2,
            model_patch_info=model_patch_info,
            bias_token_p=bias_token_p,
            add_random_p=add_random_p,
            model=model,
        )

        tds_masked = tds_masked_attn

        computer = partial(
            masked_predict,
            # tds_masked=tds_masked,
            output_precision=output_precision,
            transform_only_p=transform_only_p,
        )

        dataset_compute_gen(
            # name="A",
            # name="tmp",
            name=name,
            computer=computer,
            batch_size=batch_size,
            tds_patches=tds_masked,
            dataset_start=dataset_start,
            dataset_end=dataset_end,
            model=model,
            # load_from_cache_file=False,
            # **kwargs,
        )


## * Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("mode", help="mode of operation")
    parser.add_argument("--submode1", help="description for submode1", default=None)
    args = parser.parse_args()
    mode = args.mode
    submode1 = args.submode1

    print(f"Faith Submode1: {submode1}", file=sys.stderr)

    #: @todo filter out the unwanted from-to aggregations
    #: E.g., =t11= is the same as to the last layer for CAT
    if mode == "attnv1":
        attn_transforms = attn_transforms_get()

        attn_compute_global(
            name=mode,
            attn_transforms=attn_transforms,
            attn_prepare_attribution_columns=attn_prepare_attribution_columns,
            add_random_p=True,
            selected_layers=[
                0,
                -1,
                -2,
                -3,
                -4,
            ],
        )
    elif mode == "attnv2":
        attn_transforms = attn_transforms_get()

        attn_compute_global(
            name=mode,
            attn_transforms=attn_transforms,
            attn_prepare_attribution_columns=attn_prepare_attribution_columns,
            add_random_p=True,
        )
    elif mode == "catv1":
        attn_transforms = attn_transforms_get()

        my_attn_prepare_attribution_columns = partial(
            attn_prepare_attribution_columns,
            include_patterns=".*CAT.*",
        )
        attn_compute_global(
            name=mode,
            attn_transforms=attn_transforms,
            attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
            add_random_p=True,
        )
    elif mode == "smallv1":
        attn_transforms = attn_transforms_get()

        my_attn_prepare_attribution_columns = partial(
            attn_prepare_attribution_columns,
            include_patterns=[
                # "^._logits",
                "^CAT_sum_f1_to8",
                "^IxG[^_]*",
            ],
        )
        attn_compute_global(
            name=mode,
            attn_transforms=attn_transforms,
            attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
            add_random_p=True,
        )
    elif mode == "mainv1":
        if submode1 == None:
            ensemble_mode = None
            store_blocks_p = False
        elif submode1 == "f1":
            ensemble_mode = True
            store_blocks_p = True

        attn_transforms = attn_transforms_get(
            residual_strengths=[
                # 0,
                # 0.25,
                0.5,
                # 0.85,
                # 0.95,
            ],
            sum_from_layers=None,
            sum_to_layers=None,
            ensemble_mode=ensemble_mode,
        )
        ic(len(attn_transforms), attn_transforms, len(attn_transforms))

        def my_filter(*, k, v):
            if any(
                re.search(pat, k)
                for pat in [
                    f"^blocks__{blocks_len - 1}__AttnWHeadGrad$",
                ]
            ):
                return True

            if k.startswith("blocks__"):
                if not store_blocks_p:
                    return False

                if any(
                    re.search(pat, k)
                    for pat in [
                        r"CAT(?:_AttnFrom)?$",
                        r"Mean__AttnGrad_Attn$",
                        r"MeanReLU__AttnGrad_Attn$",
                        r"MeanAttn(?:Grad)?$",
                    ]
                ):
                    return True
                else:
                    return False

            return True

        my_attn_prepare_attribution_columns = partial(
            attn_prepare_attribution_columns,
            filter_fn=my_filter,
        )

        with DynamicVariables(
            decompv.utils.dynamic_obj,
            dsmap_predict_masked_remove_cols=dsmap_predict_masked_remove_cols,
        ):
            attn_compute_global(
                name=mode,
                attn_transforms=attn_transforms,
                attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
                add_random_p=True,
            )

    elif mode == "v3":
        if submode1 in [
            "m6",
            "m7",
        ]:
            attn_transforms = m6_attn_transforms_get()

        elif submode1 in [
            "m8",
        ]:
            attn_transforms = m8_attn_transforms_get()

        else:
            attn_transforms = attn_transforms_get(
                sum_to_layers=None,
                ensemble_mode=None,
            )

        if submode1 == None:
            include_patterns = [
                # "^._logits",
                ##
                # "^CAT(?:_RS)?(?:_AttnFrom)?_sum(?:_f\d+)?$",
                # "^CAT(?:_RS)?(?:_AttnFrom)?_sum",
                # "^(?:blocks__\d+__)?CAT(?:_s:[^_]+)?(?:_AttnFrom)?",
                "CAT_s:(sum|RS)",
                ##
                # "PosE",
                # "^IxG",
                "blocks__0__PatchGrad_s:(sum|L2)",
                "FGrad_s:(sum|RS)",
                ##
                # "^MeanReLU__AttnGrad_Attn_ro_str50$",
                # "^MeanReLU__AttnGrad_Attn_sum",
                # "^(?:blocks__\d+__)MeanReLU__AttnGrad_Attn",
                "MeanReLU__AttnGrad_Attn(?:$|_sum$|_ro)",
                "^Mean(?:Abs)?__AttnGrad_Attn(?:$|_sum$|_ro)",
                ##
                f"^blocks__{blocks_len - 1}__.*CAM_s:sum",
                f"^blocks__{blocks_len - 1}__AttnWHeadGrad$",
                ##
                # "^MeanAttn_ro_str50$",
                # "^MeanAttn_sum$",
                f"^(?:blocks__{blocks_len - 1}__)?MeanAttn(?:$|_sum$|_ro)",
                ##
                "^Mean(?:Abs)?AttnGrad(?:$|_sum$|_ro)",
                ##
            ]
            exclude_patterns = [
                # "_to\d+(?:_|$)",
                "^Image(?:Grad|IxG)",
                "_s:(?!sum)[^_]*_AttnFrom",
                "blocks__.*_AttnFrom",
                "GRCAM",
                "XRCAM",
                "XARCAM",
                "PCAM",
                "PACAM",
                "PARCAM",
            ]
        if submode1 == "m1":
            include_patterns = [
                # "^._logits",
                ##
                # "^CAT(?:_RS)?(?:_AttnFrom)?_sum(?:_f\d+)?$",
                # "^CAT(?:_RS)?(?:_AttnFrom)?_sum",
                # "^(?:blocks__\d+__)?CAT(?:_s:[^_]+)?(?:_AttnFrom)?",
                "^CAT_s:sum_sum(?:_f\d+$|$)",
                f"^CAT_s:sum_AttnFrom_sum(_to{blocks_len - 1})?$",
                ##
                # "PosE",
                # "^IxG",
                "blocks__0__(CAT|PatchGrad|FGrad)_s:(sum|L2)",
                # "^CAT_s:sum_sum_FGrad_s:sum$",
                "^FGrad_s:sum_sum$",
                ##
                # "^MeanReLU__AttnGrad_Attn_ro_str50$",
                # "^MeanReLU__AttnGrad_Attn_sum",
                # "^(?:blocks__\d+__)MeanReLU__AttnGrad_Attn",
                "^MeanReLU__AttnGrad_Attn(?:$|_sum$|_ro)",
                # "^Mean(?:Abs)?__AttnGrad_Attn(?:$|_sum$|_ro)",
                ##
                f"^blocks__{blocks_len - 1}__CAT_s:(sum|RS)",
                f"^blocks__{blocks_len - 1}__.*CAM_s:sum",
                f"^blocks__{blocks_len - 1}__AttnWHeadGrad$",
                ##
                # "^MeanAttn_ro_str50$",
                # "^MeanAttn_sum$",
                f"^(?:blocks__{blocks_len - 1}__)?MeanAttn(?:$|_sum$|_ro)",
                ##
                "^Mean(?:Abs)?AttnGrad(?:$|_sum$|_ro)",
                ##
            ]
            exclude_patterns = [
                # "_to\d+(?:_|$)",
                "^Image(?:Grad|IxG)",
                "_s:(?!sum)[^_]*_AttnFrom",
                "blocks__.*_AttnFrom",
                "GRCAM",
                "XRCAM",
                "XARCAM",
                "PCAM",
                "PACAM",
                "PARCAM",
            ]

        elif submode1 == "m2":
            include_patterns = [
                f"^CAT_s:sum_AttnFrom_sum(_to{blocks_len - 1})?$",
                ##
                "blocks__0__(FGrad)_s:(sum|L2)",
                ##
                f"^blocks__{blocks_len - 1}__AttnWHeadGrad$",
                "^(GCAM|XA?CAM)_s:(sum|RS)_sum$",
                ##
            ]
            exclude_patterns = [
                # "_to\d+(?:_|$)",
                "^Image(?:Grad|IxG)",
                "_s:(?!sum)[^_]*_AttnFrom",
                "blocks__.*_AttnFrom",
                "GRCAM",
                "XRCAM",
                "XARCAM",
                "PCAM",
                "PACAM",
                "PARCAM",
            ]

        elif submode1 == "m3":
            include_patterns = [
                f"_sum_f{blocks_len // 2}($|_)",
                "^CAT_s:sum_sum_FGrad_s:sum$",
                "blocks__0_.*_AttnFrom",
                "AttnWHeadGrad_sum$",
            ]
            ic(include_patterns)

            exclude_patterns = [
                # "_to\d+(?:_|$)",
                "_s:(?!sum)",
                "GRCAM",
                "XRCAM",
                "XARCAM",
                "PCAM",
                "PACAM",
                "PARCAM",
            ]

        elif submode1 == "full":
            include_patterns = [
                f"_sum(_f{blocks_len // 2}($|_)|$)",
                # "^._logits",
                ##
                # "^CAT(?:_RS)?(?:_AttnFrom)?_sum(?:_f\d+)?$",
                # "^CAT(?:_RS)?(?:_AttnFrom)?_sum",
                # "^(?:blocks__\d+__)?CAT(?:_s:[^_]+)?(?:_AttnFrom)?",
                "CAT",
                ##
                "PosE",
                # "^IxG",
                "PatchGrad",
                "CAM",
                "FGrad",
                ##
                # "^MeanReLU__AttnGrad_Attn_ro_str50$",
                # "^MeanReLU__AttnGrad_Attn_sum",
                # "^(?:blocks__\d+__)MeanReLU__AttnGrad_Attn",
                "Mean(?:ReLU|Abs)?__AttnGrad_Attn",
                ##
                "AttnWHeadGrad",
                # "^blocks__\d+__AttnWHeadGrad$",
                # f"^blocks__{blocks_len - 1}__AttnWHeadGrad$",
                ##
                # "^MeanAttn_ro_str50$",
                # "^MeanAttn_sum$",
                "^(?:blocks__\d+__)?MeanAttn(?:$|_sum|_ro)",
                ##
                "^(?:blocks__\d+__)?Mean(?:Abs|ReLU)?AttnGrad(?:$|_sum|_ro)",
                ##
            ]
            exclude_patterns = [
                # "_to\d+(?:_|$)",
                "^Image(?:Grad|IxG)",
            ]

        elif submode1 in ["m5", "m6"]:
            include_patterns = [
                "rnd1",
                ##
                f"_sum(_f{blocks_len // 2}($|_)|$|_to|_FGrad_)",
                f"blocks__(0|{blocks_len - 1}|{(blocks_len // 2) + 1})__",
                ##
                "PosE",
                ##
                "Mean(?:ReLU|Abs)?__AttnGrad_Attn_ro",
                ##
                "^MeanAttn_ro",
                ##
                "^TokenTM",
                ##
            ]

            exclude_patterns = [
                "Image",  #: @different from seg m5
                ##
                r"^blocks__\d+__TokenTM",
                "AttnWHeadGrad",
                "PosE",
                "_s:L1",
                "GRCAM",
                "XCAM",
                "XRCAM",
                "XARCAM",
                "PCAM",
                "PACAM",
                "PRCAM",
                "PARCAM",
            ]

        elif submode1 == "cat_sum":
            include_patterns = [
                f"CAT.*_s:sum",
            ]
            exclude_patterns = []

        elif submode1 in ["m7"]:
            if run_check_completeness_mode_p or run_compute_completeness_mode_p:
                include_patterns = [
                    ##
                    # "^blocks__0__FGrad_s:sum$",
                    ##
                    "^NON_EXISTENT$",
                    #: We don't want anything included.
                ]

            else:
                include_patterns = m7_include_patterns

            exclude_patterns = m7_exclude_patterns

        elif submode1 in ["m8"]:
            include_patterns = m8_include_patterns
            exclude_patterns = m8_exclude_patterns

        my_attn_prepare_attribution_columns = partial(
            attn_prepare_attribution_columns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            # exclude_images_p=False,
            exclude_images_p=True,
            # keep_as_is_patterns=[
            #     r"(?i)blocks__\d+__Attn",
            # ],
            # verbose=False,
            # verbose=True,
            # corr_mode="Kendall",
        )

        with DynamicVariables(
            decompv.utils.dynamic_obj,
            dsmap_predict_masked_remove_cols=dsmap_predict_masked_remove_cols,
        ):
            attn_compute_global(
                # name=mode,
                name=f"{compact_gbrand}",
                attn_transforms=attn_transforms,
                attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
                add_random_p=True,
            )

    else:
        raise ValueError(f"compute_attention: unsupported mode {mode}")

##
