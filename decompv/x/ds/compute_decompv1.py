##
import argparse
from decompv.x.bootstrap import (
    dsmap_predict_masked_remove_cols,
)
from decompv.x.ds.main import *


## * Utils
def decompv_compute_global_v2(
    *,
    name,
    decomposition_config,
    attn_prepare_attribution_columns,
    transform_only_p=True,
    dataset_start=None,
    dataset_end=None,
    tds_patches: TransformedDataset = None,
    add_random_p=False,  #: only True for the first dataset
    output_precision="bfloat16",
    #: @globals
    #: model,
    #: model_name,
    compute_gen_opts=None,
    mask_p=True,
    batch_size=None,
    **decompv_opts,
):
    if batch_size is None:
        batch_size = global_batch_size

    if compute_gen_opts is None:
        compute_gen_opts = dict()

    if dataset_start is None:
        dataset_start = dataset_start_global

    if dataset_end is None:
        dataset_end = dataset_end_global

    if tds_patches is None:
        tds_patches = tds_patches_imagenet

    bias_token_p = False
    model_patch_info = patch_info_from_name(model_name, bias_token_p=bias_token_p)

    tds_decompv = tds_decompv_get(
        decomposition_config=decomposition_config,
        model=model,
        model_name=model_name,
        tds_patches=tds_patches,
        **decompv_opts,
    )

    tds_decompv = tds_decompv.transform(
        partial(
            attn_prepare_attribution_columns,
            model=model,
        ),
    )

    if mask_p:
        tds_masked = masker_entrypoint(
            tds_scalar=tds_decompv,
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
    else:
        tds_masked = tds_decompv

        computer = partial(
            compute_tds_identity,
            # output_precision=output_precision,
        )

    return dataset_compute_gen(
        # name="D",
        # name="tmp",
        name=name,
        computer=computer,
        batch_size=batch_size,
        tds_patches=tds_masked,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        model=model,
        # load_from_cache_file=False,
        **compute_gen_opts,
    )


##
if __name__ == "__main__":
    metrics_target = metrics_target_get()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--agg-strategy", help="attributions_aggregation_strategy", default="vector"
    )
    parser.add_argument("--mode", help="name", default="DecompV_old")
    args = parser.parse_args()
    mode = args.mode
    attributions_aggregation_strategy = args.agg_strategy

    blocks_len = len(model.blocks)

    if mode == "DecompV_old":
        name = "DecompV"

        decomposition_config = DecompositionConfig(
            device=device,
            attributions_aggregation_strategy=attributions_aggregation_strategy,
            name=name,
        )

        decompv_compute_global(
            decomposition_config=decomposition_config,
        )
    elif mode == "DecompFast":
        name = mode
        metadata_name = ""

        decomposition_config = DecompositionConfig(
            device=device,
            attributions_aggregation_strategy=f"vector_f{blocks_len - 1}",
            mlp_decompose_p=False,
            name=name,
        )

        my_attn_prepare_attribution_columns = partial(
            attn_prepare_attribution_columns,
            # filter_fn=my_filter,
            keep_as_is_patterns=[
                # r"^attributions_s_logit(?:_|$)",
            ],
            # exclude_patterns=[],
        )

        with DynamicVariables(
            decompv.utils.dynamic_obj,
            dsmap_predict_masked_remove_cols=dsmap_predict_masked_remove_cols,
        ):
            decompv_compute_global_v2(
                name=name,
                decomposition_config=decomposition_config,
                attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
                add_random_p=False,
                compute_gen_opts=dict(
                    metadata_name=metadata_name,
                ),
            )
    elif mode == "DecompV":
        name = mode
        # if attributions_aggregation_strategy != "vector":
        #     name += f"_{attributions_aggregation_strategy}"
        metadata_name = f"{name}_{attributions_aggregation_strategy}"

        decomposition_config = DecompositionConfig(
            device=device,
            attributions_aggregation_strategy=attributions_aggregation_strategy,
            name=name,
        )

        my_attn_prepare_attribution_columns = partial(
            attn_prepare_attribution_columns,
            # filter_fn=my_filter,
            keep_as_is_patterns=[
                # r"^attributions_s_logit(?:_|$)",
            ],
            # exclude_patterns=[],
        )

        with DynamicVariables(
            decompv.utils.dynamic_obj,
            dsmap_predict_masked_remove_cols=dsmap_predict_masked_remove_cols,
        ):
            decompv_compute_global_v2(
                name=name,
                decomposition_config=decomposition_config,
                attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
                add_random_p=False,
                compute_gen_opts=dict(
                    metadata_name=metadata_name,
                ),
            )
    elif mode == "DecompC":
        name = mode
        metadata_name = f"{name}_{attributions_aggregation_strategy}"

        decomposition_config = DecompositionConfig(
            device=device,
            attributions_aggregation_strategy=attributions_aggregation_strategy,
            name=name,
        )

        my_attn_prepare_attribution_columns = partial(
            attn_prepare_attribution_columns,
            # filter_fn=my_filter,
            keep_as_is_patterns=[
                # r"^attributions_s_logit(?:_|$)",
            ],
            # exclude_patterns=[],
        )

        with DynamicVariables(
            decompv.utils.dynamic_obj,
            dsmap_predict_masked_remove_cols=dsmap_predict_masked_remove_cols,
        ):
            after_transforms = []
            for my_attributions_v_key in [
                "autodetect",
            ]:
                for topk in [
                    1,
                    2,
                    3,
                    4,
                    5,
                    10,
                    20,
                    50,
                    100,
                    250,
                    500,
                    750,
                ]:
                    for delta_scale in [
                        0.25,
                        0.5,
                        0.75,
                        0.8,
                        0.85,
                        0.9,
                        0.95,
                        1,
                        1.05,
                        1.25,
                        1.5,
                        2.5,
                        5,
                        10,
                        20,
                        50,
                        # 100,
                    ]:
                        after_transforms += [
                            partial(
                                target_name2fn[metrics_target],
                                delta_scale=delta_scale,
                                topk=topk,
                                attributions_v_key=my_attributions_v_key,
                            )
                        ]

            if metrics_target == ground_truth_mode_cst:
                name_dir = name
            else:
                name_dir = f"{name}_{metrics_target}"

            decompv_compute_global_v2(
                name=name_dir,
                decomposition_config=decomposition_config,
                attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
                add_random_p=False,
                compute_gen_opts=dict(
                    metadata_name=metadata_name,
                ),
                after_transforms2=after_transforms,
            )
    else:
        raise ValueError(f"unsupported mode {mode}")
    ##
