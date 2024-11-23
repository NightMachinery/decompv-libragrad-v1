#!/usr/bin/env python3

import pynight
from pynight.common_debugging import reload_modules
import re
from IPython import embed

##
import decompv
from decompv.x.bootstrap import *
from decompv.x.ds.main import *
from pynight.common_icecream import ic

##
my_tds_indexed = tds_indexed_imagenet
my_tds_torch_cpu = tds_torch_cpu_imagenet
my_tds_patches = tds_patches_imagenet

bias_token_p = True

model_patch_info = patch_info_from_name(model_name, bias_token_p=bias_token_p)
ic(model_patch_info)

##
decomposition_config = DecompositionConfig(
    device=device,
    attributions_aggregation_strategy="vector",
    save_intermediate_p=True,
    name="DecompV",
)

tds_decompv = my_tds_patches.transform(
    partial_dynamic(
        partial(
            dsmap_decompv_attributions,
            model=model,
            decomposition_config=decomposition_config,
            raw_attention_store_mode="full",
            store_cls_only_p=True,
            store_perf_p=False,
            attr_name_mode="v1",
        ),
        dynamic_dict=decomposition.dynamic_obj,
        print_diag_enabled_groups=lst_filter_out(
            decomposition.dynamic_obj.print_diag_enabled_groups,
            [
                "warning.nondecomposed",
                "check_attributions",
            ],
        ),
    ),
)

tds_scalar = tds_decompv
# tds_scalar = tds_scalar.transform(decompv.x.ds.utils.transform_softmax)
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
##

batch_example = tds_scalar[:2]
print(torch_shape_get(batch_example))
##
embed()
