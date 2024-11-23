from decompv.early_boot import *
import evaluate
from pynight.common_json import (
    json_save,
    json_load,
    json_save_update,
)
from pynight.common_clip import (
    open_clip_sep,
)
from pynight.common_model_name import (
    model_name_clip_p,
    model_name_eva2_p,
)
from brish import *
### * constants
from decompv.x.gbrand import gbrands_from

globals().update(
    gbrands_from(
        model_name=model_name,
        # configure_p=False,
        configure_p=True,
    )
)
print(f"compact_gbrand: {compact_gbrand}")
##
MODEL_CLS_METRICS_ROOT = f"{CLS_METRICS_ROOT}/{model_name}/{compact_gbrand}"
MODEL_CLS_METRICS_ROOT = getenv(
    "DECOMPV_MODEL_CLS_METRICS_ROOT",
    default=MODEL_CLS_METRICS_ROOT,
)

MODEL_COMPLETENESS_METRICS_ROOT = f"{COMPLETENESS_METRICS_ROOT}/{model_name}/{compact_gbrand}"
MODEL_COMPLETENESS_METRICS_ROOT = getenv(
    "DECOMPV_MODEL_COMPLETENESS_METRICS_ROOT",
    default=MODEL_COMPLETENESS_METRICS_ROOT,
)
##
## ** Load the Rest
from functools import partial
import time
import gc
import socket
import itertools

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import captum.attr

import torch
import torch.nn

nn = torch.nn
import torch.nn.functional as F

import numpy

np = numpy

import timm

import torchvision.transforms
from torchvision.transforms import ToPILImage

import timm.models.decomposition
import timm.models.decomposition as decomposition
from timm.models.decomposition import (
    fair_model_get,
    configure_gradient_modes,
    check_attributions,
    image_url2np,
    image_from_url,
    image_batch_from_urls,
    DecompositionConfig,
    attributions_normify,
    overlay_colored_grid,
    attributions_n_get,
    attributions_scalarify,
    attributions_scalarify_v2,
    mean_normalize_last_dim,
    vis_normatt_vs_rawatt,
    model_transform_get,
    image_from_url,
    entrypoint_normatt,
    entrypoint_decompv,
    MLPDecomposed,
    SoftmaxDecomposed,
    check_attributions_v2,
    multiply_attributions,
    QuickGELUDecomposed,
    GELUDecomposed,
    lowpass_filter,
    threshold_filter,
    simple_obj,
    simple_obj_update,
    vis_attr,
    attributions_distribute_biascls,
    h_exponent,
    create_model,
    print_diag,
    print_diag_sep,
)

from timm.models.decomposition_x import (
    model_decompose,
)

from timm.models.clip_wrapper import (
    TimmCLIPModel,
)

from pynight.common_numpy import (
    image_url2pil,
)

from pynight.common_timm import (
    model_name_get,
    patch_info_from_name,
)

from pynight.common_benchmark import (
    timed,
    Timed,
)

from pynight.common_dynamic import (
    DynamicVariables,
    dynamic_set,
    dynamic_get,
)

from pynight.common_datasets import (
    TransformedDataset,
    ConcatenatedTransformedDataset,
    dataset_cache_filenames,
    dataset_push_from_disk_to_hub,
    mapconcat,
    save_and_delete,
)

import pynight.common_telegram as common_telegram
import pynight.common_dict
from pynight.common_dict import BatchedDict
from pynight.common_functional import fn_name
import requests

import os
import sys

import humanize
import datasets

import pynight.common_torch as common_torch
from pynight.common_torch import (
    img_tensor_show,
    torch_shape_get,
    no_grad_maybe,
    model_device_get,
    drop_tokens,
    seed_set,
)

from pynight.common_files import mkdir
from pynight.common_datetime import datetime_dir_name
from pynight.common_regex import rget
from pynight.common_iterable import (
    dir_grep,
    dg,
    IndexableList,
    to_iterable,
)

# from transformers.utils import logging as hf_logging
# hf_logging.disable_progress_bar
##
exact_reproducibility = getenv(
    "DECOMPV_EREPRO",
    default=False,
)
if exact_reproducibility == "y":
    exact_reproducibility = True
else:
    exact_reproducibility = False

seed_set(
    81,
    cuda_deterministic=exact_reproducibility,
)

force_cpu_p = bool_from_str(
    getenv(
        "DECOMPV_FORCE_CPU_P",
        default=False,
    ),
)
if not force_cpu_p and torch.cuda.is_available():
    device = "cuda"

else:
    device = "cpu"

ic(device)
##
print("decompv/bootstrap finished")
