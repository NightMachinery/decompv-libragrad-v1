## * Load First
import os

os.environ["HF_EVALUATE_OFFLINE"] = os.environ.get("HF_EVALUATE_OFFLINE", "1")

from pynight.common_debugging2 import ipdb_enable

ipdb_enable()

from pynight.common_bells import (
    bell_call_remote,
)

from IPython import embed
##
from decompv.constants import *
import os
import pathlib

from pynight.common_shell import (
    str_falsey_to_none,
    getenv2,
)
from pynight.common_icecream import ic

HOME = pathlib.Path.home()
##
tlg_me = os.environ.get("DECOMPV_TLG_LOG_ID", -1001948197369)
##
import os
from os import getenv
import pprint
import argparse
from brish import bool_from_str

from pynight.common_files import (
    rm,
    mkdir,
    list_children,
    open_file,
)
from pynight.common_hosts import mb2_p

##
run_check_completeness_mode_p = bool_from_str(
    getenv(
        "DECOMPV_RUN_COMPLETENESS_P",
        default=False,
    ),
)

run_compute_completeness_mode_p = bool_from_str(
    getenv(
        "DECOMPV_RUN_COMPUTE_COMPLETENESS_P",
        default=False,
    ),
)

global_force_pruned_mode_p = bool_from_str(
    getenv(
        "DECOMPV_FORCE_PRUNE_P",
        default=False,
    ),
)

qual_prototype_p = bool_from_str(
    getenv(
        "DECOMPV_QUAL_PROTOTYPE_P",
        default=False,
    ),
)

run_skip_main_p = bool_from_str(
    getenv(
        "DECOMPV_RUN_SKIP_MAIN_P",
        default=False,
    ),
)

run_postprocess_deus_p = bool_from_str(
    getenv(
        "DECOMPV_RUN_POSTPROCESS_DEUS_P",
        default=False,
    ),
)

run_deus_p = bool_from_str(
    getenv(
        "DECOMPV_RUN_DEUS_P",
        default=False,
    ),
)
run_deus_p = run_deus_p or qual_prototype_p or run_check_completeness_mode_p
##
run_force_submode = getenv(
    "DECOMPV_RUN_FORCE_SUBMODE",
    default=None,
)
##
qual_submode = getenv(
    "DECOMPV_QUAL_SUBMODE",
    default=None,
)
##
seg_dataset_end_global = getenv(
    "DECOMPV_SEG_DATASET_END",
    default=None,
)
##
ARTIFACTS_ROOT = os.path.join(os.environ["HOME"], "code", "decompv_artifacts")
path = ARTIFACTS_ROOT

mkdir(ARTIFACTS_ROOT)

SHARED_ROOT = "/opt/decompv"
_local_shared_root = os.path.join(HOME, "dv")
if os.path.exists(_local_shared_root):
    assert not os.path.exists(
        SHARED_ROOT
    ), f"Local shared root '{_local_shared_root}' found, but '{SHARED_ROOT}' also exists. This is likely an error. Please remove one of them."

    print("Auto-detected usage of local shared root")

    SHARED_ROOT = _local_shared_root

SHARED_ROOT = getenv(
    "DECOMPV_SHARED_ROOT",
    default=SHARED_ROOT,
)

DATASET_NAME = "ImageNetVal"
DATASET_NAME = getenv(
    "DECOMPV_DATASET_NAME",
    default=DATASET_NAME,
)


def imagenet_p(
    *,
    dataset_name=None,
):
    if dataset_name is None:
        dataset_name = DATASET_NAME

    if dataset_name in [
        "ImageNetVal",
        "ImageNet-Hard",
    ]:
        return True


##
SEG_DATASET_NAME = "ImageNetS"
SEG_DATASET_NAME = getenv(
    "DECOMPV_SEG_DATASET_NAME",
    default=SEG_DATASET_NAME,
)
##
model_name = getenv(
    "DECOMPV_MODEL_NAME",
    "vit_base_patch16_clip_224.openai_ft_in12k_in1k",
)
##
DS_ROOT = f"{SHARED_ROOT}/datasets"
DS_ROOT = getenv(
    "DECOMPV_DS_ROOT",
    default=DS_ROOT,
)
if DATASET_NAME != "ImageNetVal":
    #: @backcompat
    DS_ROOT += f"/{DATASET_NAME}"

DS_MODEL_ROOT = f"{DS_ROOT}/{model_name}"
##
DONE_EXPERIMENTS = f"{HOME}/done_experiments.jsonl"
DONE_EXPERIMENTS = getenv(
    "DECOMPV_DONE_EXPERIMENTS",
    default=DONE_EXPERIMENTS,
)
DONE_EXPERIMENTS_TO_MERGE = f"{HOME}/done_experiments_to_merge.jsonl"
DONE_EXPERIMENTS_TO_MERGE = getenv(
    "DECOMPV_DONE_EXPERIMENTS_TO_MERGE",
    default=DONE_EXPERIMENTS_TO_MERGE,
)

DONE_EXPERIMENTS_REMOVED = f"{HOME}/done_experiments_removed.jsonl"
DONE_EXPERIMENTS_REMOVED = getenv(
    "DECOMPV_DONE_EXPERIMENTS_REMOVED",
    default=DONE_EXPERIMENTS_REMOVED,
)
##
model_load_p = bool_from_str(
    getenv(
        "DECOMPV_MODEL_LOAD_P",
        # default=False,
        default=True,
    ),
)
dataset_load_p = bool_from_str(
    getenv(
        "DECOMPV_DATASET_LOAD_P",
        # default=False,
        default=True,
    ),
)

save_attr_mode = getenv(
    "DECOMPV_SAVE_ATTR_MODE",
    # default="ALL",
    default=None,
)

if save_attr_mode == "ALL":
    dsmap_predict_masked_remove_cols = None
else:
    dsmap_predict_masked_remove_cols = "ALL"

if mb2_p():
    METRICS_ROOT = f"{HOME}/code/uni/FairGrad_Metrics/metrics"
    # METRICS_ROOT = f"{HOME}/code/uni/DecompV-Notebooks/metrics"
    assert os.path.exists(METRICS_ROOT), f"Metrics root does not exist: {METRICS_ROOT}"

else:
    METRICS_ROOT = f"{SHARED_ROOT}/metrics"

METRICS_ROOT = getenv(
    "DECOMPV_METRICS_ROOT",
    default=METRICS_ROOT,
)
SEG_METRICS_ROOT = f"{METRICS_ROOT}/s"
#: The seg evals have a separate dataset from the faithfulnes evals, so we set SEG_METRICS_ROOT before adding DATASET_NAME.
if DATASET_NAME != "ImageNetVal":
    METRICS_ROOT += f"/{DATASET_NAME}"

if SEG_DATASET_NAME != "ImageNetS":
    SEG_METRICS_ROOT += f"/{SEG_DATASET_NAME}"
MODEL_SEG_METRICS_ROOT = f"{SEG_METRICS_ROOT}/{model_name}"

CLS_METRICS_ROOT = f"{METRICS_ROOT}/cls_v3"
CLS_METRICS_ROOT = getenv(
    "DECOMPV_CLS_METRICS_ROOT",
    default=CLS_METRICS_ROOT,
)

COMPLETENESS_METRICS_ROOT = f"{METRICS_ROOT}/CE"
COMPLETENESS_METRICS_ROOT = getenv(
    "DECOMPV_COMPLETENESS_METRICS_ROOT",
    default=COMPLETENESS_METRICS_ROOT,
)

if DATASET_NAME == "ImageNetVal":
    DS_INDEXED_PATH = f"{DS_ROOT}/ImageNet1K-val_indexed"
    #: Since we are storing ImageNetVal in the root without creating a subdirectory for it, we should store the name of the dataset in the tail.
else:
    DS_INDEXED_PATH = f"{DS_ROOT}/indexed"
    #: DS_ROOT already contains the name of the dataset

DS_INDEXED_PATH = getenv(
    "DECOMPV_DS_INDEXED",
    default=DS_INDEXED_PATH,
)

if DATASET_NAME == "ImageNetVal":
    DS_PATCHIFIED_PATH = f"{DS_ROOT}/ImageNet1K-val_patchified"
    #: Since we are storing ImageNetVal in the root without creating a subdirectory for it, we should store the name of the dataset in the tail.
else:
    DS_PATCHIFIED_PATH = f"{DS_ROOT}/patchified"
    #: DS_ROOT already contains the name of the dataset

DS_PATCHIFIED_PATH = getenv(
    "DECOMPV_DS_PATCHIFIED",
    default=DS_PATCHIFIED_PATH,
)

##
benchmark_mode_p = bool_from_str(
    getenv(
        "DECOMPV_BENCHMARK_P",
        default=False,
    )
)


##
###
def metrics_target_get():
    if not hasattr(metrics_target_get, "first_p"):
        #: @hack static variable by using the function itself as the object
        metrics_target_get.first_p = False
        first_p = True

    else:
        first_p = False

    if first_p:
        target_mode = (
            getenv(
                "DECOMPV_METRICS_TARGET",
                default=None,
            )
            or ground_truth_mode_cst
        )
        ic(target_mode)

        metrics_target_get.target_mode = target_mode

    return metrics_target_get.target_mode


## * Imports
import pynight
from pynight.common_debugging import reload_modules
import re

# reload_modules(re.compile("^(timm|pynight)"))
##
from pytictoc import TicToc

tictoc = TicToc()

from rich import inspect


def h(x, **kwargs):
    return inspect(x, help=True, **kwargs)


##
#########
