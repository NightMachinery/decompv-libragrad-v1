import functools
import math
import numpy as np
import pandas
import pandas as pd
from decompv.early_boot import (
    run_check_completeness_mode_p,
)
from decompv.x.bootstrap import *
import pprint
from IPython import embed
import decompv.utils
from decompv.utils import (
    transform_torch_save,
    save_tds_torch,
    attr_name_official_get,
    h_attr_sort_key,
    transform_cpu,
    batch_size_for,
)
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from contextlib import nullcontext
from pynight.common_icecream import ic
import torch
from pynight.common_dynamic import (
    partial_dynamic,
    DynamicVariables,
    dynamic_get,
    dynamic_set,
)
from pynight.common_faith import (
    cls_metrics_get,
    compute_aopc_lodds,
)
from pynight.common_tqdm import (
    tqdm,
)
from pynight.common_tqdm2 import (
    redirect_print_to_tqdm,
)
from tqdm import tqdm as tqdm_orig
import timm.models.decomposition
import timm.models.decomposition as decomposition
from timm.models.decomposition import (
    sum_attributions,
    NightSoftmax,
    normalize_to_unit_vector,
)
import glob
from typing import List
from brish import bool_from_str
import sys
import os
from os import getenv
import json
import datasets
from pynight.common_iterable import (
    to_iterable,
    IndexableList,
    list_dup_rm,
)
from pynight.common_debugging import fn_name_current
from pynight.common_datasets import dataset_cache_filenames
from pynight.common_benchmark import (
    timed,
    Timed,
)
from pynight.common_files import (
    rm,
    mkdir,
    list_children,
    open_file,
)
from pynight.common_dict import (
    simple_obj,
    key_del,
)
from pynight.common_torch import (
    torch_device_name_get,
    model_device_get,
    host_info_get,
    TorchBenchmarker,
    rank_tensor,
    unique_first_indices,
    scale_patch_to_pixel,
    torch_gpu_empty_cache,
    get_k_predicted_indices,
    get_first_predicted_indices,
    get_second_predicted_indices,
)
from pynight.common_package import packages_commit_get
from pynight.common_sort import version_sort_key
from pynight.common_iterable import (
    lst_filter_out,
)
from pynight.common_torch import (
    drop_mask,
    drop_topk,
    keep_topk,
    drop_from_dim,
)

# from pynight.common_timm import (
#     patch_info_from_name,
# )

from torchmetrics.functional.regression import kendall_rank_corrcoef
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


## * Constants
patch_grad_raw_name = "PatchGrad_s:raw"
## * Channel Mixing
channel_mixers = set(
    [
        "sum",
        "RS",
        # "NS",
        # "QS25",
        # "QS50",
        # "QS75",
        "L1",
        "L2",
        # "L4",
        # "LInf",
    ]
)
channel_mixers_positive = set(
    [
        "sum",
        # "QS25",
        # "QS50",
        # "QS75",
        "L2",
        # "L4",
        # "LInf",
    ]
)
# channel_mixers_essential = set(
#     [
#         "sum",
#         "RS",
#         # "NS",
#         # "QS25",
#         # "QS50",
#         # "QS75",
#         "L1",
#         "L2",
#         # "L4",
#         # "LInf",
#     ]
# )
channel_mixers_no_sum = channel_mixers - set(["sum"])


## * Prepare Inputs
def label_natural_get():
    import decompv.x.imagenet

    # assert imagenet_p(), "The function label_natural_get returns ImageNet labels, but the current dataset is not an ImageNet dataset."

    # return decompv.x.imagenet.IMAGENET_1k_LABELS

    return decompv.x.imagenet.imagenet_labels_human


def dsmap_input_filter(batch):
    return [img.mode == "RGB" for img in (batch["image"])]


def transform_input_prepare(
    batch,
    model,
    accept_gray_p=False,
    device=None,
):
    ##
    #: @duplicateCode/3659dfc93b398f1ce73fe835dac21b82
    if device is None:
        device = model_device_get(model)

    elif device == "NA":
        device = None
    ##

    image_batch = image_batch_from_urls(
        model=model,
        urls=batch["image"],
        device=device,
        accept_gray_p=accept_gray_p,
    )

    del batch["image"]

    batch["image_array"] = image_batch.image_batch_cpu
    if image_batch.image_batch_dv is not None:
        batch["image_array_dv"] = image_batch.image_batch_dv

    batch["image_natural"] = image_batch.image_natural

    ##
    if "label" in batch:
        if imagenet_p():
            label_natural = label_natural_get()[batch["label"]]
            batch["label_natural"] = label_natural

        label_cpu = torch.tensor(batch["label"])
        batch["label_cpu"] = label_cpu
        # del batch['label']

        if device:
            label_dv = label_cpu.to(device)
            batch["label_dv"] = label_dv
        ##

    return batch


def transform_pixels2patches(
    batch,
    model,
    grad_p=False,
):
    batch = dict(batch)

    batch_size = len(batch["image_array"])

    image_batch_dv = batch["image_array_dv"]

    if grad_p:
        image_batch_dv.requires_grad_(True)
        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            model.pos_embed.requires_grad_(True)
            model.pos_embed.retain_grad()

    with no_grad_maybe(not grad_p):
        patches_batch = model.pixels2patches(image_batch_dv)
        # patches_batch_cpu = patches_batch.cpu()

    if not grad_p:
        patches_batch = patches_batch.detach()

    batch["patches_dv"] = patches_batch

    return batch


def transform_image2patches(
    batch,
    model,
    device=None,
    **kwargs,
):
    batch = transform_input_prepare(
        batch,
        model=model,
        device=device,
    )
    batch = transform_pixels2patches(
        batch,
        model=model,
        **kwargs,
    )

    return batch


## * Captum Attributions
def dsmap_captum_attributions(
    batch, batch_transformed, model, captum_attributors, store_perf_p=True
):
    batch_size = len(batch["id"])
    # ic(torch_shape_get(batch))

    # batch_transformed = transform_image2patches(dict(batch), model=model)

    # ic(torch_shape_get(batch_transformed))

    device = model_device_get(model)

    label_dv = batch_transformed["label_dv"]

    patches_batch = batch_transformed["patches_dv"]
    # patches_batch = patches_batch.to(device)
    patches_batch.requires_grad_(True)  # @?

    additional_forward_args = None
    for attributor in captum_attributors:
        if not attributor.get("enabled_p", True):
            continue

        attributor_obj = attributor["obj"]
        if "kwargs" in attributor:
            attributor_kwargs = attributor["kwargs"]
        else:
            attributor_kwargs = dict()

        model.zero_grad()  #: @redundant?

        #: This approach for profiling the memory used didn't quite work.
        # gc.collect()
        # torch.cuda.empty_cache()
        # start_memory = torch.cuda.memory_allocated(device)

        start_time = time.time()  #: the current time in seconds since the Epoch
        attributions = attributor_obj.attribute(
            inputs=patches_batch,
            target=label_dv,
            additional_forward_args=additional_forward_args,
            **attributor_kwargs,
        )
        end_time = time.time()
        time_taken = end_time - start_time

        # end_memory = torch.cuda.memory_allocated(device)
        # memory_used = end_memory - start_memory

        attributions = attributions.detach().cpu()
        # ic(torch_shape_get(attributions))

        if "name" in attributor:
            attributor_name = attributor["name"]

            batch[f"attributions_{attributor_name}"] = attributions
            # batch[attributor_name] = [attributions[i] for i in range(attributions.shape[0])]

            print_diag(
                f"attributor_name={attributor_name}, time_taken={time_taken}",
                group="time",
            )

            # batch[f"time_{attributor_name}_batch{batch_size}"] = [time_taken] * batch_size
            # batch[f"time_{attributor_name}"] = [(time_taken, batch_size)] * batch_size
            if store_perf_p:
                batch[f"perf_{attributor_name}"] = [
                    dict(time_taken=time_taken, batch_size=batch_size)
                ] * batch_size
            #: We are storing the time it took to process the whole batch for each element.
            #: I think this is better than dividing it, as the whole batch should be processed more or less in parallel:
            # batch[f"time_{attributor_name}"] = [time_taken / batch_size] * batch_size

            # ic(memory_used)
            # batch[f"memory_{attributor_name}"] = memory_used

    model.zero_grad()

    # ic(type(batch), torch_shape_get(vars(batch), type_only_p=True))
    return batch


def h_attr_captum_compute(
    *,
    captum_attributors,
    tds_patches,
    batch_size,
    model,
    mapconcat_opts=None,
    **dummy,
):
    if mapconcat_opts is None:
        mapconcat_opts = dict()

    dataset_after = mapconcat(
        tds_patches.dataset,
        tds_patches.fn_with_transforms(
            partial(
                dsmap_captum_attributions,
                model=model,
                captum_attributors=captum_attributors,
                # store_perf_p=False,
            ),
        ),
        unchanged_keep_columns=["id"],
        batched=True,
        batch_size=batch_size,
        **mapconcat_opts,
    )

    dataset_after.set_format(
        "torch",
        columns=[
            col
            for col in dataset_after.column_names
            if not (col.startswith("perf_") or col == "id")
        ],
        output_all_columns=True,
    )

    return simple_obj(
        dataset_all=dataset_after,
    )


##
def dataset_compute_gen(
    *,
    name,
    metadata_name="",
    computer,
    tds_patches,
    dataset_start,
    dataset_end,
    model,
    print_diag_file="auto",
    # new_fingerprint='auto',
    new_fingerprint=None,
    dataset_path="auto",
    #: set =dataset_path= to None to disable saving
    batch_size=1000,
    measure_carbon_p=True,
    delete_p=True,
    keep_in_memory="auto",
    return_p=False,
    save_p=True,
    **mapconcat_opts,
):
    model_name = model_name_get(model)
    device = model_device_get(model)

    if benchmark_mode_p:
        dataset_start = 0
        dataset_end = 10
        batch_size = 10

        if save_p:
            save_p = False
            print("disabled save_p because benchmark_mode_p")

    if dataset_end <= 0:
        dataset_end += len(tds_patches)
        #: so 0 means all of the dataset

    dataset_end = min(dataset_end, len(tds_patches))
    tds_patches = tds_patches.select(range(dataset_start, dataset_end))

    metadata = host_info_get()
    metadata["name"] = name
    metadata.update(
        gradient_mode_brand=gradient_mode_brand,
        softmax_mode=softmax_mode,
        backward_softmax_p=decomposition.dynamic_obj.backward_softmax_p,
        attention_gradient_mode=decomposition.dynamic_obj.attention_gradient_mode,
        layer_norm_gradient_mode=decomposition.dynamic_obj.layer_norm_gradient_mode,
        gelu_gradient_mode=decomposition.dynamic_obj.gelu_gradient_mode,
    )

    metadata["cuda_gc_mode"] = dynamic_get(
        timm.models.decomposition.dynamic_vars, "cuda_gc_mode"
    )

    dataset_len = len(tds_patches)
    assert dataset_len == (
        dataset_end - dataset_start
    ), "dataset_start/dataset_end invalid"

    metadata["len"] = dataset_len
    metadata["dataset_start"] = dataset_start
    metadata["dataset_end"] = dataset_end

    metadata["model_name"] = model_name
    metadata["batch_size"] = batch_size

    if new_fingerprint == "auto":
        new_fingerprint = f"{name}_{model_name}"

    if dataset_path == "auto":
        dataset_path = f"{DS_MODEL_ROOT}/{name}/"

    if not dataset_path:
        print("dataset_compute_gen: no dataset_path provided, NOT saving")

    if print_diag_file == "auto":
        if dataset_path:
            print_diag_file = f"{dataset_path}/log_diag"
            rm(print_diag_file)  #: remove old logs
        else:
            print_diag_file = "log"

    if keep_in_memory == "auto":
        keep_in_memory = bool_from_str(
            getenv(
                "DECOMPV_KEEP_IN_MEMORY",
                default="y",
            ),
        )

    assert new_fingerprint != "auto"
    assert dataset_path != "auto"

    mapconcat_opts["new_fingerprint"] = new_fingerprint
    mapconcat_opts["keep_in_memory"] = keep_in_memory
    metadata["keep_in_memory"] = keep_in_memory
    ic(metadata)

    metadata["pkgs"] = packages_commit_get(
        [
            "captum",
            "timm",
            "decompv",
            "pynight",
        ],
        import_p=True,
    )

    dataset_all = None
    batch_all = None
    with DynamicVariables(
        decomposition.dynamic_vars,
        print_diag_file=print_diag_file,
    ), TorchBenchmarker(
        output_dict=metadata,
        output_dir=dataset_path,
        device=device,
    ):
        result = computer(
            batch_size=batch_size,
            model=model,
            tds_patches=tds_patches,
            mapconcat_opts=mapconcat_opts,
            dataset_path=dataset_path,
        )

        if dataset_path:
            metrics_target = metrics_target_get()
            tds_path = f"{dataset_path}/{metrics_target}"
            # tds_path = f"{dataset_path}/T"

            #: TransformedDatasets save each batch separately, so no need to indicate the start/end IDs in the directory name.

            mkdir(tds_path)

            if "tds_after" in result and result.tds_after:
                with DynamicVariables(
                    decompv.utils.dynamic_obj,
                    tds_path=tds_path,
                ):
                    if run_compute_completeness_mode_p:
                        tqdm_name = f"Completeness: {tds_path}"

                    elif run_check_completeness_mode_p:
                        tqdm_name = f"Checking Completeness: {tds_path}"

                    else:
                        tqdm_name = f"Faith: {tds_path}"

                    #: Note that TransformedDatasets are lazy so without saving them no actual processing takes place.
                    save_result = save_tds_torch(
                        result.tds_after,
                        output_dir=tds_path,
                        batch_size=batch_size,
                        tqdm=tqdm,
                        name=name,
                        return_p=return_p,
                        save_p=save_p,
                        tqdm_name=tqdm_name,
                    )

                if return_p:
                    batch_all = save_result.batch_all
                if "metadata" in save_result:
                    metadata.update(save_result.metadata)

    if "metadata" in result:
        metadata.update(result.metadata)

    if "dataset_all" in result and result.dataset_all is not None:
        assert (
            dataset_all is None
        ), f"two 'dataset_all' values supplied: {dataset_all}, {result.dataset_all}"

        dataset_all = result.dataset_all

    dataset_afters = None
    if "dataset_afters" not in result:
        if dataset_all:
            dataset_afters = {"": result.dataset_all}
    else:
        dataset_afters = result.dataset_afters

    if dataset_path:
        dataset_path2 = f"{dataset_path}/{dataset_start}_{dataset_end}"
        mkdir(dataset_path2)

        metadata_dir = f"{dataset_path2}/{metadata_name}/"
        mkdir(metadata_dir)
        with open_file(
            f"{metadata_dir}/night_metadata.json",
            "w",
            exists="increment_number",
        ) as json_file:
            json.dump(metadata, json_file)

        if dataset_afters:
            to_delete = []
            for name, dataset in dataset_afters.items():
                dataset_path2_current = dataset_path2
                if name:
                    dataset_path2_current = f"{dataset_path2_current}/{name}"
                save_res = save_and_delete(
                    dataset,
                    dataset_path2_current,
                    delete_p=False,
                    max_shard_size=999999999999,
                )
                dataset_afters[name] = save_res.dataset
                to_delete += save_res.to_delete
                del save_res

            to_delete += dataset_cache_filenames(result.dataset_all, cache_only_p=True)

            if delete_p:
                #: These datasets can share cache files, so we need to delete the caches after saving all the datasets.
                for path in set(to_delete):
                    rm_res = rm(path)
                    if rm_res.retcode != 1:  #: 1 : non-existent
                        print(rm_res.msg)

    return simple_obj(
        dataset_all=dataset_all,
        #: dataset_all can have its cache files deleted if saving is enabled and delete_p!
        datasets=dataset_afters,
        batch_all=batch_all,
        metadata=metadata,
    )


def compute_tds_identity(
    *,
    tds_patches,
    **dummy,
):
    return simple_obj(
        tds_after=tds_patches,
    )


## * GlobEnc/ALTI
def transform_globalti_attributions(
    batch_transformed,
    *,
    model=None,
    batch=None,
    # name=None,
    # decomposition_config,
    del_p=True,
    device=None,
):
    device = device or torch.device("cpu")

    with torch.no_grad():
        outputs_final = dict()

        ###
        sum_dim = [-2]
        for k, v in batch_transformed.items():
            # v = batch_transformed[k]
            ##
            block_i = rget(k, "^blocks__(\d+)__attributions_v$")
            if block_i:
                block_i = int(block_i)

                v = v.to(device)  #: no-op if the same device
                #: copying to GPU takes a lot of time (~2.25s), so not worth it

                v = sum_attributions(v, sum_dim=sum_dim)
                #: (batch, token_to, token_from+error_token, hidden_dim)

                v = v[..., :-1, :]  #: removes the error/bias source
                #: (batch, token_to, token_from, hidden_dim)

                ###
                mode = "norm"
                ord = 2
                v_globenc = attributions_scalarify(
                    v,
                    sum_dim=None,
                    mode=mode,
                    ord=ord,
                )
                v_globenc = v_globenc.cpu()
                #: (batch, token_to, token_from)

                outputs_final[f"blocks__{block_i}__globenc_{mode}{ord}"] = v_globenc
                ###
                o = batch_transformed[f"blocks__{block_i}__output"]
                o = o.to(device)
                #: (batch, token, hidden_dim)

                o_norm1 = torch.linalg.vector_norm(
                    o,
                    dim=[-1],
                    ord=1,
                )
                #: (batch, token,)

                sources_count = v.shape[-2]
                ##
                o_expanded = o.unsqueeze(-2)
                #: (batch, token_to, 1, hidden_dim)

                diff = o_expanded - v  #: broadcasting the subtraction
                #: (batch, token_to, token_from, hidden_dim)

                alti = torch.linalg.vector_norm(
                    diff,
                    dim=[-1],
                    ord=1,
                )
                #: (batch, token_to, token_from)

                o_norm1_expanded = o_norm1.unsqueeze(-1)
                #: (batch, token, 1)

                alti_normalized = o_norm1_expanded - alti
                alti_normalized = torch.relu(alti_normalized)
                #: (batch, token_to, token_from)

                alti_all = alti_normalized
                ##
                # alti_all = torch.zeros(v.shape[:-1], dtype=torch.float32)
                # #: (batch, token_to, token_from)

                # for src_i in range(sources_count):
                #     src_v = v[..., src_i, :]
                #     #: (batch, token_to, hidden_dim)

                #     diff = o - src_v
                #     alti = torch.linalg.vector_norm(
                #         diff,
                #         dim=[-1],
                #         ord=1,
                #     )
                #     #: (batch, token_to,)

                #     alti_normalized = o_norm1 - alti
                #     alti_normalized = torch.relu(alti_normalized)
                #     #: (batch, token_to,)

                #     alti_all[..., :, src_i] = alti_normalized
                ##

                alti_all = torch.div(
                    alti_all,
                    alti_all.sum(dim=[-1], keepdim=True),
                )
                #: normalized attributions to sum to one
                #: (batch, token_to, token_from)

                alti_all = alti_all.cpu()
                outputs_final[f"blocks__{block_i}__ALTI"] = alti_all
                ###
                if del_p:
                    continue
                    #: skips saving the attributions into the outputs_final
                else:
                    v = v.cpu()

            if del_p and any(
                re.search(pat, k)
                for pat in [
                    # r'^attributions_',
                    r"^attributions_v_cls$",
                    r"^blocks__(\d+)__output$",
                ]
            ):
                continue
                #: The attributions we have from 'DecompV' here are useless, as a bad config was used to generate them.

            outputs_final[k] = v
        ###
        return outputs_final


## * DecompV Attributors
def dsmap_decompv_attributions(
    batch_transformed,
    *,
    model,
    batch=None,
    decomposition_config,
    raw_attention_store_mode="n",
    store_cls_only_p=False,
    store_perf_p=True,
    attr_name_mode=None,
    return_features_p=True,
    name=None,
):
    with torch.no_grad():
        outputs_final = dict()
        if batch is not None:
            outputs_final.update(batch)
        else:
            outputs_final.update(batch_transformed)

        for k in [
            #: These two are later needed for masked_predict.
            # 'patches_dv',
            # 'label_dv',
        ]:
            key_del(outputs_final, key=k)

        if name is None:
            # name = decomposition_config.name
            name = f"{decomposition_config.name}_{decomposition_config.attributions_aggregation_strategy}"

        batch_size = len(batch_transformed["id"])

        device = model_device_get(model)

        blocks_len = len(model.blocks)

        # label_dv = batch_transformed["label_dv"]

        patches_batch = batch_transformed["patches_dv"]

        start_time = time.time()  #: the current time in seconds since the Epoch
        result = model.forward_patch_level_decomposed(
            patches_batch,
            decomposition_config=decomposition_config,
            return_features_p=return_features_p,
        )
        end_time = time.time()
        time_taken = end_time - start_time

        if return_features_p:
            outputs = vars(result.outputs)
        else:
            outputs = vars(result)

        outputs = transform_cpu(outputs)

        if return_features_p:
            outputs_features = vars(result.features)
            outputs_features = transform_cpu(outputs_features)
        else:
            outputs_features = None
        del result
        del outputs["decomposition_config"]  #: saved separately for the whole dataset
        attr_name = "attributions_v_logits"
        if attr_name_mode == "v1":
            attr_name += f"_{name}"
        outputs[attr_name] = outputs.pop("attributions_v")

        logits = outputs.pop("features")

        if logits.shape[-1] == 1:
            #: Two Classes Using Sigmoid
            #: @duplicateCode/df023e9a4445d7cbfe8f574de6bd4878

            ic("Sigmoid (pre)", torch_shape_get(logits))

            #: Create logits for the sigmoid case
            zero_logits = torch.zeros_like(logits)
            logits = torch.cat([zero_logits, logits], dim=-1)
            outputs["logits"] = logits.detach().cpu()
            ic("Sigmoid (post)", torch_shape_get(logits))
            del zero_logits, logits
        else:
            outputs["logits"] = logits

        if raw_attention_store_mode == "full":
            pass
        elif raw_attention_store_mode == "last":
            k = f"blocks__{blocks_len - 1}__attn__rawattn"
            outputs_final[k] = outputs[k]
        elif raw_attention_store_mode == "last_average":
            outputs_final[f"blocks__{blocks_len - 1}__avgattn"] = torch.mean(
                outputs[f"blocks__{blocks_len - 1}__attn__rawattn"], dim=1
            )

        if raw_attention_store_mode in ("n", "last", "last_average"):
            outputs = {k: v for k, v in outputs.items() if not k.endswith("__rawattn")}

        outputs_final.update(outputs)

        if outputs_features is not None:
            # ic(torch_shape_get(outputs_features, type_only_p=True, size_p=True))
            outputs_final["attributions_v_features"] = outputs_features.pop(
                "attributions_v"
            )
            outputs_final["features"] = outputs_features.pop("features")

            if store_cls_only_p:
                outputs_final = transform_add_attributions_v_cls(
                    outputs_final, remove_p=True
                )

        if store_perf_p:
            outputs_final[f"perf_{name}"] = [
                dict(time_taken=time_taken, batch_size=batch_size)
            ] * batch_size

        # ic(type(outputs_final), torch_shape_get(vars(outputs_final), type_only_p=True))
        ##
        # return outputs_final
        ##
        #: [jalali:1403/01/03]
        batch_transformed.update(outputs_final)

        # assert "segmasks" in batch_transformed
        # assert "label_cpu" in batch_transformed

        return batch_transformed
        ##


def transform_add_attributions_v_cls(batch, remove_p=True):
    if "attributions_v_features" in batch:
        batch["attributions_v_cls"] = batch["attributions_v_features"][:, 0]

        if remove_p:
            del batch["attributions_v_features"]

    return batch


def dataset_format_set1(dataset):
    return dataset.with_format(
        "torch",
        columns=[
            col
            for col in dataset.column_names
            if not (col.startswith("perf_") or col == "id")
        ],
        output_all_columns=True,
    )


def h_decompv_compute(
    *,
    tds_patches,
    batch_size,
    model,
    decomposition_config,
    store_cls_only_p,
    raw_attention_store_mode,
    mapconcat_opts=None,
    **dummy,
):
    metadata = dict()
    metadata["decompv_compute_version"] = 0.1
    if mapconcat_opts is None:
        mapconcat_opts = dict()

    start_time = time.time()
    dataset_after = mapconcat(
        tds_patches.dataset,
        tds_patches.fn_with_transforms(
            partial(
                dsmap_decompv_attributions,
                model=model,
                decomposition_config=decomposition_config,
                store_cls_only_p=store_cls_only_p,
                raw_attention_store_mode=raw_attention_store_mode,
                # store_perf_p=False,
            ),
        ),
        unchanged_keep_columns=["id"],
        batched=True,
        batch_size=batch_size,
        **mapconcat_opts,
    )
    end_time = time.time()
    time_taken = end_time - start_time
    metadata["time_dsmap_decompv_attributions"] = time_taken
    print(f"dsmap_decompv_attributions: took {time_taken}", flush=True)

    dataset_after = dataset_format_set1(dataset_after)
    cols = dataset_after.column_names
    column_sets = []

    if raw_attention_store_mode in ("last", "full"):
        column_sets.append(
            simple_obj(
                name=f"rawattn_{raw_attention_store_mode}",
                columns=[c for c in cols if c.endswith("__rawattn")],
                _readonly_p=False,
            )
        )
    elif raw_attention_store_mode == "last_average":
        column_sets.append(
            simple_obj(
                name="avgattn_last",
                columns=[c for c in cols if c.endswith("__avgattn")],
                _readonly_p=False,
            )
        )
    elif raw_attention_store_mode == "n":
        pass
    else:
        raise ValueError(
            f"Unknown raw_attention_store_mode: {raw_attention_store_mode}"
        )

    column_sets.append(
        simple_obj(
            name="perf",
            columns=[c for c in cols if c.startswith("perf_")],
            _readonly_p=False,
        )
    )
    column_sets.append(
        simple_obj(
            name="features",
            columns=["features"],
            _readonly_p=False,
        )
    )
    column_sets.append(
        simple_obj(
            name="logits",
            columns=["logits"],
            _readonly_p=False,
        )
    )
    column_sets.append(
        simple_obj(
            name="attributions_v_logits",
            columns=["attributions_v_logits"],
            _readonly_p=False,
        )
    )
    if store_cls_only_p:
        if False:
            start_time = time.time()
            dataset_after = mapconcat(
                dataset_after,
                transform_add_attributions_v_cls,
                batched=True,
                batch_size=1024,
            )
            end_time = time.time()
            time_taken = end_time - start_time
            metadata["time_transform_add_attributions_v_cls"] = time_taken
            print(f"transform_add_attributions_v_cls: took {time_taken}", flush=True)

        column_sets.append(
            simple_obj(
                name="attributions_v_cls",
                columns=["attributions_v_cls"],
                _readonly_p=False,
            )
        )
    else:
        column_sets.append(
            simple_obj(
                name="attributions_v_features",
                columns=["attributions_v_features"],
                _readonly_p=False,
            )
        )
    for c in column_sets:
        c.columns += ["id"]

    dataset_afters = dict()
    for c in column_sets:
        dataset_current = dataset_after.select_columns(c.columns)
        dataset_afters[c.name] = dataset_current
    # dataset_afters = simple_obj(**dataset_afters)

    return simple_obj(
        metadata=metadata,
        dataset_all=dataset_after,
        dataset_afters=dataset_afters,
    )


def decompv_v0_to_v0_1(dataset):
    if "attributions_s_logits" in dataset.column_names:
        dataset = dataset.rename_columns(
            {
                "attributions_s_logits": "attributions_v_logits",
            },
        )

    return dataset


## * Using Attributions
def transform_attributions_scalarify(
    batch,
    mode="norm",
    sum_dim=None,
    keep_mode="all",
    **kwargs,
):
    if keep_mode == "all":
        new_batch = dict(batch)
    elif keep_mode is None:
        new_batch = dict()
    else:
        raise ValueError(f"Unsupported keep_mode: {keep_mode}")

    for k, v in batch.items():
        if k.startswith("attributions_"):
            if mode == "norm":
                k_new = f"attributions_n_{k[13:]}"
            elif mode == "identity":
                k_new = k
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            new_batch[k_new] = attributions_scalarify(
                v,
                sum_dim=sum_dim,
                mode=mode,
                **kwargs,
            )

    # ic(
    #     torch_shape_get(batch, type_only_p=True),
    #     torch_shape_get(new_batch, type_only_p=True),
    # )
    return new_batch


def transform_scalarify(batch, methods, name=None, mode="norm", sum_dim=None, **kwargs):
    #: @mutates its input =batch=
    ##
    new_batch = dict()

    if name is None:
        name = f"_s:{mode}"

    for method in methods:
        v = batch[method]
        method_name = method

        raw_suffix = "_s:raw"
        if method_name.endswith(raw_suffix):
            method_name = method_name[: -len(raw_suffix)]

        new_batch[f"{method_name}{name}"] = attributions_scalarify(
            v, sum_dim=sum_dim, mode=mode, **kwargs
        )

    # ic(torch_shape_get(new_batch, type_only_p=True))
    batch.update(new_batch)

    return batch


def random_attributions_n_generate(batch_size, attribution_sources_len):
    # Create a random tensor of shape [batch_size, attribution_sources_len]
    attributions_n = torch.rand(batch_size, attribution_sources_len)
    #: Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).

    # Ensure the attributions are positive and sum up to one
    attributions_n = attributions_n / attributions_n.sum(dim=-1, keepdim=True)

    return attributions_n


def transform_random_attributions_n(batch, attribution_sources_len):
    ids = batch["id"]
    batch_size = len(ids)

    batch["attributions_s_rnd1"] = random_attributions_n_generate(
        batch_size,
        attribution_sources_len,
    )

    # ic(
    #     batch_size,
    #     attribution_sources_len,
    #     batch["attributions_s_rnd1"].shape,
    # )

    return batch


def transform_mask_by_attributions(
    batch,
    *,
    model,
    top_ratio,
    drop_bias_token_p=False,
    drop_bias_token_patterns=[
        r"^attributions_s_logit(?:_|$)",
    ],
    **kwargs,
):
    num_prefix_tokens = getattr(model, "num_prefix_tokens", 0)

    drop_bias_token_patterns = to_iterable(drop_bias_token_patterns)

    ids = batch["id"]
    start_id = ids[0]
    end_id = ids[-1] + 1  #: exclusive end

    batch_update = dict()
    #: Adding new keys in the loop can cause an exception, so we store the new keys in 'batch_update'.
    for k, v in batch.items():
        if k.startswith("attributions_n_") or k.startswith("attributions_s_"):
            #: n: norm
            #: s: signed scalar
            ##
            top_ratio_100 = top_ratio * 100
            if top_ratio_100.is_integer():
                top_ratio_str = f"{int(top_ratio_100):02d}"
            else:
                top_ratio_str = f"{top_ratio_100:04.1f}"
                #: =04= means at least 4 chars in the output (i.e., =02.5=)

            attributions = v
            attributions_no_cls = v[..., num_prefix_tokens:]
            if drop_bias_token_p or any(
                re.search(pat, k) for pat in drop_bias_token_patterns
            ):
                attributions_no_cls = attributions_no_cls[..., :-1]
                #: No need to add a mask value for the bias token

            # ic(torch_shape_get(attributions_no_cls))
            mask = threshold_filter(
                attributions_no_cls,
                top_fill=1,
                top_ratio=top_ratio,
                **kwargs,
            )
            mask = mask.bool()  #: @aesthetic
            mask_n = torch.logical_not(mask)

            for i_ in range(num_prefix_tokens):
                mask = common_torch.prepend_value(mask, 1)  #: includes CLS
                mask_n = common_torch.prepend_value(mask_n, 1)  #: includes CLS

            # ic(torch_shape_get(mask))
            # ic(torch_shape_get(attributions))
            assert (
                mask.shape == attributions.shape[:2]
            ), f"mask.shape {mask.shape} does not match attributions.shape {attributions.shape} (num_prefix_tokens={num_prefix_tokens})"

            topratio_k = f"mask_topratio{top_ratio_str}_{k}"
            nratio_k = f"mask_nratio{top_ratio_str}_{k}"

            tds_path = decompv.utils.dynamic_obj.tds_path

            #: Normalize path:
            tds_path = os.path.normpath(tds_path)

            if tds_path:

                def handle_directory(tds_path, key, start_id, end_id):
                    dir_ = f"{tds_path}/logits_{key}"
                    if os.path.exists(dir_):
                        if id_range_checker(dir_)(start_id, end_id):
                            print(
                                f"skipped: start_id={start_id}, end_id={end_id}, {key}",
                                flush=True,
                            )
                            return None
                    return key

                keys = [topratio_k, nratio_k]

                for i, key in enumerate(keys):
                    keys[i] = handle_directory(tds_path, key, start_id, end_id)

                topratio_k, nratio_k = keys
            ##

            if topratio_k:
                batch_update[topratio_k] = mask

            if nratio_k:
                batch_update[nratio_k] = mask_n

    batch.update(batch_update)

    return batch


def dsmap_predict_masked(
    batch_transformed,
    *,
    batch=None,
    model,
    tds_patches=None,
    time_p=True,
    output_precision=None,
    remove_cols="DYN",
):
    device = model_device_get(model)

    ids = batch_transformed["id"]
    if "patches_dv" in batch_transformed:
        patches = batch_transformed["patches_dv"]
    else:
        patches = tds_patches[ids]["patches_dv"]

    outputs = dict()
    if remove_cols == "DYN":
        remove_cols = dynamic_get(
            decompv.utils.dynamic_obj, "dsmap_predict_masked_remove_cols", default="ALL"
        )

    if remove_cols == "ALL":
        for k, v in batch_transformed.items():
            if k in [
                "id",
                "logits",
            ] or any(
                re.search(pat, k)
                for pat in [
                    r"^(?:perf|time)_",
                    completeness_error_regex,
                ]
            ):
                outputs[k] = batch_transformed[k]

    elif remove_cols is None:
        for k, v in batch_transformed.items():
            if k in [
                "patches",
                "patches_dv",
                "label_dv",
            ]:
                continue

            outputs[k] = v
    else:
        raise ValueError(f"Unsupported 'remove_cols': {remove_cols}")

    for k, v in batch_transformed.items():
        if k.startswith("mask_"):
            # ic(k)

            mask = v
            #: [batch, tokens]

            #: Nonzero elements show places where we should keep the tokens.
            keep_indices = torch.nonzero(mask, as_tuple=True)
            #: [[https://pytorch.org/docs/stable/generated/torch.nonzero.html][torch.nonzero — PyTorch 2.4 documentation]]
            #: If :attr:`as_tuple` is ``True``, one 1-D tensor for  each dimension, containing the indices of each nonzero element along that  dimension.

            #: Reshape the indices to match patch_drop's output format
            #: `keep_indices[1]` contains the indices of nonzero tokens along the token dim?
            #: keep_indices[1] is then reshaped to [batch_size, num_patches_to_zero], assuming the number of non-zero elements (tokens to be zeroed out) is consistent across batches.
            keep_indices = keep_indices[1].view(patches.shape[0], -1)

            num_patches = patches.shape[1]
            assert (
                keep_indices.min() >= 0 and keep_indices.max() < num_patches
            ), f"Indices out of bounds: min={keep_indices.min()}, max={keep_indices.max()}, num_patches={num_patches}"

            with Timed(name=f"drop_tokens ({k})", enabled_p=False):
                #: Time taken by drop_tokens (mask_topratio01_attributions_n_rnd1): 0.0001385211944580078 seconds
                ##
                if getattr(model, "droppable_tokens_p", True):
                    patches_masked = drop_tokens(
                        tokens=patches,
                        mask=mask,
                        num_prefix_tokens=0,
                        #: CLS included in the mask
                    )
                    #: no need to store these patches_masked

                else:
                    patches_masked = patches.to(device) * mask.unsqueeze(-1).to(device)
                    # ic(torch_shape_get(mask), torch_shape_get(patches), torch_shape_get(patches_masked))
                    #: ic| torch_shape_get(mask): (torch.bool, torch.Size([45, 196]), device(type='cpu'))
                    #: torch_shape_get(patches): (torch.float32, torch.Size([45, 196, 384]), device(type='cuda', index=0))
                    #: torch_shape_get(patches_masked): (torch.float32, torch.Size([45, 196, 384]), device(type='cuda', index=0))

            # ic(torch_shape_get((patches, patches_masked), size_p=True))

            with torch.no_grad(), Timed(
                name=f"forward ({k})", enabled_p=time_p
            ), DynamicVariables(
                decomposition.dynamic_obj,
                raw_attention_store_p=False,
                raw_attention_grad_store_p=False,
                ##
                #: @duplicateCode/83cf7ddcbd7f6b5ea1b6bf051ff6de14
                block_output_store_p=False,
                block_output_grad_store_p=False,
                layer_norm_attribute_bias_p=False,
                linear_attribute_bias_p=False,
                gelu_attribute_bias_p=False,
                ##
                print_diag_enabled_groups=lst_filter_out(
                    decomposition.dynamic_obj.print_diag_enabled_groups,
                    [
                        "gradient_mode",
                    ],
                ),
                ##
            ):
                output = (
                    model.forward_patch_level(
                        patches_masked.to(device),
                        keep_indices=keep_indices.to(device),
                    )
                    .detach()
                    .cpu()
                )
                #: The output is a tensor of (batched) logits.
                # ic(torch_shape_get(output, type_only_p=True))

            output_col = f"logits_{k}"
            if output_precision == "bfloat16":
                output = output.bfloat16()
            #   File "/home/mehri/micromamba/envs/p310/lib/python3.10/site-packages/datasets/features/features.py", line 323, in _cast_to_python_objects
            #     for x in obj.detach().cpu().numpy()
            # TypeError: Got unsupported ScalarType BFloat16
            elif output_precision == "float16":
                output = output.half()
            #   File "pyarrow/array.pxi", line 1498, in pyarrow.lib.Array.to_numpy
            #   File "pyarrow/error.pxi", line 121, in pyarrow.lib.check_status
            # pyarrow.lib.ArrowNotImplementedError: Not implemented type for Arrow list to pandas: halffloat

            outputs[output_col] = output

    return outputs


## * Loading the Datasets
### ** Loading Torch Tensors
def tensor_partitioned_load(
    directory,
    *,
    name=None,
    ignore_load_failure_p=True,
    ##
    # start_limit=None,
    # end_limit=None,
    ##
    start_limit="AUTO",
    end_limit="AUTO",
):
    if start_limit == "AUTO":
        start_limit = os.environ.get(
            "DECOMPV_START_LIMIT",
            None,
        )
        if start_limit is not None:
            start_limit = int(start_limit)

    if end_limit == "AUTO":
        end_limit = os.environ.get(
            "DECOMPV_END_LIMIT",
            None,
        )
        if end_limit is not None:
            end_limit = int(end_limit)

    tensor_dict = {}
    partition_dict = {}

    name = name or os.path.basename(directory)

    for filename in os.listdir(directory):
        # ic(filename)
        if filename.endswith(".pt"):
            start_index, end_index = h_extract_indices_from_filename(filename)
            #: [start_index, end_index)
            # ic(filename, start_index, end_index, name)

            if (start_limit is not None and end_index <= start_limit) or (
                end_limit is not None and start_index >= end_limit
            ):
                print(
                    f"{fn_name_current()}: skipped (outside limits):\n\t{filename}",
                    flush=True,
                )
                continue

            #: Load the tensor from the file
            tensor_path = os.path.join(directory, filename)
            try:
                tensor = torch.load(tensor_path)
            except:
                print(
                    f"failed to load tensor: {tensor_path}", file=sys.stderr, flush=True
                )
                if ignore_load_failure_p:
                    continue
                else:
                    raise
            # print(f"loaded: {tensor_path}")

            #: Drop extra items outside the start and end limits
            if start_limit is not None:
                if start_limit >= start_index:
                    tensor = tensor[start_limit - start_index :]
                    start_index = start_limit
            if end_limit is not None:
                if end_limit < end_index:
                    tensor = tensor[: end_limit - start_index]
                    end_index = end_limit

            # Store the partition information in a dictionary
            if name not in partition_dict:
                partition_dict[name] = []

            partition_dict[name].append(
                simple_obj(
                    tensor=tensor,
                    start_index=start_index,
                    end_index=end_index,
                    id=range(start_index, end_index),
                ),
            )

    for name, partition_data in partition_dict.items():
        partition_data = sorted(
            partition_data,
            key=lambda d: d["start_index"],
        )

        with Timed(
            name="torch.cat",
            enabled_p=False,
            # enabled_p=True,
        ):
            all_tensors = torch.cat([d.tensor for d in partition_data])
            all_ids = torch.cat([torch.tensor(d.id) for d in partition_data])

            first_indices = unique_first_indices(all_ids, dim=0)

            unique_tensors = all_tensors[first_indices]
            unique_ids = all_ids[first_indices]

            tensor_dict = BatchedDict(
                id=unique_ids,
            )
            tensor_dict[name] = unique_tensors

    return BatchedDict(tensor_dict)


def h_extract_indices_from_filename(filename):
    # Extract start_index, and end_index from the filename
    parts = filename[:-3].split("_")  # [:-3] removes '.pt'
    return int(parts[-2]), int(parts[-1])  # start_index, end_index


def id_range_checker(directory):
    partition_list = []

    for filename in [file for file in os.listdir(directory) if file.endswith(".pt")]:
        start_index, end_index = h_extract_indices_from_filename(filename)
        partition_list.append((start_index, end_index))

    def is_range_present(start, end):
        return any(s <= start <= end <= e for s, e in partition_list)

    return is_range_present


### _
def transform_patchify_load(batch, device):
    patches_cpu = batch["patches"]
    batch["patches_dv"] = patches_cpu.to(device)

    batch["label_dv"] = batch["label"].to(device)

    return batch


def datasets_load1(
    *,
    dataset_root=None,
    model_name,
    name,
    sub_names=None,
    keep_in_memory=False,
    primary_key=["id"],
    deduplicate_p=False,
):
    dataset_root = dataset_root or DS_ROOT
    sub_names = sub_names or [""]

    dataset_concat1 = []
    for i, sub_name in enumerate(sub_names):
        glob_path = f"{dataset_root}/{model_name}/{name}/*_*"
        if sub_name:
            glob_path += f"/{sub_name}/"

        # ic(glob_path)
        dataset_paths = list(glob.glob(glob_path))
        dataset_paths = [path for path in dataset_paths if os.path.isdir(path)]
        dataset_paths.sort()
        ic(glob_path, dataset_paths)

        dataset_concat0 = [
            datasets.load_from_disk(p, keep_in_memory=keep_in_memory)
            for p in dataset_paths
        ]
        # ic(dataset_concat0[0].format)

        dataset_sub = datasets.concatenate_datasets(dataset_concat0)
        # ic(dataset_sub.format)
        #: Concatenating datasets with different formats will reset the format.
        #: This is not trivial to fix.

        if i >= 1:
            dataset_sub = dataset_sub.remove_columns(["id"])
            #: concatenate_datasets(axis=1) cannot handle duplicate columns
            #: @todo This assumes that all the sub datasets have exactly the same slices.
        dataset_concat1.append(dataset_sub)

    dataset_full = datasets.concatenate_datasets(dataset_concat1, axis=1)

    if deduplicate_p and primary_key:
        dataset_full = dataset_duplicate_rm(
            dataset_full,
            primary_key=primary_key,
        )

    return dataset_full


def dataset_duplicate_rm(
    dataset,
    primary_key: List[str],
):
    #: This function takes some time to complete.
    ##
    seen = set()

    def canonicalize(x):
        if isinstance(x, torch.Tensor):
            if x.nelement() == 1:
                return x.item()
            else:
                return x.tolist()
        else:
            return x

    def filter_seen(
        batch,
    ):
        primary_key_instance = frozenset(
            (k, canonicalize(v)) for k, v in batch.items() if k in primary_key
        )
        if primary_key_instance not in seen:
            seen.add(primary_key_instance)
            return True
        else:
            return False

    dataset_unique = dataset.filter(filter_seen, batched=False)
    # ic(seen)

    return dataset_unique


def transform_softmax(batch):
    inputs = simple_obj(
        attributions_v=batch["attributions_v_logits"],
        features=batch["logits"],
        decomposition_config=DecompositionConfig(
            device=None,
            softmax_decompose_p=True,
        ),
    )
    softmax = SoftmaxDecomposed(dim=-1)
    inputs_sm = softmax.forward(inputs)
    batch["attributions_v_softmax"] = inputs_sm.attributions_v.detach().cpu()
    batch["probability"] = inputs_sm.features.detach().cpu()

    batch["attributions_v_logits"] = batch["attributions_v_logits"].detach().cpu()
    batch["logits"] = batch["logits"].detach().cpu()

    return batch


def transform_attributions_select_label(batch, dataset_indexed):
    if DATASET_NAME == "MURA":
        print(f"{fn_name_current()}: MURA: not supported, skipped", flush=True)
        return batch

    ids = batch["id"]

    # dataset_indexed_batch = dataset_indexed[ids]
    # labels = dataset_indexed_batch["label"]
    labels = IndexableList(dataset_indexed["label"])[ids]

    key_mapping = dict(
        attributions_v_logits="attributions_s_logit",
        attributions_v_softmax="attributions_s_softmax",
    )
    p1_mapping = dict(
        logits="logit",
    )

    new_batch = dict()
    for k, v in batch.items():
        k_new = None

        p1 = rget(k, r"attributions_v_(logits|softmax)")
        name = rget(k, r"attributions_v_(?:logits|softmax)_(.*)")
        if name and p1:
            if p1 in p1_mapping:
                p1 = p1_mapping[p1]

            k_new = f"attributions_s_{p1}_{name}"
        elif k in key_mapping:
            k_new = key_mapping[k]

        if k_new:
            try:
                #: 'attributions_v_logits': (torch.float32, torch.Size([2, 198, 1, 1000]))
                new_batch[k_new] = v[torch.arange(v.shape[0]), ..., labels]
            except:
                ic(k, k_new, v.shape, labels)
                raise
    batch.update(new_batch)

    # ic(torch_shape_get(batch, type_only_p=True))
    return batch


def add_random_baseline(
    tds_scalar,
    *,
    model_patch_info,
    model,
):
    num_prefix_tokens = getattr(model, "num_prefix_tokens", 0)
    assert (
        num_prefix_tokens == model_patch_info.num_prefix_tokens
    ), f"You have not set the correct num_prefix_tokens in the model_patch_info. Actual num_prefix_tokens: {num_prefix_tokens}, model_patch_info:\n{model_patch_info}"

    return tds_scalar.transform(
        partial(
            transform_random_attributions_n,
            attribution_sources_len=model_patch_info.source_count,
        )
    )


def masker_entrypoint(
    tds_scalar,
    model_patch_info,
    bias_token_p,
    add_random_p=False,
    *,
    model,
):
    if benchmark_mode_p:
        return tds_scalar

    if add_random_p:
        tds_scalar = add_random_baseline(
            tds_scalar,
            model_patch_info=model_patch_info,
            model=model,
        )

    # ic(tds_scalar.preview())

    tds_masked = tds_scalar
    for top_ratio in [
        # 0.01,
        # 0.025,
        # 0.05,
        # 0.075,
        0.1,
        # 0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        # 0.95,
        # 0.99,
    ]:
        tds_masked = tds_masked.transform(
            partial(
                transform_mask_by_attributions,
                model=model,
                top_ratio=top_ratio,
                drop_bias_token_p=bias_token_p,
            ),
        )

    # ic(tds_masked.preview())
    return tds_masked


## * Attention-Based Attributions
def h_ig_accumulate_results(
    *,
    accumulator,
    current_step,
    step_result,
    ig_steps,
):
    ##
    # def recursive_accumulate(acc, result, steps):
    #     if isinstance(result, (int, float, torch.Tensor)):
    #         if acc is None:
    #             return result / steps

    #         return acc + result / steps

    #     elif isinstance(result, dict):
    #         if acc is None:
    #             acc = {}

    #         for key, value in result.items():
    #             acc[key] = recursive_accumulate(acc.get(key), value, steps)

    #         return acc

    #     elif isinstance(result, list):
    #         if acc is None:
    #             acc = [None] * len(result)

    #         return [recursive_accumulate(a, r, steps) for a, r in zip(acc, result)]

    #     else:
    #         return result  # For non-numeric values, just keep the last one

    # return recursive_accumulate(accumulator, step_result, ig_steps)
    ##
    for key, value in step_result.items():
        if (
            "completeness_error" in key or key.startswith("attributions_")
        ) and isinstance(value, torch.Tensor):
            value = value.detach().cpu()

            if key not in accumulator:
                accumulator[key] = value / ig_steps
            else:
                accumulator[key] += value / ig_steps

        else:
            if current_step <= 0:
                print(f"IG: non-IG key: {key}")

            # :\n\t{torch_shape_get(key)}
            accumulator[key] = value  # For non-numeric values, just keep the last one

    return accumulator
    ##


def dsmap_attn(
    batch_transformed,
    *,
    ig_steps="from_dynamic",
    batch=None,
    attn_prepare_attribution_columns=None,
    model,
    store_perf_p=True,
    **kwargs,
):
    if ig_steps == "from_dynamic":
        ig_steps = decomposition.dynamic_obj.ig_steps

    print_diag(
        f"dsmap_attn: ig_steps: {ig_steps}",
        group="gradient_mode",
    )

    if not ig_steps:
        res = h_dsmap_attn(
            batch_transformed,
            batch=batch,
            model=model,
            store_perf_p=store_perf_p,
            **kwargs,
        )

        if attn_prepare_attribution_columns is not None:
            res = attn_prepare_attribution_columns(
                res,
                model=model,
            )

        return res

    else:
        if "image_array_dv" in batch_transformed:
            #: Currently just ignore the pixel-level stuff
            if True:
                del batch_transformed["image_array_dv"]

                image_array_dv = None

            else:
                raise NotImplementedError(
                    "Pixel-level currently not implemented for Integrated Gradient."
                )

                image_array_dv = batch_transformed["image_array_dv"]

        else:
            image_array_dv = None

        patches_batch = batch_transformed["patches_dv"]
        baseline = torch.zeros_like(patches_batch)

        #: Initialize accumulator dict
        accumulator = {}

        for step in range(ig_steps):
            print_diag(
                f"current ig_step: {step}",
                group="gradient_mode",
            )

            #: Interpolate between baseline and original patches
            alpha = (step + 1) / ig_steps
            interpolated_patches = (
                baseline + alpha * (patches_batch - baseline)
            ).detach()

            #: Create a new batch_transformed with interpolated patches
            current_batch = {**batch_transformed, "patches_dv": interpolated_patches}

            #: Get the result for the current step
            step_result = h_dsmap_attn(
                current_batch,
                batch=batch,
                model=model,
                store_perf_p=False,
                **kwargs,
            )

            if attn_prepare_attribution_columns is not None:
                step_result = attn_prepare_attribution_columns(
                    step_result,
                    model=model,
                )

            else:
                raise Exception("IG: attn_prepare_attribution_columns must be given")

            # step_result = transform_cpu(step_result)

            #: Accumulate the results
            accumulator = h_ig_accumulate_results(
                accumulator=accumulator,
                current_step=step,
                step_result=step_result,
                ig_steps=ig_steps,
            )

        return accumulator


def h_dsmap_attn(
    batch_transformed,
    *,
    batch=None,
    model,
    # raw_attention_store_mode,
    store_perf_p=True,
    name="attndata",
    # target_labels=ground_truth_mode_cst,
    target_labels="from_config",
    remove_cols="MAGIC_OLD",
    after_transforms=None,
    after_transforms_time_p=False,
    # after_transforms_time_p=True,
    text_grad_p=False,
):
    torch_gpu_empty_cache()

    if target_labels == "from_config":
        target_labels = metrics_target_get()

    ic(target_labels)

    # ic(type(batch), torch_shape_get(vars(batch), type_only_p=True), type(batch_transformed), torch_shape_get(batch_transformed, type_only_p=True))

    if after_transforms is None:
        after_transforms = []

    batch_size = len(batch_transformed["id"])

    device = model_device_get(model)

    blocks_len = len(model.blocks)

    if "image_array_dv" in batch_transformed:
        image_array_dv = batch_transformed["image_array_dv"]
    else:
        image_array_dv = None

    patches_batch = batch_transformed["patches_dv"]
    patches_batch.requires_grad_(True)
    patches_batch.retain_grad()

    if remove_cols is None:
        output = dict(batch_transformed)
    else:
        output = dict()

    clip_p = getattr(model, "clip_p", False)

    with DynamicVariables(
        decomposition.dynamic_obj,
        raw_attention_store_p=True,
        raw_attention_grad_store_p=True,
        block_output_store_p=True,
        block_output_grad_store_p=True,
        print_diag_enabled_groups=lst_filter_out(
            decomposition.dynamic_obj.print_diag_enabled_groups,
            [
                "progress",
            ],
        ),
    ):
        start_time = time.time()  #: the current time in seconds since the Epoch

        model.zero_grad()  #: might help with GC
        model.eval()  #: @redundant to make sure

        if clip_p:
            output["image_id"] = batch_transformed["image_id"]
            output["clip_text"] = batch_transformed["clip_text"]

            if "label_natural" in batch_transformed:
                #: @untested
                ##
                assert (
                    "clip_text" not in batch_transformed
                ), "Both clip_text and label_natural provided for a CLIP model!"

                texts = batch_transformed["label_natural"]

                if target_labels == ground_truth_mode_cst:
                    target_labels = torch.arange(len(texts))
                    batch_transformed["label"] = target_labels
                    batch_transformed["label_dv"] = target_labels.to(device)
                    target_labels = batch_transformed["label_dv"]

                else:
                    raise NotImplementedError(
                        f"target_labels not yet supported here: {target_labels}"
                    )

            else:
                texts = batch_transformed["clip_text"]

            image_features = model.forward_patch_level(
                patches_batch,
            )
            #: (batch, d)

            tokenizer = model.tokenizer

            with no_grad_maybe(not text_grad_p):
                text_tokens = tokenizer(texts).to(device)
                text_features = model.encode_text(text_tokens)
                text_features = normalize_to_unit_vector(text_features)
                #: (batch, d)

            image_features = normalize_to_unit_vector(image_features)

            logits = 100.0 * image_features @ text_features.T
            #: (batch, d) @ (d, batch) = (batch, batch)

            output["logits"] = logits.detach().cpu()

            reward_batched = torch.diag(logits)
            reward = reward_batched.sum()
            output["grad_target_batch"] = reward_batched.detach().cpu()
            del reward_batched

        else:
            result = model.forward_patch_level(
                patches_batch,
            )
            sigmoid_model_p = result.shape[-1] == 1
            #: [[id:fc18ef3e-6afc-40d6-94b0-3ddf08a1d61b][KhalfounMehdi/MURA · Datasets at Hugging Face]]
            #: MURA has 0 and 1 for labels.

            if sigmoid_model_p:
                #: Two Classes Using Sigmoid
                #: @duplicateCode/df023e9a4445d7cbfe8f574de6bd4878

                #: Create logits for the sigmoid case
                zero_logits = torch.zeros_like(result)
                logits = torch.cat([zero_logits, result], dim=-1)
                output["logits"] = logits.detach().cpu()
                # ic("Sigmoid", torch_shape_get(result), torch_shape_get(logits))
                #: torch_shape_get(result): (torch.float32, torch.Size([150, 1]), device(type='cuda', index=0))  torch_shape_get(logits): (torch.float32, torch.Size([150, 2]), device(type='cuda', index=0))

                del zero_logits, logits

            else:
                output["logits"] = result.detach().cpu()

            ##
            # label_dv = batch_transformed["label_dv"]
            if target_labels == ground_truth_mode_cst:
                target_labels = batch_transformed["label"]

            elif target_labels == first_highest_predicted_mode_cst:
                target_labels = get_first_predicted_indices(output["logits"])

            elif target_labels == second_highest_predicted_mode_cst:
                target_labels = get_second_predicted_indices(output["logits"])

            else:
                #: I added this exception later. I am confused why we didn't support this other branch here. How do we support any alternative then?!
                raise NotImplementedError(
                    f"target_labels not yet supported here: {target_labels}"
                )

            if sigmoid_model_p:
                if ic(decomposition.dynamic_obj.backward_softmax_p):
                    print_diag(
                        "Skipping the head-softmax gradient for sigmoid model",
                        group="gradient_mode",
                    )

                    # result = torch.sigmoid(result)

                if ic(decomposition.dynamic_obj.head_contrastive_n):
                    print_diag(
                        "Skipping the head contrastive method for sigmoid model",
                        group="gradient_mode",
                    )

                # ic(target_labels)
                reward_batched = torch.where(
                    (target_labels.to(device)) == 1, result[:, 0], -result[:, 0]
                )

            else:
                if decomposition.dynamic_obj.backward_softmax_p:
                    head_softmax = NightSoftmax(dim=-1)
                    head_softmax.prefix = "head_softmax."

                    head_softmax_gbrand = (
                        decomposition.dynamic_obj.head_softmax_gradient_mode
                    )
                    head_softmax_competition_scale = (
                        decomposition.dynamic_obj.head_softmax_competition_scale
                    )
                    ic(head_softmax_gbrand, head_softmax_competition_scale)

                    result = head_softmax(
                        result,
                        gradient_mode=head_softmax_gbrand,
                        competition_scale=head_softmax_competition_scale,
                    )
                    # result = torch.softmax(result, dim=-1)

                # ic(target_labels)
                batch_indices = torch.arange(len(target_labels))
                reward_batched = result[batch_indices, target_labels]
                #: [batch]

                ##
                head_contrastive_n = decomposition.dynamic_obj.head_contrastive_n
                head_contrastive_scale = (
                    decomposition.dynamic_obj.head_contrastive_scale
                )

                ic(head_contrastive_n, head_contrastive_scale)

                if head_contrastive_n:
                    # Use a detached clone to find top-k indices without affecting gradients
                    with torch.no_grad():
                        result_no_target = result.clone()
                        result_no_target[batch_indices, target_labels] = float("-inf")
                        _, topk_indices = torch.topk(
                            result_no_target, head_contrastive_n, dim=1
                        )

                    # Gather the top-k logits from the original result tensor
                    topk_values = result.gather(1, topk_indices)
                    # Compute the mean of the top logits for each sample
                    mean_topk = topk_values.mean(dim=1)
                    # Compute the scaled contrastive penalty
                    contrastive_penalty = mean_topk * head_contrastive_scale

                    ic(
                        reward_batched.shape,
                        contrastive_penalty.shape,
                        topk_indices.shape,
                        topk_values.shape,
                    )
                    # ic| head_contrastive_n: 2, head_contrastive_scale: 0.5
                    # ic| reward_batched.shape: torch.Size([103])
                    # contrastive_penalty.shape: torch.Size([103])
                    # topk_indices.shape: torch.Size([103, 2])
                    # topk_values.shape: torch.Size([103, 2])

                    # Subtract the penalty from the reward
                    reward_batched = reward_batched - contrastive_penalty
                ##

            reward = reward_batched.sum()
            output["grad_target_batch"] = reward_batched.detach().cpu()
            del reward_batched

        if image_array_dv is not None:
            image_array_dv.grad = None

        patches_batch.grad = None
        model.zero_grad()
        reward.backward()
        end_time = time.time()
        time_taken = end_time - start_time

    output["id"] = batch_transformed["id"]
    output["patches_dv"] = batch_transformed["patches_dv"].detach()
    output[patch_grad_raw_name] = patches_batch.grad.detach()
    patches_batch.grad = None

    if (
        image_array_dv is not None
        and hasattr(image_array_dv, "grad")
        and image_array_dv.grad is not None
    ):
        print("ImageIxG: computing ...", flush=True)

        image_grad_raw_name = "ImageGrad_s:raw"
        output[image_grad_raw_name] = image_array_dv.grad.detach()
        image_array_dv.grad = None

        image_IxG_raw_name = "ImageIxG_s:raw"
        output[image_IxG_raw_name] = (
            torch.mul(output[image_grad_raw_name], image_array_dv).detach().cpu()
        )
        output[image_grad_raw_name] = output[image_grad_raw_name].cpu()

        for mode in channel_mixers:
            output = transform_scalarify(
                output,
                methods=[
                    image_IxG_raw_name,
                    image_grad_raw_name,
                ],
                mode=mode,
                dim=1,  #: batch, channel, x, y
            )

        # ic(list(output.keys()))
    else:
        print("ImageIxG: NA", flush=True)
        # ic(image_array_dv.grad, image_array_dv)
        pass

    if (
        hasattr(model, "pos_embed")
        and hasattr(model.pos_embed, "grad")
        and model.pos_embed.grad is not None
        and (not getattr(model, "no_embed_class", False))
    ):
        print("PosEIxG: computing ...", flush=True)

        # if hasattr(model, "no_embed_class"):
        #     assert not model.no_embed_class, "no_embed_class is not yet supported"

        pos_grad_raw_name = "PosEGrad_s:raw"
        output[pos_grad_raw_name] = model.pos_embed.grad.detach()
        model.pos_embed.grad = None

        pos_embed_IxG_raw_name = "PosEIxG_s:raw"
        output[pos_embed_IxG_raw_name] = (
            torch.mul(output[pos_grad_raw_name], model.pos_embed).detach().cpu()
        )

        # output[pos_grad_raw_name] = output[pos_grad_raw_name].cpu()
        del output[pos_grad_raw_name]
        #: PosEGrad should be the same as PatchGrad, as pos_embed is summed with the patches. So there is no need to keep it.

        for mode in channel_mixers:
            output = transform_scalarify(
                output,
                methods=[
                    pos_embed_IxG_raw_name,
                ],
                mode=mode,
            )

    # output["IxG"] = torch.mul(output[patch_grad_raw_name], output["patches_dv"]).detach().cpu()
    #: IxG is now blocks__0__CAT

    output[patch_grad_raw_name] = output[patch_grad_raw_name].cpu()

    output_basic_cols = list(output.keys())

    for block_i in range(blocks_len):
        block_current = model.blocks[block_i]

        ##
        if hasattr(block_current, "attn"):
            rawattn = block_current.attn.stored_rawattn
            output[f"blocks__{block_i}__attn__rawattn"] = rawattn

            rawattn_grad = block_current.attn.stored_rawattn_grad
            output[f"blocks__{block_i}__attn__rawattn_grad"] = rawattn_grad
            ##

            ##
            if hasattr(block_current.attn, "stored_mha"):
                MHA = block_current.attn.stored_mha
                output[f"blocks__{block_i}__MHA"] = MHA

            if hasattr(block_current.attn, "stored_mha_grad"):
                MHAGrad = block_current.attn.stored_mha_grad
                output[f"blocks__{block_i}__MHAGrad"] = MHAGrad
            ##

            ##
            if hasattr(block_current.attn, "stored_value"):
                value = block_current.attn.stored_value
                output[f"blocks__{block_i}__Value"] = value

            if hasattr(block_current.attn, "stored_value_grad"):
                ValueGrad = block_current.attn.stored_value_grad
                output[f"blocks__{block_i}__ValueGrad"] = ValueGrad
            ##

        output[f"blocks__{block_i}__output"] = block_current.stored_output
        output[f"blocks__{block_i}__output_grad"] = block_current.stored_output_grad

    perf_data = dict()
    if after_transforms:
        with Timed(
            name="dsmap_attn: All Transforms", enabled_p=after_transforms_time_p
        ):
            old_columns = list(output.keys())

            for transform in after_transforms:
                tmp = dict()
                with Timed(print_p=False, output_dict=tmp):
                    output = transform(output)

                if "_name" in output:
                    transform_name = output["_name"]
                    del output["_name"]
                else:
                    transform_name = fn_name(transform, module_p=False)

                perf_data[f"time_{transform_name}"] = tmp["time"]

            # print(f"dsmap_attn: remove_cols={remove_cols}")
            if remove_cols == "MAGIC_OLD":

                def col_p(k):
                    res = k not in old_columns

                    # res = res and not k.endswith('__MeanReLUAttnGrad')

                    res = res or k.startswith("perf_") or k in output_basic_cols
                    return res

                output = {k: v for k, v in output.items() if col_p(k)}

    if store_perf_p:
        output[f"perf_{name}"] = [
            dict(time_taken=time_taken, batch_size=batch_size, **perf_data)
        ] * batch_size

    # ic(torch_shape_get(output, type_only_p=True))
    return output


def h_attn_compute(
    *,
    tds_patches,
    batch_size,
    model,
    # raw_attention_store_mode,
    inner_opts=None,
    mapconcat_opts=None,
    dataset_path=None,
    transform_only_p=True,
    **dummy,
):
    raise NotImplementedError("h_attn_compute needs a rewrite")
    #: Remove transform_only_p.
    #: We can move the benchmarking options to save_tds_torch instead.
    #: The output of this function will be TransformedDataset if transform_only_p, and a Dataset otherwise. Obviously this output type union is nasty and unusable downstream.
    ##
    metadata = dict()
    metadata["attn_compute_version"] = 0.1
    if mapconcat_opts is None:
        mapconcat_opts = dict()

    ic(mapconcat_opts)

    if inner_opts is None:
        inner_opts = dict()

    start_time = time.time()
    if transform_only_p:
        assert dataset_path, "transform_only_p without saving is meaningless"
        dataset_after = None

        save_tds_torch(
            tds_patches,
            output_dir=dataset_path,
            batch_size=batch_size,
            tqdm=tqdm,
        )
    else:
        dataset_after = mapconcat(
            tds_patches.dataset,
            tds_patches.fn_with_transforms(
                partial(
                    dsmap_attn,
                    model=model,
                    # raw_attention_store_mode=raw_attention_store_mode,
                    # store_perf_p=False,
                    **inner_opts,
                ),
                time_p=True,
            ),
            unchanged_keep_columns=["id"],
            batched=True,
            time_p=True,
            batch_size=batch_size,
            **mapconcat_opts,
        )
        dataset_after = dataset_format_set1(dataset_after)

    end_time = time.time()
    time_taken = end_time - start_time
    metadata["time_dsmap_attn"] = time_taken
    print(f"dsmap_attn: took {time_taken}", flush=True)

    return simple_obj(
        metadata=metadata,
        dataset_all=dataset_after,
    )


## ** Raw Attention and Gradients into Attributions
## *** Aggregators
import torch
import einops


def fromto_indices_normalize(
    *,
    from_layer=None,
    to_layer=None,
    model=None,
    blocks_len=None,
):
    if blocks_len is None:
        blocks_len = len(model.blocks)

    if to_layer is None:
        to_layer = blocks_len
    elif to_layer <= 0:
        to_layer += blocks_len
    to_layer = min(to_layer, blocks_len)

    if from_layer is None:
        from_layer = 0
    elif from_layer < 0:
        from_layer += blocks_len

    return simple_obj(
        from_layer=from_layer,
        to_layer=to_layer,
    )


def fromto_postfix(
    from_layer,
    to_layer,
    blocks_len,
):
    output_col = ""
    if from_layer > 0:
        output_col += f"_f{from_layer}"
    if to_layer != blocks_len:
        output_col += f"_to{to_layer}"

    return output_col


def transform_aggregate_layers_gen(
    batch,
    model,
    output_name,
    name_fn,
    from_layer=0,
    to_layer=None,
    relu_p=False,
    normalize=None,
    agg_fn=None,
    agg_name=None,
    agg_name_fn=None,
    **agg_opts,
):
    output_name_lower = output_name.lower()
    if "attn" in output_name_lower:
        if "attnfrom" in output_name_lower:
            if not model_attn_p(batch, model):
                return batch

        else:
            if not model_cls_attn_p(batch, model):
                return batch

    assert agg_fn is not None, "Aggregation function (agg_fn) must be provided."

    output = dict()

    blocks_len = len(model.blocks)
    fromto_res = fromto_indices_normalize(
        from_layer=from_layer,
        to_layer=to_layer,
        blocks_len=blocks_len,
    )
    from_layer, to_layer = fromto_res.from_layer, fromto_res.to_layer

    # ic(from_layer, to_layer, blocks_len)
    assert (
        from_layer < to_layer
    ), f"from_layer ({from_layer}) must be smaller than to_layer ({to_layer})"

    agg = None
    for block_i in range(from_layer, to_layer):
        current = batch[name_fn(block_i)]

        if relu_p:
            current = torch.relu(current)

        if normalize == "to1":
            #: (..., attr_from)
            current = torch.divide(
                current,
                torch.abs(current.sum(dim=(-1,), keepdim=True)) + 1e-12,
            )  #: element-wise

        agg = agg_fn(current, agg, **agg_opts)

    output_col = output_name
    if relu_p:
        output_col += "_relu"
    if normalize:
        output_col += f"_{normalize}"
    if agg_name:
        output_col += f"_{agg_name}"
    if agg_name_fn:
        output_col = agg_name_fn(current_name=output_col, agg_opts=agg_opts)

    output_col += fromto_postfix(
        from_layer,
        to_layer,
        blocks_len=blocks_len,
    )
    output[output_col] = agg
    output["_name"] = output_col

    batch.update(output)
    return batch


def sum_agg_fn(current, agg):
    if agg is None:
        return current
    else:
        return current + agg


def einsum_agg_fn(
    current,
    agg,
    residual_strength,
    average_dim=None,
    scale=1,  #: might help with the vanishing attributions problem in Rollout
):
    if average_dim:  #: @untested @unused
        current = current.mean(
            dim=average_dim,
        )

    assert (
        current.ndim == 3
    ), f"Invalid input shape for rollout: {torch_shape_get(current)}"
    #: (batch, attention_to, attention_from)

    if residual_strength:
        identity = torch.eye(current.shape[-1]).unsqueeze(0)
        current = (scale * (1 - residual_strength)) * current + (
            scale * residual_strength
        ) * identity
    else:
        current = scale * current

    if agg is None:
        return current
    else:
        return einops.einsum(
            current,
            agg,
            "batch attr_to middleman, batch middleman attr_from -> batch attr_to attr_from",
        )


def transform_aggregate_layers_sum(*args, **kwargs):
    return transform_aggregate_layers_gen(
        *args, **kwargs, agg_fn=sum_agg_fn, agg_name="sum"
    )


def h_rollout_agg_name(current_name, *, agg_opts=None, **kwargs):
    if agg_opts is None:
        agg_opts = dict()

    agg_opts.update(kwargs)

    residual_strength = agg_opts.get("residual_strength", None)
    if residual_strength:
        current_name += f"_str{int(residual_strength*100)}"

    scale = agg_opts.get("scale", None)
    if scale is not None and scale != 1:
        current_name += f"_scl{scale}"

    return current_name


def transform_aggregate_layers_rollout(*args, **kwargs):
    return transform_aggregate_layers_gen(
        *args,
        **kwargs,
        agg_fn=einsum_agg_fn,
        agg_name="ro",
        agg_name_fn=h_rollout_agg_name,
    )


## **** Combine Attributions
## ***** Sum
def transform_attr_combine_sum(batch, methods, normalize=None):
    normalize = to_iterable(normalize)

    name = ""
    agg = None
    for method in methods:
        if name:
            name += f"+{method}"
        else:
            name += f"{method}"

        current = batch[method]

        try:
            current = attributions_autoget(current)
            assert current is not None
            for fn in normalize:
                current = fn(current)

            agg = sum_agg_fn(current=current, agg=agg)
        except:
            ic(method, methods, torch_shape_get((current, agg)))
            raise

    for fn in normalize:
        name += f"|{fn_name(fn, module_p=False)}"

    batch[name] = agg

    return batch


## **** Rollouts
transform_rawattn_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanAttn",
    name_fn=(lambda i: f"blocks__{i}__MeanAttn"),
)

transform_AttnGrad_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanAttnGrad",
    name_fn=(lambda i: f"blocks__{i}__MeanAttnGrad"),
)

transform_AbsAttnGrad_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanAbsAttnGrad",
    name_fn=(lambda i: f"blocks__{i}__MeanAbsAttnGrad"),
)

transform_MeanAttn_CAT_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanAttn_CAT",
    name_fn=(lambda i: f"blocks__{i}__CAT_MeanAttn"),
    relu_p=False,
    normalize=None,
)
transform_MeanAttn_CAT_relu_to1_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanAttn_CAT",
    name_fn=(lambda i: f"blocks__{i}__CAT_MeanAttn"),
    relu_p=True,
    normalize="to1",
)

transform_MeanReLUAttnGrad_MeanAttn_CAT_relu_to1_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanReLUAttnGrad_MeanAttn_CAT",
    name_fn=(lambda i: f"blocks__{i}__CAT_MeanReLUAttnGrad_MeanAttn"),
    relu_p=True,
    normalize="to1",
)

transform_Mean__AttnGrad_Attn_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="Mean__AttnGrad_Attn",
    name_fn=(lambda i: f"blocks__{i}__Mean__AttnGrad_Attn"),
    relu_p=False,
    normalize=None,
)

transform_MeanReLU__AttnGrad_Attn_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanReLU__AttnGrad_Attn",
    name_fn=(lambda i: f"blocks__{i}__MeanReLU__AttnGrad_Attn"),
    relu_p=False,
    normalize=None,
)
transform_MeanAbs__AttnGrad_Attn_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanAbs__AttnGrad_Attn",
    name_fn=(lambda i: f"blocks__{i}__MeanAbs__AttnGrad_Attn"),
    relu_p=False,
    normalize=None,
)

transform_MeanReLU__AttnGrad_Attn_relu_to1_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanReLU__AttnGrad_Attn",
    name_fn=(lambda i: f"blocks__{i}__MeanReLU__AttnGrad_Attn"),
    relu_p=True,
    normalize="to1",
)

transform_MeanAttnGrad_MeanAttn_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanAttnGrad_MeanAttn",
    name_fn=(lambda i: f"blocks__{i}__MeanAttnGrad_MeanAttn"),
    relu_p=False,
    normalize=None,
)

transform_MeanAttnGrad_MeanAttn_relu_to1_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanAttnGrad_MeanAttn",
    name_fn=(lambda i: f"blocks__{i}__MeanAttnGrad_MeanAttn"),
    relu_p=True,
    normalize="to1",
)

transform_MeanReLUAttnGrad_MeanAttn_relu_to1_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanReLUAttnGrad_MeanAttn",
    name_fn=(lambda i: f"blocks__{i}__MeanReLUAttnGrad_MeanAttn"),
    relu_p=True,
    normalize="to1",
)

transform_MeanReLUAttnGrad_MeanAttn_relu_rollout = partial(
    transform_aggregate_layers_rollout,
    output_name="MeanReLUAttnGrad_MeanAttn",
    name_fn=(lambda i: f"blocks__{i}__MeanReLUAttnGrad_MeanAttn"),
    relu_p=True,
)


## *** Attn
def model_attn_p(batch, model):
    return "blocks__0__attn__rawattn" in batch


def model_cls_p(batch, model):
    cls_p = getattr(model, "has_class_token", False)
    # ic(model, cls_p)

    return cls_p


def model_cls_attn_p(batch, model):
    return model_cls_p(batch, model) and model_attn_p(batch, model)


def skip_if_no_attn(func):
    """
    Decorator that skips the execution of the decorated function if the batch does not contain
    attention data as determined by the `model_attn_p` predicate.

    Args:
        func (callable): The function to be decorated. It should accept at least two arguments: `batch` and `model`.

    Returns:
        callable: The wrapped function that includes the attention check.
    """

    @functools.wraps(func)
    def wrapper(batch, model, *args, **kwargs):
        if not model_attn_p(batch, model):
            # print("Attention data not found. Skipping the transformation.")
            return batch

        return func(*args, batch=batch, model=model, **kwargs)

    return wrapper


def skip_if_no_cls_attn(func):
    @functools.wraps(func)
    def wrapper(batch, model, *args, **kwargs):
        if not model_cls_attn_p(batch, model):
            return batch

        return func(*args, batch=batch, model=model, **kwargs)

    return wrapper


@skip_if_no_cls_attn
def transform_MeanAttn(batch, model):
    output = dict()

    blocks_len = len(model.blocks)
    for block_i in range(blocks_len):
        rawattn = batch[f"blocks__{block_i}__attn__rawattn"]
        #: (batch, head, attention_to, attention_from)

        rawattn = torch.mean(
            rawattn,
            dim=(1,),
            keepdim=False,
        )

        output[f"blocks__{block_i}__MeanAttn"] = rawattn

    batch.update(output)
    return batch


@skip_if_no_cls_attn
def transform_AttnGrad(batch, model):
    output = dict()

    blocks_len = len(model.blocks)
    for block_i in range(blocks_len):
        rawattn_grad = batch[f"blocks__{block_i}__attn__rawattn_grad"]
        #: (batch, head, attention_to, attention_from)

        ##
        #: I have no idea why we had this here. We already have =Mean__AttnGrad_Attn=, this should have been just the gradient ...

        # rawattn = batch[f"blocks__{block_i}__attn__rawattn"]
        # AttnGrad = torch.mul(
        #     rawattn,
        #     rawattn_grad,
        # )
        # #: (batch, head, attention_to, attention_from)
        ##
        #: [jalali:1402/12/21/17:44]
        #: I fixed this.
        #: This function is now pretty useless, we are just renaming (aliasing) the columns.
        AttnGrad = rawattn_grad
        ##

        output[f"blocks__{block_i}__AttnGrad"] = AttnGrad

    batch.update(output)
    return batch


@skip_if_no_cls_attn
def transform_MeanAttnGrad(batch, model, relu_p=False, del_p=False):
    output = dict()

    blocks_len = len(model.blocks)
    for block_i in range(blocks_len):
        AttnGrad = batch[f"blocks__{block_i}__AttnGrad"]
        #: (batch, head, attention_to, attention_from)
        if del_p:
            del batch[f"blocks__{block_i}__AttnGrad"]

        MeanAbsAttnGrad = torch.mean(torch.abs(AttnGrad), dim=(1,), keepdim=False)
        #: (batch, attention_to, attention_from)

        output[f"blocks__{block_i}__MeanAbsAttnGrad"] = MeanAbsAttnGrad

        if relu_p:
            AttnGrad = torch.relu(AttnGrad)

        MeanAttnGrad = torch.mean(AttnGrad, dim=(1,), keepdim=False)

        if relu_p:
            output_name = f"blocks__{block_i}__MeanReLUAttnGrad"
        else:
            output_name = f"blocks__{block_i}__MeanAttnGrad"

        output[output_name] = MeanAttnGrad

    batch.update(output)
    return batch


## *** AttnCAM (Chefer's GradCAM)
@skip_if_no_cls_attn
def transform_AttnWHeadGrad(batch, model):
    output = dict()

    blocks_len = len(model.blocks)
    for block_i in range(blocks_len):
        rawattn = batch[f"blocks__{block_i}__attn__rawattn"]
        rawattn_grad = batch[f"blocks__{block_i}__attn__rawattn_grad"]
        #: (batch, head, attention_to, attention_from)

        rawattn_grad = rawattn_grad[..., 1:]
        #: (batch, head, attention_to, attention_from_without_CLS)
        #: @variant Include the CLS as well.

        rawattn_grad = rawattn_grad.mean(dim=(-1,), keepdim=True)
        #: (batch, head, attention_to, 1)
        #: The reason we only take the mean over =attention_from=, is because the original method from Chefer only used the last layer, and it was only interested in the CLS token. So it didn't make sense to take the mean over all =attention_to= tokens.

        AttnWHeadGrad = torch.mul(
            rawattn,
            rawattn_grad,
        )
        #: (batch, head, attention_to, attention_from)

        AttnWHeadGrad = AttnWHeadGrad.mean(dim=(1,), keepdim=False)
        #: (batch, attention_to, attention_from)
        #: multi-head attention weighted by mean gradient of attention of each head to each output

        # ic(torch_shape_get(AttnWHeadGrad, size_p=True))
        output[f"blocks__{block_i}__AttnWHeadGrad"] = AttnWHeadGrad

    batch.update(output)
    return batch


## *** Chefer2


@skip_if_no_cls_attn
def transform_TokenTM(
    batch,
    model,
):
    output = dict()

    epsilon = 1e-8

    blocks_len = len(model.blocks)

    def mat_combine(*, current, agg):
        #: batched matrix multiply
        #: out = current x agg
        ##
        return einsum_agg_fn(
            agg=agg,
            current=current,
            ##
            scale=1,
            residual_strength=0,
            average_dim=None,
        )

    def L(embedding):
        #: L2 norm (p=2) along the last dimension (dim=-1)
        return torch.norm(embedding, p=2, dim=-1)

    def diag_values_to_matrix(diag_values):
        """
        Convert diagonal values to a batched diagonal matrix.

        Args:
            diag_values (torch.Tensor): Tensor of diagonal values with shape [batch, token].

        Returns:
            torch.Tensor: Batched diagonal matrix with shape [batch, token, token].
        """
        #: [[https://pytorch.org/docs/stable/generated/torch.diag_embed.html][torch.diag_embed — PyTorch 2.4 documentation]]
        return torch.diag_embed(diag_values)

        ##
        # batch_size, num_tokens = diag_values.shape
        # #: [batch, token]

        # # Create a batched identity matrix
        # batched_eye = (
        #     torch.eye(num_tokens, device=diag_values.device)
        #     .unsqueeze(0)
        #     .expand(batch_size, -1, -1)
        # )
        # #: [batch, token, token]

        # # Multiply the batched identity matrix with the diagonal values
        # mat = batched_eye * diag_values.unsqueeze(-1)
        # #: [batch, token, token] * [batch, token, 1]

        # return mat  # [batch, token, token]
        ##

    def create_C0(initial_embeddings):
        # Apply L function to each initial embedding
        ##
        # diagonal_values = torch.stack(
        #     [L(embedding) for embedding in initial_embeddings]
        # )
        ##
        diagonal_values = L(initial_embeddings)  # Assuming L can handle batched input
        # [batch, token]

        C0 = diag_values_to_matrix(diagonal_values)

        return C0
        # [batch, token, token]

    def NECC_compute(*, tokens_before, tokens_after):
        """
        Compute the Normalized Embedding Cosine Coherence (NECC) using softmax.

        Args:
            tokens_before (torch.Tensor): Tensor of original token embeddings with shape [batch, token, embedding_dim].
            tokens_after (torch.Tensor): Tensor of transformed token embeddings with shape [batch, token, embedding_dim].

        Returns:
            torch.Tensor: NECC scores with shape [batch, token].
        """
        #: [[https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html][torch.nn.functional.cosine_similarity — PyTorch 2.4 documentation]]
        #: Dimension dim of the output is squeezed (see torch.squeeze()), resulting in the output tensor having 1 fewer dimension.
        C_E = F.cosine_similarity(tokens_before, tokens_after, dim=-1)  # [batch, token]

        # Apply softmax to the cosine similarities along the token dimension
        NECC = F.softmax(C_E, dim=-1)  # [batch, token]

        return NECC

    def W_compute(*, tokens_before, tokens_after):
        """
        Compute the transformation weights W.

        Args:
            tokens_before (torch.Tensor): Tensor of original token embeddings with shape [batch, token, embedding_dim].
            tokens_after (torch.Tensor): Tensor of transformed token embeddings with shape [batch, token, embedding_dim].
            L (callable): A function that takes a tensor of token embeddings and returns a tensor of shape [batch, token].

        Returns:
            torch.Tensor: Transformation weights W with shape [batch, token, token].
        """
        # Compute the L function on original and transformed embeddings
        L_E = L(tokens_before)  # [batch, token]
        L_E_tilde = L(tokens_after)  # [batch, token]

        # Compute NECC scores using the modified NECC_compute with softmax
        NECC = NECC_compute(
            tokens_before=tokens_before, tokens_after=tokens_after
        )  # [batch, token]

        # Compute the transformation weights
        W = (L_E_tilde / (L_E + epsilon)) * NECC
        # [batch, token]

        W = diag_values_to_matrix(W)

        return W  # [batch, token, token]

    input_patches = get_input_patches(batch)
    # [batch, token, embedding]

    C0 = create_C0(input_patches)
    C_curr = C0

    for l in range(1, blocks_len + 1):
        block_i = l - 1
        #: TokenTM uses 1-indexing for the layers

        ##
        if block_i == 0:
            tokens_pre_attn = get_input_patches(batch)

        else:
            tokens_pre_attn = batch[f"blocks__{block_i - 1}__output"]

        tokens_post_attn = model.blocks[block_i].stored_post_attn

        tokens_out = batch[f"blocks__{block_i}__output"]
        ##

        C_prev = C_curr

        O_identity = torch.eye(
            C_prev.shape[-1],
            device=C_prev.device,
            dtype=C_prev.dtype,
        ).unsqueeze(0)
        #: [1, token, token]

        rawattn = batch[f"blocks__{block_i}__attn__rawattn"]
        A = torch.mean(
            rawattn,
            dim=(1,),
            keepdim=False,
        )
        #: (batch, attention_to, attention_from)

        rawattn_grad = batch[f"blocks__{block_i}__attn__rawattn_grad"]
        A_grad_relu = torch.mean(
            rawattn_grad.relu(),
            dim=(1,),
            keepdim=False,
        )
        #: (batch, attention_to, attention_from)

        ##
        W_attn = W_compute(
            tokens_before=tokens_pre_attn,
            tokens_after=tokens_post_attn,
        )
        T = mat_combine(
            agg=W_attn,
            current=A,
        )

        U_attn = O_identity + torch.mul(T, A_grad_relu)
        ##
        W_MLP = W_compute(
            tokens_before=tokens_post_attn,
            tokens_after=tokens_out,
        )

        U_MLP = O_identity + W_MLP
        ##
        U_curr = mat_combine(
            agg=U_attn,
            current=U_MLP,
        )

        out_key = f"blocks__{block_i}__TokenTM"

        output[out_key] = U_curr
        ##
        C_curr = mat_combine(
            agg=C_prev,
            current=U_curr,
        )
        ##

    out_key = f"TokenTM"
    output[out_key] = C_curr

    batch.update(output)
    return batch


@skip_if_no_cls_attn
def transform_Mean__AttnGrad_Attn(
    batch,
    model,
):
    #: @Chefer2/layerwise @GradSAM @AttGrads/layerwise
    ##
    output = dict()

    blocks_len = len(model.blocks)
    for block_i in range(blocks_len):
        rawattn = batch[f"blocks__{block_i}__attn__rawattn"]
        rawattn_grad = batch[f"blocks__{block_i}__attn__rawattn_grad"]
        #: (batch, head, attention_to, attention_from)

        for mix_mode in [
            # True,
            # False,
            "relu",
            "abs",
            None,
        ]:
            rawattn_rawattngrad = torch.mul(rawattn, rawattn_grad)
            #: (batch, head, attention_to, attention_from)

            if mix_mode == "relu":
                rawattn_rawattngrad = torch.relu(rawattn_rawattngrad)
            elif mix_mode == "abs":
                rawattn_rawattngrad = torch.abs(rawattn_rawattngrad)

            mean__attn_attngrad = torch.mean(
                rawattn_rawattngrad,
                dim=(1,),
                keepdim=False,
            )
            #: (batch, attention_to, attention_from)

            if mix_mode == "relu":
                out_key = f"blocks__{block_i}__MeanReLU__AttnGrad_Attn"
            elif mix_mode == "abs":
                out_key = f"blocks__{block_i}__MeanAbs__AttnGrad_Attn"
            else:
                out_key = f"blocks__{block_i}__Mean__AttnGrad_Attn"

            output[out_key] = mean__attn_attngrad

    batch.update(output)
    return batch


@skip_if_no_cls_attn
def transform_MeanAttnGrad_MeanAttn(
    batch,
    model,
):
    output = dict()

    blocks_len = len(model.blocks)
    for block_i in range(blocks_len):
        MeanAttn = batch[f"blocks__{block_i}__MeanAttn"]
        #: (batch, attention_to, attention_from)

        for k in [
            "MeanAttnGrad",
            "MeanReLUAttnGrad",
        ]:
            input_name = f"blocks__{block_i}__{k}"

            MeanAttnGrad = batch[input_name]
            #: (batch, attention_to, attention_from)

            MeanAttnGrad_MeanAttn = torch.mul(MeanAttnGrad, MeanAttn)

            output[f"blocks__{block_i}__{k}_MeanAttn"] = MeanAttnGrad_MeanAttn

    batch.update(output)
    return batch


## *** CAT
def cat_channel_mixed_name_get(*, channel_mixer, name_prefix="CAT"):
    if False and channel_mixer == "sum":
        #: We have abandoned backcompat for this.
        ##
        return name_prefix
    else:
        return f"{name_prefix}_s:{channel_mixer}"


def cam_weights_get(
    *,
    gradients,
    activations,
    #: (batch, token_sequence, hidden)
    grad_relu_p=False,
    activations_abs_p=False,
    mode="GCAM",
    eps=1e-7,
):
    if grad_relu_p:
        gradients = torch.nn.functional.relu(gradients)

    if activations_abs_p:
        activations = torch.abs(activations)

    if mode == "XCAM":
        channel_activations_sum = activations.sum(dim=1, keepdim=True)
        #: (batch, 1, hidden)

        weights = gradients * activations
        #: (batch, token_sequence, hidden)

        weights = weights.sum(dim=1, keepdim=True)
        #: (batch, 1, hidden)

        weights = weights / (channel_activations_sum + eps)
        #: (batch, 1, hidden)

    elif mode == "GCAM":
        weights = torch.mean(gradients, dim=1, keepdim=True)
        #: (batch, 1, hidden)

    elif mode == "PCAM":
        #: GradCAM++

        channel_activations_sum = activations.sum(dim=1, keepdim=True)
        #: (batch, 1, hidden)

        alpha_numer = gradients.pow(2)

        alpha_denom = 2 * gradients.pow(2) + channel_activations_sum * gradients.pow(3)
        alpha = alpha_numer.div(alpha_denom + eps)

        weights = (alpha * gradients).sum(dim=1, keepdim=True)
        # ic(
        #     alpha.shape,
        #     weights.shape,
        #     gradients.shape,
        #     channel_activations_sum.shape,
        #     activations.shape,
        # )
        # ic| alpha.shape: torch.Size([50, 197, 768])
        # weights.shape: torch.Size([50, 1, 768])
        # gradients.shape: torch.Size([50, 197, 768])
        # channel_activations_sum.shape: torch.Size([50, 1, 768])
        # activations.shape: torch.Size([50, 197, 768])

    else:
        raise ValueError(f"{fn_name_current()}: Unsupported mode: {mode}")

    return weights


def cam_calculate(
    *,
    activations,
    weights,
    method_name,
    block_i,
    output,
    channel_mixers,
):
    weighted_feature_maps = activations * weights  #: Element-wise multiplication
    #: (batch, token_sequence, hidden)

    raw_name = f"blocks__{block_i}__{method_name}_s:raw"
    output[raw_name] = weighted_feature_maps

    for mode in channel_mixers:
        output = transform_scalarify(
            output,
            methods=[raw_name],
            mode=mode,
        )
        #: (batch, token_sequence)

    #: We do NOT do destructive visual post-processings such as ReLU and normalization here, as that's dumb. So our method here is strictly superior to the original GradCAM.
    #: (Same for AttnCAM)

    return output


def get_input_patches(batch):
    if "patches" in batch:
        h = batch["patches"]
    else:
        h = batch["patches_dv"].cpu()

    return h


def transform_CAT(
    batch,
    model,
    naming_mode="start",
    # naming_mode="end",
    # include_last_p=True,
    include_last_p=False,
    channel_mixers=None,
    compute_mode="full",
):
    output = dict()

    if channel_mixers is None:
        channel_mixers = globals()["channel_mixers"]

    grad_target_batch = batch["grad_target_batch"]
    #: [batch]

    blocks_len = len(model.blocks)

    if naming_mode == "start":
        if include_last_p:
            blocks_len += 1

    elif naming_mode == "end":
        if not include_last_p:
            blocks_len -= 1

    else:
        raise ValueError(f"{fn_name_current()}: Invalid naming_mode: {naming_mode}")

    for block_i in range(blocks_len):
        if naming_mode == "start":
            if block_i == 0:
                h = get_input_patches(batch)

                h_grad = batch[patch_grad_raw_name]
            else:
                h = batch[f"blocks__{block_i - 1}__output"]
                h_grad = batch[f"blocks__{block_i - 1}__output_grad"]
        elif naming_mode == "end":
            h = batch[f"blocks__{block_i}__output"]
            h_grad = batch[f"blocks__{block_i}__output_grad"]
            #: (batch, token_sequence, hidden)

        ##
        simple_grad_raw_name = f"blocks__{block_i}__PatchGrad_s:raw"
        output[simple_grad_raw_name] = h_grad
        for mode in channel_mixers:
            output = transform_scalarify(
                output,
                methods=[simple_grad_raw_name],
                mode=mode,
            )
            #: (batch, token_sequence)

        # del output[simple_grad_raw_name]
        ##

        if compute_mode == "full":
            ## Original GradCAM (GCAM)
            channel_grad_weights = cam_weights_get(
                gradients=h_grad,
                activations=h,
                grad_relu_p=False,
                activations_abs_p=False,
                mode="GCAM",
            )
            output = cam_calculate(
                activations=h,
                weights=channel_grad_weights,
                method_name="GCAM",
                block_i=block_i,
                output=output,
                channel_mixers=channel_mixers,
            )

            ## ReLUGradCAM (GRCAM)
            # relu_channel_grad_weights = cam_weights_get(
            #     gradients=h_grad,
            #     activations=h,
            #     grad_relu_p=True,
            #     activations_abs_p=False,
            #     mode="GCAM",
            # )
            # output = cam_calculate(
            #     activations=h,
            #     weights=relu_channel_grad_weights,
            #     method_name="GRCAM",
            #     block_i=block_i,
            #     output=output,
            #     channel_mixers=channel_mixers_positive,
            # )

            ## XGradCAM
            # xgradcam_weights = cam_weights_get(
            #     gradients=h_grad,
            #     activations=h,
            #     grad_relu_p=False,
            #     activations_abs_p=False,
            #     mode="XCAM",
            # )
            # output = cam_calculate(
            #     activations=h,
            #     weights=xgradcam_weights,
            #     method_name="XCAM",
            #     block_i=block_i,
            #     output=output,
            #     channel_mixers=channel_mixers,
            # )

            ## ReLUXGradCAM (XRCAM)
            # relu_xgradcam_weights = cam_weights_get(
            #     gradients=h_grad,
            #     activations=h,
            #     grad_relu_p=True,
            #     activations_abs_p=False,
            #     mode="XCAM",
            # )
            # output = cam_calculate(
            #     activations=h,
            #     weights=relu_xgradcam_weights,
            #     method_name="XRCAM",
            #     block_i=block_i,
            #     output=output,
            #     channel_mixers=channel_mixers,
            # )

            ## XAbsGradCAM (XACAM)
            xabs_gradcam_weights = cam_weights_get(
                gradients=h_grad,
                activations=h,
                grad_relu_p=False,
                activations_abs_p=True,
                mode="XCAM",
            )
            output = cam_calculate(
                activations=h,
                weights=xabs_gradcam_weights,
                method_name="XACAM",
                block_i=block_i,
                output=output,
                channel_mixers=channel_mixers,
            )

            ## ReLUXAbsGradCAM (XARCAM)
            # relu_xabs_gradcam_weights = cam_weights_get(
            #     gradients=h_grad,
            #     activations=h,
            #     grad_relu_p=True,
            #     activations_abs_p=True,
            #     mode="XCAM",
            # )
            # output = cam_calculate(
            #     activations=h,
            #     weights=relu_xabs_gradcam_weights,
            #     method_name="XARCAM",
            #     block_i=block_i,
            #     output=output,
            #     channel_mixers=channel_mixers_positive,
            # )
            ##

            ## GradCAM++ (PRCAM)
            # weights = cam_weights_get(
            #     gradients=h_grad,
            #     activations=h,
            #     grad_relu_p=True,
            #     activations_abs_p=False,
            #     mode="PCAM",
            # )
            # output = cam_calculate(
            #     activations=h,
            #     weights=weights,
            #     method_name="PRCAM",
            #     block_i=block_i,
            #     output=output,
            #     channel_mixers=channel_mixers,
            # )
            ##
            ## AbsGradCAM++ (PARCAM)
            # weights = cam_weights_get(
            #     gradients=h_grad,
            #     activations=h,
            #     grad_relu_p=True,
            #     activations_abs_p=True,
            #     mode="PCAM",
            # )
            # output = cam_calculate(
            #     activations=h,
            #     weights=weights,
            #     method_name="PARCAM",
            #     block_i=block_i,
            #     output=output,
            #     channel_mixers=channel_mixers,
            # )
            ##
            ## No-ReLU GradCAM++ (PCAM)
            # weights = cam_weights_get(
            #     gradients=h_grad,
            #     activations=h,
            #     grad_relu_p=False,
            #     activations_abs_p=False,
            #     mode="PCAM",
            # )
            # output = cam_calculate(
            #     activations=h,
            #     weights=weights,
            #     method_name="PCAM",
            #     block_i=block_i,
            #     output=output,
            #     channel_mixers=channel_mixers,
            # )
            ##
            ## No-ReLU AbsGradCAM++ (PACAM)
            # weights = cam_weights_get(
            #     gradients=h_grad,
            #     activations=h,
            #     grad_relu_p=False,
            #     activations_abs_p=True,
            #     mode="PCAM",
            # )
            # output = cam_calculate(
            #     activations=h,
            #     weights=weights,
            #     method_name="PACAM",
            #     block_i=block_i,
            #     output=output,
            #     channel_mixers=channel_mixers,
            # )
            ##

        IxG_raw = torch.mul(h, h_grad)  #: element-wise
        #: (batch, token_sequence, hidden)

        cat_name = f"blocks__{block_i}__CAT"
        cat_raw_name = f"blocks__{block_i}__CAT_s:raw"

        output[cat_name] = IxG_raw
        # output[cat_raw_name] = IxG_raw
        #: We would need to handle the raw version in =attn_prepare_attribution_columns=.

        #: IxG Completeness Errors
        IxG_sum = IxG_raw.sum(dim=tuple(range(1, IxG_raw.dim())))
        #: sum all dims except first (batch)

        completeness_error = IxG_sum - grad_target_batch
        #: We store the raw error per batch item.

        # if block_i == 0:
        #     ic(torch_shape_get(IxG_sum), torch_shape_get(grad_target_batch))
        #     ic| torch_shape_get(IxG_sum): (torch.float32, torch.Size([48]), device(type='cpu'))
        #         torch_shape_get(grad_target_batch): (torch.float32, torch.Size([48]), device(type='cpu'))

        del IxG_sum

        completeness_error_i_name = f"blocks__{block_i}__IxG_completeness_error"
        output[completeness_error_i_name] = completeness_error

        ##
        if True:
            #: See `cat_channel_mixed_name_get'.

            for mode in channel_mixers:
                output = transform_scalarify(
                    output,
                    methods=[cat_name],
                    mode=mode,
                )
                #: (batch, token_sequence)
        else:
            for mode in channel_mixers_no_sum:
                output = transform_scalarify(
                    output,
                    methods=[cat_name],
                    mode=mode,
                )
                #: (batch, token_sequence)

            #: For backcompat, we name =CAT_s:sum= as =CAT=:
            output = transform_scalarify(
                output,
                methods=[cat_name],
                mode="sum",
                name="",  #: replaces the old value
            )
        ##
        # cat_relusum = F.relu(IxG_raw).sum(dim=-1)
        # #: (batch, token_sequence)

        # output[f"blocks__{block_i}__CAT_RS"] = cat_relusum
        ##

        ## IxG on MHA
        #: @NotImplemented for now
        # MHA = batch[f"blocks__{block_i}__MHA"]
        # MHAGrad = batch[f"blocks__{block_i}__MHAGrad"]
        # ic(MHA.shape, MHAGrad.shape)
        # #: @? (batch, token_sequence, head, hidden)

        # MHACLS_raw_name = f"blocks__{block_i}__MHACLS_s:raw"

        # MHA_IxG_raw = torch.mul(MHA, MHAGrad)  #: element-wise
        # MHACLS_raw = MHA_IxG_raw[:, 0]
        # output[cat_raw_name] = MHA_IxG_raw
        ##
        if compute_mode == "IxG":
            break
            #: No need to compute other layers

    batch.update(output)
    return batch


@skip_if_no_attn
def transform_CAT_AttnFrom(
    batch,
    model,
    naming_mode="start",
    #: CAT's naming mode
    #: AttCAT's uses the IxG (CAT) started from the next layer together with the current layer's attention
    include_last_p=False,
    multiply_with=None,
    # multiply_with=[
    #     "MeanAttn",
    #     # "AttnWHeadGrad",
    #     "MeanAttnGrad",
    #     "MeanReLUAttnGrad",
    #     "MeanAttnGrad_MeanAttn",
    #     "MeanReLUAttnGrad_MeanAttn",
    # ],
):
    output = dict()

    multiply_with = to_iterable(multiply_with)

    blocks_len = len(model.blocks)
    if not include_last_p:
        blocks_len -= 1

    for block_i in range(blocks_len):
        rawattn = batch[f"blocks__{block_i}__attn__rawattn"]
        #: (batch, head, attention_to, attention_from)

        attn_mean_from = rawattn.mean(dim=(1, 2), keepdim=False)
        #: (batch, attention_from)

        names = [
            cat_channel_mixed_name_get(channel_mixer=channel_mixer, name_prefix="CAT")
            for channel_mixer in channel_mixers
        ]
        # names = ["CAT"]
        # names += [f"CAT_s:{name}" for name in channel_mixers_no_sum]

        for name in names:
            if naming_mode == "start":
                cat = batch[f"blocks__{block_i+1}__{name}"]
            elif naming_mode == "end":
                cat = batch[f"blocks__{block_i}__{name}"]
                #: (batch, token_sequence)
            else:
                raise ValueError(
                    f"{fn_name_current()}: Invalid naming_mode: {naming_mode}"
                )

            CAT_AttnFrom = torch.mul(attn_mean_from, cat)
            #: (batch, token_sequence)

            output[f"blocks__{block_i}__{name}_AttnFrom"] = CAT_AttnFrom

            for k in multiply_with:
                #: used to be =transform_MeanAttn_CAT=
                #: @novel
                ##
                MeanAttn = batch[f"blocks__{block_i}__{k}"]
                #: (batch, attention_to, attention_from)

                MeanAttn_CAT = einops.einsum(
                    MeanAttn,
                    cat,
                    "batch attention_to attention_from, batch attention_to -> batch attention_to attention_from",
                )

                output[f"blocks__{block_i}__{name}_{k}"] = MeanAttn_CAT

    batch.update(output)
    return batch


## * Visualization
def attributions_show2(
    *,
    batch,
    model,
    attributions_cols,
    attributions_col_patterns=None,
    # bias_token_p=False,
    tds_torch_cpu=None,
    model_patch_info,
    token_i="auto",
    # token_i=0,
    title=None,
    title_mode="verbose",
    normalize=["rank_uniform"],
    color_positive="viridis",
    scale=1,
    batch_from=0,
    batch_to=999,
    batch_size=None,
    tlg_msg="",
    export_dir=None,
    label_topk=4,
    plot_output_p=False,
    export_tlg_id=tlg_me,
    enabled_p=True,
    compact_gbrand="",
    tqdm_name=None,
    coco_p=False,
    **kwargs,
):
    #: @assume the input is a TransformedDataset
    #: We can later on convert dict, BatchedDict to TransformedDataset, too
    ##
    if not enabled_p:
        return

    colormap_p = isinstance(color_positive, str)

    clip_p = getattr(model, "clip_p", False)

    label_topk_orig = label_topk

    data_size = len(batch)
    if batch_to is None:
        batch_to = data_size
    else:
        batch_to = min(batch_to, data_size)

    # ic(batch_from, batch_to, batch_size,)

    # batch = batch[batch_from:batch_to]
    batch = batch.select(range(batch_from, batch_to))
    data_n = batch_to - batch_from

    batch_size = batch_size or len(batch)
    batch_n = math.ceil(data_n / batch_size)
    # ic(torch_shape_get(batch))

    if tlg_msg and export_tlg_id:
        common_telegram.send(export_tlg_id, msg=tlg_msg)

    attributions_cols = to_iterable(attributions_cols)
    attributions_col_patterns = to_iterable(attributions_col_patterns)

    title = title or ""

    # if model_patch_info is None:
    #     print(f"{fn_name_current()}: model_patch_info set from main")

    model_name = model_name_get(model)
    # model_name = model_patch_info.model_name

    print(f"{fn_name_current()}: batch_size={batch_size}, batch_n={batch_n}")
    for batch in tqdm(
        batch.batched_iterator(batch_size),
        total=(batch_n),
        name=tqdm_name,
    ):
        ids = batch["id"]
        # ic(ids)

        if clip_p:
            label_topk = min(label_topk_orig, len(batch))
            #: CLIP models produce n=batch_size logits.

            # assert tds_torch_cpu is None
            #: =tds_torch_cpu= is required for =image_natural=.

        if coco_p or tds_torch_cpu is None:
            #: COCO produces multiple images from each item in the initial input of the transform pipeline, so getting `batch_inputs` from it is challenging. It's best to take care not to remove columns during the transforms instead.
            batch_inputs = batch

        else:
            batch_inputs = tds_torch_cpu[ids]

        logits = batch["logits"]
        probs = F.softmax(logits, dim=-1)

        for image_i in range(0, len(batch)):
            id_current = ids[image_i]
            if "image_id" in batch_inputs:
                image_id_current = batch_inputs["image_id"][image_i]

            elif "imagenet_id" in batch_inputs:
                image_id_current = batch_inputs["imagenet_id"][image_i]

            else:
                image_id_current = None

            # ic(image_i)
            if "clip_text" in batch_inputs:
                assert clip_p

                label_natural = batch_inputs["clip_text"][image_i]

            elif "label_natural" in batch_inputs:
                label_natural = batch_inputs["label_natural"][image_i]

            else:
                label_natural = "label_natural not found"

            print(
                f"\nid_current: {id_current}\nimage_id_current: {image_id_current}\nimage_i: {image_i}\nlabel_natural: {label_natural}\n"
            )

            logits_current = logits[image_i]
            probs_current = probs[image_i]
            try:
                probs_topk, labels_topk = torch.topk(
                    probs_current,
                    label_topk,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )

            except:
                ic(
                    id_current,
                    image_i,
                    data_size,
                    len(batch),
                    label_topk,
                    torch_shape_get(logits),
                    torch_shape_get(logits_current),
                    torch_shape_get(probs),
                    torch_shape_get(probs_current),
                )

                raise

            attributions_cols_current = list(attributions_cols)
            for k in sorted(
                list(batch.keys()),
                key=partial(
                    version_sort_key,
                    float_p=True,
                ),
            ):
                if k not in attributions_cols_current and any(
                    re.search(p, k) for p in attributions_col_patterns
                ):
                    attributions_cols_current.append(k)

            attributions_cols_current = list_dup_rm(attributions_cols_current)
            for attributions_col in attributions_cols_current:
                export_dir_current = export_dir

                if attributions_col not in batch:
                    print(
                        f"{fn_name_current()}: non-existent attributions_col: {attributions_col}"
                    )
                    continue

                normalize_ = normalize
                export_name_normalize_part = None
                if normalize_ == "auto":
                    if any(
                        pat in attributions_col
                        for pat in [
                            "XACAM",
                            "GCAM",
                        ]
                    ):
                        export_name_normalize_part = "shift2zero"
                        normalize_ = [
                            "shift_min_to_zero",
                            "scale_by_max_abs_attr",
                        ]

                    else:
                        normalize_ = [
                            "relu",
                            "scale_by_max_signed_attr",
                        ]

                attributions = batch[attributions_col]
                # ic(len(ids), torch_shape_get(attributions))

                if title_mode == "verbose":
                    title_current = f"{title}{model_name}\nlabel: {label_natural}\nAttr: {attributions_col}\nnormalize_: {','.join(normalize_)}"

                elif title_mode == "minimal":
                    title_current = f"{title}\nTarget: {label_natural}"

                else:
                    title_current = title

                for top_i, label_id in enumerate(labels_topk):
                    prob = probs_topk[top_i]
                    if prob < 0.001:
                        prob_str = "{:.2e}".format(prob)

                    else:
                        prob_str = "{:.3f}".format(prob)

                    if clip_p:
                        top_label_natural = batch_inputs["clip_text"][label_id]

                        # ic(batch_inputs["clip_text"], top_i, label_id, prob, top_label_natural)

                    elif imagenet_p():
                        top_label_natural = label_natural_get()[label_id]

                    else:
                        raise NotImplementedError(
                            f"{fn_name_current()}: DATASET_NAME not supported: {DATASET_NAME}"
                        )

                    if title_mode == "verbose":
                        title_current += (
                            f"\n{top_i + 1}. {prob_str}: {top_label_natural}"
                        )

                attributions_current = attributions[image_i]

                token_i_current = token_i

                if token_i_current == "auto":
                    ##
                    attributions_current = attributions_autoget(
                        attributions_current.unsqueeze(0),
                        name=attributions_col,
                    ).squeeze(0)
                    #: 'attributions_autoget' expects a batch dim
                    ##
                    # if attributions.ndim == 3:
                    #     token_i_current = 0
                    # elif attributions.ndim == 2:
                    #     token_i_current = None
                    # else:
                    #     raise ValueError(
                    #         f"attributions.ndim not supported: {attributions.ndim} "
                    #     )
                    ##
                else:
                    if token_i_current:  #: not None or 0
                        if title_mode == "verbose":
                            title_current += "\ntokei_i: {token_i_current}"

                    if token_i_current is not None:
                        attributions_current = attributions_current[token_i_current]

                attributions_col_export = attributions_col
                if attributions_col_export.startswith("attributions_s_"):
                    attributions_col_export = attributions_col_export[
                        len("attributions_s_") :
                    ]

                if export_dir_current:
                    export_dir_current += (
                        f"/{model_name}/{compact_gbrand}/{attributions_col_export}"
                    )

                    if colormap_p:
                        export_dir_current += f"/{color_positive}"

                    if export_name_normalize_part:
                        export_dir_current += f"/{export_name_normalize_part}"

                if image_id_current is not None:
                    export_name = f"{image_id_current}"

                else:
                    export_name = f"{id_current}"

                export_name += f"/{label_natural}"

                overlay_colored_grid(
                    pixel_p=attr_pixel_level_p(attributions_col),
                    image_natural_tensor=batch_inputs["image_natural"][image_i],
                    # image_natural=batch_inputs["image_natural"][image_i],
                    # image_concats_right=[batch_inputs["image_natural"][image_i]],
                    model_patch_info=model_patch_info,
                    # patch_size=model_patch_info.patch_resolution,
                    # bias_token_p=bias_token_p,
                    attributions_n=attributions_current,
                    normalize=normalize_,
                    color_positive=color_positive,
                    scale=scale,
                    title=title_current,
                    plot_output_p=plot_output_p,
                    export_tlg_id=export_tlg_id,
                    export_dir=export_dir_current,
                    export_name=export_name,
                    **kwargs,
                )

    print(f"{fn_name_current()}: finished")


def attributions_show1(batch, i, attributions, bias_token_p=False, **kwargs):
    #: @deprecated
    ##
    # ic(torch_shape_get(attributions))
    # ic(attributions.sum(dim=-1))

    overlay_colored_grid(
        image_natural_tensor=batch["image_natural"][i],
        image_concats_right=[batch["image_natural"][i]],
        patch_size=model_patch_info.patch_resolution,
        attributions_n=(attributions[i]),
        plot_output_p=False,
        export_tlg_id=tlg_me,
        normalize=["rank_uniform"],
        color_positive="viridis",
        scale=1,
        bias_token_p=bias_token_p,
        **kwargs,
    )


## * Masked Prediction
def masked_predict(
    *,
    tds_patches,
    tds_masked=None,
    batch_size,
    model,
    output_precision=None,
    # output_precision="float16",
    # output_precision='bfloat16',
    transform_only_p=False,
    mapconcat_opts=None,
    **dummy,
):
    if tds_masked is None:
        tds_masked = tds_patches
        #: For when all the processing is happening on the fly and we aren't saving the intermediate transforms.

    if mapconcat_opts is None:
        mapconcat_opts = dict()

    metadata = dict()
    metadata["output_precision"] = output_precision
    ic(metadata)

    process_fn = partial(
        dsmap_predict_masked,
        model=model,
        tds_patches=tds_patches,
        #: @perf/time Beware that if tds_patches is used by 'dsmap_predict_masked', we will calculate all the transforms twice!
        output_precision=output_precision,
    )
    process_fn = partial_dynamic(
        process_fn,
        dynamic_dict=decomposition.dynamic_vars,
        print_diag_file="log",
    )

    tds_after = None
    dataset_masked_logits = None
    if transform_only_p:
        tds_after = tds_masked.transform(
            process_fn,
        )
    else:
        # print_diag_sep()

        if isinstance(
            tds_masked, pynight.common_datasets.ConcatenatedTransformedDataset
        ):
            tds_masked_dataset = tds_masked.datasets_concatenated().with_format("torch")
        else:
            tds_masked_dataset = tds_masked.dataset

        dataset_masked_logits = mapconcat(
            tds_masked_dataset,
            tds_masked.fn_with_transforms(process_fn),
            unchanged_keep_columns=["id"],
            # new_fingerprint=f'masked_log_{model_name_get(model)}',
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=False,
            **mapconcat_opts,
        )

        # print_diag_sep()

    return simple_obj(
        dataset_all=dataset_masked_logits,
        tds_after=tds_after,
        metadata=metadata,
    )


def transform_probability(
    batch,
    unchanged_keep_columns=None,
):
    batch_new = dict()
    for k, v in batch.items():
        if k == "logits":
            name = k
        else:
            name = rget(k, "^logits_(.*)$")

        if name:
            logits = v
            ##
            #: @duplicateCode/df023e9a4445d7cbfe8f574de6bd4878
            if logits.shape[-1] == 1:
                #: Two Classes Using Sigmoid

                # ic("Sigmoid (pre)", torch_shape_get(logits))

                #: Create logits for the sigmoid case
                zero_logits = torch.zeros_like(logits)
                logits = torch.cat([zero_logits, logits], dim=-1)
                del zero_logits
            ##

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            batch_new[f"prob_{name}"] = probabilities

            predicted_label = torch.argmax(logits, dim=-1)
            batch_new[f"predictedLabel_{name}"] = predicted_label

    # ic(torch_shape_get((batch, batch_new)))
    if unchanged_keep_columns is not None:
        for col in unchanged_keep_columns:
            batch_new[col] = batch[col]

        batch = batch_new
    else:
        batch.update(batch_new)

    return batch


## ** Masked CLS Metrics
from pynight.common_evaluate import ConfiguredMetric


def xp_groups_get(tds_masked_pred):
    xp_groups = []

    if isinstance(tds_masked_pred, dict):
        #: We have been given an example transformed minibatch.
        keyholder = tds_masked_pred
    else:
        keyholder = tds_masked_pred.dataset

    for k in list(keyholder.keys()):
        if k == "logits":
            name = k
        else:
            name = rget(k, "^logits_(mask_(?:topratio|nratio).*)$")

        if name:
            xp_groups.append(name)

    xp_groups.sort()
    return xp_groups


def compute_cls_metrics(
    dataset_indexed,
    tds_masked_pred,
    logits_orig_dict,
    xp_groups,
    kfold=10,
):
    xp_metrics = dict()

    hf_evaluate_p = False
    if hf_evaluate_p:
        clf_metrics = [
            evaluate.load(
                "accuracy",
            ),
        ]
        for avg_mode in ["macro", "micro", "weighted"]:
            clf_metrics += [
                ConfiguredMetric(
                    evaluate.load(
                        "f1",
                    ),
                    average=avg_mode,
                ),
                ConfiguredMetric(
                    evaluate.load(
                        "precision",
                    ),
                    average=avg_mode,
                    zero_division=1,
                ),
                ConfiguredMetric(
                    evaluate.load(
                        "recall",
                    ),
                    average=avg_mode,
                    zero_division=1,
                ),
            ]

        clf_metrics = evaluate.combine(clf_metrics)

    with Timed(
        name="""tds_masked_pred.dataset["id"]""",
        enabled_p=False,
        # enabled_p=True,
    ):
        ids = tds_masked_pred.dataset["id"]
        ids = ids.tolist()

    prob_logits = logits_orig_dict["prob_logits"]

    target_mode = metrics_target_get()

    if target_mode == ground_truth_mode_cst:
        with Timed(
            name="""dataset_indexed[ids]["label"]""",
            enabled_p=False,
            # enabled_p=True,
        ):
            refs = IndexableList(dataset_indexed["label"])[ids]
            #: `dataset_indexed[ids]["label"]` decodes images and is thus very slow!
        ic(len(refs))

    elif target_mode == first_highest_predicted_mode_cst:
        refs = get_first_predicted_indices(prob_logits)

    elif target_mode == second_highest_predicted_mode_cst:
        refs = get_second_predicted_indices(prob_logits)

    else:
        raise ValueError(f"Unsupported metrics target_mode: {target_mode}")

    with Timed(
        name="""tds_masked_pred[:]""",
        enabled_p=False,
        # enabled_p=True,
    ):
        batch = tds_masked_pred[:]

    orig_probs = prob_logits[torch.arange(len(refs)), refs]

    for g in xp_groups:
        ic(g)
        res = dict()
        res["len"] = len(refs)

        predicted_probs = batch[f"prob_{g}"]
        #: (batch_size, 1000)

        if len(predicted_probs) != len(refs):
            print_diag(
                f"compute_cls_metrics: g={g}: len(predicted_probs)={len(predicted_probs)} != len(refs)={len(refs)}, skipping g"
            )
            continue

        predicted_probs = predicted_probs[torch.arange(len(refs)), refs]
        #: (batch_size,)

        aopc_results = compute_aopc_lodds(
            refs=refs,
            orig_probs=orig_probs,
            predicted_probs=predicted_probs,
            kfold=kfold,
        )
        res.update(aopc_results)
        ###
        preds = batch[f"predictedLabel_{g}"]
        assert len(preds) == len(refs)

        if hf_evaluate_p:
            with Timed(
                name="clf_metrics",
            ):
                res.update(
                    clf_metrics.compute(
                        references=refs,
                        predictions=preds,
                    ),
                )

        else:
            with Timed(
                name="sklearn_metrics",
            ):
                #: [[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html][accuracy_score — scikit-learn 1.5.0 documentation]]
                accuracy = accuracy_score(refs, preds)
                # accuracy_var = np.var([accuracy_score(refs[i:i+1], preds[i:i+1]) for i in range(len(refs))])

                ##: Calulate direct variances:
                accuracy_var = accuracy * (1 - accuracy)
                #: Variance of Bernoulli distribution
                #: [[https://en.wikipedia.org/wiki/Bernoulli_distribution][Bernoulli distribution - Wikipedia]]

                accuracy_var_of_sample_mean = accuracy_var / len(preds)
                ic(accuracy, accuracy_var, accuracy_var_of_sample_mean, len(preds))

                res.update(
                    {
                        "accuracy": accuracy,
                        "accuracy_var": accuracy_var,
                        "accuracy_var_of_sample_mean": accuracy_var_of_sample_mean,
                    }
                )
                ##
                #: [[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html][f1_score — scikit-learn 1.5.0 documentation]]
                #: Macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                #: weighted: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
                #: zero_division: Sets the value to return when there is a zero division. If set to “warn”, this acts like 0, but a warning is also raised. - If set to np.nan, such values will be excluded from the average.
                average_mode = "weighted"
                metrics = {
                    # "accuracy": accuracy_score,
                    "precision": lambda y_true, y_pred: precision_score(
                        y_true,
                        y_pred,
                        average=average_mode,
                        zero_division=np.nan,
                    ),
                    "recall": lambda y_true, y_pred: recall_score(
                        y_true,
                        y_pred,
                        average=average_mode,
                        zero_division=np.nan,
                    ),
                    "f1": lambda y_true, y_pred: f1_score(
                        y_true,
                        y_pred,
                        average=average_mode,
                        zero_division=np.nan,
                    ),
                }

                res_other_metrics = cls_metrics_get(
                    refs=refs,
                    preds=preds,
                    metrics=metrics,
                    kfold=kfold,
                )
                res.update(res_other_metrics)
                ##

        # if g == 'mask_topratio10_attributions_n_rnd1':
        #     ic(g, res)

        ###

        xp_metrics[g] = res

        # break

    return xp_metrics


def logits_to_cls_metrics(
    dataset_masked_logits,
    *,
    logits_orig_dict=None,
    dataset_indexed,
):
    tds_masked_pred = TransformedDataset(dataset_masked_logits)
    tds_masked_pred = tds_masked_pred.transform(transform_probability)

    # ic(len(tds_masked_pred), len(tds_masked_pred.dataset), len(dataset_masked_logits))

    xp_groups = xp_groups_get(tds_masked_pred)
    # embed()

    xp_cls_metrics = compute_cls_metrics(
        dataset_indexed=dataset_indexed,
        tds_masked_pred=tds_masked_pred,
        logits_orig_dict=logits_orig_dict,
        xp_groups=xp_groups,
    )

    return xp_cls_metrics


def pt_logits_to_cls_metrics(
    directory,
    *,
    json_path="auto",
    name=None,
    logits_orig_dict=None,
    directories_unmasked=None,
    **kwargs,
):
    metrics_target = metrics_target_get()

    names = []
    if directories_unmasked is not None:
        logits_dict = logits_orig_dict

        names += [os.path.basename(d) for d in directories_unmasked]
        # embed()
    else:
        name = name or os.path.basename(directory)
        names.append(name)
        # print(f"pt_logits_to_cls_metrics: starting: {name}")

        logits_dict = tensor_partitioned_load(directory, name=name)

        # ic(directory, torch_shape_get(logits_dict))
        # os._exit(0)

        # print(f"pt_logits_to_cls_metrics: loaded tensors: {name}")

    xp_cls_metrics = logits_to_cls_metrics(
        logits_dict,
        logits_orig_dict=logits_orig_dict,
        **kwargs,
    )

    for name in names:
        xp_cls_metrics_current = dict(xp_cls_metrics)
        if "logits" in xp_cls_metrics_current:
            xp_cls_metrics_current[name[7:]] = xp_cls_metrics_current["logits"]
            #: name[7:] removes 'logits_' from name.
            del xp_cls_metrics_current["logits"]

        json_path_current = json_path
        if json_path_current == "auto":
            start_index = int(logits_dict["id"][0])
            end_index = int(logits_dict["id"][-1]) + 1
            #: @assuming The IDs are contiguous, which they might not be.

            json_path_current = f"{MODEL_CLS_METRICS_ROOT}/{name}/{metrics_target}/{start_index}_{end_index}.json"
        else:
            assert len(names) == 1

        # ic(json_path_current)
        mkdir(json_path_current, do_dirname=True)
        with open(json_path_current, "w") as f:
            json.dump(xp_cls_metrics_current, f)

    return simple_obj(
        xp_cls_metrics=xp_cls_metrics,
        # json_path=json_path_current,
    )


def funcall1(args):
    fn, *args = args
    return fn(*args)


def pt_children_logits_to_cls_metrics(
    directory,
    debug_p=False,
    # debug_p=True,
    **kwargs,
):
    logits_dir = f"{directory}/logits"
    if not os.path.exists(logits_dir):
        logits_dir = f"{directory}/../../A/{ground_truth_mode_cst}/logits"

    if not os.path.exists(logits_dir):
        logits_dir = f"{directory}/../../A/T/logits"
        #: From, e.g., 'combinedv1/T'

    assert os.path.exists(logits_dir)

    directories = list_children(
        directory,
        include_patterns=r"^logits_",
    )
    # ic(directories[:2])

    tasks = []
    ##
    logits_orig_dict = None
    if os.path.exists(logits_dir):
        logits_orig_dict = tensor_partitioned_load(logits_dir)
        #: contains IDs and the predicted logits without masking

        logits_orig_dict = transform_probability(logits_orig_dict)
        # embed()

        nratio_pattern = r"nratio(\d+(?:\.\d+)?)"
        topratio_pattern = r"topratio(\d+(?:\.\d+)?)"

        # Filter directories that match either of the patterns
        directories_filtered = [
            d
            for d in directories
            if re.search(nratio_pattern, d) or re.search(topratio_pattern, d)
        ]

        # Apply substitutions to the filtered directories
        directories_unmasked = set(
            [
                re.sub(
                    nratio_pattern,
                    "nratio0",
                    re.sub(
                        topratio_pattern,
                        "topratio100",
                        d,
                    ),
                )
                for d in directories_filtered
            ]
        )

        process_fn2 = partial(
            pt_logits_to_cls_metrics,
            logits_orig_dict=logits_orig_dict,
            directories_unmasked=directories_unmasked,
            **kwargs,
        )

        tasks.append((process_fn2, None))

        # if debug_p:
        #     return [process_fn2(None)]

        # embed()
    else:
        raise Exception("Unmasked logits not found!")
    ##
    process_fn = partial(
        pt_logits_to_cls_metrics,
        logits_orig_dict=logits_orig_dict,
        **kwargs,
    )
    tasks += [(process_fn, directory) for directory in directories]

    if debug_p:
        return [process_fn(ic(directories[0]))]

    with multiprocessing.Pool() as pool:
        metrics = list(
            tqdm(
                pool.imap_unordered(funcall1, tasks),
                total=len(tasks),
            ),
        )

    # with multiprocessing.Pool() as pool:
    #     metrics = list(
    #         tqdm(pool.imap_unordered(process_fn, directories), total=len(directories))
    #     )
    ##
    # with multiprocessing.Pool() as pool:
    #     metrics = list(tqdm_orig(pool.imap_unordered(process_fn, directories), total=len(directories)))
    ###
    #: This threaded paralellization did not help much.
    # def process_directory(d):
    #     # with redirect_print_to_tqdm():
    #     with nullcontext():
    #         return pt_logits_to_cls_metrics(d, **kwargs)
    ##
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     metrics = list(tqdm_orig(executor.map(process_directory, directories), total=len(directories)))
    ##
    # with ThreadPoolExecutor(max_workers=48) as executor:
    #     futures = []
    #     for d in tqdm_orig(directories, position=-1):
    #         futures.append(executor.submit(process_directory, d))
    #     metrics = [f.result() for f in futures]
    ###
    # metrics = []
    # for d in tqdm_orig(directories, position=-1):
    #     with redirect_print_to_tqdm():
    #         metrics.append(pt_logits_to_cls_metrics(d, **kwargs))
    ##
    return metrics


##
def attr_pixel_level_p(name):
    if name is not None:
        return re.search(r"^(?:attributions_[^_]+_)?Image", name)

    else:
        return False


def attr_pos_embed_p(name):
    if name is not None:
        return re.search(r"^(?:attributions_[^_]+_)?PosE", name)
    else:
        return False


def attributions_autoget(
    v,
    name=None,
    warn_cls_p=False,
):
    try:
        # ic(name, torch_shape_get(v))

        if v.ndim == 2:  #: (batch, attr_of_input_token)
            return v

        elif v.ndim == 3:
            if attr_pixel_level_p(name):
                #: batch, x, y
                return v

            if v.shape[-1] == v.shape[-2]:  #: (batch, attr_to, attr_from,)
                if warn_cls_p and name:
                    print(f"Selected CLS: {name}")

                return v[..., 0, :]  #: Select attributions for the CLS token
            # elif v.shape[-1] == 1000:  #: (batch, attr, class)
            #     return v
            else:  #: (batch, token, hidden)
                return None

        elif v.ndim == 4:  #: (batch, head, attr_to, attr_from)
            if name:
                print(f"Dropped non-averaged heads: {name}")

            return None

    except:
        if name is not None:
            ic(name, torch_shape_get(v))

        raise


## *** Metric Dict to Dataframe
def df_sort(
    df,
    sort_by="name",
    ascending=True,
):
    if sort_by == "name":

        def sort_key(series):
            if series.name == "method":
                series = series.apply(h_attr_sort_key)

            # ic(series)
            return series

        sort_columns = ["method"]
        if isinstance(df.index, pandas.MultiIndex):
            if "metric" in df.index.names:
                if True:

                    def metric_sort_key(series):
                        def h(x):
                            x = x.lower()
                            if x == "accuracy":
                                return (0, x)
                            else:
                                return (1, x)

                        return series.apply(h)

                    df = df.sort_values(
                        by="metric", key=metric_sort_key, ascending=True
                    )
                else:
                    sort_columns.append("metric")
                    #: triggers an upstream bug:
                    #: `unhashable type: 'list'`

        df = df.sort_values(
            by=sort_columns,
            key=sort_key,
            ascending=True,
            kind="stable",
        )
        # df = df.sort_index(ascending=True)
        # df = df.sort_values(
        #     "method",
        #     key=sort_key,
        #     ascending=True,
        # )
    else:
        df = df.sort_values(sort_by, ascending=ascending)

    return df


def metric_dict_to_df(
    xp_metrics,
    *,
    ratio_pattern,
    plabel="top-ratio",
    sort_opts="auto",
    official_only_p=False,
    officialize_names=True,
    scale_by=100,
    #: multiplies all values by this much
):
    rows = []
    for method, metrics in xp_metrics.items():
        #: 'method' starts with 'mask_...'
        method_type = rget(method, "attributions_(?:n|s)_(.*)$")
        # ic(method, method_type)
        if officialize_names:
            method_type_official = attr_name_official_get(method_type, mode="ALL")
        else:
            method_type_official = None
            official_only_p = False

        if method_type_official is None:
            if official_only_p:
                print(
                    f"Warning: method_type dropped because no official name provided: {method_type}"
                )
                continue
        else:
            method_type = method_type_official
        # ic(method_type)

        for metric, value in metrics.items():
            topratio = rget(method, ratio_pattern)
            if topratio is None:
                continue

            topratio = float(topratio)

            rows.append([topratio, method_type, metric, value * scale_by])

    df = pd.DataFrame(rows, columns=[plabel, "method", "metric", "value"])

    if sort_opts == "auto":
        sort_opts = dict()

    if sort_opts is not None:
        df = df_sort(
            df,
            **sort_opts,
        )

    return df


metric_dict_to_df_topratio = partial(
    metric_dict_to_df,
    ratio_pattern=r"_topratio(\d+(?:\.\d+)?)_",
    # plabel="Percentage of Lowest Patches Discarded",
)

metric_dict_to_df_nratio = partial(
    metric_dict_to_df,
    ratio_pattern=r"_nratio(\d+(?:\.\d+)?)_",
    # plabel="Percentage of Top Patches Discarded",
)
## *** Plotting
import hashlib
import matplotlib
import matplotlib.pyplot as plt
import tempfile


def xp_cls_metrics_plot(
    df,
    #: Selected metrics to plot
    metrics_to_plot=[
        "accuracy",
        # "f1_average_macro",
        "f1_average_weighted",
        # "precision_average_macro",
        # "precision_average_weighted",
        # "recall_average_macro",
        "recall_average_weighted",
        "aopc_mean",
        "aopc_var",
        "lodds_mean",
        "lodds_var",
    ],
    alpha=0.75,
    linewidth=2,
    length=None,
    color_deterministic_p=False,
    xlabel="Percentage of Top Patches Kept",
    export_tlg_id="auto",
    show_p="auto",
    backend="matplotlib",
    tlg_msg="",
    title="",
):
    #: The plot assumes the values to be between 0 and 1, so be careful with the normalization. Currently we are assuming the dataframe stores values between 0-100.
    ##
    import plotly.io as pio
    import plotly.graph_objects as go

    tlg_msg = str(tlg_msg)
    title = str(title)

    df = df.sort_values("top-ratio", ascending=True)

    if export_tlg_id == "auto":
        export_tlg_id = tlg_me

    if show_p == "auto":
        show_p = not export_tlg_id

    ##
    linestyles = [
        "--",
        "-.",
        ":",
    ]
    # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    linestyles_plotly = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
    markers = ["o", "v", "^", "<", ">", "s", "p", "*"]

    open_dot_markers = [
        "circle-open-dot",
        "square-open-dot",
        "diamond-open-dot",
        "cross-open-dot",
        "x-open-dot",
        "triangle-up-open-dot",
        "triangle-down-open-dot",
        "triangle-left-open-dot",
        "triangle-right-open-dot",
        "triangle-ne-open-dot",
        "triangle-se-open-dot",
        "triangle-sw-open-dot",
        "triangle-nw-open-dot",
        "pentagon-open-dot",
        "hexagon-open-dot",
        "hexagon2-open-dot",
        "octagon-open-dot",
        "star-open-dot",
    ]
    markers_plotly = open_dot_markers

    color_len = 20  #: 10 or 20
    colormap = matplotlib.colormaps.get_cmap(f"tab{color_len}b")

    def get_color(method):
        hash_object = hashlib.md5(method.encode())
        return int(hash_object.hexdigest(), 16) % color_len

    ##
    title += f"ImageNet Validation"
    if length:
        title += f" (len={length})"

    for metric in metrics_to_plot:
        if any(
            p in metric
            for p in [
                "lodds",
                "_var",
            ]
        ):
            ylims = None
        elif "aopc" in metric:
            ylims = [-1, 1]
        else:
            ylims = [0, 1]

        df_metric = df[df["metric"] == metric]

        methods = df_metric["method"].unique().tolist()
        methods.sort(
            key=partial(
                version_sort_key,
                float_p=True,
            ),
        )
        # ic(methods)

        if backend == "matplotlib":
            # plt.figure(figsize=(14.5, 8.2))
            plt.figure(figsize=(14.5, 12))
            # plt.figure(figsize=(16, 9))
        elif backend == "plotly":
            fig = go.Figure()

        for i, method in enumerate(methods):
            zorder = len(methods) - i
            if "attributions_s_logit" in method:
                zorder += 999

            # ic(method, zorder)

            df_method = df_metric[df_metric["method"] == method]

            if color_deterministic_p:
                color = colormap(get_color(method))
            else:
                color = None

            value = df_method["value"] * 0.01
            # value += np.random.normal(0, 0.01, len(df_method['value'])
            # value = list(value)

            xs = df_method["top-ratio"]
            # xs = xs.astype(str)
            # xs = list(xs)

            # ic(xs, value,)
            method_shortened = rget(method, r"(?:attributions?_._)?(.*)")
            if backend == "matplotlib":
                linestyle = linestyles[i % len(linestyles)]
                marker = markers[i % len(markers)]

                plt.plot(
                    xs,
                    value,
                    label=method_shortened,
                    alpha=alpha,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    marker=marker,
                    zorder=zorder,
                )
                # break
            elif backend == "plotly":
                linestyle = linestyles_plotly[i % len(linestyles_plotly)]
                marker = markers_plotly[i % len(markers_plotly)]

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=value,
                        mode="lines+markers",
                        name=method_shortened,
                        line=dict(color=color, width=linewidth, dash=linestyle),
                        marker=dict(symbol=marker),
                        hoverinfo="name",
                    )
                )

        tlg_msg_curr = f"{tlg_msg}{title}\n\n{xlabel}\n`{metric}`"
        if backend == "matplotlib":
            plt.title(title)

            plt.xlabel(xlabel)

            plt.ylabel(metric.title())
            if True:
                if ylims:
                    plt.ylim(ylims)
            else:
                plt.yscale("log")

            plt.grid(True)

            # plt.legend()
            plt.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.15), shadow=False, ncol=2
            )
            # Position the legend below the plot

            plt.tight_layout()

            if export_tlg_id:
                wait_p = False
                common_telegram.send(
                    files=(plt.gcf()),
                    msg=tlg_msg_curr,
                    chat_id=export_tlg_id,
                    wait_p=wait_p,
                    lock_key="metrics",
                    autobatch=True,
                    # autobatch=False,
                    # savefig_opts=dict(
                    #     bbox_inches='tight',
                    # ),
                )

            if show_p:
                plt.show()

            plt.close()
        elif backend == "plotly":
            plotly_opts = dict()
            if ylims:
                plotly_opts["yaxis"] = dict(range=ylims)

            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=metric.title(),
                # legend=dict(x=0.5, y=-0.2, xanchor='center', orientation='h')
                **plotly_opts,
            )

            # Create a temporary file name for the plotly figure
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp:
                temp_filename = temp.name

                # Save the plotly figure as HTML
                pio.write_html(
                    fig,
                    file=temp_filename,
                    include_plotlyjs="cdn",
                )

                # Send the HTML file to Telegram (assuming you have a function to send files)
                if export_tlg_id:
                    wait_p = False
                    common_telegram.send(
                        files=(temp_filename,),
                        msg=tlg_msg_curr,
                        chat_id=export_tlg_id,
                        wait_p=wait_p,
                        lock_key="metrics",
                        autobatch=True,
                        # autobatch=False,
                    )

                if show_p:
                    print(temp_filename)
                    # fig.show()

    if not show_p:
        print(f"{fn_name_current()}: finished")


## *** Tables
def df_to_html(df, file_path, *args, **kwargs):
    # Convert DataFrame to HTML
    html_table = df.to_html(*args, **kwargs)

    # Write HTML string to file
    with open(file_path, "w") as f:
        f.write(html_table)


def df_to_html2(df, file_path):
    import panel

    mkdir(file_path, do_dirname=True)

    df = panel.widgets.Tabulator(df)
    df.save(file_path)


def xp_cls_metrics_to_table(
    df,
    metric="accuracy",
    sort_opts="auto",
    remove_nan_p=True,
):
    #: @assumes topratios are from 100
    ##
    # Filter dataframe by the 'accuracy' metric
    df_accuracy = df[df["metric"] == metric]

    # Pivot the table to get methods as rows and top-ratio as columns
    pivot_df = df_accuracy.pivot(index="method", columns="top-ratio", values="value")

    # Convert the column names to numeric
    x_values = pivot_df.columns.values.astype(float) / 100

    # Calculate area under curve
    pivot_df["area-under-curve"] = pivot_df.apply(
        lambda row: np.trapz(row, x=x_values), axis=1
    )
    # pivot_df['area-under-curve'] = pivot_df.apply(lambda row: np.trapz(row, dx=0.1), axis=1)

    pivot_df["average"] = pivot_df.drop(columns=["area-under-curve"]).mean(axis=1)

    if remove_nan_p:
        pivot_df = pivot_df.dropna()

    if sort_opts == "auto":
        sort_opts = dict()

    if sort_opts is not None:
        pivot_df = df_sort(
            pivot_df,
            **sort_opts,
        )

    return pivot_df


##
def attn_prepare_attribution_columns(
    batch,
    model,
    selected_layers=None,
    include_patterns=None,
    exclude_patterns=None,
    exclude_images_p=True,
    filter_fn=None,
    keep_as_is_patterns=None,
    keep_logits_p=True,
    verbose="first_p",
    corr_mode=None,
    kendall_include=[
        "AbsAttnGrad",
        "ReLUAttnGrad",
        "MeanReLU__AttnGrad_Attn",
    ],
):
    if not hasattr(attn_prepare_attribution_columns, "first_p"):
        #: @hack static variable by using the function itself as the object
        attn_prepare_attribution_columns.first_p = False
        first_p = True
    else:
        first_p = False

    verbose_p = verbose is True or (verbose == "first_p" and first_p)

    selected_layers = to_iterable(selected_layers)
    include_patterns = to_iterable(include_patterns)
    include_patterns = [re.compile(pat) for pat in include_patterns]

    exclude_patterns = to_iterable(exclude_patterns)
    keep_as_is_patterns = to_iterable(keep_as_is_patterns)

    keep_as_is_patterns.append(r"^(?:imagenet_|image_)?id$")
    keep_as_is_patterns.append(completeness_error_regex)
    keep_as_is_patterns.append(r"^clip_text$")
    keep_as_is_patterns.append(r"^grad_target(?:_batch)?$")
    if keep_logits_p:
        keep_as_is_patterns.append(r"^logits$")
    # else:
    #     exclude_patterns.append(r"^logits$")
    #: not needed, as attributions_autoget drops them automatically

    keep_as_is_patterns.append(r"^patches_dv$")

    if "label_cpu" in batch:
        batch["label"] = batch["label_cpu"]

    keep_as_is_patterns.append(r"^segmasks(?:_dv|_overlay_pil)?$")
    keep_as_is_patterns.append(r"^semantic_seg_arr$")
    keep_as_is_patterns.append(r"^class_ids(?:_dv)?$")
    keep_as_is_patterns.append(r"^label(?:_dv)?$")
    keep_as_is_patterns.append(r"^comparative_labels$")
    keep_as_is_patterns.append(r"^competing[^_]*_label_.*$")
    keep_as_is_patterns.append(r"^attributions_s_logit(?:_|$)")

    keep_as_is_patterns = [re.compile(pat) for pat in keep_as_is_patterns]

    image_pattern = r"^image(?:_(?:natural|cpu|dv|array|array_dv))?$"
    label_pattern1 = r"^(?:english_)?label(?:_(?:natural|cpu|cat_dog))?$"
    #: =label_cat_dog= is included in the oxford_pet dataset.

    other_misc = r"^origin$"  #: ImageNet-Hard's dataset source of each image
    for p in [
        image_pattern,
        label_pattern1,
        other_misc,
    ]:
        if exclude_images_p:
            exclude_patterns.append(p)

        else:
            keep_as_is_patterns.append(p)

    exclude_patterns.append(r"^_name$")
    #: used in 'transform_aggregate_layers_gen' to enable storing each transform's performance

    #: Exclude the raw (unaggregated, vector) versions of methods such as GradCAM, IxG, etc.:
    exclude_patterns.append(r"_s:raw$")
    keep_as_is_patterns.append(r"^ImageIxG_s:raw$")

    #: I don't know if =attributions_autoget= can handle these raw values correctly, so i am explicitly excluding them:
    #: [[file:::output\[f"blocks__{block_i}__MHA"\] =]]
    exclude_patterns.append(r"(?:MHA|[Vv]alue)(?:Grad|_grad)?$")

    #: @metadata @config
    keep_as_is_patterns.append(r"^(?:perf|time)_")
    # exclude_patterns.append(r"^(?:perf|time)_")

    exclude_patterns = [re.compile(pat) for pat in exclude_patterns]

    blocks_len = len(model.blocks)

    selected_layers = [
        fromto_indices_normalize(from_layer=i).from_layer for i in selected_layers
    ]
    # selected_layers = [i + (blocks_len - 1) for i in selected_layers]
    selected_layers_prefixes = [f"blocks__{i}" for i in selected_layers]

    selected_names_raw = []
    selected_names = []
    new_batch = dict()
    for k, v in batch.items():
        # if k == 'attributions_s_logit_DecompV_vector_f11':
        #     embed()

        if any(re.search(pattern, k) for pattern in keep_as_is_patterns):
            if verbose_p:
                print(f"{fn_name_current()}: keep_as: {k}")

            new_batch[k] = v

            if k.startswith("attributions_"):
                selected_names_raw.append(k)

        elif any(re.search(pattern, k) for pattern in exclude_patterns):
            if verbose_p:
                print(f"{fn_name_current()}: exclude: {k}")

            continue

        elif not include_patterns or any(
            re.search(pattern, k) for pattern in include_patterns
        ):
            if filter_fn:
                if not filter_fn(k=k, v=v):
                    if verbose_p:
                        print(f"{fn_name_current()}: filtered out: {k}")

                    continue

            if (
                selected_layers
                and k.startswith("blocks__")
                and not any(k.startswith(p) for p in selected_layers_prefixes)
            ):
                if verbose_p:
                    print(f"{fn_name_current()}: excluded block: {k}")

                continue

            v = attributions_autoget(
                v,
                name=k,
            )
            if v is None:
                if verbose_p:
                    print(f"{fn_name_current()}: skipped bad attribution method: {k}")

                continue

            method_name = f"attributions_s_{k}"
            new_batch[method_name] = v

            selected_names_raw.append(k)
            selected_names.append(method_name)
        else:
            if verbose_p:
                print(f"{fn_name_current()}: not included: {k}")

            continue

    selected_count = len(selected_names_raw)
    if verbose_p:
        if selected_count <= 2000:
            print(f"selected attributions methods (len={selected_count}):")
            pprint.pprint(selected_names_raw)

    print(f"selected attribution methods' count: {selected_count}", flush=True)
    # embed()

    if corr_mode == "Kendall":
        with Timed(
            name=f"Kendall",
            # enabled_p=False,
        ):
            kendall_corrs = dict()
            for i, method_1 in enumerate(selected_names):
                print(f"Kendall: computing {i}/{len(selected_names)} ...")

                for j in range(i + 1, len(selected_names)):
                    method_2 = selected_names[j]

                    if not (
                        any(re.search(pat, method_1) for pat in kendall_include)
                        and any(re.search(pat, method_2) for pat in kendall_include)
                    ):
                        continue
                        #: Kendall is way too expensive, we can only compute it for methods we really want to compare to each other ...

                    method_1_block_i = rget(method_1, r"\bblocks__(\d+)\b")
                    method_2_block_i = rget(method_2, r"\bblocks__(\d+)\b")
                    if method_1_block_i != method_2_block_i:
                        continue

                    attr_1 = new_batch[method_1]
                    attr_2 = new_batch[method_2]

                    #: Moving them to GPU makes things much slower!
                    # attr_1 = attr_1.to(device)
                    # attr_2 = attr_2.to(device)

                    # ic(torch_shape_get([attr_1, attr_2]))

                    if attr_pixel_level_p(method_1):
                        if attr_pixel_level_p(method_2):
                            continue
                            #: Computing Kendall correlations for pixel-level attributions is too expensive ...
                            ##
                            #: ic| torch_shape_get([new_batch[method_2], new_batch[method_1]]): [(torch.float32, torch.Size([2, 224, 224]), device(type='cpu')), (torch.float32, torch.Size([2, 224, 224]), device(type='cpu'))]

                            attr_1 = attr_1.reshape(attr_1.shape[0], -1)
                            attr_2 = attr_2.reshape(attr_2.shape[0], -1)
                        else:
                            continue

                    if len(attr_1.shape) == 1:
                        raise ValueError(
                            f"{fn_name_current()}: attr_1.shape not supported: {attr_1.shape}"
                        )

                    if attr_pos_embed_p(method_1) or attr_pos_embed_p(method_2):
                        #: only meaningful when the batch size is one
                        continue

                    current_pair = (method_1, method_2)
                    # ic(current_pair)
                    assert (
                        current_pair not in kendall_corrs
                    ), f"{fn_name_current()}: Duplicate pair: {current_pair}"

                    #: kendall_rank_corrcoef(preds, target, variant='b', t_test=False, alternative='two-sided')
                    kendall_corrs[current_pair] = kendall_rank_corrcoef(
                        attr_2.T,
                        attr_1.T,
                        #: Somehow the batch dimension is the last.
                        ##
                        variant="a",
                        # variant="b",
                        #: Variant b is way more expensive, as it computes ties.
                        ##
                        t_test=False,
                    )

            new_batch["kendall_corrs"] = kendall_corrs

    return new_batch


## * Contrastive
def attributions_v_key_get(
    batch,
    attributions_v_key="autodetect",
):
    found_p = False
    if attributions_v_key == "autodetect":
        for k in list(batch.keys()):
            if k.startswith("attributions_v_"):
                assert (
                    found_p == False
                ), "attributions_v_key_get: At least two possible candidates were found: {attributions_v_key}, {k}"
                found_p = True

                attributions_v_key = k

    return attributions_v_key


def transform_competing_second_predicted(
    batch,
    **kwargs,
):
    logits = batch["logits"]
    second_predicted_indices = get_second_predicted_indices(logits)

    # ic(second_predicted_indices.shape)
    #: (batch)
    second_predicted_indices = second_predicted_indices.unsqueeze(-1)
    # ic(second_predicted_indices.shape)
    #: (batch, 1)

    batch = transform_attrv_competing(
        batch,
        target_label=second_predicted_indices,
        name=second_highest_predicted_mode_cst,
        **kwargs,
    )

    return batch


def transform_competing_ground_truth(
    batch,
    **kwargs,
):
    if "label_cpu" in batch:
        ground_truth = batch["label_cpu"]
    elif isinstance(batch["label"], torch.Tensor):
        ground_truth = batch["label"]
    else:
        ground_truth = torch.tensor(batch["label"])

    ground_truth = ground_truth.unsqueeze(-1)
    #: (batch, label)

    batch = transform_attrv_competing(
        batch,
        target_label=ground_truth,
        name=ground_truth_mode_cst,
        **kwargs,
    )

    return batch


def transform_attrv_competing(
    batch,
    *,
    target_label,
    attributions_v_key="autodetect",
    name="",
    delta_scale=5,
    topk=1,
    to_gpu_p=False,
):
    postfix = ""
    if name:
        postfix += f"_{name}"

    attributions_v_key = attributions_v_key_get(
        batch=batch,
        attributions_v_key=attributions_v_key,
    )
    attributions_s_key = re.sub(
        r"^attributions_v_",
        "attributions_s_",
        attributions_v_key,
    )
    attributions_s_key = re.sub(
        r"^attributions_._logits?(_|$)",
        "attributions_s_logit\\1",
        attributions_v_key,
    )
    attributions_s_key += postfix

    if to_gpu_p:
        batch[attributions_v_key] = batch[attributions_v_key].cuda()
        batch["logits"] = batch["logits"].cuda()
        target_label = target_label.cuda()

    attributions_logits = batch[attributions_v_key]

    logits = batch["logits"]

    def label_to_indices(label):
        label_as_indices = label.unsqueeze(-2)
        label_as_indices = label_as_indices.expand(
            *(
                label_as_indices.shape[:-2]
                + (attributions_logits.shape[-2],)
                + label_as_indices.shape[-1:]
            )
        )
        return label_as_indices

    target_label_as_indices = label_to_indices(target_label)
    # ic(target_label_as_indices.shape)

    # ic(
    #     attributions_logits.shape,
    #     target_label.shape,
    # )
    attributions_logit = torch.gather(
        attributions_logits,
        -1,
        target_label_as_indices,
    )
    # ic(attributions_logit.shape)

    attributions_logit, _ = attributions_logit.max(dim=-1, keepdim=False)
    #: =max= is indeed much better than mean
    # attributions_logit = attributions_logit.mean(dim=(-1,))
    # ic(attributions_logit.shape)

    batch[f"{attributions_s_key}"] = attributions_logit
    # ic(attributions_s_key, attributions_logit.shape)

    attributions_to_others = drop_from_dim(
        tensor=attributions_logits,
        indices=target_label_as_indices,
        dim=-1,
        keep_p=False,
    )
    # ic(attributions_to_others.shape)

    ##
    if True:
        logits_others_infinity = torch.scatter(
            logits,
            -1,
            target_label,
            -float("inf"),
        )
        competing_res = keep_topk(
            logits_others_infinity,
            k=topk,
            dim=-1,
            largest=True,
        )
        # ic(torch_shape_get(competing_res))

        competing_label = competing_res.indices

        competing_label_as_indices = label_to_indices(competing_label)

        attributions_s_logits_competing_topk = torch.gather(
            attributions_logits,
            -1,
            competing_label_as_indices,
        )
        # ic(attributions_s_logits_competing_topk.shape)
    else:
        #: This section computes the topk competing label per attribution source, which a priori seems silly.

        attributions_to_others_infinity = torch.scatter(
            attributions_logits,
            -1,
            target_label_as_indices,
            -float("inf"),
        )

        #: =attributions_to_others_infinity= is needed to return correct indices.
        competing_res = keep_topk(
            attributions_to_others_infinity,
            k=topk,
            dim=-1,
            largest=True,
        )
        # ic(torch_shape_get(competing_res))

        competing_label = competing_res.indices
        #: (batch, from_token, label)

        attributions_s_logits_competing_topk = competing_res.values
    ##

    batch[f"competing{topk}_label{postfix}"] = [competing_label]

    assert imagenet_p()
    batch[f"competing{topk}_label_natural{postfix}"] = [
        label_natural_get()[competing_label]
    ]
    # print(
    #     f"label: {label_natural_get()[target_label]}\ncompeting: {label_natural_get()[competing_label]}"
    # )

    ###
    attr_competing_dict = dict()
    ##
    attributions_to_others_relu = F.relu(attributions_to_others)
    attr_competing_dict["All"] = attributions_to_others.mean(dim=(-1,))
    attr_competing_dict["ReLUAll"] = F.relu(attr_competing_dict["All"])

    attr_competing_dict["MReLUAll"] = attributions_to_others_relu.mean(dim=(-1,))

    attr_competing_dict[f"SumAll"] = attributions_to_others.sum(
        dim=-1,
        keepdim=False,
    )
    attr_competing_dict[f"SumReLUAll"] = attributions_to_others_relu.sum(
        dim=-1,
        keepdim=False,
    )

    attr_competing_dict[f"MaxAll"], _ = attributions_to_others.max(
        dim=-1,
        keepdim=False,
    )

    attr_competing_dict[f"ReLUMaxAll"] = F.relu(attr_competing_dict[f"MaxAll"])
    ##
    attr_competing_dict[f"MReLU{topk}"] = F.relu(
        attributions_s_logits_competing_topk
    ).mean(dim=(-1,))

    attr_competing_dict[f"{topk}"] = attributions_s_logits_competing_topk.mean(
        dim=(-1,)
    )

    attr_competing_dict[f"ReLU{topk}"] = F.relu(attr_competing_dict[f"{topk}"])

    attr_competing_dict[f"Max{topk}"], _ = attributions_s_logits_competing_topk.max(
        dim=-1,
        keepdim=False,
    )

    attr_competing_dict[f"ReLUMax{topk}"] = F.relu(attr_competing_dict[f"Max{topk}"])
    ###

    for competing_name, attributions_competing in attr_competing_dict.items():
        batch[f"{attributions_s_key}_competing{competing_name}_ds{delta_scale}"] = (
            attributions_logit - delta_scale * attributions_competing
        )

    return batch


## ** Contrastive Config
target_name2fn = {
    second_highest_predicted_mode_cst: transform_competing_second_predicted,
    ground_truth_mode_cst: transform_competing_ground_truth,
}


## * Segmentation
def segmask_patchwise_to_pixelwise(
    segmask,
    model_patch_info,
    binarize_mode="positive",
    interpolate_mode="nearest-exact",
    verbose=False,
    name=None,
):
    """
    binarize_mode="positive": Assumes `segmask` is either a bool tensor or a float tensor whose positive values indicates the mask.
    """

    if segmask.dtype != torch.bool:
        if binarize_mode == "positive":
            segmask = segmask > 0
        elif binarize_mode is None:
            pass
        else:
            raise ValueError(f"Invalid binarize_mode: {binarize_mode}")

    segmask = segmask.float()

    # ic(torch_shape_get(segmask))
    # ic(segmask)

    segmask = scale_patch_to_pixel(
        segmask,
        output_channel_dim_p=True,
        output_width=model_patch_info.image_resolution,
        output_height=model_patch_info.image_resolution,
        interpolate_mode=interpolate_mode,
        verbose=verbose,
    )

    # ic(torch_shape_get(segmask))
    # ic(segmask)

    return segmask


def filter_to_unique_methods(method2metrics):
    #: remove duplicates from method2metrics based on a key

    unique_methods = set()
    method2metrics_unique = {}
    for method, metrics in method2metrics.items():
        method_key = rget(method, "^(.*)_N:.*")
        if not method_key:
            method_key = method

        if method_key not in unique_methods:
            unique_methods.add(method_key)
            method2metrics_unique[method] = metrics

    return method2metrics_unique


## * LineX-Complete
def layer_name_to_bam(
    name,
    postfix="_s:raw",
):
    #: @idempotent
    if name.startswith("BAM_"):
        assert name.endswith(
            postfix
        ), f"{fn_name_current()}: name does not end with {postfix}"

        return name

    name_2 = f"BAM_{name}"
    name_2 += postfix

    return name_2


def transform_fullgrad(
    batch,
    *,
    model,
    channel_mixer,
    prefix="auto",
    #: @todo Make this work for other classes, too?
    #: The `prefix' can be easily determined through the BAM names in `batch', but some of layer names have been hardcoded. We would need to inspect if they need changing for each new model.
):
    if prefix == "auto":
        prefix = f"{model.__class__.__name__}."
        if prefix == "CustomTextCLIP.":
            prefix = "CustomTextCLIP.visual.trunk."

        print(f"{fn_name_current()}: detected prefix of model: {prefix}")

    output = dict()

    grad_target_batch = batch["grad_target_batch"]
    #: [batch]
    # ic(torch_shape_get(grad_target_batch))

    blocks_len = len(model.blocks)

    cat_name = cat_channel_mixed_name_get(
        channel_mixer=channel_mixer,
        name_prefix="CAT",
    )
    fullgrad_name = cat_channel_mixed_name_get(
        channel_mixer=channel_mixer,
        name_prefix="FGrad",
    )

    unused_bam_names = [k for k in batch.keys() if k.startswith("BAM_")]

    def get_bam(
        name,
        default="MAGIC_ERROR",
    ):
        bam_name = layer_name_to_bam(name)
        if bam_name not in batch:
            if default == "MAGIC_ERROR":
                raise ValueError(f"bam_name not found in batch: {bam_name}")
            else:
                return default

        unused_bam_names.remove(bam_name)

        raw = batch[bam_name]
        if "mlp_tokens.fc2" in name:
            #: batch['BAM_MlpMixer.blocks.23.mlp_tokens.fc2_s:raw'].shape
            #: torch.Size([45, 384, 196])
            #: [batch, channel, token]
            raw = raw.transpose(1, 2)
            #: [batch, token, channel]

        channel_mixed = attributions_scalarify_v2(
            raw,
            mode=channel_mixer,
        )

        # if 'softmax' in bam_name:
        #     ic(bam_name, raw.shape, channel_mixed.shape)

        return channel_mixed

    end_layers = [
        f"{prefix}norm",
        ##
        f"{prefix}attn_pool.kv",
        f"{prefix}attn_pool.q",
        #: torch.Size([Batch, Token])
        ##
    ]

    expected_unattributed_sum = torch.zeros(batch["patches_dv"].shape[:-2])
    #: [batch,]

    all_biases_map = torch.zeros(batch["patches_dv"].shape[:-1])
    # ic(torch_shape_get(all_biases_map))
    #: [batch, tokens]
    #: ic| torch_shape_get(all_biases_map): (torch.float32, torch.Size([1, 785]), device(type='cpu'))

    for name in reversed(end_layers):
        bam = get_bam(name, default=None)
        if bam is None:
            continue

        try:
            all_biases_map += bam

        except:
            ic(
                name,
                torch_shape_get(bam),
                torch_shape_get(all_biases_map),
            )

            raise

    head_layers = [
        f"{prefix}head",
        f"{prefix}fc_norm",
        ##
        f"{prefix}attn_pool.proj",
        f"{prefix}attn_pool.norm",
        f"{prefix}attn_pool.mlp.fc1",
        f"{prefix}attn_pool.mlp.fc2",
        #: [batch, 1]
        #: The assignment of these actually depends on whether `pool == "token"` in FairAttentionPoolLatent. Assuming it is "token" for now. (I have added an assertion for this.)
        #: @IDLink/ede5263a574877350366c9fbf569144c
        ##
    ]
    ##
    #: You can use the below for debugging in ipdb:
    # batch_size = len(batch["id"])
    # name = f"{prefix}attn_pool.mlp.fc2"
    # unused_bam_names.append(layer_name_to_bam(name))
    # bam_ = get_bam(name, default=None)
    # bam_.shape
    # ic(torch_shape_get())
    ##
    for name in reversed(head_layers):
        head_bias = get_bam(name, default=None)
        if head_bias is None:
            continue

        if head_bias.ndim == 2 and head_bias.shape[-1] == 1:
            #: attn_pool biases have [batch, 1] shape.

            head_bias = head_bias.squeeze(-1)

        # ic(torch_shape_get(all_biases_map), torch_shape_get(head_bias))
        #: head_bias: [batch]
        #: torch_shape_get(head_bias): (torch.float32, torch.Size([1]), device(type='cpu'))

        all_biases_map[:, 0] += head_bias
        #: We will assign the head's bias to the CLS token.
        #: @todo/useless Just use =expected_unattributed_sum=.

        del head_bias

    output[f"{fullgrad_name}_end"] = all_biases_map.clone()

    block_bams = dict()
    for bam_name in unused_bam_names:
        block_i = rget(bam_name, r"\.blocks\.(\d+)\.")
        if not block_i:
            continue

        block_i = int(block_i)
        block_bams.setdefault(block_i, []).append(bam_name)

    for block_i in reversed(range(blocks_len)):
        block_bam_names = block_bams.get(block_i, [])

        for name in block_bam_names:
            bam = get_bam(name)
            if "mlp_tokens.fc1" in name:
                #: batch['BAM_MlpMixer.blocks.23.mlp_tokens.fc1_s:raw'].shape
                #: torch.Size([45, 384, 384])

                #: `get_bam` has already summed over the last dim, so we need to sum over the remaining dim:
                expected_unattributed_sum += bam.sum(dim=-1)
                #: There is no easy way to attribute these to the tokens.

            else:
                all_biases_map += bam

            del bam

        if block_i == 0:
            #: Our blocks__0__* start from the very beginning of the model, not from the start of the 0th block.

            start_layers = [
                f"{prefix}norm_pre",
            ]

            for name in reversed(start_layers):
                bam = get_bam(
                    name,
                    default=None,
                )
                if bam is not None:
                    all_biases_map += bam

            output[f"{fullgrad_name}_start"] = all_biases_map.clone()

        block_cat_name = f"blocks__{block_i}__{cat_name}"
        cat = batch[block_cat_name]

        fullgrad_i_name = f"blocks__{block_i}__{fullgrad_name}"

        fullgrad_i = cat + all_biases_map
        # ic(torch_shape_get(fullgrad_i))
        #: ic| torch_shape_get(fullgrad_i): (torch.float32, torch.Size([1, 785]), device(type='cpu'))
        #: [batch, tokens]

        output[fullgrad_i_name] = fullgrad_i

        ##
        fullgrad_i_sum = fullgrad_i.sum(dim=-1)
        if torch.any(expected_unattributed_sum != 0):
            # ic(
            #     grad_target_batch.abs().mean(),
            #     fullgrad_i_sum.abs().mean(),
            #     expected_unattributed_sum.abs().mean(),
            # )
            #: ic| grad_target_batch.abs().mean(): tensor(8.1763)
            #: fullgrad_i_sum.abs().mean(): tensor(4.7763)
            #: expected_unattributed_sum.abs().mean(): tensor(3.3999)

            fullgrad_i_sum += expected_unattributed_sum

        if channel_mixer == "sum":
            completeness_error = fullgrad_i_sum - grad_target_batch
            #: We store the raw error per batch item.

            completeness_error_i_name = f"blocks__{block_i}__FGrad_completeness_error"
            output[completeness_error_i_name] = completeness_error

        if channel_mixer == "sum" and dynamic_get(
            timm.models.decomposition.dynamic_vars,
            "fullgrad_completeness_check_p",
            default=False,
        ):
            atol = timm.models.decomposition.dynamic_obj.get("completeness_atol", 1e-02)
            rtol = timm.models.decomposition.dynamic_obj.get("completeness_rtol", 1e-03)
            completeness_layers = timm.models.decomposition.dynamic_obj.get(
                "completeness_layers", None
            )
            # ic(atol, rtol)
            if not torch.allclose(
                grad_target_batch,
                fullgrad_i_sum,
                atol=atol,
                rtol=rtol,
            ):
                #: I increased the tolerances after seeing this error in batch 116 (batch size 10):
                #: ic| sys.last_value: ValueError("FullGrad is NOT complete!
                #: ic| fullgrad_i_name: 'blocks__3__FGrad_s:sum'
                #: fullgrad_i_sum: tensor([ 7.1941, 8.8166, 4.7995, 9.3478, 8.2407, 9.3045, 9.2572, 9.6106,  -0.0531, 9.1845])
                #: grad_target_batch: tensor([ 7.1941, 8.8166, 4.7995, 9.3478, 8.2407, 9.3045, 9.2572, 9.6106,  -0.0531, 9.1845])
                #: fullgrad_i_sum - grad_target_batch: tensor([-4.7684e-06, 2.8610e-06, 1.4305e-06, -1.9073e-06, 0.0000e+00,  -1.9073e-06, 1.9073e-06, 9.5367e-07, 1.0692e-06, 9.5367e-07])")
                #:
                #: increased the tol again
                # ic| fullgrad_i_name: 'blocks__0__FGrad_s:sum'
                # torch.abs(fullgrad_i_sum - grad_target_batch).max(): tensor(0.0085)
                #:
                #: I increased the tolerances again for EVA Large on =blocks__2__FGrad_s:sum=:
                #: torch.abs(fullgrad_i_sum - grad_target_batch).max(): 0.0004

                ic_msg = ic.format(
                    fullgrad_i_name,
                    # fullgrad_i_sum,
                    # grad_target_batch,
                    # fullgrad_i_sum - grad_target_batch,
                    torch.abs(fullgrad_i_sum - grad_target_batch).max(),
                )
                msg = f"FullGrad is NOT complete!\n{ic_msg}"

                if (
                    completeness_layers is None
                    or (blocks_len - block_i) <= completeness_layers
                ):
                    raise ValueError(msg)
                else:
                    print_diag(
                        msg,
                        group="fullgrad_completeness_check",
                    )

            elif True:
                ic_msg = ic.format(
                    fullgrad_i_name,
                    # fullgrad_i_sum,
                    # grad_target_batch,
                    torch.abs(fullgrad_i_sum - grad_target_batch).max(),
                )
                print_diag(
                    f"""FullGrad is complete:\n{ic_msg}""",
                    group="fullgrad_completeness_check_success",
                    force=run_check_completeness_mode_p,
                )

        ##
        if block_i == 0:
            cat_ensemble_name = f"{cat_name}_sum"
        else:
            cat_ensemble_name = f"{cat_name}_sum_f{block_i}"

        if cat_ensemble_name in batch:
            cat_ensemble = batch[cat_ensemble_name]

            fullgrad_i_name = f"{cat_ensemble_name}_{fullgrad_name}"

            fullgrad_i = cat_ensemble + all_biases_map
            #: [batch, tokens]

            output[fullgrad_i_name] = fullgrad_i
        ##

    assert (
        unused_bam_names == []
    ), f"{fn_name_current()}: Unused BAMs: {unused_bam_names}"

    batch.update(output)
    return batch


def transform_add_bamaps(batch, model):
    #: BAMap: Bias Attribution Map
    ##
    bamaps_obj = get_bias_attributions(
        model,
    )
    named_bias_attributions_raw = bamaps_obj.named_bias_attributions_raw
    # ic(torch_shape_get(named_bias_attributions_raw))

    for name, bamap in named_bias_attributions_raw.items():
        name_2 = layer_name_to_bam(name)

        assert (
            name_2 not in batch
        ), f"{fn_name_current()}: key already exists in batch: {name_2}"
        batch[name_2] = bamap

    return batch


def get_bias_attributions(
    root_module,
    *,
    all_bias_attributions_raw=None,
    seen=None,
    prefix="",
    discard_return_p=False,
):
    if all_bias_attributions_raw is None:
        all_bias_attributions_raw = dict()

    if seen is None:
        seen = set()

    if root_module in seen:
        return

    if not prefix:
        root_module_name = root_module.__class__.__name__
        prefix = f"{root_module_name}."

    name = prefix[:-1]  #: removes the last '.'

    if hasattr(root_module, "stored_bias_attributions_raw"):
        seen.add(root_module)

        # ic(list(all_bias_attributions.keys()))
        if name in all_bias_attributions_raw:
            for i in range(2, 1000):
                name_new = f"{name}{i}"
                if name_new not in all_bias_attributions_raw:
                    name = name_new
                    prefix = f"{name}."
                    break

        assert name not in all_bias_attributions_raw
        all_bias_attributions_raw[name] = root_module.stored_bias_attributions_raw

    for child_name, child in root_module.named_children():
        # ic(child_name)

        get_bias_attributions(
            child,
            all_bias_attributions_raw=all_bias_attributions_raw,
            seen=seen,
            prefix=f"{prefix}{child_name}.",
            discard_return_p=True,
        )

    if discard_return_p:
        return None

    result = dict(
        named_bias_attributions_raw=all_bias_attributions_raw,
    )

    return simple_obj(
        **result,
    )


def get_bias_attributions_v1(
    root_module,
    all_bias_attributions=None,
    all_bias_attributions_raw=None,
    seen=None,
    prefix="",
    compute_sum_p=True,
    discard_return_p=False,
):
    #: @deprecated @backcompat/frozen
    ##
    if all_bias_attributions is None:
        all_bias_attributions = dict()

    if all_bias_attributions_raw is None:
        all_bias_attributions_raw = dict()

    if seen is None:
        seen = set()

    if root_module in seen:
        return

    if not prefix:
        root_module_name = root_module.__class__.__name__
        prefix = f"{root_module_name}."

    name = prefix[:-1]  #: removes the last '.'

    if hasattr(root_module, "stored_bias_attributions"):
        seen.add(root_module)

        # ic(list(all_bias_attributions.keys()))
        if name in all_bias_attributions:
            for i in range(2, 1000):
                name_new = f"{name}{i}"
                if name_new not in all_bias_attributions:
                    name = name_new
                    prefix = f"{name}."
                    break

        assert name not in all_bias_attributions
        all_bias_attributions[name] = root_module.stored_bias_attributions
        all_bias_attributions_raw[name] = root_module.stored_bias_attributions_raw

    for child_name, child in root_module.named_children():
        # ic(child_name)

        get_bias_attributions(
            child,
            all_bias_attributions=all_bias_attributions,
            all_bias_attributions_raw=all_bias_attributions_raw,
            seen=seen,
            prefix=f"{prefix}{child_name}.",
            compute_sum_p=False,  #: No need for a recursive solution
            discard_return_p=True,
        )

    if discard_return_p:
        return None

    result = dict(
        named_bias_attributions=all_bias_attributions,
        named_bias_attributions_raw=all_bias_attributions_raw,
    )

    if compute_sum_p:
        if len(all_bias_attributions.values()):
            #: Sum all bias attribution tensors
            bias_attributions_sum = torch.zeros_like(
                next(iter(all_bias_attributions.values()))
            )

            for name, tensor in all_bias_attributions.items():
                if "head" in name:
                    ic(name, bias_attributions_sum.shape, tensor.shape)
                    #: head_s:raw: [batch, 1000]
                    #: All of these except the target logit should be zero as their gradient is zero.
                    #: Note that here we have the channel-mixed version which is simply [batch].

                    bias_attributions_sum[:, 0] += tensor
                else:
                    bias_attributions_sum += tensor

            result["bias_attributions_sum"] = bias_attributions_sum
        else:
            result["bias_attributions_sum"] = torch.zeros([])

    return simple_obj(
        **result,
    )


##
def completeness_error_postprocess_entrypoint(
    *,
    # debug_p=False,
    # debug_p=True,
    tds_patches=None,
    tds_torch_cpu=None,
    device=None,
    model,
    **kwargs,
):
    from decompv.x.ds.main import (
        tds_patches_imagenet,
        tds_torch_cpu_imagenet,
    )

    model_name = model_name_get(model)

    completeness_error_root = f"{DS_MODEL_ROOT}/{compact_gbrand}/CE/1p/"
    completeness_error_directories = list_children(
        completeness_error_root,
        include_patterns=r"completeness_error$",
    )
    ic(completeness_error_root, completeness_error_directories[:2])
    ####
    ##
    #: @duplicateCode/3659dfc93b398f1ce73fe835dac21b82
    if device is None:
        device = model_device_get(model)

    elif device == "NA":
        device = None
    ##
    ig_steps = decomposition.dynamic_obj.ig_steps

    if ig_steps:
        print(f"{fn_name_current()}: IG detected! ig_steps={ig_steps}")

        assert DATASET_NAME == "ImageNetVal", "Not implemented for other datasets"

        if tds_patches is None:
            tds_patches = tds_patches_imagenet

        if tds_torch_cpu is None:
            tds_torch_cpu = tds_torch_cpu_imagenet

        my_batch = tds_patches_imagenet[:1]
        input_patches = my_batch["patches_dv"]
        baseline_patches = torch.zeros_like(input_patches).to(device)

        target = tds_torch_cpu[:2000]["label_cpu"]
        target = target.to(device)

        with Timed(name="computing the zero baseline"):
            out_baseline = model.forward_patch_level(baseline_patches)

        ic(torch_shape_get(target), torch_shape_get(out_baseline))
        #: target: [N]
        #: out_baseline: [1, num_classes] (ImageNet has 1000 classes)

        assert out_baseline.shape == (
            1,
            1000,
        ), f"Expected shape (1, 1000), got {out_baseline.shape}"

        baseline_target_logits = out_baseline[0, target]

        assert (
            baseline_target_logits.shape == target.shape
        ), f"Expected shape {target.shape}, got {baseline_target_logits.shape}"

        baseline_target_logits = baseline_target_logits.cpu()
        del my_batch, input_patches, baseline_patches, target, out_baseline

    else:
        baseline_target_logits = None

    ####
    tqdm_name = f"Completeness Metrics: {completeness_error_root}"
    for d in tqdm(
        completeness_error_directories,
        name=tqdm_name,
    ):
        process_fn2 = partial(
            completeness_error_from_dir,
            completeness_error_dir_path=d,
            baseline_target_logits=baseline_target_logits,
            **kwargs,
        )

        process_fn2()


def completeness_error_from_dir(
    *,
    completeness_error_dir_path,
    baseline_target_logits,
    name=None,
    json_path="auto",
):
    name = name or os.path.basename(completeness_error_dir_path)

    completeness_errors_dict = tensor_partitioned_load(
        completeness_error_dir_path,
        name="res",
    )
    completeness_errors = completeness_errors_dict["res"]
    n_samples = len(completeness_errors)

    if baseline_target_logits is not None:
        #: IG: sum(Attr) = Out - Baseline
        #: IG: sum(Attr) - Out + Baseline = 0
        #:
        #: completeness_error = IxG_sum - grad_target_batch
        #: completeness_error = sum(Attr) - Out
        #: completeness_error = sum(Attr) - Out + Baseline

        baseline_target_logits = baseline_target_logits[:n_samples]
        assert (
            completeness_errors.shape == baseline_target_logits.shape
        ), f"Shape mismatch: completeness_error shape {completeness_errors.shape} vs baseline_target_logits shape {baseline_target_logits.shape}"

        completeness_errors += baseline_target_logits

    completeness_errors = completeness_errors.abs()
    mean = float(completeness_errors.mean())
    var = float(completeness_errors.var())

    #: Variance of sample mean (VSM = var/n)
    vsm = var / len(completeness_errors)

    stats = {
        "mean": mean,
        "var": var,
        "vsm": vsm,
        "n_samples": n_samples,
        "metadata": {
            "version": 2,
            #: V2: uses abs
            ##
            "name": name,
            "baseline_applied": baseline_target_logits is not None,
        },
    }
    ###
    if json_path == "auto":
        start_index = int(completeness_errors_dict["id"][0])
        end_index = int(completeness_errors_dict["id"][-1]) + 1
        #: @assuming The IDs are contiguous, which they might not be.

        json_path = f"{MODEL_CLS_METRICS_ROOT}/CE/{name}/{start_index}_{end_index}.json"

    ic(stats, json_path)

    json_save(
        stats,
        file=json_path,
        exists="overwrite",
    )


##
