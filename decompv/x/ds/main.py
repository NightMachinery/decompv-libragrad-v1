## * @pre
from decompv.utils import batch_size_for
from pynight.common_datasets import TransformedDataset
from brish import bool_from_str
import datasets
from os import getenv
from pathlib import Path

from pynight.common_datasets import (
    mapconcat,
    dataset_index_add,
)

from decompv.x.ds.utils import (
    dsmap_input_filter,
    dataset_format_set1,
)

from decompv.x.bootstrap import *
from decompv.x.ds.utils import *


## * User Inputs
global_batch_size = getenv(
    "DECOMPV_ATTR_BATCH_SIZE",
    default="auto",
    # default="100",
)

if global_batch_size == "auto":
    global_batch_size = batch_size_for(
        model_name=model_name,
        all_gbrands=all_gbrands,
        override=None,
    )
else:
    global_batch_size = int(global_batch_size)

batch_size = global_batch_size  #: @alias

batch_size_inference = int(
    getenv(
        "DECOMPV_INFER_BATCH_SIZE",
        default="100",
    )
)

dataset_start_global = int(
    getenv(
        "DECOMPV_DATASET_START",
        default="0",
    )
)

dataset_end_global = getenv(
    "DECOMPV_DATASET_END",
    default=None,
)
if dataset_end_global is None:
    dataset_end_global = getenv(
        "DECOMPV_DATASET_SIZE",
        default="4",
    )
dataset_end_global = int(dataset_end_global)

##
import os
import sys

print(f"ds.main: Current process PID: {os.getpid()}", file=sys.stderr)
## * Model
if model_load_p and model_name:
    if DATASET_NAME in [
        "MURA",
        "oxford_pet",
    ]:
        assert model_name == "vit_base_patch16_224"
        #: I am not sure if we use model_name elsewhere, so I am not just setting it here.

        if DATASET_NAME == "MURA":
            num_classes = 1
            checkpoint_path = f"{HOME}/models/vit-shapley/MURA_vit_base_patch16_224_surrogate_22ompjqu_state_dict.ckpt"

        elif DATASET_NAME == "oxford_pet":
            num_classes = 37
            checkpoint_path = f"{HOME}/models/vit-shapley/Pet_vit_base_patch16_224_surrogate_146vf465_state_dict.ckpt"

        else:
            raise ValueError(f"DATASET_NAME={DATASET_NAME} is not supported.")

        # Load the state_dict from the checkpoint
        state_dict = torch.load(checkpoint_path)
        # ic(torch_shape_get(state_dict))

        # Create a new state_dict with modified keys
        modified_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                modified_key = key[len("backbone.") :]  # Remove the 'backbone.' prefix
                modified_state_dict[modified_key] = value
            else:
                modified_state_dict[key] = value

        # Create the model
        model = timm.create_model(
            model_name,
            num_classes=num_classes,
        )  # Adjust num_classes if needed

        # Load the modified state_dict into the model
        model.load_state_dict(modified_state_dict)
        model.to(device)
        model.eval()
        model.name = model_name

    else:
        model = create_model(
            device=device,
            model_name=model_name,
            ##
            # act_layer=QuickGELUDecomposed,
        )

    model_patch_info = patch_info_from_name(model_name)
    model_num_prefix_tokens = getattr(model, "num_prefix_tokens", 0)
    assert (
        model_num_prefix_tokens == model_patch_info.num_prefix_tokens
    ), f"You have not set the correct num_prefix_tokens in the model_patch_info. Actual num_prefix_tokens: {model_num_prefix_tokens}, model_patch_info:\n{model_patch_info}"
    del model_num_prefix_tokens

else:
    print(
        "Skipped creating the model. Set DECOMPV_MODEL_LOAD_P=y and DECOMPV_MODEL_NAME to load the model."
    )

    model = None
    model_patch_info = None
###
## * Load the Dataset
if dataset_load_p:
    if not Path(DS_INDEXED_PATH).exists():
        print(f"{DS_INDEXED_PATH} does not exist; creating the dataset ...")
        from decompv.x.ds.indexed import dataset_indexed_save

        if DATASET_NAME == "ImageNetVal":
            from decompv.x.ds.imagenetval import dataset, dataset_name

        elif DATASET_NAME == "ImageNet-Hard":
            from decompv.x.ds.imagenet_hard import dataset, dataset_name

        elif DATASET_NAME == "MURA":
            from decompv.x.ds.mura import dataset, dataset_name

        elif DATASET_NAME == "oxford_pet":
            from decompv.x.ds.oxford_pet import dataset, dataset_name

        else:
            raise ValueError(f"DATASET_NAME={DATASET_NAME} is not supported.")

        dataset_indexed_save(
            dataset=dataset,
            dataset_name=dataset_name,
            dest=DS_INDEXED_PATH,
        )

        print(f"Created the indexed dataset!")

    dataset_indexed = datasets.load_from_disk(DS_INDEXED_PATH)
    #: 'set_format' with `columns=['label']` still changed the type of the image column from PIL.JpegImagePlugin.JpegImageFile to dict(bytes=..., path=...)

    # dataset_small = dataset_indexed.select(range(dataset_start, dataset_end))
    # tds_small = TransformedDataset(dataset_small)
    # tds_torch_cpu = tds_small.transform(
    #     partial(transform_input_prepare, model=model, device="NA")
    # )
    # tds_patches = tds_small.transform(partial(transform_image2patches, model=model))
    ##
    tds_indexed = TransformedDataset(dataset_indexed)
    tds_indexed_imagenet = tds_indexed
    if model:
        tds_torch_cpu_imagenet = tds_indexed.transform(
            partial(transform_input_prepare, model=model, device="NA")
        )
        tds_patches_lazy_imagenet = tds_indexed.transform(
            partial(transform_image2patches, model=model)
        )

        ##
        #: @duplicateCode/6a3ad540403ebe103105ff868160268f
        dataset_patchified_path = f"{DS_PATCHIFIED_PATH}/{model_name}"
        ##
        if os.path.exists(dataset_patchified_path):
            dataset_patchified = datasets.load_from_disk(dataset_patchified_path)
            dataset_patchified = dataset_format_set1(dataset_patchified)
            tds_patches_imagenet = TransformedDataset(dataset_patchified)
            tds_patches_imagenet = tds_patches_imagenet.transform(
                partial(
                    transform_patchify_load,
                    device=device,
                )
            )
            # ic(tds_patches_imagenet.preview())
        else:
            print(f"dataset_patchified_path does not exist: {dataset_patchified_path}")
            tds_patches_imagenet = tds_patches_lazy_imagenet


##
# ic(tds_patches.preview())
###
def attr_captum_compute_global(
    *,
    captum_attributors,
    name="auto",
    tds_patches: TransformedDataset = None,
    dataset_start=None,
    dataset_end=None,
    **kwargs,
):
    if tds_patches is None:
        tds_patches = tds_patches_imagenet

    if dataset_start is None:
        dataset_start = dataset_start_global

    if dataset_end is None:
        dataset_end = dataset_end_global

    if name == "auto":
        name = ""
        for attributor in captum_attributors:
            name += f"""{attributor["name"]},"""
        name = name[:-1]  #: removes the last comma
    ic(name)

    computer = partial(
        h_attr_captum_compute,
        captum_attributors=captum_attributors,
    )

    return dataset_compute_gen(
        name=name,
        computer=computer,
        batch_size=batch_size,
        tds_patches=tds_patches,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        model=model,
        # load_from_cache_file=False,
        **kwargs,
    )


###
def decompv_compute_global(
    *,
    decomposition_config=None,
    store_cls_only_p=None,
    raw_attention_store_mode=None,
    tds_patches: TransformedDataset = None,
    dataset_start=None,
    dataset_end=None,
    **kwargs,
):
    if tds_patches is None:
        tds_patches = tds_patches_imagenet

    if dataset_start is None:
        dataset_start = dataset_start_global

    if dataset_end is None:
        dataset_end = dataset_end_global

    if decomposition_config is None:
        decomposition_config = DecompositionConfig(
            device=device,
            attributions_aggregation_strategy="vector",
        )

    if store_cls_only_p is None:
        store_cls_only_p = getenv(
            "DECOMPV_STORE_CLS_ONLY_P",
            default="y",
        )
        store_cls_only_p = bool_from_str(store_cls_only_p)

    if raw_attention_store_mode is None:
        raw_attention_store_mode = getenv(
            "DECOMPV_RAW_ATTENTION_STORE_MODE",
            default="last_average",
        )

    name = decomposition_config.name

    computer = partial(
        h_decompv_compute,
        decomposition_config=decomposition_config,
        store_cls_only_p=store_cls_only_p,
        raw_attention_store_mode=raw_attention_store_mode,
    )

    return dataset_compute_gen(
        name=name,
        computer=computer,
        batch_size=batch_size,
        tds_patches=tds_patches,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        model=model,
        # load_from_cache_file=False,
        **kwargs,
    )


def tds_decompv_get(
    *,
    decomposition_config,
    model=model,
    # my_tds_indexed=tds_indexed_imagenet,
    # my_tds_torch_cpu=tds_torch_cpu_imagenet,
    tds_patches=None,
    model_name=model_name,
    after_transforms=None,
    after_transforms_get=None,
    after_transforms2=None,
    raw_attention_store_mode="n",
    store_cls_only_p=True,
    store_perf_p=True,
    attr_name_mode="v1",
    select_label="legacy",
):
    if tds_patches is None:
        tds_patches = tds_patches_imagenet

    tds_decompv = tds_patches.transform(
        partial_dynamic(
            partial(
                dsmap_decompv_attributions,
                model=model,
                decomposition_config=decomposition_config,
                raw_attention_store_mode=raw_attention_store_mode,
                store_cls_only_p=store_cls_only_p,
                store_perf_p=store_perf_p,
                attr_name_mode=attr_name_mode,
            ),
            dynamic_dict=decomposition.dynamic_obj,
            print_diag_enabled_groups=lst_filter_out(
                decomposition.dynamic_obj.print_diag_enabled_groups,
                [
                    "warning.nondecomposed",
                    "check_attributions",
                    "check_attributions.end",
                    "config",
                ],
            ),
        ),
    )

    tds_decompv = tds_decompv.transform(
        partial(
            transform_globalti_attributions,
            model=model,
            # device=device,
            # decomposition_config=decomposition_config,
        )
    )

    after_transforms = to_iterable(after_transforms)
    if after_transforms_get is not None:
        after_transforms += after_transforms_get(
            model=model,
        )
    for transform in after_transforms:
        #: @metadata We can add the ability to store the time each transform took to TransformedDataset. These transforms might set '_name' in their output to signal their name for metadata purposes, but I think we need a better API if adding this to TransformedDataset.
        tds_decompv = tds_decompv.transform(transform)

    tds_scalar = tds_decompv
    # tds_scalar = tds_scalar.transform(decompv.x.ds.utils.transform_softmax)
    tds_scalar = tds_scalar.transform(
        partial(
            transform_attributions_scalarify,
            mode="identity",
            sum_dim=(-2,),  #: sum over extra attribution sources
            keep_mode="all",
        ),
    )

    after_transforms2 = to_iterable(after_transforms2)
    for transform in after_transforms2:
        tds_scalar = tds_scalar.transform(transform)

    if select_label == "legacy":
        tds_scalar = tds_scalar.transform(
            partial(
                transform_attributions_select_label,
                dataset_indexed=dataset_indexed,
            ),
        )
    ##

    return tds_scalar


###
