from decompv.utils import batch_size_for
from decompv.x.ds.main import *
from pynight.common_icecream import ic

from pynight.common_tqdm import (
    tqdm,
)
from pynight.common_tqdm2 import (
    redirect_print_to_tqdm,
)
from tqdm import tqdm as tqdm_orig

from IPython import embed

import os

import math

from collections import defaultdict
import pprint

from PIL import Image
import torchvision.transforms

from sklearn.metrics import PrecisionRecallDisplay
from torcheval.metrics.functional import binary_auprc

from pynight.common_files import list_children
from pynight.common_datasets import TransformedDataset

from pynight.common_torch import (
    model_device_get,
    swap_interpolation_to,
    scale_patch_to_pixel,
    torch_gpu_empty_cache,
    gpu_memory_get,
    rank_tensor,
    nan_to_0,
)
from pynight.common_seg import (
    overlay_masks_on_image,
    seg_id_to_mask_dict,
    compute_segmentation_metrics,
)
from pynight.common_iterable import (
    list_of_dict_to_dict_of_list,
)
from pynight.common_dict import (
    key_del,
)
from pynight.common_attr import (
    normalize_map,
    transform_attr_threshold,
)

from pynight.common_benchmark import (
    timed,
    Timed,
)

from pynight.common_json import (
    json_save,
    json_load,
    json_save_update,
)

import decompv.x.imagenet


# ic(decompv.x.imagenet.IMAGENET_1k_LABELS[0])
###
# Global class_colors dict
class_colors = {
    3: [0, 0, 1],  # car: blue
    8: [0, 1, 0],  # truck: green
    17: [0, 0, 1],  # blue
    # 17: [139/255, 69/255, 19/255],  # brown
    18: [0, 1, 0],  # green
    # 18: [128/255, 128/255, 128/255]  # gray
}


def imagenet_s_load(
    batch_list,
    *,
    model,
    class_colors=None,
    display_p=False,
    device="cpu",
    model_transforms,
    multiclass_n=1,
    least_area_percent=5,
    verbose=0,
    return_mode="full",
):
    ##
    #: @duplicateCode/3659dfc93b398f1ce73fe835dac21b82
    if device is None:
        device = model_device_get(model)

    elif device == "NA":
        device = None
    ##

    import decompv.x.imagenet_s

    assert model_transforms is not None
    assert (
        least_area_percent > 1
    ), "least_area_percent is scaled 1 to 100, but you have provided a value less than 1."

    if class_colors is None:
        class_colors = dict()

    result = []
    for id_, path in batch_list:
        base_name = os.path.basename(path)
        #: e.g., 'ILSVRC2012_val_00029930.png'
        #: Remove the extension:
        base_name = os.path.splitext(base_name)[0]
        imagenet_id = int(base_name.split("_")[-1])

        imagenet_label = decompv.x.imagenet.imagenet_val_labels[imagenet_id]
        imagenet_label_natural = label_natural_get()[imagenet_label]

        imagenet_img_path = f"{decompv.x.imagenet.imagenet_val_dir}/{base_name}.JPEG"

        if os.path.exists(imagenet_img_path):
            imagenet_img = Image.open(imagenet_img_path).convert("RGB")

            if model_transforms is not None:
                image_natural = model_transforms.transform_tensor(imagenet_img)
                #: (channels, height, width)

                image_array = model_transforms.transform(imagenet_img)
            else:
                assert False, "Dead Codepath!"

                image_natural = np.array(imagenet_img)
                #: (height, width, channels)

                image_natural = image_natural.transpose(2, 0, 1)
                #: (channels, height, width)

            # ic(image_natural.dtype, image_natural.shape)
        else:
            print(f"Missing: {imagenet_img_path}", flush=True)
            continue

        segmentation = Image.open(path).convert("RGB")
        segmentation = np.array(segmentation)
        segmentation_id = (
            segmentation[:, :, 1] * 256 + segmentation[:, :, 0]
        )  #: R+G*256

        # ic(torch_shape_get(segmentation_id), torch_shape_get(image_natural))

        classes_present = np.unique(segmentation_id)

        #: remove 0 from classes_present if present:
        classes_present = classes_present[classes_present != 0]

        if len(classes_present) < multiclass_n:
            if verbose >= 1:
                print(
                    f"Skipping {base_name} as it has less than {multiclass_n} classes.",
                    flush=True,
                )
            continue

        if model_transforms is not None:
            segmentation_id = Image.fromarray(
                segmentation_id.astype(np.int32), mode="I"
            )
            #: Since the segmentation_id can take a value in the range 0-1000, we can't use uint8 for the image. So we are using "I" which means signed int32. "I" also works with the transforms, unlike some other modes.
            #: Weirdly, this does not need/want the channel dimension, hence we can directly feed `segmentation_id` to it.

            segment_transform = swap_interpolation_to(
                model_transforms.transform_tensor,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT,
                #: [jalali:1403/01/23/20:48] changed from NEAREST to NEAREST_EXACT
            )
            segmentation_id = segment_transform(segmentation_id)

            # ic(torch_shape_get(segmentation_id), torch_shape_get(image_natural))

            classes_present_after = np.unique(segmentation_id)
            classes_present_after = classes_present_after[classes_present_after != 0]

            if not np.array_equal(classes_present, classes_present_after):
                print(
                    f"Transforms have changed the classes present in the segmentation map: {classes_present} vs {classes_present_after}"
                )

        if len(classes_present_after) == 0:
            if verbose >= 1:
                print(
                    f"Skipping {base_name} as it has no classes present after being transformed."
                )
            continue

        segmasks = seg_id_to_mask_dict(
            segmentation_id,
            exclude_ids=[0, 1000],
        )
        #: The ignored part is annotated as 1000, and the other category is annotated as 0.

        segmasks = {
            decompv.x.imagenet_s.imagenet_s_id_to_imagenet_id(k): v
            for k, v in segmasks.items()
        }

        # ic(torch_shape_get(segmasks))

        total_pixels = math.prod(segmentation_id.shape)
        #: no batch dim here

        min_pixels = total_pixels * (least_area_percent / 100)
        class_counts = {
            k: v.sum() for k, v in segmasks.items() if v.sum() >= min_pixels
        }
        if len(class_counts) < multiclass_n:
            if verbose >= 1:
                print(
                    f"Skipping {base_name} as it has less than {multiclass_n} classes "
                    f"with at least {least_area_percent}% area after being transformed."
                )
            continue

        #: Sort classes by area and take the top multiclass_n
        top_classes = sorted(class_counts, key=class_counts.get, reverse=True)[
            :multiclass_n
        ]

        # # Find the biggest segmask and use its class as the label:
        # segmask_sizes = {k: v.sum() for k, v in segmasks.items()}
        # largest_segmask_class_id = max(segmask_sizes, key=segmask_sizes.get)
        # largest_segmask_class_natural = decompv.x.imagenet.IMAGENET_1k_LABELS[
        #     largest_segmask_class_id
        # ]

        if return_mode in ["full"]:
            segmasks_overlay = overlay_masks_on_image(
                image_natural,
                segmasks,
                class_colors=class_colors,
                input_dim_mode="chw",
                # input_dim_mode=None,
                # input_range="255",
                input_range="1",
                alpha=0.75,
            )
            #: returns PIL.Image

            if display_p:
                ic(base_name, imagenet_id, imagenet_label, imagenet_label_natural)
                img_tensor_show(segmasks_overlay)
        else:
            segmasks_overlay = None

        if return_mode in ["filter"]:
            result.append((id_, path))
        elif return_mode in ["full"]:
            for class_id in top_classes:
                result.append(
                    dict(
                        id=id_,
                        imagenet_id=imagenet_id,
                        segmasks_overlay_pil=segmasks_overlay,
                        segmasks=segmasks,
                        semantic_seg_arr=segmentation_id,
                        image_natural=image_natural,
                        image_array=image_array,
                        ##
                        # label_natural=imagenet_label_natural,
                        # label=imagenet_label,
                        ##
                        label=class_id,
                        label_natural=label_natural_get()[class_id],
                        ##
                    )
                )

        # result.append(
        #     dict(
        #         id=id_,
        #         imagenet_id=imagenet_id,
        #         segmasks_overlay_pil=segmasks_overlay,
        #         segmasks=segmasks,
        #         semantic_seg_arr=segmentation_id,
        #         image_natural=image_natural,
        #         image_array=image_array,
        #         ##
        #         # label_natural=imagenet_label_natural,
        #         # label=imagenet_label,
        #         ##
        #         label=largest_segmask_class_id,
        #         label_natural=largest_segmask_class_natural,
        #         ##
        #     )
        # )

    if len(result) == 0:
        return None

    if return_mode in ["filter"]:
        return result

    result = list_of_dict_to_dict_of_list(result)

    #: Concat the images to a batch:
    result["image_array"] = torch.stack(result["image_array"], dim=0)
    if device:
        result["image_array_dv"] = result["image_array"].to(device)

    else:
        result["image_array_dv"] = result["image_array"]

    label_cpu = torch.tensor(result["label"])
    result["label_cpu"] = label_cpu

    if device:
        label_dv = label_cpu.to(device)
        result["label_dv"] = label_dv

    return result


###
def method2metrics_process(method2metrics, *, sort_by):
    method2metrics = {
        k: list_of_dict_to_dict_of_list(v)
        for k, v in method2metrics.items()
        if len(v) >= 1
    }

    # use torch.mean() to compute the mean of the list of tensors
    # [[id:e3679b11-4f8a-4a79-8e1e-25339dceba5b][torch.std() returns nan for single item tensors. · Issue #29372 · pytorch/pytorch]]
    method2metrics_mean = {
        k: {
            k2: torch.mean(nan_to_0(torch.tensor(v2).float())).item()
            for k2, v2 in v.items()
        }
        for k, v in method2metrics.items()
    }

    def srt(item):
        try:
            return item[1][sort_by]
        except:
            ic(item)
            raise

    method2metrics_mean = dict(
        sorted(
            method2metrics_mean.items(),
            key=srt,
            reverse=True,
        )
    )

    method2metrics_mean_unique = filter_to_unique_methods(method2metrics_mean)

    ##
    #: Note that this std is between the batches, and not individual images. So it's a bit useless.
    method2metrics_std = {
        k: {
            k2: torch.std(nan_to_0(torch.tensor(v2).float())).item()
            for k2, v2 in v.items()
        }
        for k, v in method2metrics.items()
    }

    method2metrics_std = dict(
        sorted(
            method2metrics_std.items(),
            key=lambda item: item[1][sort_by],
            reverse=False,
            #: less variance is better
        )
    )

    method2metrics_std_unique = filter_to_unique_methods(method2metrics_std)
    ##

    return simple_obj(
        mean=method2metrics_mean,
        mean_unique=method2metrics_mean_unique,
        std=method2metrics_std,
        std_unique=method2metrics_std_unique,
    )


###
def seg_compute(
    *,
    tds_seg,
    model,
    model_name,
    model_patch_info,
    name,
    ds_name,
    submode,
    all_gbrands,
    select_p,
    seg_ablation_p,
    metadata,
    data_n,
    seg_batch_size=None,
    show_images_p=False,
    no_threshold_p=True,
    threshold_p=False,
    interpolate_mode="bicubic",
    embed_p=False,
    extra_tqdm_name=None,
    ##
    to_device_p=True,
    #: CPU: (batch_size=60) Time: batch=0: computing seg metrics: 23.58258080482483 seconds
    #: CUDA: (batch_size=60) Time: batch=0: computing seg metrics: 10.130741834640503 seconds
    ##
):
    if metadata is None:
        metadata = dict()

    # export_dir = f"{ARTIFACTS_ROOT}/plots_v4"
    # export_dir = f"{ARTIFACTS_ROOT}/plots_coco"
    export_dir = f"{ARTIFACTS_ROOT}/plots_v5/"
    if ds_name:
        export_dir += f"/{ds_name}/"

    if name:
        export_dir += f"/{name}/"

    tqdm_name = "Seg"
    if ds_name:
        tqdm_name += f"-{ds_name}"

    tqdm_name += f" {all_gbrands.compact_gbrand} {model_name} "
    if extra_tqdm_name:
        tqdm_name += extra_tqdm_name
    ##
    growable_batch_p = True
    #: multiclass_p can make the batch sizes grow to be more than 1!
    ##
    seg_batch_size = batch_size_for(
        model_name=model_name,
        all_gbrands=all_gbrands,
        override=seg_batch_size,
        seg_p=True,
    )

    batch_n = math.ceil(data_n / seg_batch_size)
    # batch_n =  20
    # batch_n =  200
    # batch_n =  200
    # batch_n =  500
    # batch_n =  200
    # batch_n = 86 * 6
    # batch_n = 2
    # batch_n = 20

    ##

    if show_images_p:
        ic(export_dir)

        threshold_p = False

        seg_batch_size = 1

        # batch_n = 2
        # batch_n = 40
        batch_n = 5
        # batch_n = 20
        # batch_n = 20

    else:
        selected_methods_best = None

        normalizers = [
            # None,
            [
                "shift_min_to_zero",
                "scale_by_max_abs_attr",
            ],
            # ["rank_uniform"],
        ]

    metadata["naming_mode"] = "start"
    metadata["submode"] = submode
    metadata["seg_batch_size"] = seg_batch_size
    metadata["data_n"] = data_n
    metadata["batch_n"] = batch_n
    metadata["threshold_p"] = threshold_p
    metadata["no_threshold_p"] = no_threshold_p
    metadata["interpolate_mode"] = interpolate_mode
    ic(metadata)

    method2metrics = defaultdict(list)
    method2metrics_no_threshold = defaultdict(list)
    attr_p_name_to_orig = dict()
    #: method -> [dict_of_metrics]
    ##
    torch_gpu_empty_cache()

    data_n_real = 0

    with DynamicVariables(
        decomposition.dynamic_obj,
        print_diag_enabled_groups=lst_filter_out(
            decomposition.dynamic_obj.print_diag_enabled_groups,
            [
                # "gradient_mode",
            ],
        )
        + [
            "fullgrad_completeness_check_success",
        ],
    ), Timed(
        name=f"""computing seg metrics (all)""",
        # enabled_p=False,
        enabled_p=True,
    ):
        for i, d in enumerate(
            tqdm(
                tds_seg.batched_iterator(
                    seg_batch_size,
                    autoadjust_batch_size_mode="shrink",
                    # autoadjust_batch_size_mode=True,
                ),
                name=tqdm_name,
                total=(batch_n),
            )
        ):
            print(f"started {i} ...", flush=True)
            ic(type(d), len(d))
            # if i >= 3:
            #     break
            # else:
            #     continue

            data_n_real += len(d)

            with Timed(
                name=f"""batch={i}: computing seg metrics""",
                # enabled_p=False,
                enabled_p=True,
            ):
                label_cpu = d["label_cpu"]
                # ic(torch_shape_get(d['segmasks']))

                segmask_gt = [
                    d["segmasks"][batch_i][label.item()]
                    for batch_i, label in enumerate(label_cpu)
                ]
                # ic(len(segmask_gt))

                segmask_gt = torch.stack(segmask_gt)
                #: Concatenates a sequence of tensors along a new dimension.
                # ic(torch_shape_get(segmask_gt))

                if to_device_p:
                    segmask_gt = segmask_gt.to(device)

                batch_len = segmask_gt.shape[0]

                method2segmask = dict()
                selected_attribution_methods = [
                    k for k in d.keys() if k.startswith("attributions_s_")
                ]
                if i == 0:
                    ic(
                        len(selected_attribution_methods),
                        selected_attribution_methods,
                    )

                for attr_name in selected_attribution_methods:
                    selected_p = select_p(attr_name)
                    if (show_images_p) and not selected_p:
                        continue

                    pixel_p = attr_pixel_level_p(attr_name)
                    if attr_pos_embed_p(attr_name):
                        if batch_len != 1 or growable_batch_p:
                            # print("PosEmbed attributions are computed per the whole batch, so the only way to compute them per image is to have a batch size of one.")
                            continue

                    if no_threshold_p:
                        attr = d[attr_name]
                        if not pixel_p:
                            #: Since we don't do any normalizations here (as BAUPRC is scale and shift invariant), we only need normalize_map to skip CLS etc. But when pixel-level, we don't have these extra tokens in the first place. So there is no need to run this code.
                            attr = normalize_map(
                                attr,
                                normalize=None,
                                # normalize=[
                                #     "shift_min_to_zero",
                                #     "scale_by_max_abs_attr",
                                # ],
                                num_prefix_tokens=model_patch_info.num_prefix_tokens,
                                bias_token_p=False,
                                clone_p=False,
                                pixel_p=pixel_p,
                            ).attributions_skipped

                        # ic(attr.shape)
                        # attr.shape: torch.Size([10, 784])

                        # ic(torch_shape_get(attr))
                        # ic(torch.unique(attr[0])[:10])
                        try:
                            attr = segmask_patchwise_to_pixelwise(
                                attr,
                                model_patch_info,
                                binarize_mode=None,
                                interpolate_mode=interpolate_mode,
                                # verbose=True,
                            )
                        except:
                            ic(attr_name, torch_shape_get(attr))
                            raise

                        # ic(torch.unique(attr[0])[:10])
                        # ic(torch_shape_get(attr))
                        # ic(segmask_gt.shape)
                        # attr.shape: torch.Size([10, 1, 224, 224])
                        # segmask_gt.shape: torch.Size([10, 1, 224, 224])

                        #: [[https://pytorch.org/torcheval/main/generated/torcheval.metrics.functional.binary_auprc.html#torcheval.metrics.functional.binary_auprc][torcheval.metrics.functional.binary_auprc — TorchEval main documentation]]
                        try:
                            auprc = binary_auprc(
                                attr.reshape(batch_len, -1),
                                segmask_gt.reshape(batch_len, -1),
                                num_tasks=batch_len,
                            )
                        except:
                            ic(attr_name, torch_shape_get(attr))
                            raise

                        #: binary_auprc(  input: torch.Tensor,  target: torch.Tensor,  *,  num_tasks: int = 1, ) -> torch.Tensor
                        #: returns a batch
                        auprc_mean = auprc.mean().item()
                        auprc_std = auprc.std().item()
                        if selected_p:
                            ic(attr_name, auprc, auprc_mean)

                            if show_images_p:
                                if False:
                                    assert seg_batch_size == 1

                                    fig, ax = plt.subplots(figsize=(6, 3))
                                    #: [[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay.from_predictions][sklearn.metrics.PrecisionRecallDisplay — scikit-learn 1.4.1 documentation]]
                                    PrecisionRecallDisplay.from_predictions(
                                        segmask_gt.reshape(-1).cpu(),
                                        attr.reshape(-1).cpu(),
                                        ax=ax,
                                    )
                                    plt.show()
                                    plt.close()
                                    print("", flush=True)

                                image_i = 0
                                image_nat = d["image_natural"][image_i]
                                imagenet_id = d["imagenet_id"][image_i]
                                label = d["label"][image_i]
                                label_natural = d["label_natural"][image_i]
                                label = int(label)
                                ic(imagenet_id)

                                image_masked_gt = image_nat * segmask_gt[image_i].cpu()
                                ic(attr.shape)
                                overlay_colored_grid(
                                    image_nat,
                                    # d[attr_name][image_i],
                                    attr[image_i].flatten(),
                                    label=label,
                                    label_natural=label_natural,
                                    pixel_p=True,
                                    # pixel_p=pixel_p,
                                    scale=1,
                                    # normalize=["rank_uniform"],
                                    normalize=[
                                        "shift_min_to_zero",
                                        "scale_by_max_abs_attr",
                                    ],
                                    color_positive="viridis",
                                    model_patch_info=model_patch_info,
                                    title=f"{model_name}\n{compact_gbrand}\n{attr_name}\nSS/{interpolate_mode}    label:{label_natural}",
                                    image_concats=[
                                        image_masked_gt,
                                        # attr[image_i].cpu(),
                                    ],
                                    image_concats_right=[image_nat],
                                    export_dir=f"{export_dir}/{model_name}/{compact_gbrand}/{attr_name}/{interpolate_mode}/i{imagenet_id}/{label}",
                                    export_name="",
                                    export_name_postfix="/",
                                )
                                print("", flush=True)

                        del attr
                        del auprc

                        seg_metrics = dict(
                            BAUPRC=auprc_mean,
                            BAUPRC_std=auprc_std,
                        )
                        method2metrics_no_threshold[attr_name].append(seg_metrics)

                    if threshold_p:
                        for normalize in normalizers:
                            if normalize is None:
                                gt_thresholds = list(np.arange(-0.1, 0.2, 0.01))
                            else:
                                gt_thresholds = []

                            gt_thresholds += [
                                # 0.0001,
                                # 0.0005,
                                # 0.001,
                                # 0.005,
                                # 0.025,
                                # 0.05,
                                0.1,
                                # 0.125,
                                # 0.15,
                                # 0.175,
                                0.2,
                                # 0.225,
                                # 0.25,
                                0.3,
                                0.4,
                                0.5,  #: from TransAtt
                                # 0.6,
                                # 0.7,
                                # 0.8,
                                # 0.9,
                            ]
                            for gt_threshold in set(gt_thresholds):
                                if selected_methods_best is not None:
                                    if attr_name in selected_methods_best:
                                        qual_wanted = selected_methods_best[attr_name]

                                        if gt_threshold in qual_wanted["gt_thresholds"]:
                                            pass
                                        else:
                                            continue
                                    else:
                                        # print(f"skipped attr: {attr_name}")
                                        continue

                                if to_device_p:
                                    tmp_device = device
                                else:
                                    tmp_device = None

                                res = transform_attr_threshold(
                                    d,
                                    normalize_opts=dict(
                                        # bias_token_p=model_patch_info.bias_token_p,
                                        bias_token_p=False,
                                        num_prefix_tokens=model_patch_info.num_prefix_tokens,
                                        normalize=normalize,
                                        # outlier_quantile=0.1,
                                        outlier_quantile=0,
                                        pixel_p=pixel_p,
                                    ),
                                    attr_name=attr_name,
                                    gt_threshold=gt_threshold,
                                    device=tmp_device,
                                    return_mode="new_only",
                                )
                                tmp_batch = res.result
                                attr_p_name = res.name
                                attr_p_name_to_orig[attr_p_name] = attr_name
                                segmask = tmp_batch[attr_p_name]
                                del tmp_batch

                                segmask = segmask_patchwise_to_pixelwise(
                                    segmask,
                                    model_patch_info,
                                    interpolate_mode=interpolate_mode,
                                )

                                seg_metrics = compute_segmentation_metrics(
                                    predicted_mask=segmask,
                                    ground_truth_mask=segmask_gt,
                                )
                                method2metrics[attr_p_name].append(seg_metrics)

                                if show_images_p:
                                    #: @warn We CANNOT store the segmasks even in CPU memory if we have too many methods selected!

                                    segmask = segmask.cpu()  #: releasing the memory
                                    method2segmask[attr_p_name] = segmask
                                else:
                                    del segmask
                                    # torch_gpu_empty_cache(gc_mode=None)

            del segmask_gt
            torch_gpu_empty_cache()
            #: This wasn't needed when we weren't using LineX, so our LineX code probably has a memory leak somewhere?

            # if i >= 0:
            if (i + 1) >= batch_n:
                break

    metadata["batch_n_real"] = i + 1  #: i is zero-indexed
    metadata["data_n_real"] = data_n_real
    del d

    print("seg's main loop finished", flush=True)
    ###
    method2metrics_obj = method2metrics_process(
        method2metrics,
        sort_by="IoU",
    )
    method2metrics_mean = method2metrics_obj.mean
    method2metrics_mean_unique = method2metrics_obj.mean_unique
    method2metrics_std = method2metrics_obj.std
    method2metrics_std_unique = method2metrics_obj.std_unique

    method2metrics_no_threshold_obj = method2metrics_process(
        method2metrics_no_threshold,
        sort_by="BAUPRC",
    )
    method2metrics_no_threshold_mean = method2metrics_no_threshold_obj.mean
    method2metrics_no_threshold_mean_unique = (
        method2metrics_no_threshold_obj.mean_unique
    )
    method2metrics_no_threshold_std = method2metrics_no_threshold_obj.std
    method2metrics_no_threshold_std_unique = method2metrics_no_threshold_obj.std_unique
    ###
    if show_images_p:
        print("show_images_p is true, not saving metrics")
    else:
        if seg_ablation_p:
            metrics_save_dir = f"""{MODEL_SEG_METRICS_ROOT.replace('/metrics/s/', '/metrics/x/')}/{submode}/{compact_gbrand}"""
        else:
            metrics_save_dir = f"{MODEL_SEG_METRICS_ROOT}/{submode}/{compact_gbrand}"
        ic(compact_gbrand, submode, metrics_save_dir)

        exists = "error"
        # exists = "ignore"
        # update_mode = "skip"
        # update_mode = "file_exists_error"
        update_mode = "overwrite"

        if no_threshold_p and len(method2metrics_no_threshold_mean) == 0:
            print("method2metrics_no_threshold_mean is empty, skipping save")

        else:
            json_save(
                metadata,
                file=f"{metrics_save_dir}/metadata_{submode}.json",
                exists="increment_number",
            )
            if threshold_p:
                if False:
                    #: 205M for 86*6 batches
                    json_save_update(
                        method2metrics,
                        file=f"{metrics_save_dir}/method2metrics.json",
                        exists=exists,
                        update_mode=update_mode,
                    )

                json_save_update(
                    method2metrics_mean,
                    file=f"{metrics_save_dir}/method2metrics_mean.json",
                    exists=exists,
                    update_mode=update_mode,
                )
                json_save_update(
                    method2metrics_mean_unique,
                    file=f"{metrics_save_dir}/method2metrics_mean_unique.json",
                    exists=exists,
                    update_mode=update_mode,
                )
                json_save_update(
                    method2metrics_std,
                    file=f"{metrics_save_dir}/method2metrics_std.json",
                    exists=exists,
                    update_mode=update_mode,
                )
                json_save_update(
                    method2metrics_std_unique,
                    file=f"{metrics_save_dir}/method2metrics_std_unique.json",
                    exists=exists,
                    update_mode=update_mode,
                )

            if no_threshold_p:
                #: 22M
                json_save_update(
                    method2metrics_no_threshold,
                    file=f"{metrics_save_dir}/method2metrics_no_threshold.json",
                    exists=exists,
                    update_mode=update_mode,
                )

                json_save_update(
                    method2metrics_no_threshold_mean,
                    file=f"{metrics_save_dir}/method2metrics_no_threshold_mean.json",
                    exists=exists,
                    update_mode=update_mode,
                )
                json_save_update(
                    method2metrics_no_threshold_mean_unique,
                    file=f"{metrics_save_dir}/method2metrics_no_threshold_mean_unique.json",
                    exists=exists,
                    update_mode=update_mode,
                )
                json_save_update(
                    method2metrics_no_threshold_std,
                    file=f"{metrics_save_dir}/method2metrics_no_threshold_std.json",
                    exists=exists,
                    update_mode=update_mode,
                )
                json_save_update(
                    method2metrics_no_threshold_std_unique,
                    file=f"{metrics_save_dir}/method2metrics_no_threshold_std_unique.json",
                    exists=exists,
                    update_mode=update_mode,
                )

            print("seg metrics saved")
    ###
    if embed_p:
        embed()


###
