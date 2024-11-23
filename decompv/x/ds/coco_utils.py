###
from collections import defaultdict
from pyexfiltrator import exfiltrate, exfiltrated
from functools import partial
import json5
from pynight.common_numpy import (
    image_url2np,
)
from pynight.common_regex import (
    float_pattern,
)
from pynight.common_benchmark import (
    Timed,
)
from pynight.common_torch import (
    torch_shape_get,
    torch_shape_get_hidden,
    torch_shape_get_hidden_ids,
    img_tensor_show,
    scale_patch_to_pixel,
)
from pynight.common_dict import list_of_dict_to_bacthed_dict
from pynight.common_icecream import ic
from pynight.common_dict import simple_obj
from pynight.common_datasets import (
    TransformedDataset,
    TransformResult,
)
from pynight.common_iterable import (
    HiddenList,
)
import pynight.common_jax

from pynight.common_timm import (
    patch_info_from_name,
)
import decompv
from decompv.x.bootstrap import *
from decompv.x.ds.main import *
import decompv.x.ds.utils
import decompv.x.imagenet
import pynight.common_telegram as common_telegram
from decompv.x.ds.compute_decompv1 import (
    decompv_compute_global_v2,
)

###
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

###
from pathlib import Path

HOME = Path.home()

with open(f"{HOME}/datasets/COCO/mappings/coco2imagenet_v3.json", "r") as file:
    coco2imagenet = json5.load(file)

tmp = dict()
for k, v in coco2imagenet.items():
    if "quality" in v and v["quality"] in [
        "terrible",
        "bad",
    ]:
        continue

    if "good_p" in v and not v["good_p"]:
        continue

    tmp[k] = v

coco2imagenet = tmp
###
#: When mapping to ImageNet, use the mapping of all the classes
coco_label_groups = [
    [
        "car",
        "truck",
    ],
    # ["bus", "train",]
]
###
coco_image_dir = f"{HOME}/COCO dataset"

# dataType = "val2017"
dataType = "train2017"

coco_annotations_dir = f"{HOME}/datasets/COCO/annotations"
annFile = f"{coco_annotations_dir}/instances_{dataType}.json"


###
def image_metadata_path_get(
    *,
    image_metadata=None,
    # filename=None,
    split="auto",
    coco_image_dir=None,
):
    if coco_image_dir is None:
        coco_image_dir = globals()["coco_image_dir"]

    filename = None
    if filename is None:
        assert image_metadata is not None

        filename = image_metadata["file_name"]

    if split == "auto":
        url = image_metadata["coco_url"]

        if "train2017" in url:
            split = "train2017"
        elif "val2017" in url:
            split = "val2017"
        elif "test2017" in url:
            split = "test2017"
        elif "train2014" in url:
            split = "train2014"
        elif "val2014" in url:
            split = "val2014"
        elif "test2014" in url:
            split = "test2014"
        else:
            raise ValueError(f"Unknown split for url: {url}")

    full_path = f"{coco_image_dir}/{split}/{image_metadata['file_name']}"

    return full_path


###
# initialize COCO api for instance annotations
coco = COCO(annFile)
###
coco_class_ids = coco.getCatIds()

coco_classes_lst = coco.loadCats(coco_class_ids)
coco_classes = dict()

for coco_class in coco_classes_lst:
    coco_classes[coco_class["id"]] = coco_class


# ic(coco_classes)
###
def count_images_for_class_pairs(
    coco_classes,
    include_supercategories=None,
):
    # Filter classes based on the specified supercategories if provided
    if include_supercategories:
        class_names = [
            cls["name"]
            for class_id, cls in coco_classes.items()
            if cls["supercategory"] in include_supercategories
        ]
    else:
        class_names = [cls["name"] for cls in coco_classes.values()]

    class_names = [
        class_name
        for class_name in class_names
        if class_name
        not in (
            "person",
            "giraffe",
        )
    ]

    # ic(class_names)
    pair_counts = {}

    # Iterate over every possible pair of COCO classes
    for i in range(len(class_names)):
        for j in range(
            i + 1, len(class_names)
        ):  # j starts from i+1 to avoid duplicate pairs and self-pairing
            pair = (class_names[i], class_names[j])
            result = coco_get_by_class(class_names=pair)
            pair_counts[pair] = len(result.image_ids)

    pair_counts = dict(sorted(pair_counts.items(), key=lambda x: x[1], reverse=True))

    return pair_counts


###
# get all images containing given categories, select one at random
def coco_get_by_class(
    *,
    class_names=None,
    superclass_names=None,
    class_ids=None,
    limit_n=None,
):
    class_names = to_iterable(class_names)
    superclass_names = to_iterable(superclass_names)
    class_ids = to_iterable(class_ids)

    class_ids = coco.getCatIds(
        catNms=class_names,
        supNms=superclass_names,
        catIds=class_ids,
    )

    image_ids = coco.getImgIds(catIds=class_ids)
    image_ids = image_ids[:limit_n]

    return simple_obj(
        class_ids=class_ids,
        image_ids=image_ids,
    )


def coco_get_by_image_ids(image_ids):
    images = coco.loadImgs(image_ids)

    return images


def coco_image_get_by_class(
    *,
    class_names=None,
    superclass_names=None,
    class_ids=None,
    limit_n=None,
    **kwargs,
):
    r = coco_get_by_class(
        class_names=class_names,
        superclass_names=superclass_names,
        class_ids=class_ids,
        # limit_n=limit_n,
    )

    images = coco_get_by_image_ids(r.image_ids)
    #: A list of dictionaries, each dictionary contains metadata about one image.

    return coco_image_metadata_process(
        images,
        include_classes=r.class_ids,
        limit_n=limit_n,
        **kwargs,
    )


def segmasks_overlap_get(mask1, mask2):
    """Computes overlap for two binary masks."""

    intersection = np.sum(np.logical_and(mask1, mask2))
    all = min(np.sum(mask1), np.sum(mask2))
    return intersection / all
    ##
    # union = np.sum(np.logical_or(mask1, mask2))
    # iou = intersection / union
    # return iou
    ##


def coco_image_metadata_process(
    images,
    # min_seg_coverage=0.02,
    min_seg_coverage=0.05,
    #: classes having segmasks less than this many percent of the whole image will be assumed non-existent
    min_segmented_classes=2,
    max_unmerged_class_overlap=0.7,
    max_class_overlap=0.7,
    include_classes=None,
    limit_n=None,
):
    if max_class_overlap > 1:
        max_class_overlap /= 100
        #: supports both [0,1], and [0, 100]

    if min_seg_coverage > 1:
        min_seg_coverage /= 100

    filtered_images = []

    for image_metadata in images:
        image_metadata = dict(image_metadata)  #: copies the data
        # ic(image_metadata)

        class_ids = set()
        seg_class_ids = []

        image_metadata["file_path"] = image_metadata_path_get(
            image_metadata=image_metadata
        )

        annIds = coco.getAnnIds(
            imgIds=image_metadata["id"],
            # iscrowd=None,
        )

        anns = coco.loadAnns(annIds)

        for ann in anns:
            class_id = ann["category_id"]
            if include_classes is None or class_id in include_classes:
                class_ids.add(class_id)

            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):
                    torch_shape_get_hidden_ids.add(id(ann["segmentation"]))

        image_metadata["anns"] = anns

        segmasks = dict()
        unmerged_segmasks = dict()
        for class_id in class_ids:
            if False:  #: might not be needed
                #: Check if the image has a segmentation mask annotation for ALL specified classes
                has_mask = all(
                    any(
                        ann["category_id"] == class_id and ann.get("segmentation")
                        for ann in anns
                    )
                )

                if not has_mask:
                    continue

            masks = [
                coco.annToMask(ann)
                for ann in image_metadata["anns"]
                if ann["category_id"] == class_id
            ]
            if any(mask is None for mask in masks):
                #: @todo What happens in =coco.annToMmask= if there is no mask?
                exfiltrate()
                raise Exception("debug id 98127826")

            unmerged_segmasks[class_id] = masks
            if masks:
                merged_mask = masks[0]
                for mask in masks[1:]:
                    merged_mask = np.maximum(merged_mask, mask)
                merged_mask[merged_mask > 0] = (
                    1.0  #: @redundant as the masks here are binary
                )
                segmasks[class_id] = merged_mask

                if min_seg_coverage is not None:
                    # Count the number of pixels in segmask that are set to 1
                    num_mask_pixels = np.sum(segmasks[class_id])

                    # Count the total number of pixels in the image.
                    # Since segmasks[class_id] is a 2D array, its shape will give us the image dimensions
                    total_pixels = (
                        segmasks[class_id].shape[0] * segmasks[class_id].shape[1]
                    )

                    # Calculate the percentage
                    coverage_percentage = num_mask_pixels / total_pixels

                    # Check if this is greater than or equal to 2%
                    if coverage_percentage > min_seg_coverage:
                        seg_class_ids.append(class_id)

        image_metadata["segmasks"] = segmasks
        image_metadata["class_ids"] = seg_class_ids

        # Check for mask overlap between classes
        if max_unmerged_class_overlap is not None:
            skip_image = False
            for class_id1, masks1 in unmerged_segmasks.items():
                for class_id2, masks2 in unmerged_segmasks.items():
                    if class_id1 != class_id2:
                        for mask1 in masks1:
                            for mask2 in masks2:
                                iou = segmasks_overlap_get(mask1, mask2)
                                if iou > max_unmerged_class_overlap:
                                    skip_image = True
                                    break
                            if skip_image:
                                break
                    if skip_image:
                        break
                if skip_image:
                    break

            if skip_image:
                print(
                    f"""skipped due to (unmerged) class overlap: {image_metadata["id"]}, IoU={iou}"""
                )
                continue

        if max_class_overlap is not None:
            skip_image = False
            for class_id1 in segmasks:
                for class_id2 in segmasks:
                    if class_id1 != class_id2:
                        iou = segmasks_overlap_get(
                            segmasks[class_id1], segmasks[class_id2]
                        )
                        # ic(iou, class_id1, class_id2)

                        if iou > max_class_overlap:
                            skip_image = True
                            break
                if skip_image:
                    break

            if skip_image:
                print(
                    f"""skipped due to class overlap: {image_metadata["id"]}, IoU={iou}"""
                )
                continue

        if len(seg_class_ids) >= min_segmented_classes:
            filtered_images.append(image_metadata)
            # print(f"""accepted: {image_metadata["id"]}, seg_class_ids={seg_class_ids}, last_coverage_percentage={coverage_percentage}""")

            if limit_n is not None and len(filtered_images) >= limit_n:
                break
        else:
            # print(f"""skipped: {image_metadata["id"]}, seg_class_ids={seg_class_ids}, last_coverage_percentage={coverage_percentage}""")
            pass

    return filtered_images


def image_metadata_to_input(
    *,
    image_metadata,
    model,
    device=None,
    min_seg_coverage=0.02,
    ordinal_id=None
):
    if device is None:
        device = model_device_get(model)

    elif device == "NA":
        device = None

    if min_seg_coverage > 1:
        min_seg_coverage /= 100
    ##
    result = dict()
    result["ordinal_id"] = ordinal_id
    for k, v in image_metadata.items():
        if k in [
            "id",
            "class_ids",
        ]:
            result[k] = v
    ##
    image_np = image_url2np(image_metadata["file_path"])
    if image_np is None:
        return None
    ##
    to_pil = torchvision.transforms.ToPILImage()
    transforms = model_transform_get(model)
    ##
    image_pil = to_pil(image_np)
    image_transformed_tensor = transforms.transform_tensor(image_pil).cpu()
    image_transformed = transforms.transform(image_pil)

    image_cpu_squeezed = image_transformed
    image_cpu = image_cpu_squeezed.unsqueeze(0)

    if device:
        image_dv = image_cpu.to(device)
    else:
        image_dv = None
    ##
    segmasks = image_metadata["segmasks"]
    segmasks_transformed = dict()
    for k, v in segmasks.items():
        v = (
            v * 255
        )  #: The mask is either 0 or 1, but the image functions expect the maximum value to be 255.
        transformed_mask = transforms.transform_tensor(to_pil(v)).cpu()
        transformed_mask[transformed_mask > 0] = 1.0
        #: The transforms seem to add a "soft" edge to the binary masks which we don't want.

        # Calculate the percentage coverage
        num_mask_pixels = transformed_mask.sum().item()
        total_pixels = transformed_mask.shape[1] * transformed_mask.shape[2]  # H x W
        coverage_percentage = num_mask_pixels / total_pixels

        # Check if this is greater than or equal to min_seg_coverage
        if coverage_percentage >= min_seg_coverage:
            segmasks_transformed[k] = transformed_mask

    # result['segmasks_untrasnformed'] = image_metadata['segmasks']
    result["segmasks"] = segmasks_transformed
    ##
    return simple_obj(
        **result,
        # image_np=image_np,
        image_pil=image_pil,
        image_natural=image_transformed_tensor,
        image_cpu_squeezed=image_cpu_squeezed,
        image_cpu=image_cpu,
        image_dv=image_dv,
    )


def transform_input_prepare_coco(
    batch_lst,
    *,
    model,
    min_segmented_classes=2,
    mode="coco",
):
    if mode not in [
        "Narnia2",
    ]:
        # raise NotImplementedError("We have changed the expected input from list of COCO IDs to an enumerated list. Adapt your mode.")
        batch_lst = list(enumerate(batch_lst)) #: @hack

    result = dict()

    image_objects = [
        image_metadata_to_input(
            image_metadata=image_metadata,
            model=model,
            ordinal_id=ordinal_id,
        )
        for (ordinal_id, image_metadata) in batch_lst
    ]
    image_objects = [
        image_metadata
        for image_metadata in image_objects
        if image_metadata is not None
        and len(image_metadata.segmasks.keys()) >= min_segmented_classes
    ]

    if len(image_objects) == 0:
        return None
    # assert (
    #     len(image_objects) >= 1
    # ), "All images provided had their segmasks outside of the current crop. The code currently doesn't support this, you need to implement the needed logic."
    ##
    if mode in [
        "Narnia",
        "Narnia2",
    ]:
        image_objects_expanded = []
        for image_obj in image_objects:
            for class_id in image_obj["class_ids"]:
                image_obj_copy = dict(
                    image_obj
                )  #: copy image_obj, which is a SimpleObject

                #: We convert class_id to ImageNet ID:
                class_name = coco_classes[class_id]["name"]
                image_obj_copy["label_natural"] = class_name
                coco2imagenet_my_class = coco2imagenet[class_name]
                imagenet_classes = [
                    lbl["id"] for lbl in coco2imagenet_my_class["imagenet_labels"]
                ]
                image_obj_copy["label"] = imagenet_classes[0]
                segmask = image_obj_copy["segmasks"][class_id]
                image_obj_copy["segmasks"] = {image_obj_copy["label"]: segmask}

                image_objects_expanded.append(image_obj_copy)
        image_objects = image_objects_expanded

    #: Assuming all images are of same dimensions
    image_batch_cpu = torch.stack(
        [image_obj["image_cpu"].squeeze(0) for image_obj in image_objects]
    )
    if device:
        image_batch_dv = image_batch_cpu.to(device)
    else:
        image_batch_dv = None
    ##
    for k in [
        "id",
        "class_ids",
        "segmasks",
        "image_natural",
    ]:
        result[k] = [image_obj[k] for image_obj in image_objects]

    if mode in [
        "Narnia2",
    ]:
        result["id"] = [image_obj["ordinal_id"] for image_obj in image_objects]

    if mode in [
        "Narnia",
        "Narnia2",
    ]:
        result["image_id"] = [image_obj["id"] for image_obj in image_objects]

        result["imagenet_id"] = [
            image_obj["id"] for image_obj in image_objects
        ]  #: @hack @todo

        for k in [
            "label",
            "label_natural",
        ]:
            result[k] = [image_obj[k] for image_obj in image_objects]

        result["label_cpu"] = torch.tensor(result["label"]).cpu()
        if device:
            result["label_dv"] = result["label_cpu"].to(device)

        del result["class_ids"]

    return simple_obj(
        **result,
        image_array=image_batch_cpu,
        image_array_dv=image_batch_dv,
        # image_batch_cpu=image_batch_cpu,
        # image_batch_dv=image_batch_dv,
    )


##### * Run
###
