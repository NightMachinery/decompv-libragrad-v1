from decompv.x.ds.main import *
from decompv.x.ds.seg import *
from IPython import embed

###
model_patch_info = pynight.common_dict.simple_obj_update(
    model_patch_info,
    bias_token_p=False,
)

my_model_patch_info = model_patch_info
###
ic(SEG_DATASET_NAME)
if SEG_DATASET_NAME == "ImageNetS":
    #: @duplicateCode/66c312bd705e550866f1d4bfe32f590d
    ##
    from decompv.x.imagenet_s import *

    imagenet_s_all = imagenet_s_all_get()

    imagenet_s_tds = TransformedDataset(imagenet_s_all)
    imagenet_s_tds = imagenet_s_tds.transform(
        partial(
            imagenet_s_load,
            model_transforms=model_transform_get(model),
            class_colors=class_colors,
            device=device,
            model=model,
        ),
    )
    # ic(imagenet_s_tds.preview())

    imagenet_s_patches_tds = imagenet_s_tds.transform(
        partial(
            transform_pixels2patches,
            model=model,
            grad_p=True,
        )
    )
    # ic(imagenet_s_patches_tds.preview())

elif SEG_DATASET_NAME == "zele":
    ##
    #: @duplicateCode/5fd640b06397b1b38b020a1194312f08
    #: We should refactor the code here to use =zele_tds_input_get= etc.

    from decompv.x.ds.coco_utils import (
        coco_image_get_by_class,
        image_metadata_to_input,
        transform_input_prepare_coco,
    )

    images = coco_image_get_by_class(
        class_names=[
            # "cat",
            # "dog",
            "zebra",
            "elephant",
            # "car",
            # "truck",
        ],
        # limit_n=200,
    )
    ic(len(images))
    #: elephant and zebra: 10?

    tds_images = TransformedDataset(images)

    ##
    #: For testing one image:
    # image_input = image_metadata_to_input(
    #     image_metadata=images[0],
    #     model=model,
    # )

    # ic(torch_shape_get(image_input))
    #: namespace(id=98949,
    #:           class_ids=[24, 22],
    #:           segmasks={22: (torch.float32,
    #:                      torch.Size([1, 224, 224]),
    #:                      device(type='cpu')),
    #:                     24: (torch.float32,
    #:                      torch.Size([1, 224, 224]),
    #:                      device(type='cpu'))},
    #:           image_pil=<PIL.Image.Image image mode=RGB size=640x427>,
    #:           image_natural=(torch.float32,
    #:                          torch.Size([3, 224, 224]),
    #:                          device(type='cpu')),
    #:           image_cpu_squeezed=(torch.float32,
    #:                               torch.Size([3, 224, 224]),
    #:                               device(type='cpu')),
    #:           image_cpu=(torch.float32,
    #:                      torch.Size([1, 3, 224, 224]),
    #:                      device(type='cpu')),
    #:           image_dv=(torch.float32,
    #:                     torch.Size([1, 3, 224, 224]),
    #:                     device(type='cuda', index=0)))
    ##

    tds_images_prepared = tds_images.transform(
        partial(
            transform_input_prepare_coco,
            model=model,
            mode="Narnia",
        )
    )

    # ic(tds_images_prepared.preview())
    #: {'id': [98949, 98949, 279208, 279208],
    #:  'image_array': (torch.float32,
    #:   torch.Size([4, 3, 224, 224]),
    #:   device(type='cpu')),
    #:  'image_array_dv': (torch.float32,
    #:   torch.Size([4, 3, 224, 224]),
    #:   device(type='cuda', index=0)),
    #:  'image_natural': [(torch.float32,
    #:    torch.Size([3, 224, 224]),
    #:    device(type='cpu')),
    #:   (torch.float32, torch.Size([3, 224, 224]), device(type='cpu')),
    #:   (torch.float32, torch.Size([3, 224, 224]), device(type='cpu')),
    #:   (torch.float32, torch.Size([3, 224, 224]), device(type='cpu'))],
    #:  'imagenet_id': [98949, 98949, 279208, 279208],
    #:  'label': [340, 386, 340, 386],
    #:  'label_cpu': (torch.int64, torch.Size([4]), device(type='cpu')),
    #:  'label_dv': (torch.int64, torch.Size([4]), device(type='cuda', index=0)),
    #:  'label_natural': ['zebra', 'elephant', 'zebra', 'elephant'],
    #:  'segmasks': [{340: (torch.float32,
    #:     torch.Size([1, 224, 224]),
    #:     device(type='cpu'))},
    #:   {386: (torch.float32, torch.Size([1, 224, 224]), device(type='cpu'))},
    #:   {340: (torch.float32, torch.Size([1, 224, 224]), device(type='cpu'))},
    #:   {386: (torch.float32, torch.Size([1, 224, 224]), device(type='cpu'))}]}

    imagenet_s_tds = tds_images_prepared
    imagenet_s_patches_tds = imagenet_s_tds.transform(
        partial(
            transform_pixels2patches,
            model=model,
            grad_p=True,
        )
    )
    # ic(imagenet_s_patches_tds.preview())
    #: {'id': [98949, 98949, 279208, 279208],
    #:  'image_array': (torch.float32,
    #:   torch.Size([4, 3, 224, 224]),
    #:   device(type='cpu')),
    #:  'image_array_dv': (torch.float32,
    #:   torch.Size([4, 3, 224, 224]),
    #:   device(type='cuda', index=0)),
    #:  'image_natural': [(torch.float32,
    #:    torch.Size([3, 224, 224]),
    #:    device(type='cpu')),
    #:   (torch.float32, torch.Size([3, 224, 224]), device(type='cpu')),
    #:   (torch.float32, torch.Size([3, 224, 224]), device(type='cpu')),
    #:   (torch.float32, torch.Size([3, 224, 224]), device(type='cpu'))],
    #:  'imagenet_id': [98949, 98949, 279208, 279208],
    #:  'label': [340, 386, 340, 386],
    #:  'label_cpu': (torch.int64, torch.Size([4]), device(type='cpu')),
    #:  'label_dv': (torch.int64, torch.Size([4]), device(type='cuda', index=0)),
    #:  'label_natural': ['zebra', 'elephant', 'zebra', 'elephant'],
    #:  'patches_dv': (torch.float32,
    #:   torch.Size([4, 785, 768]),
    #:   device(type='cuda', index=0)),
    #:  'segmasks': [{340: (torch.float32,
    #:     torch.Size([1, 224, 224]),
    #:     device(type='cpu'))},
    #:   {386: (torch.float32, torch.Size([1, 224, 224]), device(type='cpu'))},
    #:   {340: (torch.float32, torch.Size([1, 224, 224]), device(type='cpu'))},
    #:   {386: (torch.float32, torch.Size([1, 224, 224]), device(type='cpu'))}]}

else:
    raise ValueError(f"Unknown SEG_DATASET_NAME: {SEG_DATASET_NAME}")

####
attr_mode = "grad"
submode = getenv(
    "DECOMPV_SEG_SUBMODE",
    default="full.1",
)
print(f"Seg Submode: {submode}", file=sys.stderr)

if attr_mode == "grad":
    from decompv.x.ds.compute_attention import (
        attn_transforms_get,
        attn_compute_global,
        blocks_len,
        m6_attn_transforms_get,
        m7_include_patterns,
        m7_exclude_patterns,
        m8_attn_transforms_get,
        m8_include_patterns,
        m8_exclude_patterns,
    )

    if submode in [
        "m6",
        "m7",
    ]:
        attn_transforms = m6_attn_transforms_get()

    elif submode in [
        "m8",
    ]:
        attn_transforms = m8_attn_transforms_get()

    else:
        attn_transforms = attn_transforms_get(
            sum_to_layers=None,
            ensemble_mode=None,
        )

    # submode = "full.1"
    # submode = "m2.1"
    ic(submode, blocks_len)

    exclude_patterns = []
    include_patterns = []
    if submode == "full.1":
        include_patterns = [
            "rnd1",
            # "^._logits",
            ##
            f"_sum_f{blocks_len // 2}($|_)",
            # "^CAT(?:_RS)?(?:_AttnFrom)?_sum(?:_f\d+)?$",
            # "^CAT(?:_RS)?(?:_AttnFrom)?_sum",
            # "^(?:blocks__\d+__)?CAT(?:_s:[^_]+)?(?:_AttnFrom)?",
            "CAT",
            ##
            "^Image(?:Grad|IxG)",
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
            ##
            # "^MeanAttn_ro_str50$",
            # "^MeanAttn_sum$",
            "^(?:blocks__\d+__)?MeanAttn(?:$|_sum|_ro)",
            ##
            "^(?:blocks__\d+__)?Mean(?:Abs|ReLU)?AttnGrad(?:$|_sum|_ro)",
            ##
            "^TokenTM",
            ##
        ]

    elif submode in ["m5", "m6"]:
        include_patterns = [
            "rnd1",
            ##
            f"_sum(_f{blocks_len // 2}($|_)|$|_to|_FGrad_)",
            f"blocks__(0|{blocks_len - 1}|{(blocks_len // 2) + 1})__",
            ##
            "^Image(?:Grad|IxG)",
            ##
            "Mean(?:ReLU|Abs)?__AttnGrad_Attn_ro",
            ##
            "^MeanAttn_ro",
            ##
            "^TokenTM",
            ##
        ]

        exclude_patterns = [
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

    elif submode in ["m7"]:
        include_patterns = m7_include_patterns
        exclude_patterns = m7_exclude_patterns

    elif submode in ["m8"]:
        include_patterns = m8_include_patterns
        exclude_patterns = m8_exclude_patterns

    elif False:
        include_patterns = [
            # "ImageIxG_s:(RS|sum|L1)",
            "^Image",
            "PosE",
            # "IxG",
        ]

    elif submode == "m2.1":
        include_patterns = [
            "rnd1",
            f"^CAT_s:sum_AttnFrom_sum(_to{blocks_len - 1})?$",
            ##
            f"^blocks__{blocks_len - 1}__AttnWHeadGrad$",
            ##
        ]
    else:
        raise ValueError(f"Unknown submode: {submode}")

    my_attn_prepare_attribution_columns = partial(
        attn_prepare_attribution_columns,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        exclude_images_p=False,
        # exclude_images_p=True,
        keep_as_is_patterns=[
            # r"(?i)blocks__\d+__Attn",
        ],
        verbose=False,
        # verbose=True,
        # corr_mode="Kendall",
    )

    tds_attr = attn_compute_global(
        name="TMP1",
        tds_patches=imagenet_s_patches_tds,
        attn_transforms=attn_transforms,
        attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
        add_random_p=True,
        early_return_mode="1",
        dsmap_attn_opts=dict(
            remove_cols=None,
        ),
    )
    tds_seg = tds_attr

    ###
    def select_p(attr_name):
        # if False:
        # if seg_ablation_p:
        if True:
            include_patterns = [
                # "ImageIxG_s:(RS|sum)",
                # "ImageGrad_s:(RS|sum)",
                "ImageIxG_s:RS",
                "ImageGrad_s:RS",
                "blocks__0__CAT_s:(RS|sum)$",
                r"CAT_s:sum_sum($|_FGrad_s:sum|_f(6|12))",
                r"GCAM_s:(sum)_sum($|_f(6|12))",
                # r"XACAM_s:(RS|sum)_sum($|_f(6|12))",
                r"XACAM_s:(sum)_sum($|_f(6|12))",
                r"Mean(Abs)AttnGrad(_sum($|_f(6|12)))",
            ]

        elif True:
            include_patterns = [
                # "blocks__11__AttnWHeadGrad",
                # "blocks__3__CAT(_s:(RS|sum))?$",
                # "blocks__8__CAT(_s:(RS|sum))?$",
                "ImageIxG",
                "ImageGrad",
                # "ImageGrad_s:L1",
                "PosE",
                # r"CAT_s:sum_sum_f(6|12)",
                r"CAT_s:sum_sum($|_f)",
                r"CAT_s:RS_sum_f(6|12)_FGrad_s:RS",  #: @V2
                r"CAT_s:sum_AttnFrom_sum(_f(6|12))?_to",
                r"Mean(ReLU)?__AttnGrad_Attn_ro_str50",
                r"blocks__\d+__CAT_s:sum$",
                r"blocks__(11|23)__MeanAttn$",
                r"blocks__0__FGrad_s:sum",
                r"blocks__(6|7|12|13)__FGrad_s:RS",  #: @V2
                r"MeanAttn_sum($|_f(6|12))",
                r"MeanAttn_ro_str50",
                r"GCAM_s:sum_sum($|_f(6|12))",
                r"blocks__(11|23)__GCAM_s:sum",
                r"rnd1",
                r"MeanAttn(_ro|_sum($|_f(6|12)))",
                r"(globenc|ALTI).*(_ro|_sum($|_f(6|12)))",
                r"Mean(Abs)?AttnGrad(_ro|_sum($|_f(6|12)))",
            ]
        else:
            include_patterns = [
                "ImageIxG_s:RS",
                r"CAT_s:sum_sum_f12",
                r"MeanReLU__AttnGrad_Attn_ro_str50",
                r"blocks__23__GCAM_s:sum",
                # # r"MeanAttn(_ro|_sum($|_f(6|12)))",
                r"(globenc|ALTI).*(_ro|_sum($|_f(6|12)))",
                r"Mean(Abs)?AttnGrad_sum$",
            ]

        return any(re.search(n, attr_name) for n in include_patterns)

    ###

elif attr_mode == "GlobALTI":
    from decompv.x.ds.compute_globalti import (
        globalti_compute_global,
    )

    submode = "scaled"

    my_attn_prepare_attribution_columns = partial(
        attn_prepare_attribution_columns,
        # filter_fn=my_filter,
        keep_as_is_patterns=[
            # r"^attributions_s_logit(?:_|$)",
        ],
        exclude_images_p=False,
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
    ):
        tds_attr = globalti_compute_global(
            name="G",
            tds_patches=imagenet_s_patches_tds,
            attn_prepare_attribution_columns=my_attn_prepare_attribution_columns,
            add_random_p=False,
            early_return_mode="1",
        )

    tds_seg = tds_attr

    ###
    def select_p(attr_name):
        return True

    ###
####
show_images_p = bool_from_str(
    getenv(
        "DECOMPV_SEG_SHOW_IMAGES_P",
        default=False,
    )
)
seg_ablation_p = bool_from_str(
    getenv(
        "DECOMPV_SEG_ABLATION_P",
        default=False,
    )
)
seg_embed_p = bool_from_str(
    getenv(
        "DECOMPV_SEG_EMBED_P",
        default=False,
    )
)

if seg_dataset_end_global is None:
    if seg_ablation_p:
        seg_dataset_end_global = 1000
    else:
        seg_dataset_end_global = 5000

    # if "huge" in model_name:
    #     seg_dataset_end_global = 1000

    seg_dataset_end_global = getenv(
        "DECOMPV_SEG_DATASET_SIZE",
        default=seg_dataset_end_global,
    )
seg_dataset_end_global = int(seg_dataset_end_global)
###
if __name__ == "__main__":
    ##
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("mode", help="mode of operation")
    # parser.add_argument("--submode1", help="description for submode1", default=None)
    args = parser.parse_args()
    mode = args.mode
    # submode1 = args.submode1
    ##

    metadata = dict(
        model_name=model_name,
        seg_ablation_p=seg_ablation_p,
        **all_gbrands,
        dynamic_obj=decomposition.dynamic_obj_to_metadata_dict(),
    )

    if mode == "v1":
        name = ""
    else:
        raise NotIplementedError(f"mode {mode} not implemented")

    seg_compute(
        name=name,
        model_name=model_name,
        model=model,
        ds_name=SEG_DATASET_NAME,
        submode=submode,
        all_gbrands=all_gbrands,
        tds_seg=tds_seg,
        model_patch_info=my_model_patch_info,
        select_p=select_p,
        seg_ablation_p=seg_ablation_p,
        metadata=metadata,
        show_images_p=show_images_p,
        data_n=seg_dataset_end_global,
        embed_p=seg_embed_p,
    )
