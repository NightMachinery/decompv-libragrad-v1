###
from decompv.early_boot import (
    run_check_completeness_mode_p,
    run_compute_completeness_mode_p,
    qual_prototype_p,
    global_force_pruned_mode_p,
)

import re
import os
from os import getenv
from brish import bool_from_str
from pynight.common_torch import get_compact_gbrand
from pynight.common_shell import (
    str_falsey_to_none,
    getenv2,
)
from pynight.common_model_name import (
    model_name_clip_p,
    model_name_eva2_p,
    model_needs_MLP_DU_p,
    model_name_mixer_p,
)
from pynight.common_regex import (
    rget,
    float_pattern,
    rget_percent,
    re_maybe,
)
from pynight.common_dict import (
    simple_obj,
)


def gbrands_from(
    model_name=None,
    ig_steps=None,
    gbrand_preset=None,
    gradient_mode_brand=None,
    patchifier_gbrand=None,
    linear_ds_gbrand=None,
    qkv_ds_gbrand=None,
    mlp_ds_gbrand=None,
    mlp_mul_gbrand=None,
    softmax_mode=None,
    normalize_to_unit_vector_gbrand=None,
    configure_p=False,
    pruned_mode_p=True,
):
    ##
    ig_steps = getenv2(ig_steps, "DECOMPV_IG_STEPS", "0")
    gbrand_preset = getenv2(gbrand_preset, "DECOMPV_GBRAND_PRESET", "")

    if ig_steps:
        ig_steps = int(ig_steps)

    softmax_default_mode = "S1"
    # softmax_default_mode = "S0"

    # best_softmax_mode = "S1"
    # best_softmax_mode = "XSC"
    # best_softmax_mode = "C2_1"
    # best_softmax_mode = "C50_1"
    best_softmax_mode = softmax_default_mode

    if gbrand_preset:
        pruned_mode_p = True
        force_pruned_mode_p = global_force_pruned_mode_p
        #: If `force_pruned_mode_p`, all presets will prune.

        patchifier_gbrand = "DS100"
        linear_ds_gbrand = None
        qkv_ds_gbrand = None
        mlp_ds_gbrand = None

        ###
        normalize_to_unit_vector_gbrand = None
        best_N2U = None
        # best_N2U = "LX1"

        emphasize_pos = rget(
            gbrand_preset,
            rf"_N2U.(EP{float_pattern})$",
        )

        if gbrand_preset.endswith("_N2U.SP"):
            # force_pruned_mode_p = force_pruned_mode_p or True

            normalize_to_unit_vector_gbrand = "self_pos_v1"
            best_N2U = normalize_to_unit_vector_gbrand

            gbrand_preset = gbrand_preset[: -len("_N2U.SP")]

        elif emphasize_pos:
            # force_pruned_mode_p = force_pruned_mode_p or True

            normalize_to_unit_vector_gbrand = emphasize_pos
            best_N2U = normalize_to_unit_vector_gbrand

            gbrand_preset = gbrand_preset[: -len("_N2U." + emphasize_pos)]

        elif gbrand_preset.endswith("_N2U"):
            # force_pruned_mode_p = force_pruned_mode_p or True

            normalize_to_unit_vector_gbrand = "LX1"
            best_N2U = normalize_to_unit_vector_gbrand

            gbrand_preset = gbrand_preset[:-4]

        ###
        if gbrand_preset.endswith("_S1"):
            softmax_mode_override = "S1"
            gbrand_preset = gbrand_preset[:-3]

        elif gbrand_preset.endswith("_S0"):
            softmax_mode_override = "S0"
            gbrand_preset = gbrand_preset[:-3]

        elif gbrand_preset.endswith("_XSC"):
            softmax_mode_override = "XSC"
            gbrand_preset = gbrand_preset[:-4]

        else:
            softmax_mode_override = None

        if softmax_mode_override is not None:
            force_pruned_mode_p = force_pruned_mode_p or (
                softmax_mode_override != best_softmax_mode
            )

            softmax_default_mode = softmax_mode_override
            best_softmax_mode = softmax_mode_override

        softmax_mode = softmax_default_mode
        ###

        if gbrand_preset in [
            "Fair",
            "FairGrad",
            "LibraGrad",
            "Libra",
        ]:
            softmax_mode = best_softmax_mode

            pruned_mode_p = force_pruned_mode_p

            gradient_mode_brand = "LX1"
            mlp_mul_gbrand = "DU"
            normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset in [
            "Fair-GA2",
        ]:
            pruned_mode_p = force_pruned_mode_p

            softmax_mode = best_softmax_mode

            gradient_mode_brand = "LX-GA-D2"
            mlp_mul_gbrand = "DU"
            normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset in [
            "Fair-Scale",
        ]:
            # pruned_mode_p = force_pruned_mode_p

            softmax_mode = best_softmax_mode

            gradient_mode_brand = "NG-D2"
            mlp_mul_gbrand = "DU"
            normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset in [
            "Fair-Allen",
        ]:
            # pruned_mode_p = force_pruned_mode_p

            softmax_mode = best_softmax_mode

            gradient_mode_brand = "Gate-D2"
            mlp_mul_gbrand = "DU"
            normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "AttnLRP":
            if softmax_mode_override is None:
                softmax_mode = "S0"

            gradient_mode_brand = "LX-GA-D2"
            mlp_mul_gbrand = "DU"

        elif gbrand_preset == "AliLRP":
            gradient_mode_brand = "LX-NA"
            #: No Activation

            mlp_mul_gbrand = None

        elif gbrand_preset == "DecompX":
            if softmax_mode_override is None:
                softmax_mode = "S0"

            gradient_mode_brand = "LX1"
            mlp_mul_gbrand = None

        elif gbrand_preset in [
            "NG",
            "IG",
        ]:
            pruned_mode_p = force_pruned_mode_p

            gradient_mode_brand = "NG"
            mlp_mul_gbrand = None
            if gbrand_preset == "IG":
                assert (
                    not ig_steps
                ), "You have set ig_steps manually, but are still requesting the IG preset. An error has been raised to make sure you know what you are doing."
                ig_steps = 50

        ###
        elif gbrand_preset == "Attn":
            softmax_mode = best_softmax_mode
            # softmax_mode = "XSC"

            gradient_mode_brand = "LXA"
            mlp_mul_gbrand = None

        elif gbrand_preset == "LN":
            softmax_mode = best_softmax_mode

            gradient_mode_brand = "LN"
            mlp_mul_gbrand = None
            # normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "Act":
            softmax_mode = best_softmax_mode

            gradient_mode_brand = "Gate"
            mlp_mul_gbrand = None
            # normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "MLP-Gate":
            softmax_mode = best_softmax_mode

            gradient_mode_brand = "Gate"
            mlp_mul_gbrand = "DU"
            # normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "MLP-NoGate":
            softmax_mode = best_softmax_mode
            gradient_mode_brand = "NG"
            mlp_mul_gbrand = "DU"
            # normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "N2U":
            softmax_mode = best_softmax_mode

            gradient_mode_brand = "NG"
            mlp_mul_gbrand = None
            normalize_to_unit_vector_gbrand = best_N2U

        ###
        elif gbrand_preset == "WO-Head":  #: same as "FairGrad_S0"
            softmax_mode = "S0"
            gradient_mode_brand = "LX1"
            mlp_mul_gbrand = "DU"
            normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "WO-LN":
            softmax_mode = best_softmax_mode
            gradient_mode_brand = "LX-N_LN"
            mlp_mul_gbrand = "DU"
            normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "WO-Act":
            softmax_mode = best_softmax_mode
            gradient_mode_brand = "LX-NA"
            mlp_mul_gbrand = "DU"
            normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "WO-MLP":
            softmax_mode = best_softmax_mode
            gradient_mode_brand = "LX1"
            mlp_mul_gbrand = None
            normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "WO-Att":  #: AKA Fair-GA
            softmax_mode = best_softmax_mode
            gradient_mode_brand = "LX-GA"
            mlp_mul_gbrand = "DU"
            normalize_to_unit_vector_gbrand = best_N2U

        elif gbrand_preset == "WO-N2U":
            softmax_mode = best_softmax_mode
            gradient_mode_brand = "LX1"
            mlp_mul_gbrand = "DU"
            normalize_to_unit_vector_gbrand = None

        else:
            raise ValueError(f"Unknown gbrand preset: {gbrand_preset}")

    else:
        pruned_mode_p = bool_from_str(
            getenv2(
                pruned_mode_p,
                "DECOMPV_PRUNED_P",
                True,
            )
        )

        gradient_mode_brand = getenv2(
            gradient_mode_brand, "DECOMPV_GRADIENT_BRAND", "NG"
        )
        patchifier_gbrand = getenv2(
            patchifier_gbrand, "DECOMPV_PATCHIFIER_GBRAND", None
        )
        linear_ds_gbrand = getenv2(linear_ds_gbrand, "DECOMPV_LINEAR_DS_GBRAND", None)
        qkv_ds_gbrand = getenv2(qkv_ds_gbrand, "DECOMPV_QKV_DS_GBRAND", None)
        mlp_ds_gbrand = getenv2(mlp_ds_gbrand, "DECOMPV_MLP_DS_GBRAND", None)
        mlp_mul_gbrand = getenv2(mlp_mul_gbrand, "DECOMPV_MLP_MUL_GBRAND", None)
        softmax_mode = getenv2(
            softmax_mode, "DECOMPV_SOFTMAX_MODE", softmax_default_mode
        )
        normalize_to_unit_vector_gbrand = getenv2(
            normalize_to_unit_vector_gbrand, "DECOMPV_N2U_GBRAND", None
        )

    if model_name_mixer_p(model_name):
        #: MLP-Mixer has no attention module, so we need to convert gbrands with attention to their attention-less variants.

        if any(
            re.search(pat, gradient_mode_brand)
            for pat in [
                "^NG-D",
                "^LXA",
            ]
        ):
            gradient_mode_brand = "NG"

        elif any(
            re.search(pat, gradient_mode_brand)
            for pat in [
                "^Gate-D",
            ]
        ):
            gradient_mode_brand = "Gate"

        elif any(
            re.search(pat, gradient_mode_brand)
            for pat in [
                "^LX-GA",
            ]
        ):
            gradient_mode_brand = "LX1"

    if not model_needs_MLP_DU_p(model_name):
        mlp_mul_gbrand = None

    if model_name_clip_p(model_name):
        softmax_mode = "S0"

    else:
        normalize_to_unit_vector_gbrand = None

    if (
        qual_prototype_p
        or run_check_completeness_mode_p
        or run_compute_completeness_mode_p
    ):
        pruned_mode_p = True

    compact_gbrand = get_compact_gbrand(
        ig_steps=ig_steps,
        gradient_mode_brand=gradient_mode_brand,
        patchifier_gbrand=patchifier_gbrand,
        linear_ds_gbrand=linear_ds_gbrand,
        qkv_ds_gbrand=qkv_ds_gbrand,
        mlp_ds_gbrand=mlp_ds_gbrand,
        mlp_mul_gbrand=mlp_mul_gbrand,
        softmax_mode=softmax_mode,
        normalize_to_unit_vector_gbrand=normalize_to_unit_vector_gbrand,
        # pruned_mode_p=pruned_mode_p,
    )

    if configure_p:
        from timm.models.decomposition import (
            configure_gradient_modes,
        )

        all_gbrands = configure_gradient_modes(
            model_name=model_name,
            ig_steps=ig_steps,
            gradient_mode_brand=gradient_mode_brand,
            softmax_mode=softmax_mode,
            patchifier_gbrand=patchifier_gbrand,
            linear_ds_gbrand=linear_ds_gbrand,
            qkv_ds_gbrand=qkv_ds_gbrand,
            mlp_ds_gbrand=mlp_ds_gbrand,
            mlp_mul_gbrand=mlp_mul_gbrand,
            normalize_to_unit_vector_gbrand=normalize_to_unit_vector_gbrand,
        )
        assert compact_gbrand == all_gbrands.compact_gbrand

    else:
        all_gbrands = None

    return simple_obj(
        model_name=model_name,
        gbrand_preset=gbrand_preset,
        all_gbrands=all_gbrands,
        compact_gbrand=compact_gbrand,
        ig_steps=ig_steps,
        gradient_mode_brand=gradient_mode_brand,
        patchifier_gbrand=patchifier_gbrand,
        linear_ds_gbrand=linear_ds_gbrand,
        qkv_ds_gbrand=qkv_ds_gbrand,
        mlp_ds_gbrand=mlp_ds_gbrand,
        mlp_mul_gbrand=mlp_mul_gbrand,
        softmax_mode=softmax_mode,
        normalize_to_unit_vector_gbrand=normalize_to_unit_vector_gbrand,
        pruned_mode_p=pruned_mode_p,
    )


def gbrand_from_preset(
    *,
    model_name,
    gbrand_preset,
    method_obj=None,  #: unused arg to conform to the interface
    gbrand_postfix=None,
):
    if gbrand_postfix:
        gbrand_preset += gbrand_postfix

    raw_prefix = "raw:"
    if gbrand_preset.lower().startswith(raw_prefix):
        return gbrand_preset[len(raw_prefix) :]

    compact_gbrand = gbrands_from(
        model_name=model_name,
        gbrand_preset=gbrand_preset,
    ).compact_gbrand

    return compact_gbrand


### * Main
if __name__ == "__main__":
    globals().update(gbrands_from())
    print(compact_gbrand)
