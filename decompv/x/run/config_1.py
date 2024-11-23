from decompv.x.run.run_v1_default_config import *

configs = {
    "faith": {
        # "ablation_p": True,
        # "gbrand_presets": [
        # ],
        # "manual_params": {
        #     "DECOMPV_GRADIENT_BRAND": [
        #         "LX1",
        #     ],
        #     "DECOMPV_SOFTMAX_MODE": [
        #         "S1",
        #     ],
        #     "DECOMPV_PATCHIFIER_GBRAND": ["DS100"],
        #     "DECOMPV_MLP_MUL_GBRAND": [
        #         "",
        #         "DU",
        #     ],
        #     "DECOMPV_MLP_DS_GBRAND": [""],
        #     "DECOMPV_QKV_DS_GBRAND": [""],
        #     "DECOMPV_LINEAR_DS_GBRAND": [""],
        #     "DECOMPV_N2U_GBRAND": [""],
        # },
        "additional_env_vars": {},
    },
    "qual": {
        "colormaps": [
            "magma",
        ],
        "outlier_quantiles": [
            "0.01",
        ],
        # "gbrand_presets": [
        # ],
        "additional_env_vars": {
            "DECOMPV_SAVE_ATTR_MODE": "none",
        },
    },
    "seg": {
        # "gbrand_presets": [
        # ],
        ##
        # "manual_params": {
        #     "DECOMPV_GRADIENT_BRAND": [
        #         "NG",
        #         "LX1",
        #         "LX-GA-D2",
        #     ],
        #     "DECOMPV_SOFTMAX_MODE": [
        #         "S0",
        #     ],
        #     "DECOMPV_PATCHIFIER_GBRAND": [
        #         "DS100",
        #     ],
        #     "DECOMPV_MLP_MUL_GBRAND": [
        #         "",
        #         "DU",
        #     ],
        #     "DECOMPV_MLP_DS_GBRAND": [
        #         "",
        #     ],
        #     "DECOMPV_QKV_DS_GBRAND": [
        #         "",
        #     ],
        #     "DECOMPV_LINEAR_DS_GBRAND": [
        #         "",
        #     ],
        # },
        ##
        "additional_env_vars": {
            "DECOMPV_SEG_DATASET_NAME": "ImageNetS",
            "DECOMPV_SEG_ABLATION_P": "n",
            # "DECOMPV_SEG_ABLATION_P": "y",
        },
    },
}
