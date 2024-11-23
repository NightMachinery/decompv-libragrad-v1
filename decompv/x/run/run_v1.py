#!/usr/bin/env python3

import os
import jsonlines
import sys
import time
import itertools
import subprocess

# from tqdm import tqdm
from pynight.common_tqdm import (
    tqdm,
)
import argparse
import socket
import importlib.util
import pynight.common_telegram as common_telegram
from pynight.common_telegram import (
    tlg_chat_id_me,
    tlg_chat_id_me_notif,
)
from pynight.common_icecream import ic
from pynight.common_tui import ask
from pynight.common_hosts import hostname_get
from pynight.common_files import hdd_free_get
from pynight.common_iterable import (
    list_dup_rm,
    to_iterable,
)
from pynight.common_model_name import (
    model_name_clip_p,
    model_name_eva2_p,
    model_needs_MLP_DU_p,
    model_name_mixer_p,
)

import decompv.early_boot
from decompv.early_boot import (
    DONE_EXPERIMENTS,
    DONE_EXPERIMENTS_TO_MERGE,
    tlg_me,
)
from decompv.early_boot import *

# from decompv.x.run.run_v1_default_config import (
# )
import decompv.x.run.run_v1_default_config as default_config
from decompv.x.run.sync_experiments import (
    sync_with_to_merge,
)
from decompv.x.gbrand import gbrands_from
from pynight.common_dict import simple_obj

##


def log_info(message, log_file=None):
    """Log information to both console and file if provided."""
    print(message, flush=True)

    if log_file:
        with open(log_file, "a") as f:
            f.write(f"{message}\n")


def set_env(
    *,
    key,
    value,
    env,
    # none_to=None,
    none_to="",
):
    """Set environment variable in the provided env dict."""
    if value is not None:
        env[key] = value

    elif none_to is not None:
        env[key] = none_to

    else:
        env.pop(key, None)


def config_get(config, key):
    default = getattr(default_config, key, None)

    if isinstance(config, dict):
        return config.get(key, default)

    else:
        return getattr(
            config,
            key,
            default,
        )


def generate_experiment_hash(
    *,
    gbrand_info,
    script_type,
    params,
    script_config,
    pruned_mode_p,
    run_force_submode,
    seg_dataset_end=None,
):
    """Generate a unique hash dictionary for the experiment."""
    model_name = params["DECOMPV_MODEL_NAME"]
    ig_steps = gbrand_info.ig_steps

    hash_dict = {
        "model_name": model_name,
        "script_type": script_type,
        "compact_gbrand": gbrand_info.compact_gbrand,
    }

    if run_check_completeness_mode_p:
        assert (
            script_type == "faith"
        ), "run_check_completeness_mode_p is only for faith."

    if run_compute_completeness_mode_p:
        assert (
            script_type == "faith"
        ), "run_compute_completeness_mode_p is only for faith."

        hash_dict["run_compute_completeness_mode_p"] = run_compute_completeness_mode_p
        if ig_steps:
            hash_dict["sub_version_ig"] = 2

    if not pruned_mode_p:
        hash_dict["pruned_mode_p"] = pruned_mode_p
        #: This way, full runs will have a different signature from pruned runs.

    if run_force_submode is not None:
        hash_dict["run_force_submode"] = run_force_submode

    #: Since we are going to use different submodes for different gbrand presets, but we do not want to repeat the experiments, we should not store the submode. If, later on, we add a new submode (and need to be able to run the experiments for that submode), we can conditionally add that submode to the hash.

    if script_type == "faith":
        hash_dict.update(
            {
                ##
                "version": 2,
                #: V2: has raw completeness_error
                ##
                "dataset_name": params.get("DECOMPV_DATASET_NAME", ""),
                # "submode1": script_config.get("SUBMODE1", ""),
                "metrics_target": params.get("DECOMPV_METRICS_TARGET", ""),
            }
        )

        DECOMPV_DATASET_END = config_get(script_config, "DECOMPV_DATASET_END")
        if DECOMPV_DATASET_END == 1000:
            #: We didn't store this value for 1000, so for backcompat, we skip storing it.
            pass

        else:
            hash_dict["dataset_end"] = DECOMPV_DATASET_END

    elif script_type == "seg":
        # hash_dict["submode"] = script_config.get("submode", "")
        if "huge" in model_name:
            hash_dict["sub_version_huge"] = 2
            #: V2: Huge models are run with 5000 images like the rest.

        if seg_dataset_end is not None and seg_dataset_end != 5000:
            hash_dict["seg_dataset_end"] = seg_dataset_end

    elif script_type == "qual":
        dataset_name = params.get("DECOMPV_QUAL_DATASET_NAME", "")
        if dataset_name == "zele":
            dataset_name += "_v2"
            #: previous versions were buggy and we need to redo them

        hash_dict.update(
            {
                "version": 7,
                #: 7: also exports attr only (without blending)
                ##
                "dataset": dataset_name,
                "colormap": params.get("DECOMPV_COLORMAP", ""),
                "outlier_quantile": params.get("DECOMPV_OUTLIER_QUANTILE", ""),
            }
        )

        if qual_submode:  #: @global from early_boot
            hash_dict["qual_submode"] = qual_submode

        if not pruned_mode_p and dataset_name == "CLIP6":
            hash_dict["sub_version_c6"] = 2
            #: There was a bug in `has_class_token` for CLIP models which made the non-pruned versions of CLIP6 incomplete, so we are changing the hash of these experiments.

    if "beit" in model_name:
        hash_dict["sub_version_beit"] = 3
        #: Fixed QKV biases

    return hash_dict


def experiment_done_p(
    *,
    exp_hash,
    skip_done_file=DONE_EXPERIMENTS,
):
    """Check if the experiment has already been done, regardless of hostname."""
    try:
        with jsonlines.open(skip_done_file) as reader:
            return any(item["hash"] == exp_hash for item in reader)
    except FileNotFoundError:
        return False


def mark_experiment_done(
    *,
    exp_hash,
    hostname,
    skip_done_file=DONE_EXPERIMENTS,
):
    """Mark the experiment as done, including the hostname for reference."""
    if qual_prototype_p or run_check_completeness_mode_p:
        return

    with jsonlines.open(skip_done_file, mode="a") as writer:
        writer.write(
            {
                "hash": exp_hash,
                "hostname": hostname,
                "timm": "after_99cc35d1",
                "unix_date": int(time.time()),
            }
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "models",
        nargs="+",
        help="Model names to run experiments on",
    )

    parser.add_argument(
        "--script-type",
        type=str,
        choices=["faith", "seg", "qual"],
        help="Type of script to run (optional, read from config if not provided)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        # choices=[
        #     ground_truth_mode_cst,
        #     first_highest_predicted_mode_cst,
        #     second_highest_predicted_mode_cst,
        # ],
        default="",
        help="Comma-separated list of datasets (optional)",
    )
    parser.add_argument(
        "--metrics-target",
        type=str,
        # choices=[
        #     ground_truth_mode_cst,
        #     first_highest_predicted_mode_cst,
        #     second_highest_predicted_mode_cst,
        # ],
        default=ground_truth_mode_cst,
        help="Comma-separated list of metrics target (optional)",
    )
    parser.add_argument(
        "--hostnames",
        type=str,
        help="Comma-separated list of hostnames of all servers (optional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    return parser.parse_args()


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def set_environment_variables(
    *,
    params,
    gbrand_info,
    script_config,
    env,
):
    """Set environment variables based on parameters, gbrand_info, and script_config."""
    ##
    #: These values are set from the config directly, which means they should not change between the different iterations.
    set_env(
        key="DECOMPV_DATASET_START",
        value=config_get(script_config, "DECOMPV_DATASET_START"),
        env=env,
        none_to=None,
    )

    set_env(
        key="DECOMPV_DATASET_END",
        value=config_get(script_config, "DECOMPV_DATASET_END"),
        env=env,
        none_to=None,
    )

    set_env(
        key="DECOMPV_SAVE_ATTR_MODE",
        value=config_get(script_config, "DECOMPV_SAVE_ATTR_MODE"),
        env=env,
        none_to=None,
    )

    set_env(
        key="DECOMPV_ATTR_BATCH_SIZE",
        value=config_get(script_config, "DECOMPV_ATTR_BATCH_SIZE"),
        env=env,
        none_to=None,
    )

    set_env(
        key="DECOMPV_SHARED_ROOT",
        value=config_get(script_config, "DECOMPV_SHARED_ROOT"),
        env=env,
        none_to=None,
    )
    ##
    #: =params=
    #: These values might change between the different iterations.
    set_env(
        key="DECOMPV_DATASET_NAME",
        value=params.get("DECOMPV_DATASET_NAME", None),
        env=env,
        none_to=None,
    )
    set_env(
        key="DECOMPV_METRICS_TARGET",
        value=params.get("DECOMPV_METRICS_TARGET", None),
        env=env,
        none_to=None,
    )
    set_env(
        key="DECOMPV_MODEL_NAME",
        value=params.get("DECOMPV_MODEL_NAME", None),
        env=env,
        none_to=None,
    )
    set_env(
        key="DECOMPV_COLORMAP",
        value=params.get("DECOMPV_COLORMAP", None),
        env=env,
        none_to=None,
    )
    set_env(
        key="DECOMPV_OUTLIER_QUANTILE",
        value=str(params.get("DECOMPV_OUTLIER_QUANTILE", None)),
        env=env,
        none_to=None,
    )
    ##
    #: Set variables from gbrand_info
    set_env(
        key="DECOMPV_GRADIENT_BRAND",
        value=gbrand_info.gradient_mode_brand,
        env=env,
        none_to="",
    )
    set_env(
        key="DECOMPV_SOFTMAX_MODE",
        value=gbrand_info.softmax_mode,
        env=env,
        none_to="",
    )
    set_env(
        key="DECOMPV_PATCHIFIER_GBRAND",
        value=gbrand_info.patchifier_gbrand,
        env=env,
        none_to="",
    )
    set_env(
        key="DECOMPV_LINEAR_DS_GBRAND",
        value=gbrand_info.linear_ds_gbrand,
        env=env,
        none_to="",
    )
    set_env(
        key="DECOMPV_QKV_DS_GBRAND",
        value=gbrand_info.qkv_ds_gbrand,
        env=env,
        none_to="",
    )
    set_env(
        key="DECOMPV_MLP_DS_GBRAND",
        value=gbrand_info.mlp_ds_gbrand,
        env=env,
        none_to="",
    )
    set_env(
        key="DECOMPV_MLP_MUL_GBRAND",
        value=gbrand_info.mlp_mul_gbrand,
        env=env,
        none_to="",
    )
    set_env(
        key="DECOMPV_N2U_GBRAND",
        value=gbrand_info.normalize_to_unit_vector_gbrand,
        env=env,
        none_to="",
    )
    set_env(
        key="DECOMPV_IG_STEPS",
        value=str(gbrand_info.ig_steps),
        env=env,
        none_to=None,
    )
    ##
    for key, value in script_config.get("additional_env_vars", {}).items():
        set_env(key=key, value=value, env=env, none_to="")


def run_subprocess(
    *,
    log_file_path,
    script_args,
    env,
    output_mode="direct_tee",
    min_free_hdd=None,
    export_tlg_id=tlg_chat_id_me_notif,
    # skip_done_file=None,
    # exp_hash=None,
):
    """Run the subprocess and handle streaming output."""
    current_hostname = hostname_get()

    if min_free_hdd:
        free_gb = hdd_free_get(
            unit="GB",
            path=SHARED_ROOT,
        )

        if free_gb < min_free_hdd:
            msg = f"{current_hostname}: Not enough free disk space. Required: {min_free_hdd}GB, Available: {free_gb:.1f}GB"
            log_info(msg)

            if export_tlg_id:
                common_telegram.send(
                    export_tlg_id,
                    msg=msg,
                )

            #: Wait for user confirmation before continuing
            response = None
            if ask("Do you want to continue anyway?", default=None):
                log_info("Continuing the operation.")
            else:
                log_info(
                    "Operation cancelled by the user due to insufficient disk space."
                )
                exit(1)

    ic(script_args, log_file_path, output_mode)

    if output_mode == "direct_tee":
        #: We'll later use 'tee' to write output both to stdout and to the log file
        process = subprocess.Popen(
            script_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env,
            start_new_session=True,
        )
        #: [[https://docs.python.org/3/library/subprocess.html][subprocess — Subprocess management — Python 3.13.0 documentation]]
        #: @o1 start_new_session: When set to True, the child process will be started in a new session and process group. This means that signals like SIGINT sent to the parent process won't be sent to the child process and vice versa.

        with open(log_file_path, "a") as log_file:
            #: Start the 'tee' subprocess
            tee_process = subprocess.Popen(
                ["tee", "-a", log_file_path],
                stdin=process.stdout,
                env=env,
                start_new_session=True,
            )

            process.stdout.close()  # Allow process to receive a SIGPIPE if tee_process exits.

            # Wait for the subprocesses to complete
            process_return_code = process.wait()
            tee_process_return_code = tee_process.wait()

            if process_return_code != 0:
                raise subprocess.CalledProcessError(process_return_code, script_args)
    else:
        with open(log_file_path, "a") as log_file:
            process = subprocess.Popen(
                script_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  #: @LLM Line buffered
                env=env,
                start_new_session=True,
            )

            # Stream output
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                print(line, end="")

            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, script_args)


def faith_preprocess(*, model_name, env, **kwargs):
    """Faith-specific preprocessing step."""

    if run_skip_main_p:
        return

    ##
    #: @duplicateCode/6a3ad540403ebe103105ff868160268f
    dataset_patchified_path = f"{DS_PATCHIFIED_PATH}/{model_name}"
    ##
    if os.path.exists(dataset_patchified_path):
        msg = f"compute_patchify: dataset already exists, aborting: {dataset_patchified_path}"
        print(msg, file=sys.stderr)

        return

    ##
    log_info("Starting faith preprocess step...")
    set_env(key="DECOMPV_MODEL_NAME", value=model_name, env=env, none_to=None)

    preprocess_args = [
        "python",
        os.path.expanduser("~/code/DecompV/decompv/x/ds/compute_patchify.py"),
        "v3",
    ]
    log_file = os.path.join(
        getattr(default_config, "LOG_BASE_DIR", "~/logs/"), "patchify_log"
    )

    try:
        run_subprocess(
            log_file_path=log_file,
            script_args=preprocess_args,
            env=env,
        )

    except subprocess.CalledProcessError:
        log_info(
            "\nDataset already patchified or preprocessing failed.",
        )

    else:
        log_info("\nDataset patchified successfully.")


def faith_postprocess(*, gbrand_info, env, **kwargs):
    """Faith-specific postprocessing step."""

    if run_compute_completeness_mode_p:
        postprocess_args = [
            "python",
            os.path.expanduser(
                "~/code/DecompV/decompv/x/ds/compute_completeness_metrics.py"
            ),
        ]
        log_file = os.path.join(
            getattr(default_config, "LOG_BASE_DIR", "~/logs/"), "postprocess_CE_log"
        )
        run_subprocess(
            log_file_path=log_file,
            script_args=postprocess_args,
            env=env,
        )
        return

    if run_check_completeness_mode_p:
        return

    log_info("Starting faith postprocess step...")
    compact_gbrand = gbrand_info.compact_gbrand
    model_name = env.get("DECOMPV_MODEL_NAME")
    dataset_name = env.get("DECOMPV_DATASET_NAME")
    metrics_target = env.get("DECOMPV_METRICS_TARGET")

    input_dir = f"/opt/decompv/datasets"
    if dataset_name != "ImageNetVal":
        input_dir += f"/{dataset_name}"
    input_dir += f"/{model_name}/{compact_gbrand}/{metrics_target}"

    postprocess_args = [
        "python",
        os.path.expanduser("~/code/DecompV/decompv/x/ds/compute_cls_metrics.py"),
        input_dir,
    ]
    log_file = os.path.join(
        getattr(default_config, "LOG_BASE_DIR", "~/logs/"), "postprocess_log"
    )
    run_subprocess(
        log_file_path=log_file,
        script_args=postprocess_args,
        env=env,
    )


def generate_parameter_combinations(
    *,
    models,
    script_config,
    script_type,
    metrics_target,
    datasets,
):
    ##
    gbrand_presets_main = [
        "FairGrad",
        "NG",
    ]

    if script_type == "qual":
        gbrand_presets_main += [
            "IG",
        ]

    elif script_type == "faith":
        if metrics_target == "gt":
            gbrand_presets_main += [
                # "Fair-GA2",
                # "Fair-Allen",
                # "Fair-Scale",
            ]

    gbrand_presets_other = [
        "AliLRP",
        "AttnLRP",  #: needed for Qual CLIP
        "DecompX",
    ]

    gbrand_presets_headless = [
        #: Necessary for completeness checks
        ##
        "FairGrad_S0",  #: same as "WO-Head"
        "NG_S0",
        "IG_S0",
        "AliLRP_S0",
        "AttnLRP_S0",
        "DecompX_S0",
    ]

    gbrand_presets_ablation = [
        "IG",
        "WO-LN",
        "WO-Act",
        "WO-MLP",  #: Has Activation Gate
        "WO-Att",
        "WO-N2U",
        "FairGrad_S0",  #: same as "WO-Head"
        "MLP-Gate",
        # "Fair-Allen",
        # "MLP-NoGate",
    ]

    gbrand_presets_ablation_2 = [
        "MLP-Gate",
        # "MLP-NoGate",
        # "Gate",
        # "LN",
        # "Attn",
        # "Act",
    ]

    if metrics_target == second_highest_predicted_mode_cst:
        #: No need for ablation here
        gbrand_presets_ablation = []
        gbrand_presets_ablation_2 = []

    all_gbrand_presets_no_ablation = [
        *gbrand_presets_main,
        *gbrand_presets_other,
    ]

    all_gbrand_presets = [
        *all_gbrand_presets_no_ablation,
        *gbrand_presets_ablation,
    ]
    ##
    all_parameter_combinations = []

    datasets_ = None

    for model_name in models:
        # Set manual_params and gbrand_presets
        manual_params = script_config.get("manual_params", {})
        gbrand_presets = script_config.get("gbrand_presets", [])

        # Logic to auto-set gbrand_presets based on model_name
        if not gbrand_presets and not manual_params:
            log_info(
                f"No gbrand preset or manual params set. Auto-setting the presets for the model: {model_name}"
            )
            if run_check_completeness_mode_p:
                gbrand_presets = [
                    "FairGrad_S0",
                ]

            elif run_compute_completeness_mode_p:
                gbrand_presets = [
                    *gbrand_presets_headless,
                ]

            elif model_name_clip_p(model_name) or model_name in [
                "EVA02-L-14-336.OC.merged2b_s6b_b61k",
                "vit_base_patch16_224.OC.BiomedCLIP-PubMedBERT_256",
            ]:
                #: CLIP models
                ##
                if qual_prototype_p:
                    gbrand_presets = [
                        "FairGrad",
                        "AttnLRP",
                        # "FairGrad_N2U",
                    ]

                else:
                    gbrand_presets = [
                        "FairGrad",
                        # "FairGrad_N2U",
                        # "FairGrad_N2U.EP1.125",
                        # "FairGrad_N2U.EP1.25",
                        # "FairGrad_N2U.EP1.5",
                        # "FairGrad_N2U.EP2",
                        # "FairGrad_N2U.EP2.5",
                    ]

                    if not qual_submode:  #: @global qual_submode
                        gbrand_presets += [
                            # "FairGrad_N2U.SP",
                            # "FairGrad_N2U.EP10",
                            # "FairGrad_N2U.EP5",
                            # "FairGrad_N2U.EP3.5",
                            # "FairGrad_N2U.EP100",
                            *all_gbrand_presets,
                            *gbrand_presets_ablation_2,
                        ]

            elif model_name in [
                "eva02_small_patch14_336.mim_in22k_ft_in1k",
            ]:
                gbrand_presets = [
                    # *gbrand_presets_headless,
                    *all_gbrand_presets,
                ]

            elif model_name in [
                "gmixer_24_224.ra3_in1k",
                ##
                # "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
                ##
            ]:
                gbrand_presets = all_gbrand_presets

            elif model_name in [
                "vit_base_patch16_224",  #: MURA, oxford_pet
                "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                "vit_small_patch16_224.augreg_in21k_ft_in1k",
                "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                "vit_large_patch16_224.augreg_in21k_ft_in1k",
                "deit3_huge_patch14_224.fb_in22k_ft_in1k",
                "beitv2_large_patch16_224.in1k_ft_in22k_in1k",
                "vit_large_patch14_clip_224.openai_ft_in1k",
                "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",
                "vit_so400m_patch14_siglip_gap_378.webli_ft_in1k",
                "flexivit_large.1200ep_in1k",
            ]:
                gbrand_presets = [
                    *all_gbrand_presets_no_ablation,
                    "IG",
                ]

            else:
                gbrand_presets = all_gbrand_presets_no_ablation

        # Build parameter lists for this model
        parameter_lists = {
            "DECOMPV_MODEL_NAME": [model_name],
        }

        if gbrand_presets:
            assert (
                not manual_params
            ), "Both a gbrand preset and manual params are present!"

            #: Retain only the first item in gbrand_presets, remove dups:
            gbrand_presets = list_dup_rm(
                gbrand_presets,
                keep_first_p=True,
            )

            parameter_lists["gbrand_preset"] = gbrand_presets
            ic(parameter_lists["gbrand_preset"])

        else:
            parameter_lists.update(manual_params)

        # Add other parameters based on script_type
        if script_type == "faith":
            if datasets:
                datasets_ = datasets

            else:
                datasets_ = ["ImageNetVal"]

            parameter_lists.update(
                {
                    "DECOMPV_METRICS_TARGET": metrics_target,
                    "DECOMPV_DATASET_NAME": datasets_,
                }
            )

        elif script_type == "qual":
            colormaps = config_get(script_config, "colormaps")
            outlier_quantiles = config_get(script_config, "outlier_quantiles")

            if datasets:
                datasets_ = datasets

            else:
                datasets_ = ["CLIP1"]

            parameter_lists.update(
                {
                    "DECOMPV_QUAL_DATASET_NAME": datasets_,
                    "DECOMPV_COLORMAP": colormaps or [""],
                    "DECOMPV_OUTLIER_QUANTILE": outlier_quantiles or [""],
                }
            )

        elif script_type == "seg":
            pass

        else:
            raise ValueError(f"Invalid script_type '{script_type}' provided.")

        # Generate parameter combinations for this model
        keys, values = zip(*parameter_lists.items())
        #: The zip object yields n-length tuples, where n is the number of iterables passed as positional arguments to zip(). The i-th element in every tuple comes from the i-th iterable argument to zip(). This continues until the shortest argument is exhausted.

        parameter_combinations = [
            dict(zip(keys, v)) for v in itertools.product(*values)
        ]

        all_parameter_combinations.extend(parameter_combinations)

    return simple_obj(
        parameter_combinations=all_parameter_combinations,
        datasets=datasets_,
    )


# Map script types to their model preprocessors and postprocessors
model_preprocessors = {
    "faith": [faith_preprocess],
    "seg": [],
    "qual": [],
}

postprocess_functions = {
    "faith": [faith_postprocess],
    "seg": [],
    "qual": [],
}


def main():
    log_info(f"Current process PID: {os.getpid()}")

    sync_with_to_merge()

    continue_on_fail_p = True

    args = parse_args()
    config = load_config(args.config)
    skip_done_file = DONE_EXPERIMENTS

    # Determine script_type
    script_type = args.script_type or getattr(config, "script_type", None)
    if script_type not in ["faith", "seg", "qual"]:
        raise ValueError(
            "Script type must be specified either via --script-type or in config.py"
        )

    # Determine hostname list
    if args.hostnames:
        hostname_list = [host.strip() for host in args.hostnames.split(",")]

    else:
        hostname_list = [hostname_get()]  # Only current server

    current_hostname = hostname_get()
    SERVER_COUNT = len(hostname_list)
    try:
        SERVER_INDEX = hostname_list.index(current_hostname)
    except ValueError:
        raise ValueError(
            f"Current hostname {current_hostname} not found in the list of hostnames."
        )

    log_info(f"Server index: {SERVER_INDEX} out of {SERVER_COUNT}")

    # Load configurations based on script type
    script_config = config.configs.get(script_type, None)
    if script_config is None:
        raise ValueError(f"Script type does not exist in configs: {script_type}")

    models = args.models
    if not models:
        models = script_config.get("models", [])

    if not models:
        raise ValueError(
            "No model names provided (either on the CLI or in the config). Please specify at least one model name."
        )

    if args.metrics_target:
        metrics_target = args.metrics_target.split(",")

    else:
        metrics_target = None

    if args.datasets:
        datasets = args.datasets.split(",")

    else:
        datasets = script_config.get("datasets", None)

    if script_type == "faith":
        PYTHON_SCRIPT = "~/code/DecompV/decompv/x/ds/compute_attention.py"

        ##
        #: Adjust DECOMPV_DATASET_END based on ablation_p
        # ablation_p = script_config.get("ablation_p", False)
        ablation_p = True

        if script_config.get("DECOMPV_DATASET_END", None) is None:
            if run_check_completeness_mode_p:
                script_config["DECOMPV_DATASET_END"] = "10"

            elif run_compute_completeness_mode_p:
                script_config["DECOMPV_DATASET_END"] = "100"

            elif ablation_p:
                script_config["DECOMPV_DATASET_END"] = "1000"

            else:
                script_config["DECOMPV_DATASET_END"] = "5000"
        ##
    elif script_type == "qual":
        PYTHON_SCRIPT = "~/code/DecompV/decompv/x/ds/compute_qual.py"
        #: Dataset start and end are hardcoded in `compute_qual_ImageNetS`.

    elif script_type == "seg":
        PYTHON_SCRIPT = "~/code/DecompV/decompv/x/ds/compute_seg.py"

    else:
        raise ValueError(f"Unknown script type: {script_type}")

    res = generate_parameter_combinations(
        models=models,
        script_config=script_config,
        script_type=script_type,
        metrics_target=metrics_target,
        datasets=datasets,
    )
    parameter_combinations = res.parameter_combinations
    datasets = res.datasets
    #: In generate_parameter_combinations, we populate an empty `datasets` based on script type. In the future, we might want to refactor the code such that multiple script types can run in a single invocation of this script.
    del res

    total_iterations = len(parameter_combinations)

    ##
    preprocess_steps = model_preprocessors.get(script_type, [])
    # ic(preprocess_steps)

    postprocess_steps = postprocess_functions.get(script_type, [])
    ##

    skipped_iterations = 0
    failed_iterations = 0
    i = 0  #: Iteration counter

    with tqdm(total=total_iterations, name=f"Run ", desc="Total Progress") as pbar:
        # Preprocessing steps for each model and dataset
        if script_type == "faith":
            for model_name in models:
                #: Clone the environment for preprocessing
                env_clone_pre = os.environ.copy()

                # Set DECOMPV_MODEL_NAME in the cloned environment
                set_env(
                    key="DECOMPV_MODEL_NAME",
                    value=model_name,
                    env=env_clone_pre,
                    none_to=None,
                )

                for dataset in datasets:
                    log_info(
                        f"\npreprocessing: model_name: {model_name}, dataset: {dataset}"
                    )

                    # Set DECOMPV_DATASET_NAME in the cloned environment
                    set_env(
                        key="DECOMPV_DATASET_NAME",
                        value=dataset,
                        env=env_clone_pre,
                        none_to=None,
                    )

                    for step in preprocess_steps:
                        try:
                            step(model_name=model_name, env=env_clone_pre)

                        except BaseException as e:
                            # except Exception as e:
                            log_info(
                                f"Preprocessing failed for model {model_name}: {e}",
                            )
                            sys.exit(1)

                        else:
                            log_info(f"Preprocessing completed for model {model_name}.")

        for combo in parameter_combinations:
            i += 1  # Increment iteration counter

            # Skip iterations based on server index
            if (i - 1) % SERVER_COUNT != SERVER_INDEX:
                log_info(f"Skipped iteration {i} (assigned to another server)")
                skipped_iterations += 1
                pbar.update(1)
                continue

            params = combo

            model_name = params["DECOMPV_MODEL_NAME"]

            # Get gbrand_info using gbrands_from
            gbrand_info = gbrands_from(
                model_name=model_name,
                gbrand_preset=params.get("gbrand_preset", ""),
                gradient_mode_brand=params.get("DECOMPV_GRADIENT_BRAND", ""),
                patchifier_gbrand=params.get("DECOMPV_PATCHIFIER_GBRAND", ""),
                linear_ds_gbrand=params.get("DECOMPV_LINEAR_DS_GBRAND", ""),
                qkv_ds_gbrand=params.get("DECOMPV_QKV_DS_GBRAND", ""),
                mlp_ds_gbrand=params.get("DECOMPV_MLP_DS_GBRAND", ""),
                mlp_mul_gbrand=params.get("DECOMPV_MLP_MUL_GBRAND", ""),
                softmax_mode=params.get("DECOMPV_SOFTMAX_MODE", ""),
                normalize_to_unit_vector_gbrand=params.get("DECOMPV_N2U_GBRAND", ""),
            )
            compact_gbrand = gbrand_info.compact_gbrand

            run_force_submode = decompv.early_boot.run_force_submode
            if (
                run_force_submode is None
                and gbrand_info.ig_steps
                and model_name
                not in [
                    "vit_large_patch16_224.augreg_in21k_ft_in1k",
                ]
            ):
                run_force_submode = "m8"
                #: On most models, we only need the IG itself, not its compositions.

            pruned_mode_p = gbrand_info.pruned_mode_p
            ic(run_force_submode, pruned_mode_p)

            # Clone the current environment
            env_clone = os.environ.copy()

            # Set environment variables in the cloned environment
            set_environment_variables(
                params=params,
                gbrand_info=gbrand_info,
                script_config=script_config,
                env=env_clone,
            )

            metrics_target_curr = env_clone.get("DECOMPV_METRICS_TARGET")

            # Prepare log directory
            timestamp = int(time.time())

            if script_type == "faith":
                ##
                # SUBMODE1 = config_get(script_config, "SUBMODE1")
                if run_force_submode:
                    SUBMODE1 = run_force_submode

                elif pruned_mode_p:
                    SUBMODE1 = "m7"

                else:
                    SUBMODE1 = "m6"

                ##

                log_file_curr = os.path.join(
                    getattr(
                        default_config, "LOG_BASE_DIR", os.path.expanduser("~/logs/")
                    ),
                    script_type,
                    SUBMODE1,
                    params["DECOMPV_MODEL_NAME"],
                    f"{i}_{metrics_target_curr}_{compact_gbrand}_{timestamp}",
                )

            elif script_type == "qual":
                ##
                # qual_submode_current = config_get(script_config, "qual_submode_current")
                if qual_submode:
                    qual_submode_current = qual_submode

                else:
                    if pruned_mode_p:
                        qual_submode_current = "q_pruned_1"

                    else:
                        qual_submode_current = "q_full_1"

                ##

                log_file_curr = os.path.join(
                    getattr(
                        default_config, "LOG_BASE_DIR", os.path.expanduser("~/logs/")
                    ),
                    script_type,
                    qual_submode_current,
                    params["DECOMPV_QUAL_DATASET_NAME"],
                    params["DECOMPV_MODEL_NAME"],
                    f"{i}_{compact_gbrand}_{timestamp}",
                )

            elif script_type == "seg":
                ##
                # seg_submode = config_get(script_config, "seg_submode")
                if run_force_submode:
                    seg_submode = run_force_submode

                elif pruned_mode_p:
                    seg_submode = "m7"

                else:
                    seg_submode = "m6"

                ##

                log_file_curr = os.path.join(
                    getattr(
                        default_config, "LOG_BASE_DIR", os.path.expanduser("~/logs/")
                    ),
                    script_type,
                    seg_submode,
                    params["DECOMPV_MODEL_NAME"],
                    f"{i}_{compact_gbrand}_{timestamp}",
                )

            else:
                raise ValueError(f"Unknown script type: {script_type}")

            os.makedirs(os.path.dirname(log_file_curr), exist_ok=True)

            log_info(
                f"\n\ni: {i}\nmetrics_target_curr: {metrics_target_curr}\ngbrand: {compact_gbrand}",
                log_file=log_file_curr,
            )

            exp_hash = generate_experiment_hash(
                gbrand_info=gbrand_info,
                script_type=script_type,
                params=params,
                script_config=script_config,
                pruned_mode_p=pruned_mode_p,
                run_force_submode=run_force_submode,
                seg_dataset_end=seg_dataset_end_global,
            )
            ic(exp_hash)

            if skip_done_file:
                alread_done_p = False

                seg_dataset_end_at_least_as_good = [seg_dataset_end_global]
                if seg_dataset_end_global is not None:
                    seg_dataset_end_at_least_as_good += [None]

                run_force_submode_at_least_as_good = [run_force_submode]
                if run_force_submode == "m8":
                    run_force_submode_at_least_as_good += [
                        None,
                    ]

                for run_force_submode_, seg_dataset_end_ in itertools.product(run_force_submode_at_least_as_good, seg_dataset_end_at_least_as_good):
                    if alread_done_p:
                        break

                    exp_hash_ = generate_experiment_hash(
                        gbrand_info=gbrand_info,
                        script_type=script_type,
                        params=params,
                        script_config=script_config,
                        pruned_mode_p=pruned_mode_p,
                        run_force_submode=run_force_submode_,
                        seg_dataset_end=seg_dataset_end_,
                    )

                    alread_done_p = experiment_done_p(
                        exp_hash=exp_hash_,
                        skip_done_file=skip_done_file,
                    )

                    if not alread_done_p and pruned_mode_p:
                        #: If a full (non-pruned) run of the current experiment has been done already, don't redo the experiment.
                        exp_hash_ = generate_experiment_hash(
                            gbrand_info=gbrand_info,
                            script_type=script_type,
                            params=params,
                            script_config=script_config,
                            pruned_mode_p=False,
                            run_force_submode=run_force_submode_,
                            seg_dataset_end=seg_dataset_end_,
                        )

                        alread_done_p = experiment_done_p(
                            exp_hash=exp_hash_,
                            skip_done_file=skip_done_file,
                        )

                skip_main_p = run_skip_main_p

                if alread_done_p:
                    if run_deus_p:
                        log_info(
                            f"DEUS: Experiment {exp_hash} already done, but RERUNNING it!",
                            log_file=log_file_curr,
                        )

                    elif run_postprocess_deus_p:
                        log_info(
                            f"Experiment {exp_hash} already done. Skipping main but continuing to postprocess.",
                            log_file=log_file_curr,
                        )

                        skip_main_p = True

                    else:
                        log_info(
                            f"Experiment {exp_hash} already done. Skipping.",
                            log_file=log_file_curr,
                        )

                        skip_main_p = True  #: @redundant
                        skipped_iterations += 1
                        pbar.update(1)
                        continue

                elif skip_main_p:
                    log_info(
                        f"Experiment {exp_hash} NOT done, but run_skip_main_p. Skipping.",
                        log_file=log_file_curr,
                    )
                    skipped_iterations += 1
                    pbar.update(1)
                    continue

            #: Run main script
            script_args = [
                "python",
                os.path.expanduser(PYTHON_SCRIPT),
            ]

            if script_type == "faith":
                min_free_hdd = 30

                script_args.extend(["v3", "--submode1", SUBMODE1])

            elif script_type == "seg":
                min_free_hdd = 1

                env_clone["DECOMPV_SEG_SUBMODE"] = seg_submode
                script_args.extend(["v1"])

            elif script_type == "qual":
                min_free_hdd = 30

                env_clone["DECOMPV_QUAL_SUBMODE"] = qual_submode_current
                script_args.extend(["--dataset", params["DECOMPV_QUAL_DATASET_NAME"]])

            else:
                raise ValueError(f"Unknown script type: {script_type}")

            try:
                # if False:
                if not skip_main_p:
                    run_subprocess(
                        log_file_path=log_file_curr,
                        script_args=script_args,
                        env=env_clone,
                        min_free_hdd=min_free_hdd,
                        # skip_done_file=DONE_EXPERIMENTS,
                        # exp_hash=exp_hash,
                    )

            except BaseException as e:
                # except subprocess.CalledProcessError:
                error_message = (
                    f"\n\nException:\n{e}\n\nFailed:\n\ti={i}\nmetrics_target_curr: {metrics_target_curr}\n\tmodel={params['DECOMPV_MODEL_NAME']}\n"
                    f"\tcompact_gbrand={compact_gbrand}\n"
                    # Include any other relevant parameters
                )
                log_info(
                    error_message,
                    log_file=log_file_curr,
                )

                if continue_on_fail_p:
                    log_info(
                        "Continuing to the next iteration ...",
                        log_file=log_file_curr,
                    )

                    #: sleep for 3 seconds to allow the user to C-c us if they wish to
                    time.sleep(3)

                    failed_iterations += 1
                    pbar.update(1)
                    continue

                else:
                    sys.exit(1)

            else:
                #: The else clause is run on success
                ##
                for step in postprocess_steps:
                    try:
                        step(gbrand_info=gbrand_info, env=env_clone)

                    except BaseException as e:
                        log_info(
                            f"Post-processing failed for iteration {i}: {e}",
                            log_file=log_file_curr,
                        )
                        sys.exit(1)

                if skip_done_file and exp_hash:
                    mark_experiment_done(
                        exp_hash=exp_hash,
                        hostname=current_hostname,
                        skip_done_file=skip_done_file,
                    )

            pbar.update(1)

        msg = f"{current_hostname}: All done (total iterations: {i}, skipped: {skipped_iterations}, failed: {failed_iterations})"
        log_info(msg)
        common_telegram.send(
            tlg_chat_id_me_notif,
            msg=msg,
        )


if __name__ == "__main__":
    current_hostname = hostname_get()

    try:
        main()

    except BaseException as e:
        msg = f"{current_hostname}: run_v1 encountered an error:\n  {e}"

        print(msg)
        common_telegram.send(
            tlg_chat_id_me_notif,
            msg=msg,
        )

        raise e
