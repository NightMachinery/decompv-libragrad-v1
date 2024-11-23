#!/usr/bin/env python3
## * Imports
import os

os.environ["DECOMPV_MODEL_LOAD_P"] = ""

import torch
import argparse
import re
import decompv

# import decompv.x.imagenet
from decompv.x.bootstrap import *
from decompv.x.ds.main import *
from pynight.common_icecream import ic
import multiprocessing
from tqdm.contrib.concurrent import process_map  # or thread_map


## * Main
#: With mp 48:
#: Time: cls_res: 187.38948488235474 seconds
#: With mp default number:
#: Time: cls_res: 155.37651681900024 seconds
#:
#: Without it would be ~15 minutes.
##
# "/opt/decompv/datasets/vit_base_patch16_clip_224.openai_ft_in12k_in1k/attnv1/torch"
##


def main(directory):
    torch.set_num_threads(1)
    with Timed(name="cls_res"):
        cls_res = pt_children_logits_to_cls_metrics(
            directory,
            dataset_indexed=dataset_indexed,
        )


##
if __name__ == "__main__":
    if benchmark_mode_p:
        import os

        os._exit(0)

    #: multiprocessing reimports the module (?), so without this we will have an infinite loop.
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Path to the directory")
    args = parser.parse_args()

    main(args.directory)
