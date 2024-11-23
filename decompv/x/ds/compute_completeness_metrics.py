#!/usr/bin/env python3
## * Imports
import os

# os.environ["DECOMPV_MODEL_LOAD_P"] = ""
os.environ["DECOMPV_FORCE_CPU_P"] = "y"

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
def main():
    completeness_error_postprocess_entrypoint(
        model=model,
    )


##
if __name__ == "__main__":
    main()
