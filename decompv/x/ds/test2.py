#!/usr/bin/env python3

import pynight
from pynight.common_debugging import reload_modules
import re
from IPython import embed

##
import decompv
from decompv.x.bootstrap import *
from decompv.x.ds.main import *
from pynight.common_icecream import ic

##
my_tds_indexed = tds_indexed_imagenet
my_tds_torch_cpu = tds_torch_cpu_imagenet
my_tds_patches = tds_patches_imagenet

bias_token_p = True

model_patch_info = patch_info_from_name(model_name, bias_token_p=bias_token_p)
ic(model_patch_info)

##
ic(my_tds_indexed.preview())
ic(my_tds_torch_cpu.preview())
ic(my_tds_patches.preview())
embed()
