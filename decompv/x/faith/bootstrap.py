import os
import pathlib


HOME = pathlib.Path.home()

os.environ["DECOMPV_MODEL_LOAD_P"] = "n"
os.environ["DECOMPV_DATASET_LOAD_P"] = "n"

###
import pynight
from pynight.common_debugging import reload_modules
import re

##
from pynight.common_ipython import embed_unless_jupyter
from pynight.common_json import (
    json_partitioned_load,
    json_load,
    json_save,
    json_save_update,
)

import os
import tempfile
import decompv
from decompv.x.bootstrap import *
from decompv.x.ds.main import *
from pynight.common_icecream import ic

os.chdir(ARTIFACTS_ROOT)
##
import decompv.x.imagenet
from decompv.x.imagenet import *

###
import pandas as pd
import numpy as np
from collections import defaultdict
from pynight.common_dict import defaultdict_defaultdict
import matplotlib.pyplot as plt
from math import pi

from pynight.common_json import (
    json_partitioned_load,
    json_load,
    json_save,
    json_save_update,
)


##
