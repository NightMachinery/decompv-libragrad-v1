import os

LOG_BASE_DIR = os.path.expanduser("~/logs/")

DECOMPV_SHARED_ROOT = (
    os.path.expanduser("~/dv") if os.path.exists(os.path.expanduser("~/dv")) else None
)

# faith_submode1_default = "m6"
# seg_submode1_default = "m6"

DECOMPV_SAVE_ATTR_MODE = "none"

DECOMPV_ATTR_BATCH_SIZE = "auto"

DECOMPV_DATASET_START = "0"

# DECOMPV_SEG_DATASET_END

