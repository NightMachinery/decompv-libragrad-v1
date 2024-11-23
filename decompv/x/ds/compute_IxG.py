##
from decompv.x.ds.main import *

## * User Config
captum_attributors = [
    dict(
        name="IxG",
        obj=captum.attr.InputXGradient(model.forward_patch_level),
    ),
]
##

attr_captum_compute_global(
    captum_attributors=captum_attributors,
)
