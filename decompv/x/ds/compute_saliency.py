##
from decompv.x.ds.main import *

## * User Config
captum_attributors = [
    dict(
        name="Saliency",
        obj=captum.attr.Saliency(model.forward_patch_level),
    ),
]
##

attr_captum_compute_global(
    captum_attributors=captum_attributors,
)
