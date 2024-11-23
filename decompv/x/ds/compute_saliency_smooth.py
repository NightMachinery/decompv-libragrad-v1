##
from decompv.x.ds.main import *
from decompv.x.ds.main import batch_size

## * User Config
captum_attributors = [
    dict(
        name="SaliencySmooth50",
        obj=captum.attr.NoiseTunnel(captum.attr.Saliency(model.forward_patch_level)),
        kwargs=dict(
            nt_type="smoothgrad",
            nt_samples=50,
            nt_samples_batch_size=batch_size,
        ),
    ),
]
##

attr_captum_compute_global(
    captum_attributors=captum_attributors,
)
