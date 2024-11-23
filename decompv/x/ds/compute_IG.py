##
from decompv.x.ds.main import *
from decompv.x.ds.main import batch_size

## * User Config
captum_attributors = [
    dict(
        name="IG_s50",
        obj=captum.attr.IntegratedGradients(
            model.forward_patch_level,
            multiply_by_inputs=True,
        ),
        kwargs={
            "baselines": None,
            "n_steps": 50,
            "method": "gausslegendre",
            "return_convergence_delta": False,
            "internal_batch_size": batch_size,
        },
        # enabled_p=False,
    ),
]
##

attr_captum_compute_global(
    captum_attributors=captum_attributors,
)
