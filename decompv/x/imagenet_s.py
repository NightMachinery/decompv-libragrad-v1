from decompv.x.bootstrap import *
import decompv.x.imagenet


# if SEG_DATASET_NAME == "ImageNetS":
if True:
    from decompv.x.imagenet import *

    ###
    imagenet_s_root = f"{decompv.x.imagenet.imagenet_root}/ImageNet-S"
    imagenet_s_dir = f"{imagenet_s_root}/ds/ImageNetS919/validation-segmentation"
    imagenet_s_id_to_synset_path = (
        f"{imagenet_s_root}/data/categories/ImageNetS_categories_im919.txt"
    )

    ###
    def imagenet_s_id_to_synset_get():
        if not os.path.exists(imagenet_s_id_to_synset_path):
            msg = f"imagenet_s_id_to_synset_path does not exist:\n  {imagenet_s_id_to_synset_path}"
            print(msg, file=sys.stderr)
            return None

        with open(imagenet_s_id_to_synset_path, "r") as f:
            imagenet_s_id_to_synset = f.readlines()
            imagenet_s_id_to_synset = (
                ["other"] + [x.strip() for x in imagenet_s_id_to_synset] + ["ignored"]
            )

        return imagenet_s_id_to_synset

    def imagenet_s_id_to_imagenet_id(some_id):
        return decompv.x.imagenet.imagenet_synset_to_id[
            imagenet_s_id_to_synset[some_id]
        ]

    ###
    def imagenet_s_all_get():
        imagenet_s_all = list_children(
            imagenet_s_dir,
            # include_patterns=['.*'],
            include_patterns=[".*\.png$"],
            recursive=True,
            # verbose_p=True,
        )
        imagenet_s_all.sort(key=lambda x: (os.path.splitext(os.path.basename(x))[0]))

        rng = np.random.default_rng(seed=42)
        #: We have fixed the order of the shuffle to make sure our experiments are reproducible.

        imagenet_s_all = rng.permutation(imagenet_s_all)
        imagenet_s_all = IndexableList(list(enumerate(imagenet_s_all)))
        # imagenet_s_all = np.array(list(enumerate(imagenet_s_all)))

        return imagenet_s_all

    ###
    imagenet_synset_to_id = imagenet_synset_to_id_get()
    imagenet_s_id_to_synset = imagenet_s_id_to_synset_get()
