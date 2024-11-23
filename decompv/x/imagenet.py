from decompv.x.bootstrap import *
import pynight.common_iterable
import os

if imagenet_p():
    ###
    IMAGENET_1k_PATH = f"{DS_INDEXED_PATH}/ilsvrc2012_wordnet_lemmas.txt"
    if not os.path.exists(IMAGENET_1k_PATH):
        IMAGENET_1k_URL = (
            "https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
        )
        response = requests.get(
            IMAGENET_1k_URL,
        )
        if response.status_code != 200:
            raise Exception(f"Could not download the URL: {IMAGENET_1k_URL}")
        IMAGENET_1k_LABELS = response.text

        with open(IMAGENET_1k_PATH, "w") as f:
            f.write(IMAGENET_1k_LABELS)
    else:
        with open(IMAGENET_1k_PATH, "r") as f:
            IMAGENET_1k_LABELS = f.read()

    IMAGENET_1k_LABELS = IMAGENET_1k_LABELS.strip().split("\n")

    IMAGENET_1k_LABELS = pynight.common_iterable.IndexableList(IMAGENET_1k_LABELS)

    ###
    imagenet_root = f"{os.environ['HOME']}/datasets/imagenet"

    imagenet_val_dir = f"{imagenet_root}/2012"
    imagenet_val_labels_path = (
        f"{imagenet_root}/ILSVRC2012_validation_ground_truth_caffe.txt"
    )
    imagenet_labels_human_path = f"{imagenet_root}/imagenet-simple-labels.json"
    imagenet_synset_to_id_path = f"{imagenet_root}/imagenet2coco.txt"

    assert os.path.exists(
        imagenet_labels_human_path
    ), f"imagenet_labels_human_path does not exist:\n  {imagenet_labels_human_path}"
    imagenet_labels_human = json_load(imagenet_labels_human_path)
    #: The JSON is a simple list of strings.
    imagenet_labels_human = pynight.common_iterable.IndexableList(imagenet_labels_human)

    def imagenet_val_labels_load(imagenet_val_labels_path):
        with open(imagenet_val_labels_path, "r") as f:
            imagenet_val_labels = f.readlines()
            imagenet_val_labels = [0] + [
                int(x.strip().split(" ")[-1]) for x in imagenet_val_labels
            ]
            #: The ImageNet image IDs start from 1.
            #: The label IDs in the ground truth file start from 0.

        return imagenet_val_labels

    assert os.path.exists(
        imagenet_val_labels_path
    ), f"imagenet_val_labels_path does not exist:\n  {imagenet_val_labels_path}"
    imagenet_val_labels = imagenet_val_labels_load(imagenet_val_labels_path)
    # ic(imagenet_labels_human[imagenet_val_labels[2666]])
    #: must return 'cocktail shaker'

    def imagenet_synset_to_id_get():
        if not os.path.exists(imagenet_synset_to_id_path):
            print(
                f"imagenet_synset_to_id_path does not exist:\n  {imagenet_synset_to_id_path}",
                file=sys.stderr,
            )
            return None

        imagenet_synset_to_id = dict()

        with open(imagenet_synset_to_id_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            if "\t" not in line:
                continue

            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            imagenet_id = idx
            imagenet_synset = parts[0]
            imagenet_text = parts[1]

            imagenet_synset_to_id[imagenet_synset] = imagenet_id

        return imagenet_synset_to_id

    assert os.path.exists(
        imagenet_synset_to_id_path
    ), f"imagenet_synset_to_id_path does not exist:\n  {imagenet_synset_to_id_path}"
    imagenet_synset_to_id = imagenet_synset_to_id_get()

    ###
