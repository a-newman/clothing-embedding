import json
import os

import matplotlib.pyplot as plt
from PIL import Image

import config as cfg
from data_loader import get_dataset


def plot_nearest_neighbors(run_id,
                           dataset_name,
                           analysis_tag="",
                           dset_split="validation",
                           dset_mode='color_mask_crop',
                           subset_n=100):
    analysis_id = run_id if not analysis_tag else "{}_{}".format(
        run_id, analysis_tag)
    nn_savepath = os.path.join(
        cfg.PREDS_DIR,
        "{}_{}_{}_nearest_neighbors.json".format(analysis_id, dset_split,
                                                 subset_n))
    print("Loading nearest neighbor data from: {}".format(nn_savepath))

    with open(nn_savepath) as infile:
        nn_data = json.load(infile)

    ks = list(nn_data.keys())[:20]
    nn_data = {k: nn_data[k] for k in ks}

    ds = get_dataset(dataset_name, dset_mode=dset_mode, one_sample_only=True)
    ds = ds[0] if dset_split == "train" else ds[1]

    def _load_image(imageid):
        return ds.getitem_by_id(imageid)

    im_grid = []
    metadata_grid = []

    for probe_item, neighbor_data in nn_data.items():
        row = [elt[0] for elt in neighbor_data]
        row_metadata = [
            "{}: {}".format(elt[0], elt[1]) for elt in neighbor_data
        ]
        im_grid.append(row)
        metadata_grid.append(row_metadata)

    img_rows_plt(rows=im_grid,
                 metadata=metadata_grid,
                 im_load_func=_load_image)


def img_rows_plt(rows, im_load_func, metadata=None):
    h = len(rows)
    w = max([len(row) for row in rows])
    print(h, w)

    for i in range(h):
        fig, ax = plt.subplots(1, len(rows[i]), figsize=(w * 5, h * 5))

        for j in range(len(rows[i])):
            im = im_load_func(rows[i][j])
            ax[j].imshow(im)

            if metadata:
                ax[j].set_title(metadata[i][j])
            ax[j].tick_params(labelbottom=False,
                              labeltop=False,
                              labelleft=False,
                              labelright=False,
                              bottom=False,
                              top=False,
                              left=False,
                              right=False)
        plt.show()

    # plt.show()
