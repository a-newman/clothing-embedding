import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config as cfg
from data_loader import get_dataset


def plot_nearest_neighbors(run_id,
                           dataset_name,
                           analysis_tag="",
                           dset_split="validation",
                           dset_mode='color_mask_crop',
                           subset_n=100,
                           n_to_show=20,
                           plot_metadata=True):
    analysis_id = run_id if not analysis_tag else "{}_{}".format(
        run_id, analysis_tag)
    nn_savepath = os.path.join(
        cfg.PREDS_DIR,
        "{}_{}_{}_nearest_neighbors.json".format(analysis_id, dset_split,
                                                 subset_n))
    print("Loading nearest neighbor data from: {}".format(nn_savepath))

    with open(nn_savepath) as infile:
        nn_data = json.load(infile)

    ks = list(nn_data.keys())[:n_to_show]
    nn_data = {k: nn_data[k] for k in ks}

    ds = get_dataset(dataset_name, dset_mode=dset_mode, one_sample_only=True)
    ds = ds[0] if dset_split == "train" else ds[1]

    def _load_image(imageid):
        return ds.visualize_item(imageid)

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
                 metadata=metadata_grid if plot_metadata else None,
                 im_load_func=_load_image)


def plot_principal_component_examples(run_id,
                                      dataset_name,
                                      analysis_tag="",
                                      dset_split="validation",
                                      dset_mode='color_mask_crop',
                                      n_samples_per_component=15,
                                      truncate_n=20,
                                      show_metadata=False):
    analysis_id = run_id if not analysis_tag else "{}_{}".format(
        run_id, analysis_tag)
    pca_examples_savepath = os.path.join(
        cfg.PREDS_DIR,
        "{}_{}_pca_component_examples.json".format(analysis_id, dset_split))
    print("Loading PCA axis examples from: {}".format(pca_examples_savepath))

    with open(pca_examples_savepath) as infile:
        component_examples = json.load(infile)

    ds = get_dataset(dataset_name, dset_mode=dset_mode, one_sample_only=True)
    ds = ds[0] if dset_split == "train" else ds[1]

    def _load_image(imageid):
        return ds.visualize_item(imageid)

    im_grid = []
    metadata_grid = []
    metadata_rows = []

    for component_data in component_examples[:truncate_n]:
        title = "Component {} (Var explained: {}. Sing. value: {})".format(
            component_data['component_i'],
            round(component_data['explained_variance_ratio'], 2),
            round(component_data['singular_value'], 2))
        metadata_rows.append(title)

        examples = component_data['samples_sorted']
        indices_to_sample = np.round(
            np.linspace(0,
                        len(examples) - 1, n_samples_per_component))

        examples = [examples[int(i)] for i in indices_to_sample]

        row_images = []
        row_metadata = []

        for elt in examples:
            coeff, imid = elt
            row_images.append(imid)
            elt_title = "{} ({})".format(
                imid, coeff) if show_metadata else str(round(coeff, 2))
            row_metadata.append(elt_title)

        im_grid.append(row_images)
        metadata_grid.append(row_metadata)

    img_rows_plt(rows=im_grid,
                 metadata=metadata_grid if show_metadata else None,
                 im_load_func=_load_image,
                 row_metadata=metadata_rows if show_metadata else None)


def img_rows_plt(rows, im_load_func, metadata=None, row_metadata=None):
    h = len(rows)
    w = max([len(row) for row in rows])
    print(h, w)

    for i in range(h):
        fig, ax = plt.subplots(1, len(rows[i]), figsize=(w * 5, 5))

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

        if row_metadata:
            fig.suptitle(row_metadata[i])
        plt.show()

    # plt.show()
