import json
import os
import pickle

import fire
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config as cfg
from data_loader import get_dataset
from model import get_model


def generate_embedding_vectors(run_id,
                               analysis_tag="",
                               num_workers=20,
                               use_gpu=True,
                               dset_mode=None,
                               dset_split="validation",
                               dataset_name="deepfashion",
                               principal_encoder='1',
                               *useless_args,
                               **useless_kwargs):

    ckpt_path = os.path.join(cfg.CKPT_DIR, "{}.pth".format(run_id))
    print("ckpt path: {}".format(ckpt_path))

    analysis_id = run_id if not analysis_tag else "{}_{}".format(
        run_id, analysis_tag)
    preds_path = os.path.join(
        cfg.PREDS_DIR, "{}_{}_embedding.json".format(analysis_id, dset_split))
    print("preds savepath: {}".format(preds_path))

    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda not available")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    print('DEVICE', device)

    # load the ckpt
    print("Loading model from path: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path)
    dset_mode = ckpt['dset_mode'] if dset_mode is None else dset_mode
    model_type = ckpt.get('model_type', 'siamese')

    # model
    model = get_model(model_type,
                      freeze_encoder=True,
                      train_mode=False,
                      principal_encoder=principal_encoder)
    enc_dim = model.enc_dim
    model = nn.DataParallel(model)
    model.load_state_dict(ckpt['model_state_dict'])

    model.to(device)

    print("USING MODEL TYPE {} ON DSET {}".format(model_type, dataset_name))
    print("Using dset mode: {}".format(dset_mode))

    # data loader
    ds = get_dataset(dataset_name, dset_mode, one_sample_only=True)
    ds = ds[0] if dset_split == "train" else ds[1]
    itemids = ds.get_itemids()
    # ds = Subset(ds, range(200))
    dl = DataLoader(ds,
                    batch_size=cfg.BATCH_SIZE,
                    shuffle=False,
                    num_workers=num_workers)

    encodings_arr = np.zeros((len(ds), enc_dim))

    with torch.no_grad():
        for i, x in tqdm(enumerate(dl), total=len(ds) / cfg.BATCH_SIZE):
            x = x.to(device)
            enc = model(x)
            encodings_arr[i * cfg.BATCH_SIZE:(i + 1) *
                          cfg.BATCH_SIZE, :] = enc.cpu().numpy()

    print(encodings_arr.shape)

    encodings = {}

    for i in range(len(encodings_arr)):
        k = itemids[i]
        encoding_vec = encodings_arr[i, :]
        encodings[k] = encoding_vec.tolist()

    # TODO: laod ckpt
    with open(preds_path, "w") as outfile:
        print("Saving preds to: {}".format(preds_path))
        json.dump(encodings, outfile)


def get_nearest_neighbors(run_id,
                          analysis_tag="",
                          dset_split="validation",
                          subset_n=100,
                          *useless_args,
                          **useless_kwargs):

    assert dset_split in ["validation", "train"]

    analysis_id = run_id if not analysis_tag else "{}_{}".format(
        run_id, analysis_tag)
    preds_path = os.path.join(
        cfg.PREDS_DIR, "{}_{}_embedding.json".format(analysis_id, dset_split))
    print("loading preds from: {}".format(preds_path))

    nn_ds_savepath = os.path.join(
        cfg.PREDS_DIR,
        "{}_{}_nearest_neighbors_structure.pkl".format(analysis_id,
                                                       dset_split))
    print(
        "Saving nearest neighbor data structure to: {}".format(nn_ds_savepath))

    nn_data_savepath = os.path.join(
        cfg.PREDS_DIR,
        "{}_{}_{}_nearest_neighbors.pkl".format(analysis_id, dset_split,
                                                subset_n))
    print("Saving nearest neigbhors data to: {}".format(nn_data_savepath))

    nn_savepath = os.path.join(
        cfg.PREDS_DIR,
        "{}_{}_{}_nearest_neighbors.json".format(analysis_id, dset_split,
                                                 subset_n))
    print("Saving nearest neigbhors to: {}".format(nn_savepath))

    with open(preds_path) as infile:
        preds = json.load(infile)

    # make these into features
    keys = sorted(list(preds.keys()))
    features = np.zeros((len(keys), len(preds[keys[0]])))
    print("features shape", features.shape)

    for i, k in enumerate(keys):
        features[i, :] = preds[k]

    feat_to_test = features

    if subset_n is not None:
        feat_to_test = feat_to_test[:subset_n, :]

    distances, indices = _construct_nearest_neighbors(features, feat_to_test,
                                                      nn_ds_savepath,
                                                      nn_data_savepath)

    nearest_neighbors = {}
    # put these into a convenient format

    print("Collating data...")

    for i in range(subset_n):
        k = keys[i]
        dist = distances[i].tolist()
        ind = [keys[idx] for idx in indices[i]]
        data = [(ind[idx], dist[idx]) for idx in range(len(ind))]
        nearest_neighbors[k] = data

    with open(nn_savepath, "w") as outfile:
        json.dump(nearest_neighbors, outfile)


def _construct_nearest_neighbors(features,
                                 feat_to_test,
                                 nn_ds_savepath,
                                 nn_savepath,
                                 n_neighbors=10):

    if os.path.exists(nn_ds_savepath):
        with open(nn_ds_savepath, 'rb') as infile:
            n = pickle.load(infile)
    else:
        print("Constructing nearest neighbors data structure")
        n = NearestNeighbors(n_neighbors=n_neighbors,
                             algorithm='ball_tree').fit(features)
        print("Saving nearest neighbors data structure")
        with open(nn_ds_savepath, 'wb') as outfile:
            pickle.dump(n, outfile)

    if os.path.exists(nn_savepath):
        with open(nn_savepath, 'rb') as infile:
            distances, indices = pickle.load(infile)
    else:
        print("Calculating nearest neighbors for {} features".format(
            len(feat_to_test)))
        distances, indices = n.kneighbors(feat_to_test)
        print("Saving nearest neighbors")
        with open(nn_savepath, 'wb') as outfile:
            pickle.dump((distances, indices), outfile)

    return distances, indices


def main(operation, *args, **kwargs):
    if operation == 'gen_embedding':
        generate_embedding_vectors(*args, **kwargs)
    elif operation == 'nearest_neighbors':
        get_nearest_neighbors(*args, **kwargs)
    elif operation == 'both':
        print("GEN EMBEDDING")
        generate_embedding_vectors(*args, **kwargs)
        print("GEN NEAREST NEIGHBORS")
        get_nearest_neighbors(*args, **kwargs)
    else:
        raise Exception("Unknown operation {}".format(operation))


if __name__ == "__main__":
    fire.Fire(main)
