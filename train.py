import os
import time

import fire
import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from data_loader import (TEST_TRANSFORMS, TRAIN_TRANSFORMS,
                         DeepFashionContrastiveLoader)
from model import SiameseEncodingModel


class ContrastiveLoss(nn.Module):
    """
    Based on:
    https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
    Equation from:
    https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e
    Default margin setting from:
    https://www.cs.cornell.edu/~kb/publications/SIG15ProductNet.pdf
    """
    def __init__(self, margin=(0.2)**0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        """
        x1, x2 are the two encodings
        y is 1 if the two samples match, else 0
        """
        # check stuff
        assert x0.size() == x1.shape
        assert x1.size()[0] == y.shape[0]
        assert x1.size()[0] > 0
        assert x0.dim() == 2
        assert x1.dim() == 2
        assert y.dim() == 1

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]

        return loss

        # # this gives a tensor of size b x enc_len
        # dist_squared = torch.pow(x1 - x2, 2).sum(dim=1)
        # # tensor of size b x 1
        # dist = torch.sqrt(dist_squared)
        # l_p = 0.5 * (1 - y) * dist_squared

        # print('dist', dist)
        # margin_dist = torch.clamp(self.margin - dist, min=0.0)
        # margin_dist_squared = margin_dist.pow(2)
        # l_n = 0.5 * y * margin_dist_squared

        # loss = torch.sum(l_p + l_n)

        return loss


def image_rescale_zero_to_1_transform():
    def _inner(sample):
        # oops there's no batch here
        # min = np.amin(sample, axis=(1, 2, 3), keepdims=True)
        # max = np.amax(sample, axis=(1, 2, 3), keepdims=True)
        min = np.amin(sample)
        max = np.amax(sample)

        normalized = (sample - min) / (max - min)

        return normalized.astype('uint8')

    return _inner


def save_ckpt(savepath, model, epoch, it, optimizer, dset_mode):
    torch.save(
        {
            'epoch': epoch,
            'it': it,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dset_mode': dset_mode
        }, savepath)


def main(verbose=1,
         print_freq=100,
         restore=True,
         val_freq=1,
         run_id="model",
         dset_mode="grayscale_mask",
         freeze_encoder_until_it=1000):

    ckpt_savepath = os.path.join(cfg.CKPT_DIR, "{}.pth".format(run_id))
    print("Saving ckpts to {}".format(ckpt_savepath))
    logs_savepath = os.path.join(cfg.LOGDIR, run_id)
    print("Saving logs to {}".format(logs_savepath))

    if cfg.USE_GPU:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda not available")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    print('DEVICE', device)

    # model
    model = SiameseEncodingModel(freeze_encoder=True)
    model = DataParallel(model)

    # must call this before constructing the optimizer:
    # https://pytorch.org/docs/stable/optim.html
    model.to(device)

    # set up training
    # TODO better one?
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    criterion = ContrastiveLoss()

    initial_epoch = 0
    iteration = 0
    unfrozen = False

    if restore:
        if os.path.exists(ckpt_savepath):
            print("LOADING MODEL")
            ckpt = torch.load(ckpt_savepath)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            initial_epoch = ckpt['epoch']
            iteration = ckpt['it']
            dset_mode = ckpt.get('dset_mode', dset_mode)

    print("Using dset_mode {}".format(dset_mode))

    # dataset
    train_ds = DeepFashionContrastiveLoader(split="train",
                                            transform=TRAIN_TRANSFORMS,
                                            mode=dset_mode)
    test_ds = DeepFashionContrastiveLoader(split="validation",
                                           transform=TEST_TRANSFORMS,
                                           mode=dset_mode)
    # train_ds = Subset(train_ds, range(500))
    # test_ds = Subset(test_ds, range(100))
    train_dl = DataLoader(train_ds,
                          batch_size=cfg.BATCH_SIZE,
                          shuffle=True,
                          num_workers=cfg.NUM_WORKERS)
    test_dl = DataLoader(test_ds,
                         batch_size=cfg.BATCH_SIZE,
                         shuffle=False,
                         num_workers=cfg.NUM_WORKERS)

    # training loop
    start = time.time()

    try:
        for epoch in range(initial_epoch, cfg.NUM_EPOCHS):
            logger = SummaryWriter(logs_savepath)

            # effectively puts the model in train mode.
            # Opposite of model.eval()
            model.train()

            print("Epoch {}".format(epoch))

            for i, (im1, im2, y) in tqdm(enumerate(train_dl),
                                         total=len(train_ds) / cfg.BATCH_SIZE):
                iteration += 1

                if not unfrozen and iteration > freeze_encoder_until_it:
                    print("Unfreezing encoder")
                    unfrozen = True

                    for param in model.parameters():
                        param.requires_grad = True

                logger.add_scalar('DataTime', time.time() - start, iteration)

                im1 = im1.to(device)
                im2 = im2.to(device)
                y = y.to(device)

                enc1, enc2 = model(im1, im2)
                loss = criterion(enc1, enc2, y)

                # I think this zeros out previous gradients (in case people
                # want to accumulate gradients?)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # logging
                logger.add_scalar('TrainLoss', loss.item(), iteration)
                logger.add_scalar('ItTime', time.time() - start, iteration)
                start = time.time()

                # display metrics

            # do some validation

            print("Validating...")

            if (epoch + 1) % val_freq == 0:
                model.eval()  # puts model in validation mode

                with torch.no_grad():

                    for i, (im1, im2,
                            y) in tqdm(enumerate(test_dl),
                                       total=len(test_ds) / cfg.BATCH_SIZE):
                        im1 = im1.to(device)
                        im2 = im2.to(device)
                        y = y.to(device)

                        enc1, enc2 = model(im1, im2)
                        loss = criterion(enc1, enc2, y)

                        logger.add_scalar('ValLoss', loss, iteration)

            # end of epoch
            lr_scheduler.step()

            save_ckpt(ckpt_savepath, model, epoch, iteration, optimizer,
                      dset_mode)

    except KeyboardInterrupt:
        print('Got keyboard interrupt, saving model...')
        save_ckpt(ckpt_savepath, model, epoch, iteration, optimizer, dset_mode)


if __name__ == "__main__":
    fire.Fire(main)
