import json
import os
import random

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch.utils.data as data
from torchvision import transforms as T

import config as cfg

TRAIN_TRANSFORMS = T.Compose([
    # image_rescale_zero_to_1_transform(),
    T.ToPILImage(),
    T.Resize(cfg.RESIZE),
    T.RandomCrop(cfg.CROP_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
TEST_TRANSFORMS = T.Compose([
    # image_rescale_zero_to_1_transform(),
    T.ToPILImage(),
    T.Resize(cfg.RESIZE),
    T.CenterCrop(cfg.CROP_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataset(dataset_name, dset_mode, *args, **kwargs):
    if dataset_name == "deepfashion":
        train_ds = DeepFashionContrastiveLoader(split="train",
                                                transform=TRAIN_TRANSFORMS,
                                                mode=dset_mode,
                                                *args,
                                                **kwargs)
        test_ds = DeepFashionContrastiveLoader(split="validation",
                                               transform=TEST_TRANSFORMS,
                                               mode=dset_mode,
                                               *args,
                                               **kwargs)
    elif dataset_name == "simplicity":
        train_ds = SimplicityContrastiveLoader(
            image_transforms=TRAIN_TRANSFORMS,
            line_drawing_tranforms=TRAIN_TRANSFORMS,
            mode=dset_mode,
            *args,
            **kwargs)
        test_ds = SimplicityContrastiveLoader(
            image_transforms=TEST_TRANSFORMS,
            line_drawing_tranforms=TEST_TRANSFORMS,
            mode=dset_mode,
            *args,
            **kwargs)
    elif os.path.exists(dataset_name):
        train_ds = ImageFolderLoader(root=dataset_name,
                                     transforms=TRAIN_TRANSFORMS)
        test_ds = ImageFolderLoader(root=dataset_name,
                                    transforms=TEST_TRANSFORMS)
    else:
        raise RuntimeError("Unrecognized dataset name {}".format(dataset_name))

    return train_ds, test_ds


class InvalidDataException(Exception):
    pass


class DeepFashionContrastiveLoader(data.Dataset):
    """
    Loader for the deepfashion dataset.
    """
    def __init__(self,
                 split,
                 transform=None,
                 groupid_to_itemids_fp=None,
                 mode='color_mask_crop',
                 positive_proportion=0.5,
                 debug=False,
                 crop_margin=10,
                 one_sample_only=False):
        """
        Args:
            mode (string): mode in which to open the images. options:
                - 'grayscale_mask': 2 channels of grayscale info,
                                    one channel of 255/0 mask
                - 'color_mask_crop': 3 channels of color, masked out w/ 0
                   values and then cropped to within a margin of the item
        """
        super(DeepFashionContrastiveLoader, self).__init__()
        self.mode = mode
        self.positive_proportion = positive_proportion
        self.split = split
        self.debug = debug
        self.crop_margin = crop_margin
        self.transform = transform
        self.one_sample_only = one_sample_only

        self.itemids = []
        self.itemid_to_groupid = {}

        if groupid_to_itemids_fp is None:
            groupid_to_itemids_fp = cfg.GROUPID_TO_ITEMIDS.format(split)

        with open(groupid_to_itemids_fp) as infile:
            self.groupid_to_itemids = json.load(infile)

        for groupid, itemids in self.groupid_to_itemids.items():
            self.itemids.extend(itemids)

            for itemid in itemids:
                self.itemid_to_groupid[itemid] = groupid

        self.itemids.sort()
        self.itemids_to_idx = None

    def __len__(self):
        return len(self.itemids)

    def get_itemids(self):
        return self.itemids

    def _get_groupid(self, itemid):
        return self.itemid_to_groupid[itemid]

    def _get_group_members(self, itemid):
        return self.groupid_to_itemids[self._get_groupid(itemid)]

    def _load_annotations(self, itemid, split):
        imid, itemid = itemid.split("_")
        annfp = os.path.join(cfg.DEEPFASHION_PATH, split, "annos",
                             "{}.json".format(imid))
        with open(annfp) as infile:
            data = json.load(infile)
            try:
                ann = data[itemid]
            except KeyError as ke:
                print(data)
                raise ke

        return ann

    def _load_image(self, itemid, split):
        imid, itemid = itemid.split("_")
        imfp = os.path.join(cfg.DEEPFASHION_PATH, split, "image",
                            "{}.jpg".format(imid))
        img = cv2.imread(imfp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _mask(self, image, ann, just_mask=False):
        h, w, _ = image.shape
        polygons = ann['segmentation']
        mask = mask_util.frPyObjects(polygons, h, w)
        mask = mask_util.merge(mask)
        mask = mask_util.decode(mask)

        if just_mask:
            return mask

        masked_img = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * image

        return masked_img

    def _crop(self, image, ann):
        margin = self.crop_margin
        h, w, _ = image.shape
        x1, y1, x2, y2 = ann['bounding_box']
        x1 = max(x1 - margin, 0)
        y1 = max(y1 - margin, 0)
        x2 = min(x2 + margin, w)
        y2 = min(y2 + margin, h)

        return image[y1:y2, x1:x2, :]

    def _transform(self, image, annos, torch_transform=True):
        if self.mode == "color_mask_crop":
            masked_image = self._mask(image, annos)
            cropped_image = self._crop(masked_image, annos)

            im = cropped_image

        elif self.mode == "color_crop":
            im = self._crop(image, annos)
        elif self.mode == "grayscale_mask":
            # to grayscale
            image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # copy into a second axis
            image_gs = np.repeat(image_gs[:, :, np.newaxis], 2, axis=2)
            # get the mask
            mask = 255 * np.expand_dims(
                self._mask(image, annos, just_mask=True), axis=2)
            # concat
            inp = np.concatenate((image_gs, mask), axis=2)

            # crop it
            cropped_inp = self._crop(inp, annos)

            im = cropped_inp

        else:
            raise NotImplementedError()

        if self.transform is not None and torch_transform:
            im = self.transform(im)

        return im

    def __getitem__(self, index):
        x1id = self.itemids[index]

        return self.getitem_by_id(x1id)

    def visualize_item(self, x1id):
        x1_annos = self._load_annotations(x1id, self.split)
        x1_image = self._load_image(x1id, self.split)
        x1 = self._transform(x1_image, x1_annos, torch_transform=False)

        return x1

    def getitem_by_id(self, x1id):
        x1_annos = self._load_annotations(x1id, self.split)
        x1_image = self._load_image(x1id, self.split)
        x1 = self._transform(x1_image, x1_annos)
        metadata = {
            'x1id': x1id,
            'x1groupid': self._get_groupid(x1id),
        }

        if self.one_sample_only:
            if self.debug:
                return x1, metadata
            else:
                return x1

        choose_same_class = random.random() < self.positive_proportion

        if choose_same_class:
            options = [
                elt for elt in self._get_group_members(x1id) if elt is not x1id
            ]
            x2id = random.choice(options)
        else:
            while True:
                x2id = random.choice(self.itemids)

                if self._get_groupid(x2id) != self._get_groupid(x1id):
                    break

        x2_annos = self._load_annotations(x2id, self.split)
        x2_image = self._load_image(x2id, self.split)
        x2 = self._transform(x2_image, x2_annos)
        y = int(choose_same_class)

        if self.debug:
            metadata['x2id'] = x2id
            metadata['x2groupid'] = self._get_groupid(x2id),
            metadata['same_class'] = choose_same_class

            return x1, x2, y, metadata

        return x1, x2, y


class SimplicityContrastiveLoader(data.Dataset):
    """Data loader for the Simplicity data set."""
    def __init__(self,
                 image_path=cfg.SIMPLICITY_IMAGES_PATH,
                 data_path=cfg.SIMPLICITY_DATA_PATH,
                 bbox_path=cfg.SIMPLICITY_BBOXES_PATH,
                 positive_proportion=0.5,
                 image_transforms=None,
                 line_drawing_tranforms=None,
                 one_sample_only=False,
                 mode="natural_image"):
        self.image_path = image_path
        self.data_path = data_path
        self.bbox_path = bbox_path
        self.positive_proportion = positive_proportion
        self.image_transforms = image_transforms
        self.line_drawing_transforms = line_drawing_tranforms
        self.one_sample_only = one_sample_only
        self.mode = mode
        print("Mode", self.mode)

        assert self.mode == "natural_image" or self.mode == "line_drawing"

        # figure out the data you will pull from
        self.data_files = os.listdir(self.data_path)

    def __len__(self):
        return len(self.data_files)

    def _load_image(self, imfile):
        imfp = os.path.join(self.image_path, imfile)
        img = cv2.imread(imfp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_data_file(self, index=None, data_file=None):
        assert index is not None or data_file is not None

        if data_file is None:
            data_file = self.data_files[index]

        with open(os.path.join(self.data_path, data_file)) as infile:
            data = json.load(infile)

        return data

    def _load_line_drawing_bboxes(self, index=None, data_file=None):
        assert index is not None or data_file is not None

        if data_file is None:
            data_file = self.data_files[index]
        fpath = os.path.join(self.bbox_path, data_file)

        with open(fpath) as infile:
            data = json.load(infile)

        return data

    def _load_line_drawing(self, index=None, data_file=None):
        data = self._load_data_file(index, data_file)
        line_drawing_path = os.path.join(self.image_path,
                                         data["line_image"][0])
        img = cv2.imread(line_drawing_path)

        return img

    def _crop(self, image, bbox):
        h, w, _ = image.shape
        x1, y1, x2, y2 = bbox
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)

        return image[y1:y2, x1:x2, :]

    def get_itemids(self):
        return self.data_files

    def visualize_item(self, data_file):
        if self.mode == "natural_image":
            with open(os.path.join(self.data_path, data_file)) as infile:
                data = json.load(infile)

            imfile = data['other_images'][0][0]
            img = self._load_image(imfile)

            return img
        else:
            return self._get_principal_line_drawing(data_file=data_file)

    def _get_principal_line_drawing(self, index=None, data_file=None):
        assert index is not None or data_file is not None
        bboxes = self._load_line_drawing_bboxes(index, data_file)
        bbox = bboxes[0]
        line_drawing = self._load_line_drawing(index, data_file)
        cropped_line_drawing = self._crop(line_drawing, bbox)

        return cropped_line_drawing

    def __getitem__(self, index):
        data = self._load_data_file(index)

        # choose a random image
        imfile = random.choice(data['other_images'])[0]
        img = self._load_image(imfile)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.one_sample_only:
            if self.mode == "natural_image":
                return img
            else:
                return self.line_drawing_transforms(
                    self._get_principal_line_drawing(index=index))

        # choose a line drawing
        choose_same_class = random.random() < self.positive_proportion

        bbox_index = index

        if not choose_same_class:
            while True:
                bbox_index = random.randint(0, self.__len__() - 1)

                if bbox_index != index:
                    break

        bboxes = self._load_line_drawing_bboxes(bbox_index)
        try:
            bbox = random.choice(bboxes)
        except IndexError:
            print("WARNING: no bounding boxes found for item {} ({})".format(
                bbox_index, self.data_files[bbox_index]))
            h, w = img.shape
            bbox = [0, 0, h, w]
        line_drawing = self._load_line_drawing(bbox_index)
        cropped_line_drawing = self._crop(line_drawing, bbox)

        if self.line_drawing_transforms is not None:
            cropped_line_drawing = self.line_drawing_transforms(
                cropped_line_drawing)

        y = int(choose_same_class)

        return img, cropped_line_drawing, y


class ImageFolderLoader():
    def __init__(self, root, transforms=None, verbose=0, raise_on_error=True):
        self.root = root
        self.transforms = transforms
        self.verbose = verbose
        self.raise_on_error = True

        self.ext = ['.jpg', '.png', '.jpeg']

        self.images = [
            elt for elt in os.listdir(self.root)
            if os.path.splitext(elt)[1].lower() in self.ext
        ]

        if self.verbose:
            print("Found {} images".format(len(self.images)))

    def __getitem__(self, index):
        fname = self.images[index]
        path = os.path.join(self.root, fname)

        if not os.path.exists(path):
            raise RuntimeError("No file {}".format(path))
        img = cv2.imread(path)

        if img is None:
            if self.raise_on_error:
                raise InvalidDataException(
                    "Could not open image {}".format(path))
            img = np.zeros((256, 256, 3)).astype('uint8')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.images)

    def get_fnames(self):
        return self.images
