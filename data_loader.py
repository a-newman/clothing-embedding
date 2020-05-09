import json
import os
import random

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch.utils.data as data
from torchvision import transforms as T

import config as cfg

GROUPID_TO_ITEMIDS = \
    "/home/anelise/lab/clothes_detection/embedding/scripts/groupid_to_itemids_{}.json"

DEEPFASHION_PATH = "/home/anelise/datasets/deepfashion2"

SIMPLICITY_PATH = "/home/anelise/datasets/patterns/simplicity/"
SIMPLICITY_IMAGES_PATH = os.path.join(SIMPLICITY_PATH, "pattern_images")
SIMPLICITY_DATA_PATH = os.path.join(SIMPLICITY_PATH, "pattern_clean_data")
SIMPLICITY_BBOXES_PATH = os.path.join(SIMPLICITY_PATH,
                                      "line_drawing_detections")

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


class DeepFashionContrastiveLoader(data.Dataset):
    """
    Loader for the deepfashion dataset.

    # how to do this?
    # steps:
     1. sample an appropriate pair,
        - should I / can I enumerate all pairs ahead of time?
        -
     2. load the data
     3. transform it as required
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
                - 'color': whole image, no mask
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
            groupid_to_itemids_fp = GROUPID_TO_ITEMIDS.format(split)

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
        annfp = os.path.join(DEEPFASHION_PATH, split, "annos",
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
        imfp = os.path.join(DEEPFASHION_PATH, split, "image",
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

    def _transform(self, image, annos):
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

        if self.transform is not None:
            im = self.transform(im)

        return im

    def __getitem__(self, index):
        x1id = self.itemids[index]

        return self.getitem_by_id(x1id)

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
    def __init__(self,
                 image_path=SIMPLICITY_IMAGES_PATH,
                 data_path=SIMPLICITY_DATA_PATH,
                 bbox_path=SIMPLICITY_BBOXES_PATH,
                 positive_proportion=0.5,
                 image_transforms=None,
                 line_drawing_tranforms=None):
        self.image_path = image_path
        self.data_path = data_path
        self.bbox_path = bbox_path
        self.positive_proportion = positive_proportion
        self.image_transforms = image_transforms
        self.line_drawing_transforms = line_drawing_tranforms

        # figure out the data you will pull from
        self.data_files = os.listdir(self.data_path)

    def __len__(self):
        return len(self.data_files)

    def _load_image(self, imfile):
        imfp = os.path.join(self.image_path, imfile)
        img = cv2.imread(imfp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_data_file(self, index):
        data_file = self.data_files[index]

        with open(os.path.join(self.data_path, data_file)) as infile:
            data = json.load(infile)

        return data

    def _load_line_drawing_bboxes(self, index):
        data_file = self.data_files[index]
        fpath = os.path.join(self.bbox_path, data_file)

        with open(fpath) as infile:
            data = json.load(infile)

        return data

    def _load_line_drawing(self, index):
        data = self._load_data_file(index)
        line_drawing_path = os.path.join(self.image_path,
                                         data["line_image"][0])
        img = cv2.imread(line_drawing_path)
        # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # binary = None

        return img

    def _crop(self, image, bbox):
        h, w, _ = image.shape
        x1, y1, x2, y2 = bbox
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)

        return image[y1:y2, x1:x2, :]

    def __getitem__(self, index):
        data = self._load_data_file(index)

        # choose a random image
        imfile = random.choice(data['other_images'])[0]
        img = self._load_image(imfile)

        # choose a line drawing
        choose_same_class = random.random() < self.positive_proportion

        bbox_index = index

        if not choose_same_class:
            while True:
                bbox_index = random.randint(0, self.__len__() - 1)

                if bbox_index != index:
                    break

        bboxes = self._load_line_drawing_bboxes(bbox_index)
        bbox = random.choice(bboxes)
        line_drawing = self._load_line_drawing(bbox_index)
        cropped_line_drawing = self._crop(line_drawing, bbox)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.line_drawing_transforms is not None:
            cropped_line_drawing = self.line_drawing_transforms(
                cropped_line_drawing)

        y = int(choose_same_class)

        return img, cropped_line_drawing, y
