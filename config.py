import os

GROUPID_TO_ITEMIDS = \
    "scripts/groupid_to_itemids_{}.json"

DEEPFASHION_PATH = "/home/anelise/datasets/deepfashion2"

SIMPLICITY_PATH = "/home/anelise/datasets/patterns/simplicity/"
SIMPLICITY_IMAGES_PATH = os.path.join(SIMPLICITY_PATH, "pattern_images")
SIMPLICITY_DATA_PATH = os.path.join(SIMPLICITY_PATH, "pattern_clean_data")
SIMPLICITY_BBOXES_PATH = os.path.join(SIMPLICITY_PATH,
                                      "line_drawing_detections")
BATCH_SIZE = 200
NUM_WORKERS = 20
USE_GPU = True
NUM_EPOCHS = 100
LOGDIR = "logs"
CKPT_DIR = "ckpt"
PREDS_DIR = "preds"
RESIZE = 256
CROP_SIZE = 224
