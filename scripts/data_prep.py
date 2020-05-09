import json
import sys

import numpy as np

try:
    sys.path.append("/home/anelise/lab/clothes_detection/")
    import deepfashion_detectron as dd
except Exception as e:
    raise e


def make_shop_style_mapping(split):
    """
    Prepare necessary data files for DeepFashion loading.

    Filters out items that do not have multiple exemplars.

    Args:
        split (string): "train", "validation"
    Returns:
        itemid_to_data: mapping from itemid to relevant data
        groupid_to_itemids: 1:many mapping from groups to items
    """
    itemid_to_data = dd.get_shop_style_mapping(split, None)

    # construct groupid to itemids
    groupid_to_itemids = {}

    for itemid, data in itemid_to_data.items():
        groupid = data['groupid']
        groupid_to_itemids[groupid] = groupid_to_itemids.get(groupid,
                                                             []) + [itemid]

    # filter out groups with only one exemplar
    groupid_to_itemids = {
        groupid: itemids
        for groupid, itemids in groupid_to_itemids.items() if len(itemids) > 1
    }

    itemid_to_data = {
        itemid: data
        for itemid, data in itemid_to_data.items()
        if data['groupid'] in groupid_to_itemids
    }

    # print some numbers
    print("{} items, {} groups".format(len(itemid_to_data),
                                       len(groupid_to_itemids)))
    lens = [len(v) for k, v in groupid_to_itemids.items()]
    print("Group length: min {}, max {}, avg {}".format(
        np.min(lens), np.max(lens), np.mean(lens)))

    # Save the stuff
    print("saving")
    with open("itemid_to_data_{}.json".format(split), "w") as outfile:
        json.dump(itemid_to_data, outfile)

    with open("groupid_to_itemids_{}.json".format(split), "w") as outfile:
        json.dump(groupid_to_itemids, outfile)


if __name__ == "__main__":
    make_shop_style_mapping('train')
    make_shop_style_mapping('validation')
