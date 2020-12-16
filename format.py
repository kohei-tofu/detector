

import json
import numpy as np


def make_coco_categories():

    path_coco2017_val = "/data/public_data/COCO2017/annotations/instances_val2017.json"
    anns_list_coco2017_val = json.load(open(path_coco2017_val, 'r'))
    return anns_list_coco2017_val["categories"]


def make_coco_images(imgid, fn, height, width):
    
    temp = {
        #"license": 4,
        "file_name": fn,
        "height": height,
        "width": width,
        "id": imgid,
        #"date_captured": "2013-11-14 17:02:52",
        #"flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        #"coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg"
    }
    
    return [temp]


def make_coco_annotations(annids, imgid, bbox, person, keypoints, maxvals):
    ret = list()
    k_loop = 0
    for b_p, key in zip(bbox, person):
        
        cat = int(b_p["category_id"])
        if b_p["score"] < 0.55:
            continue
        bbox = b_p["bbox"]
        x1, y1 = bbox[0], bbox[1]
        #y1, x1 = int(bbox[1] - bbox[3] / 2), int(bbox[0] - bbox[2] / 2)
        w, h = bbox[2], bbox[3]
        bbox2 = [x1, y1, w, h]
        annids += 1
        if key == True:
            _keypoint = keypoints[k_loop].astype(np.int32).ravel().tolist()
            k_loop += 1
            d = dict(id=annids, image_id=imgid, bbox=bbox2, \
                     keypoints=_keypoint, category_id=cat, iscrowd=0, keyscore=maxvals[:, :, 0].tolist())
        else:
            d = dict(id=annids, image_id=imgid, bbox=bbox2, category_id=cat, iscrowd=0)
        ret.append(d)

    return ret