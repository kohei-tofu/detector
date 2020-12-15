
from __future__ import division
import os
import json
import sys
import time
import datetime
import copy
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from img_loader.utils import operator
from img_loader.dataset import coco_loader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from easydict import EasyDict as edict
import albumentations as A 
from albumentations import Compose
from albumentations.augmentations.transforms import Resize, Normalize
from PIL import Image

from yolo import *
import yolo.config.cfg_loader as cfgs
from img_loader.utils import *



def detect_bbox_cocoimgs(args):

    #cfg_bbox = cfgs.get("setting_yolo")
    cfg_bbox = cfgs.get(args.setting)
    if int(args.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
        
    loader_aug = cfg_bbox.DATASET.TEST(cfg_bbox)
    

    path_models = cfg_bbox.MODEL.SAVE_PATH + "models/"
    path_result = cfg_bbox.MODEL.SAVE_PATH + "result/"
    model_bbox = cfg_bbox.MODEL.CLASS(cfg_bbox.MODEL)
    model_bbox.load_state_dict(torch.load(path_models + "model_ckpt_best.pth", map_location=device))
    model_bbox.set_device(device)

    #num_batch = 4
    #loss, indicator = cfg_bbox.MODEL.EVALUATE_DETAIL(cfg_bbox, model_bbox, loader_aug, 4)


    num_batch = len(loader_aug)
    loss, indicator = cfg_bbox.MODEL.EVALUATE_DETAIL(cfg_bbox, model_bbox, loader_aug, None)
    loss =  loss / num_batch
    print("loss : {0:.5f}".format(loss))

    boxes_list, precision, recall, AP, ap_class = indicator
    AP_ = np.concatenate([ap_class[:, np.newaxis], np.array(AP)], 1)

    results = list()
    for blist1 in boxes_list:
        for blist2 in blist1:
            results.extend(blist2)

    results.sort(key=lambda res : (res['image_id'], res['score']), reverse=True) 
    print("results : ", len(results))
    result_path = os.path.join(path_result, 'results.json')
    print(result_path)
    with open(result_path, 'w') as f:
        json.dump(results, f)

    np.save(path_result + "precision.npy", precision)
    np.save(path_result + "recall.npy", recall)
    np.save(path_result + "ap_class.npy", AP_)



def detect_imgs(args):

    import random
    from torch.utils.data import DataLoader
    from img_loader.dataset import dir_loader, data_loader

    __dir = "/data/public_data/COCO2017/images/train2017/"
    __dir = "/data/public_data/COCOK2020_1105/images/testK2020_1105/"
    __flip = True
    if int(args.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    bsize = 30
    transformer = Compose([Resize(416, 416, always_apply=True), \
                           Normalize(always_apply=True)])
    cfg_bbox = cfgs.get(args.setting)

    path_models = cfg_bbox.MODEL.SAVE_PATH + "models/"
    model_bbox = cfg_bbox.MODEL.CLASS(cfg_bbox.MODEL)
    model_bbox.load_state_dict(torch.load(path_models + "model_ckpt_best.pth", map_location=device))
    model_bbox.to(device)
    model_bbox.set_device(device)

    num_seed = 1234
    data_ = dir_loader.imgloader(__dir, transformer)
    loader_yolo = DataLoader(data_, batch_size=bsize,\
                            shuffle=False, num_workers=2, collate_fn=data_loader.collate_fn_images,
                            worker_init_fn=lambda x: random.seed(num_seed))

    #loss, ind = cfg_bbox.MODEL.EVALUATE_DETAIL(cfg_bbox, model_bbox, loader_yolo, 20)
    loss, ind = cfg_bbox.MODEL.EVALUATE_DETAIL(cfg_bbox, model_bbox, loader_yolo, None)
    boxes_list = ind[0]




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', '-S', type=str, default='setting_yolo', help='which setting file are you going to use for training.')
    parser.add_argument('--job', '-J', type=str, default='bbox_coco', help='')
    parser.add_argument('--gpu', '-G', type=str, default='-1', help='')
    args = parser.parse_args()

    if args.job == "bbox_coco":
        detect_bbox_cocoimgs(args)

    #detect_imgs(args)

