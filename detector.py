
from __future__ import division
from yolo import *
#from setup.utils import *
#from setup.parse_config import *
from img_loader.utils import *

import os
import json
import sys
import time
import datetime
import argparse
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
import copy


import yolo.config.cfg_loader as cfgs


def detect_bbox_cocoimgs(args):

    #cfg_bbox = cfgs.get("setting_yolo")
    cfg_bbox = cfgs.get(args.setting)

    year = '2017'
    #compose = augmentator.bbox_test()
    
    data_type = 'val'
    loader_aug = cfg_bbox.DATASET.TEST(cfg_bbox)
    
    if int(args.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_models = cfg_bbox.MODEL.SAVE_PATH + "models/"
    #path_result = cfg_bbox.MODEL.SAVE_PATH + "result/"
    model_bbox = cfg_bbox.MODEL.CLASS(cfg_bbox.MODEL)
    model_bbox.load_state_dict(torch.load(path_models + "model_ckpt_best.pth", map_location=device))
    model_bbox.set_device(device)

    #loss, indicator = cfg_bbox.MODEL.EVALUATE_DETAIL(cfg_bbox, model_bbox, loader_aug, 4)
    loss, indicator = cfg_bbox.MODEL.EVALUATE_DETAIL(cfg_bbox, model_bbox, loader_aug, None)
    boxes_list, precision, recall, AP, ap_class = indicator




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', '-S', type=str, default='setting_yolo', help='which setting file are you going to use for training.')
    parser.add_argument('--job', '-J', type=str, default='bbox_coco', help='')
    parser.add_argument('--gpu', '-G', type=str, default='-1', help='')
    args = parser.parse_args()

    if args.job == "bbox_coco":
        detect_bbox_cocoimgs(args)

    #detect_imgs(args)

