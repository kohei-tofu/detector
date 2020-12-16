
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

import yolo, keypoint_detector
#from yolo import *
from img_loader.utils import *
from visualize import draw_bbox_each, draw_bbox
from format import *
from keypoint_detector import util_keypoints

def get_cfgs(cfg_name):
    from yolo.config import cfg_loader as cfg_yolo
    from keypoint_detector.config import cfg_loader as cfg_keypoint
    if cfg_name == "yolo":
        return cfg_yolo
    elif cfg_name == "keypoint_detector":
        return cfg_keypoint
    else:
        raise ValueError


def detect_bbox_cocoimgs(args):

    cfgs = get_cfgs(args.cfg_name)    
    #cfg_bbox = cfgs.get("setting_yolo")
    cfg_bbox = cfgs.get(args.setting)
    path_models = cfg_bbox.MODEL.SAVE_PATH + "models/"
    path_result = cfg_bbox.MODEL.SAVE_PATH + "result/"

    device_name = "cuda:" + str(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    device = torch.device(device_name)
    
    loader_aug = cfg_bbox.DATASET.TEST(cfg_bbox)
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



def detect_bbox_yourimgs(args):

    import random
    from torch.utils.data import DataLoader
    from img_loader.dataset import dir_loader, data_loader

    cfgs = get_cfgs(args.cfg_name)
    num_seed = 1234
    #__dir = "/data/public_data/COCO2017/images/train2017/"
    #__dir = "/data/public_data/COCOK2020_1105/images/testK2020_1105/"
    __dir = args.path_dataset
    cfg_bbox = cfgs.get(args.setting)
    path_models = cfg_bbox.MODEL.SAVE_PATH + "models/"
    path_result = cfg_bbox.MODEL.SAVE_PATH + "your_dataset/"
    operator.make_directory(path_result)
    
    __flip = True
    device_name = "cuda:" + str(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    device = torch.device(device_name)

    bsize = args.batchsize
    transformer = Compose([Resize(416, 416, always_apply=True), \
                           Normalize(always_apply=True)])
    model_bbox = cfg_bbox.MODEL.CLASS(cfg_bbox.MODEL)
    model_bbox.load_state_dict(torch.load(path_models + "model_ckpt_best.pth", map_location=device))
    model_bbox.to(device)
    model_bbox.set_device(device)

    data_ = dir_loader.imgloader(__dir, transformer)
    loader_yolo = DataLoader(data_, batch_size=bsize,\
                            shuffle=False, num_workers=2, collate_fn=data_loader.collate_fn_images,
                            worker_init_fn=lambda x: random.seed(num_seed))

    #loss, ind = cfg_bbox.MODEL.EVALUATE_DETAIL(cfg_bbox, model_bbox, loader_yolo, 20)
    loss, ind = cfg_bbox.MODEL.EVALUATE_DETAIL(cfg_bbox, model_bbox, loader_yolo, None)
    boxes_list = ind[0]

    results = list()
    for blist1 in boxes_list:
        for blist2 in blist1:
            results.extend(blist2)

    #results.sort(key=lambda res : (res['image_id'], res['score']), reverse=True) 
    results.sort(key=lambda res : (res['image_id'])) 
    print("results : ", len(results))
    result_path = os.path.join(path_result, 'results.json')
    print(result_path)
    with open(result_path, 'w') as f:
        json.dump(results, f)





def make_persion_batch(img, bbox_p):

    img_list = list()
    leftupper_list, scale_list = list(), list()
    person_list = list()

    for b_p in bbox_p:
        cat = int(b_p["category_id"])
        if cat != 0:
            person_list.append(False)
            continue
        if b_p["score"] < 0.55:
            person_list.append(False)
            continue
        bbox = b_p["bbox"]
        #y1, x1 = int(bbox[1] - bbox[3] / 2), int(bbox[0] - bbox[2] / 2)
        x1, y1 = bbox[0], bbox[1]
        w, h = bbox[2], bbox[3]

        temp = img[y1:(y1+h), x1:(x1+w)]

        if temp.shape[0] < 5 or temp.shape[1] < 5:
            person_list.append(False)
            continue
        
        person_list.append(True)
        leftupper = np.array([x1, y1])
        scale = np.array([w, h]) 
        img_list.append(temp)
        leftupper_list.append(leftupper)
        scale_list.append(scale)

    return img_list, leftupper_list, scale_list, person_list






def extractANDdraw_keypoints(cfg_keypoints, boxes_list, loader_normal, keypoints_save, categories, device):

    #device = torch.device("cpu")
    #device = torch.device("cuda:2")
    device_cpu = torch.device("cpu")

    shape_input = cfg_keypoints.MODEL.INPUT_SHAPE
    shape_output = cfg_keypoints.MODEL.OUTPUT_SHAPE
    model_keypoints = cfg_keypoints.MODEL.CLASS(cfg_keypoints.MODEL).to(device)
    model_keypoints.set_device(device)
    path_models = cfg_keypoints.MODEL.SAVE_PATH + "models/"
    model_keypoints.load_state_dict(torch.load(path_models + "model_ckpt_best.pth", map_location=device))
    model_keypoints.to(device)
    model_keypoints.set_device(device)

    resize_to = shape_input
    compose_key = Compose([Resize(resize_to[0], resize_to[1], always_apply=True), \
                           Normalize(always_apply=True)])
    
    ofs = 0
    coco_anns = list()
    coco_images = list()
    for loop_batch, (boexs_list2, (imgs, _targets)) in enumerate(zip(boxes_list, loader_normal)):
        targets, img_id, imsize = _targets[0], _targets[1], _targets[2]
        fnames = _targets[3]
        #print(imsize[0])
        
        # deal with images one by one.
        for loop_img, (boxes_p, i_id, fn, img) in enumerate(zip(boexs_list2, img_id, fnames, imgs)):

            #
            print("loop_img", loop_img)
            height, width = img.shape[0], img.shape[1]
            #person, bbox is person or not
            img_cropped, leftupper, scale, person = make_persion_batch(img, boxes_p) 
            if len(img_cropped) == 0:
                continue

            img_cropped_list = list()
            for i in img_cropped:
                augmented = compose_key(image=i)
                img_cropped_list.append(augmented["image"])

            with torch.no_grad():
                smap = model_keypoints(img_cropped_list)
                smap = smap.to('cpu').detach().numpy().copy()

                imgs_flipped = np.flip(np.array(img_cropped_list), 2).copy()
                imgs_flipped = imgs_flipped.tolist()
                smap_flipped = model_keypoints(imgs_flipped)
                smap_flipped = smap_flipped.to(device_cpu).numpy()
                smap_flipped = util_keypoints.flip_back(smap_flipped, cfg_keypoints.DATASET.FLIP_PAIRS)
                smap_ave = (smap + smap_flipped) * 0.5
                #print(smap_ave.shape, smap.shape, smap_flipped.shape)

            keypoints, maxvals = util_keypoints.map2keypoints(smap_ave, leftupper, scale, 17, shape_input, shape_output)

            anns_temp = make_coco_annotations(ofs, i_id, boxes_p, person, keypoints, maxvals)
            coco_anns += anns_temp
            ofs += len(anns_temp)
            coco_images += make_coco_images(i_id, fn, height, width)

            #if True:
            if keypoints_save is not None:
                fname = keypoints_save + 'keypoints_' + str(i_id) + '.jpg'
                img_kb = util_keypoints.add_keypoints2img(img, keypoints)
                img_kb = draw_bbox(img_kb, boxes_p, categories, (255, 255, 255), (255, 0, 0))
                print(fname, img_kb.shape)
                im = Image.fromarray(img_kb)
                im.save(fname)

    coco_categories = make_coco_categories()
    results = {"images" : coco_images, "annotations" : coco_anns, "categories" : coco_categories}
    
    return results


def detect_bboxANDkeypoint_yourimgs(args):

    import random
    from torch.utils.data import DataLoader
    from img_loader.dataset import dir_loader, data_loader

    cfgs = get_cfgs(args.cfg_name)
    cfgs2 = get_cfgs(args.cfg_name2)
    __dir = args.path_dataset
    cfg_bbox = cfgs.get(args.setting)
    cfg_keypoints = cfgs2.get(args.setting2)
    path_models = cfg_bbox.MODEL.SAVE_PATH + "models/"
    path_result = cfg_bbox.MODEL.SAVE_PATH + "your_dataset/"
    keypoints_save = __dir + "../result/keypoints/"
    #keypoints_save = __dir

    operator.make_directory(path_result)
    __flip = True
    
    bsize = args.batchsize
    transformer = Compose([Resize(416, 416, always_apply=True), \
                           Normalize(always_apply=True)])
    
    #device = torch.device("cpu")
    #device = torch.device("cuda:2")
    device_name = "cuda:" + str(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    device = torch.device(device_name)

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

    
    
    operator.remove_files(keypoints_save)
    operator.make_directory(keypoints_save)

    data_normal = dir_loader.imgloader(__dir, None)
    loader_normal = DataLoader(data_normal, batch_size=bsize,\
                                shuffle=False, num_workers=2, collate_fn=data_loader.collate_fn_images_sub,
                                worker_init_fn=lambda x: random.seed(num_seed))

    cfg_bbox2 = copy.deepcopy(cfg_bbox)
    cfg_bbox2.DATASET.AUGMENTATOR_val = None
    loader_bbox2 = cfg_bbox2.DATASET.TEST(cfg_bbox2)
    categories = loader_bbox2.dataset.categories()
    print("categories", categories, len(categories))

    #anns_list = extractANDdraw_keypoints(cfg_keypoints, boxes_list, loader_normal, keypoints_save, categories, device)
    anns_list = extractANDdraw_keypoints(cfg_keypoints, boxes_list, loader_normal, None, categories, device)
    anns_list["annotations"].sort(key=lambda res : (res['image_id']), reverse=True) 
    anns_path = os.path.join(keypoints_save, 'results.json')
    print(anns_path)
    with open(anns_path, 'w') as f:
        json.dump(anns_list, f)


def read_results(args):

    cfgs = get_cfgs(args.cfg_name)
    __dir = args.path_dataset
    cfg_bbox = cfgs.get(args.setting)
    path_models = cfg_bbox.MODEL.SAVE_PATH + "models/"
    path_result = cfg_bbox.MODEL.SAVE_PATH + "your_dataset" + "/"
    result_path = os.path.join(path_result, 'results.json')
    print(result_path)

    
    with open(result_path, 'r') as f:
        results = json.load(f)
        print(results)




if __name__ == "__main__":

    functions_list = ["bbox_coco", "bbox_yours", "bboxkeypoint_yours", "read_bboxes"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-B', type=int, default=30, help='')
    parser.add_argument('--cfg_name', '-C1', type=str, default='yolo', help='')
    parser.add_argument('--cfg_name2', '-C2', type=str, default='keypoint_detector', help='')
    parser.add_argument('--setting', '-S1', type=str, default='setting_yolo_spp', help='which setting file are you going to use for training.')
    parser.add_argument('--setting2', '-S2', type=str, default='mspn', help='which setting file are you going to use for training.')
    parser.add_argument('--job', '-J', type=str, default='bbox_coco', help='')
    parser.add_argument('--gpu', '-G', type=int, default='-1', help='')
    parser.add_argument('--path_dataset', '-PD', type=str, default='', help='')
    parser.add_argument('--path_results', '-PR', type=str, default='your_dataset', help='')
    args = parser.parse_args()

    print("task : ", args.job)
    if args.job in functions_list:

        if args.job == "bbox_coco":
            detect_bbox_cocoimgs(args)
        elif args.job == "bbox_yours":
            detect_bbox_yourimgs(args)
        elif args.job == "bboxkeypoint_yours":
            detect_bboxANDkeypoint_yourimgs(args)
        elif args.job == "read_bboxes":
            read_results(args)

    else:
        print("Error")

    #detect_imgs(args)

