# Detector

## Clone repository
```
git clone --recursive http://10.115.1.14/kohei/detector.git 
```



### Requirement
* pycocotools
* albumentations
* numpy
* scikit-image
* pytorch
* torchvision
* easydict



### Install libraries using pip
```
conda install -c conda-forge pycocotools
conda install -c conda-forge albumentations
conda install -c conda-forge numpy
conda install -c conda-forge scikit-image
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-forge easydict
```


### Install libraries on conda environment
```
pip install pycocotools
pip install albumentations
pip install numpy
pip install scikit-image
pip install torch torchvision
pip install easydict
```


### Download trained model
Download Object detector (YOLO-v3-spp, darknet53) from 10.56.254.1
```
scp [user_name]@10.56.254.1:/data/public_data/DL_models/yolo-v3-spp/model_ckpt_best.pth ./result/yolov3_spp/models/
scp [user_name]@10.56.254.1:/data/public_data/DL_models/darknet-53/darknet53_448.weights .result/

or

cp /data/public_data/DL_models/yolo-v3-spp/model_ckpt_best.pth ./result/yolov3_spp/models/
cp /data/public_data/DL_models/darknet-53/darknet53_448.weights ./result/
```

Download Object detector (YOLO-v3, YOLO-v3-vgg) from 10.56.254.1
```
scp [user_name]@10.56.254.1:/data/public_data/DL_models/yolo-v3/model_ckpt_best.pth ./result/yolov3/models/
scp [user_name]@10.56.254.1:/data/public_data/DL_models/yolo-v3-vgg/model_ckpt_best.pth ./result/yolov3_vgg/models/

or

cp /data/public_data/DL_models/yolo-v3/model_ckpt_best.pth ./result/yolov3/models/
cp /data/public_data/DL_models/yolo-v3/model_ckpt_best.pth ./result/yolov3_vgg/models/
```

Download Keypoint detector from 10.56.254.1
```
scp [user_name]@10.56.254.1:/data/public_data/DL_models/MSPN/model_ckpt_best.pth ./result/mspn/models/
cp /data/public_data/DL_models/MSPN/model_ckpt_best.pth ./result/mspn/models/
```


### The commands that detects bboxes from image datasets.

|Arguments for program|Explanations|
|:---:|:---|
|cfg_name|the config file for model that you want to use.|
|setting|the model setting that you want to use.|
|gpu| set gpu number to use. <br> if you want to run it on cpu, set negative number.|
|job| the task you want to run. <br> ["bbox_coco", "bbox_yours", "bboxkeypoint_yours", "read_bboxes"] can be selected.|
|path_dataset| path to data. |
|path_results| path to results file. <br> it is saved on ./results/["model_name"]/["path_results"]|
|cfg_name2|the config file for model that you want to use.|
|setting2|the model2 setting that you want to use.|


### Command examples
* #### yolo-v3-spp detects bboxes from coco dataset using gpu 0.
```
python detector.py --setting yolov3_spp --gpu 0 --job bbox_coco
```

* #### yolo-v3 detects bboxes from coco dataset using gpu 1.
```
python detector.py --setting yolov3 --gpu -1 --job bbox_coco
```

* #### yolo-v3-spp detects bboxes from coco dataset using cpu.
```
python detector.py --setting yolov3_spp --gpu -1 --job bbox_coco
```

* #### yolo-v3-spp detects bboxes from your own images.
```
python detector.py --setting yolov3_spp --gpu 0 --job bbox_yours --path_dataset [path for your own images] --path_results [path to save results json file]

python detector.py --setting yolov3_spp --gpu 0 --job bbox_yours --path_dataset /data/public_data/COCOK2020_1105/images/testK2020_1105/ --path_results your_dataset
```

* #### detects bboxes from your dataset
```
python detector.py --setting yolov3_spp --job read_bboxes --path_results your_dataset
```

* #### detects bboxes and keypoints from your dataset
```
python detector.py --setting yolov3_spp --setting2 mspn --gpu 0 --job bboxkeypoint_yours --path_dataset /data/public_data/COCOK2020_1105/images/testK2020_1105/ --path_results your_dataset

python detector.py --setting yolov3_spp --setting2 mspn --gpu 0 --job bboxkeypoint_yours --path_dataset /data/public_data/KARD/imgs/a01_s05_e03/ --path_results your_dataset
```
