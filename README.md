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


### Install libraries on conda environment
```
conda install -c conda-forge pycocotools
conda install -c conda-forge albumentations
conda install -c conda-forge numpy
conda install -c conda-forge scikit-image
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-forge easydict
```



### Download trained model
Download Object detector from 10.56.254.1
```
scp [user_name]@10.56.254.1:/data/public_data/DL_models/yolo-v3/model_ckpt_best.pth ./result/yolov3/models/
scp [user_name]@10.56.254.1:/data/public_data/DL_models/darknet-53/darknet53_448.weights .result/

or

cp /data/public_data/DL_models/yolo-v3/model_ckpt_best.pth ./result/yolov3/models/
cp /data/public_data/DL_models/darknet-53/darknet53_448.weights ./result/
```

### The commands that detects bboxes from image datasets.
#### yolo-v3 detect bboxes from coco dataset using gpu 0.
```
python detector.py --gpu 0 --job bbox_coco
```

#### yolo-v3 detect bboxes from coco dataset using cpu.
```
python detector.py --job bbox_coco
```

#### yolo-v3 detect bboxes from your own images.
```
python detector.py --gpu 0 --job bbox_yours --path_dataset [path for your own images]
python detector.py --gpu 0 --job bbox_yours --path_dataset /data/public_data/COCOK2020_1105/images/testK2020_1105/
```


