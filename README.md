# RetinaFace in PyTorch

A [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). Model size only 1.7M, when Retinaface use mobilenet0.25 as backbone net. We also provide resnet50 as backbone net to get better result. The official code in Mxnet can be found [here](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

## Mobile or Edge device deploy
We also provide a set of Face Detector for edge device in [here](https://github.com/biubug6/Face-Detector-1MB-with-landmark) from python training to C++ inference.

## WiderFace Val Performance in single scale When using Resnet50 as backbone net.
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Pytorch (same parameter with Mxnet) | 94.82 % | 93.84% | 89.60% |
| Pytorch (original image scale) | 95.48% | 94.04% | 84.43% |
| Mxnet | 94.86% | 93.87% | 88.33% |
| Mxnet(original image scale) | 94.97% | 93.89% | 82.27% |

## WiderFace Val Performance in single scale When using Mobilenet0.25 as backbone net.
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Pytorch (same parameter with Mxnet) | 88.67% | 87.09% | 80.99% |
| Pytorch (original image scale) | 90.70% | 88.16% | 73.82% |
| Mxnet | 88.72% | 86.97% | 79.19% |
| Mxnet(original image scale) | 89.58% | 87.11% | 69.12% |
<p align="center"><img src="curve/Widerface.jpg" width="640"\></p>

## FDDB Performance.
| FDDB(pytorch) | performance |
|:-|:-:|
| Mobilenet0.25 | 98.64% |
| Resnet50 | 99.22% |
<p align="center"><img src="curve/FDDB.png" width="640"\></p>

### Contents
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [TensorRT](#tensorrt)
- [References](#references)

## Installation
##### Clone and install
1. git clone https://github.com/biubug6/Pytorch_Retinaface.git

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

##### Data
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

##### Data1
We also provide the organized dataset we used as in the above directory structure.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## Training
We provide restnet50 and mobilenet0.25 as backbone network to train model.
We trained Mobilenet0.25 on imagenet dataset and get 46.58%  in top 1. If you do not wish to train the model, we also provide trained model. Pretrain model  and trained model are put in [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and [baidu cloud](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq . The model could be put as follows:
```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``data/config.py and train.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
  ```


## Evaluation
### Evaluation widerface val
1. Generate txt file
```Shell
python test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)
### Evaluation FDDB

1. Download the images [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to:
```Shell
./data/FDDB/images/
```

2. Evaluate the trained model using:
```Shell
python test_fddb.py --trained_model weight_file --network mobile0.25 or resnet50
```

3. Download [eval_tool](https://bitbucket.org/marcopede/face-eval) to evaluate the performance.

<p align="center"><img src="curve/1.jpg" width="640"\></p>

## TensorRT
-[TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)

## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
# RetinaFaceProject-EL7008

Este repositorio corresponde a un clon del repositorio original de RetinaFace: https://github.com/biubug6/Pytorch_Retinaface.

En este se realizaron modificaciones con el fin de adaptar el código para utilizar distintos modelos de redes neuronales como backbone para el algoritmo de detección de rostros de RetinaFace, usando además diversos algoritmos de optimización. El dataset utilizado es el WiderFace, disponible en http://shuoyang1213.me/WIDERFACE/index.html.

Las nuevas redes agregadas corresponde a la MobileNet v3 y la EfficientNet b2, ambas implementadas en la librería timm del repositorio https://github.com/rwightman/pytorch-image-models.

Los algoritmos de optimización añadidos son el SGDP y AdamP, modificaciones de los clásicos algoritmos SGD y Adam, implementados en la librería adamp del repositorio https://github.com/clovaai/AdamP.

Se entrenaron un total de 4 modelos, haciendo uso de una GPU Nvidia v100 con 32gb de memoria, la cual es uno de los 8 módulos disponibles que ofrece el supercomputador Nvidia DGX1.

El primer modelo utiliza la red mobilenet v3 como backbone y el optimizador Adam, tamaño de batch 32 y tamaño de imagen 640. Entrenado durante 170 épocas, aproximadamente 22 horas y 30 minutos, arrojando los siguientes resultados de accuracy de validación: 

Easy   Val AP: 0.9245397516986055
Medium Val AP: 0.9076881989570095
Hard   Val AP: 0.7926642019968118
