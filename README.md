

# HyperMamba: A Spectral-Spatial Adaptive Mamba for Hyperspectral Image Classification
Qiang Liu, Jun Yue, Yi Fang, Shaobo Xia, and Leyuan Fang, Senior Member, IEEE

![framework](/figure\framework.png)

# Getting Started

## step1: Environment Setup:

To get started, we ecommend setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment

```bash
conda create -n hypermamba python==3.10
conda activate hypermamba
pip install -r requirements.txt
```
install ```mmcv```
```bash
pip install -U openmim
mim install mmcv==2.1.0
```

download vmamba dependencies at https://github.com/MzeroMiko/VMamba/archive/refs/tags/%2320240220.tar.gz 

unzip and run:
```bash
# Install selective_scan and its dependencies
cd selective_scan && pip install .
```

## step2: Model Training and Inference:

Our work is evaluated on three pulic hyperspectral dataset

To train Hypermamba for classification on those datasets, 
you should change```include_path``` for different dataset in code file```workflow.py ```
use the following commands for model training.

```bash
python workflow.py
```

the reults are saved in ```res``` folder and are saved at ```ckpt``` folder

## Moreover 
if you want to change the data path or model settings, please go to ``` params_use``` folder.


## Citation
If this code is useful for your research, please cite this paper.
```
Q. Liu, J. Yue, Y. Fang, S. Xia and L. Fang, "HyperMamba: A Spectral-Spatial Adaptive Mamba for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-14, 2024, Art no. 5536514, doi: 10.1109/TGRS.2024.3482473.
```

```
@ARTICLE{10720896,
  author={Liu, Qiang and Yue, Jun and Fang, Yi and Xia, Shaobo and Fang, Leyuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={HyperMamba: A Spectral-Spatial Adaptive Mamba for Hyperspectral Image Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-14},
  keywords={Computational modeling;Adaptation models;Transformers;Training;Feature extraction;Accuracy;Quaternions;Context modeling;Hyperspectral imaging;Convolutional neural networks;Deep neural network;hyperspectral image (HSI) classification;Mamba},
  doi={10.1109/TGRS.2024.3482473}}

```

## Acknowledgment

This code is mainly built upon [SQSFormer](https://github.com/chenning0115/SQSFormer) and [VMamba](https://github.com/MzeroMiko/VMamba) repositories.


