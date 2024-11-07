
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

```
Q. Liu, J. Yue, Y. Fang, S. Xia and L. Fang, "HyperMamba: A Spectral-Spatial Adaptive Mamba for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-14, 2024, Art no. 5536514, doi: 10.1109/TGRS.2024.3482473.
```

## Acknowledgment



