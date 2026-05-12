# NYCU Computer Vision 2026 HW3

* **Student ID**: 314552048
* **Name**: 蔡旺霖

## Introduction
This report addresses the task of multi-class cell instance segmentation on colored medical images.
Given an input image, the model must localize each cell instance and classify it as one of four
cell types (class1–class4). The task is challenging because (i) cells are small and densely packed, (ii)
boundaries between adjacent cells are often unclear, (iii) the dataset contains only 209 training/validation
images, which introduces a high risk of overfitting, and (iv) evaluation is performed under the strict AP50
metric on segmentation masks.


## Environment Setup

Ensure your environment has installed `Python 3.8+` and CUDA.

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data_root ../data --epochs 50 --batch_size 4
```
### Inference

```bash
python inference.py
```

## Performance Snapshot
![image.png](https://github.com/wonlin91/Visual-Recognition-using-Deep-Learning/blob/main/HW3/leaderboard.png)
