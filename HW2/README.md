# NYCU Computer Vision 2026 HW2

* **Student ID**: 314552048
* **Name**: 蔡旺霖

## Introduction
This report presents a digit detection system built for a street-view number recognition benchmark. The dataset consists of 30,062 training images, 3,340 validation images, and 13,068 test images in COCO format. Each image may contain multiple overlapping digits, and the task requires predicting both the class (digit 0–9, encoded as category IDs 1–10) and the tight bounding box for every digit present in 
the scene. 
The core idea of our approach is to leverage Deformable DETR, an end-to-end transformer-based object detector that replaces the computationally expensive global attention of vanilla DETR with sparse, data-dependent deformable attention over multi-scale feature maps. This design makes the model significantly more efficient and faster to converge, which is critical given the 30-epoch training budget. We augment the base architecture with EMA model averaging, a warmup-cosine learning-rate schedule, and mixed-precision training to maximise stability and final performance.


## Environment Setup

Ensure your environment has installed `Python 3.8+` and CUDA.

```bash
pip install -r requirements.txt
```

## Usage

Open main.py and modify the configuration above, changing the mode to “train/test” and other arguments as needed.

```bash
python main.py --mode train/test --epoch arg1 --batch_size arg2
```

## Performance Snapshot
![image.png](https://github.com/wonlin91/Visual-Recognition-using-Deep-Learning/blob/main/HW2/leaderboard.png)
