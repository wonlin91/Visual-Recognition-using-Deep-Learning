# NYCU Computer Vision 2026 HW1

* **Student ID**: 123456789
* **Name**: Your Name

## Introduction
This report presents an image classification system for a 100-class object recognition task. Given an RGB image, the model must predict the correct category from a fixed label set — a standard closed-set classification problem. The training and validation split comprises 21,024 labeled images, while the held-out test set contains 2,344 unlabeled images.
The task is subject to three constraints: only ResNet-based backbone architectures are permitted, no external data may be used, and the total number of model parameters must remain below 100 million.
The core idea of our method is to combine a modernized ResNet backbone (**resnet50d**) with a suite of strong regularization strategies to maximize generalization under these constraints. Specifically, we apply **Mixup and CutMix** augmentation at the batch level to prevent the model from memorizing training samples, **label smoothing** to reduce overconfidence in predictions, and Cosine Annealing learning rate scheduling to allow stable convergence over extended training. At inference, we further apply **Test Time Augmentation (TTA)** via horizontal flipping to reduce prediction variance without any additional training cost. Together, these choices are motivated by empirical findings showing that strong augmentation strategies significantly close the generalization gap on mid-scale datasets where overfitting is a primary concern.


## Environment Setup

Ensure your environment has installed `Python 3.8+` and CUDA.

```bash
pip install -r requirements.txt
```

## Usage

Open main.py and modify the configuration above, changing the mode to “train” or “test” as needed.

```bash
python main.py
```

## Performance Snapshot
![alt text](image.png)