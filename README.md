# CISC3024 AI Assignment 1 — StarNet on CIFAR-10

Reproduction of **StarNet** from Ma et al., *Rewrite the Stars* (CVPR 2024), applied to CIFAR-10 image classification. The same architecture with the star operation replaced by summation (**SumNet**) and a CIFAR-adapted **ResNet-18** are trained as controls.

Paper: https://arxiv.org/abs/2403.19967
Official code: https://github.com/ma-xu/Rewrite-the-Stars

The student did not write this code. It was produced by an AI agent for the assignment.

## Setup

Python 3.14 with PyTorch 2.14 (Apple MPS on this machine):

```bash
pip3 install -r requirements.txt
```

## Train

```bash
python src/train.py --model starnet_s1
python src/train.py --model sumnet_s1
python src/train.py --model resnet18
```

Checkpoints go to `checkpoints/{model}_best.pt`. Epoch logs go to `results/{model}/history.csv`.

## Evaluate and plots

```bash
python src/evaluate.py --model starnet_s1
python src/evaluate.py --model sumnet_s1
python src/evaluate.py --model resnet18 --compare
python src/visualize_2d.py
```

## Layout

```
src/models/starnet.py   StarNet block, SumNet flag, ResNet-18 factory
src/data.py             CIFAR-10 loaders
src/train.py            training loop
src/evaluate.py         test metrics and figures
src/visualize_2d.py     star vs sum vs polynomial SVM on 2D moons
configs/cifar10.yaml    shared hyperparameters
report/                 assignment report source and figures
```
