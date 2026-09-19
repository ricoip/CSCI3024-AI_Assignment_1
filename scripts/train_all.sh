#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python src/train.py --model starnet_s1
python src/train.py --model sumnet_s1
python src/train.py --model resnet18
python src/evaluate.py --model starnet_s1
python src/evaluate.py --model sumnet_s1
python src/evaluate.py --model resnet18 --compare
python src/visualize_2d.py
