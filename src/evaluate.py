#!/usr/bin/env python3
"""Evaluate a trained checkpoint: test accuracy, per-class metrics, plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import CIFAR10_CLASSES, get_cifar10_loaders
from src.models.starnet import build_model
from src.utils import ensure_dir, save_json, select_device


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def collect_predictions(model, loader, device):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            ys.append(targets.numpy())
            ps.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def plot_confusion(cm: np.ndarray, labels, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_history(history_csv: Path, out_path: Path, title: str) -> None:
    if not history_csv.exists():
        return
    data = np.genfromtxt(history_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(data["epoch"], data["train_loss"], label="train")
    axes[0].plot(data["epoch"], data["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(data["epoch"], data["train_acc"], label="train")
    axes[1].plot(data["epoch"], data["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_comparison(result_root: Path, out_path: Path, models: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in models:
        csv_path = result_root / name / "history.csv"
        if not csv_path.exists():
            continue
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        axes[0].plot(data["epoch"], data["val_loss"], label=name)
        axes[1].plot(data["epoch"], data["val_acc"], label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val loss")
    axes[0].legend()
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val accuracy")
    axes[1].legend()
    fig.suptitle("CIFAR-10 validation curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cifar10.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = select_device(args.device or cfg.get("device", "auto"))
    model_name = args.model
    ckpt_path = args.checkpoint or (ROOT / "checkpoints" / f"{model_name}_best.pt")
    run_dir = ensure_dir(ROOT / "results" / model_name)
    fig_dir = ensure_dir(ROOT / "report" / "figures")

    _, _, test_loader = get_cifar10_loaders(
        data_dir=cfg["data"].get("root", "data"),
        batch_size=int(cfg["data"]["batch_size"]),
        val_ratio=float(cfg["data"].get("val_ratio", 0.1)),
        num_workers=int(cfg["data"].get("num_workers", 2)),
        seed=int(cfg.get("seed", 42)),
    )

    model = build_model(
        model_name,
        num_classes=int(cfg["model"].get("num_classes", 10)),
        stem_stride=int(cfg["model"].get("stem_stride", 1)),
        drop_path_rate=float(cfg["model"].get("drop_path_rate", 0.0)),
    ).to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])

    y_true, y_pred = collect_predictions(model, test_loader, device)
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=list(CIFAR10_CLASSES), output_dict=True
    )
    per_class = {
        cls: float((y_pred[y_true == i] == i).mean()) if np.any(y_true == i) else 0.0
        for i, cls in enumerate(CIFAR10_CLASSES)
    }
    metrics = {
        "model": model_name,
        "checkpoint": str(ckpt_path),
        "test_accuracy": acc,
        "per_class_accuracy": per_class,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    save_json(run_dir / "test_metrics.json", metrics)
    plot_confusion(cm, CIFAR10_CLASSES, run_dir / "confusion_matrix.png")
    plot_confusion(cm, CIFAR10_CLASSES, fig_dir / f"{model_name}_confusion.png")
    plot_history(run_dir / "history.csv", run_dir / "curves.png", model_name)
    plot_history(run_dir / "history.csv", fig_dir / f"{model_name}_curves.png", model_name)
    print(json.dumps({"model": model_name, "test_accuracy": acc, "per_class": per_class}, indent=2))

    if args.compare:
        names = ["starnet_s1", "sumnet_s1", "resnet18"]
        plot_comparison(ROOT / "results", fig_dir / "val_curves_compare.png", names)
        print(f"wrote {fig_dir / 'val_curves_compare.png'}")


if __name__ == "__main__":
    main()
