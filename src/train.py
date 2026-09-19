#!/usr/bin/env python3
"""Train StarNet / SumNet / ResNet-18 on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_cifar10_loaders
from src.models.starnet import build_model, count_flops, count_parameters
from src.utils import (
    accuracy_from_logits,
    cosine_warmup_lambda,
    ensure_dir,
    save_json,
    select_device,
    set_seed,
)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, targets in loader:
            images = images.to(device, non_blocking=False)
            targets = targets.to(device, non_blocking=False)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits.detach(), targets)
            n_batches += 1
    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cifar10.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override config model name.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg["model"]["name"]
    epochs = args.epochs if args.epochs is not None else int(cfg["train"]["epochs"])
    batch_size = args.batch_size or int(cfg["data"]["batch_size"])
    lr = args.lr if args.lr is not None else float(cfg["train"]["lr"])
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    device = select_device(args.device or cfg.get("device", "auto"))
    set_seed(seed)

    run_dir = args.output_dir or (ROOT / "results" / model_name)
    ckpt_dir = ROOT / "checkpoints"
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=cfg["data"].get("root", "data"),
        batch_size=batch_size,
        val_ratio=float(cfg["data"].get("val_ratio", 0.1)),
        num_workers=int(cfg["data"].get("num_workers", 2)),
        seed=seed,
    )

    model = build_model(
        model_name,
        num_classes=int(cfg["model"].get("num_classes", 10)),
        stem_stride=int(cfg["model"].get("stem_stride", 1)),
        drop_path_rate=float(cfg["model"].get("drop_path_rate", 0.0)),
    ).to(device)

    n_params = count_parameters(model)
    try:
        n_flops = count_flops(model, (1, 3, 32, 32))
    except Exception:
        n_flops = -1

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg["train"].get("weight_decay", 0.05)),
    )
    warmup = int(cfg["train"].get("warmup_epochs", 5))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_warmup_lambda(epoch, warmup, epochs),
    )

    start_epoch = 0
    best_val_acc = -1.0
    if args.resume and args.resume.exists():
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        start_epoch = int(blob.get("epoch", 0)) + 1
        best_val_acc = float(blob.get("best_val_acc", -1.0))

    history_path = run_dir / "history.csv"
    new_history = start_epoch == 0
    if new_history:
        with history_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]
            )

    print(
        f"model={model_name} device={device} params={n_params:,} "
        f"flops≈{n_flops:,} epochs={epochs} batch={batch_size} lr={lr}"
    )

    best_path = ckpt_dir / f"{model_name}_best.pt"
    last_path = ckpt_dir / f"{model_name}_last.pt"

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        with history_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch + 1,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{lr_now:.8f}",
                    f"{elapsed:.2f}",
                ]
            )
        print(
            f"epoch {epoch + 1:03d}/{epochs}  "
            f"train {train_loss:.4f}/{train_acc:.4f}  "
            f"val {val_loss:.4f}/{val_acc:.4f}  "
            f"lr {lr_now:.2e}  {elapsed:.1f}s"
        )
        payload = {
            "epoch": epoch,
            "model_name": model_name,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "val_acc": val_acc,
        }
        torch.save(payload, last_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            payload["best_val_acc"] = best_val_acc
            torch.save(payload, best_path)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)
    # Re-evaluate the best checkpoint on the test set.
    if best_path.exists():
        blob = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        best_test_loss, best_test_acc = run_epoch(
            model, test_loader, criterion, optimizer, device, False
        )
    else:
        best_test_loss, best_test_acc = test_loss, test_acc

    summary = {
        "model": model_name,
        "device": str(device),
        "params": n_params,
        "flops_macs": n_flops,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "best_val_acc": best_val_acc,
        "last_test_acc": test_acc,
        "best_ckpt_test_acc": best_test_acc,
        "best_ckpt_test_loss": best_test_loss,
        "history_csv": str(history_path),
        "best_checkpoint": str(best_path),
    }
    save_json(run_dir / "summary.json", summary)
    print(f"done. best_val_acc={best_val_acc:.4f} best_ckpt_test_acc={best_test_acc:.4f}")
    print(f"wrote {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
