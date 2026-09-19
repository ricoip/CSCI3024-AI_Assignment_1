#!/usr/bin/env python3
"""2D noisy-moons decision boundaries: star vs sum vs polynomial SVM.

This is a compact reproduction of the qualitative experiment in Figure 2 of
Ma et al., 'Rewrite the Stars' (CVPR 2024). Convolution and normalization are
removed; the remaining MLP block still uses ReLU6(f1(x)) ⋆ f2(x).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, set_seed


class DemoBlock(nn.Module):
    def __init__(self, dim: int, use_star: bool) -> None:
        super().__init__()
        self.use_star = use_star
        self.f1 = nn.Linear(dim, dim)
        self.f2 = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1, x2 = self.f1(x), self.f2(x)
        fused = self.act(x1) * x2 if self.use_star else self.act(x1) + x2
        return residual + self.g(fused)


class DemoNet(nn.Module):
    def __init__(self, dim: int = 64, depth: int = 4, use_star: bool = True) -> None:
        super().__init__()
        self.stem = nn.Linear(2, dim)
        self.blocks = nn.Sequential(*[DemoBlock(dim, use_star) for _ in range(depth)])
        self.head = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(torch.relu(self.stem(x))))


def train_demonet(X, y, use_star: bool, steps: int, lr: float, device: torch.device) -> DemoNet:
    model = DemoNet(dim=64, depth=4, use_star=use_star).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def mesh_from_points(X: np.ndarray, n: int = 400):
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


@torch.no_grad()
def predict_net(model: DemoNet, grid: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(grid, dtype=torch.float32, device=device)
    logits = model(xt)
    return logits.argmax(dim=1).cpu().numpy()


def plot_panel(ax, xx, yy, Z, X, y, title: str) -> None:
    ax.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.35, levels=1, cmap="coolwarm")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, c="#1f77b4", edgecolors="none", label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, c="#d62728", edgecolors="none", label="class 1")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")
    X, y = make_moons(n_samples=400, noise=0.25, random_state=args.seed)
    X = StandardScaler().fit_transform(X)
    xx, yy, grid = mesh_from_points(X)

    star = train_demonet(X, y, True, args.steps, 3e-3, device)
    summer = train_demonet(X, y, False, args.steps, 3e-3, device)
    svm = SVC(kernel="poly", degree=3, C=1.0, gamma="scale")
    svm.fit(X, y)

    z_star = predict_net(star, grid, device)
    z_sum = predict_net(summer, grid, device)
    z_svm = svm.predict(grid)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    plot_panel(axes[0], xx, yy, z_star, X, y, "Star operation")
    plot_panel(axes[1], xx, yy, z_sum, X, y, "Summation")
    plot_panel(axes[2], xx, yy, z_svm, X, y, "Polynomial-kernel SVM")
    fig.tight_layout()

    fig_dir = ensure_dir(ROOT / "report" / "figures")
    out = args.out or (fig_dir / "decision_boundary_2d.png")
    fig.savefig(out, dpi=160)
    (ROOT / "results").mkdir(exist_ok=True)
    fig.savefig(ROOT / "results" / "decision_boundary_2d.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
