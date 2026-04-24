"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SELF-PRUNING NEURAL NETWORK  ·  TREDENCE AI CASE STUDY             ║
║              Dynamic Weight Pruning via Learnable Sigmoid Gates              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Task    : Build a network that prunes itself during training                ║
║  Dataset : CIFAR-10 (10-class image classification)                          ║
║  Method  : Per-weight learnable gates + L1 sparsity regularisation           ║
║                                                                              ║
║  Core Idea:                                                                  ║
║    Forward:  effective_weight = W ⊙ σ(gate_scores)                          ║
║    Loss:     Total = CrossEntropy(ŷ, y) + λ · Σ σ(gate_scores)              ║
║    Result:   L1 penalty drives many gates → 0, pruning those weights         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python self_pruning_network.py                          # defaults
    python self_pruning_network.py --epochs 50 --lr 1e-3
    python self_pruning_network.py --lambdas 1e-4 1e-3 1e-2
"""

import os
import math
import time
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 · PRUNABLE LINEAR LAYER
# ══════════════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear with learnable per-weight gates.

    Each weight w_ij has an associated scalar gate score g_ij. The effective
    weight used in the forward pass is:

        gates           = σ(gate_scores)          ← sigmoid squashes to (0, 1)
        pruned_weights  = weight  ⊙  gates        ← element-wise multiply
        output          = x @ pruned_weights.T + bias

    When combined with an L1 penalty (λ · Σ gates) in the total loss, the
    optimiser is incentivised to push gate values toward 0.  The sigmoid means
    that once a gate_score becomes sufficiently negative (~< −5), σ(g) ≈ 0 and
    the weight is effectively removed from the network — i.e. "pruned".

    Gradients flow through BOTH self.weight and self.gate_scores automatically
    via PyTorch autograd, since every operation (sigmoid, multiply, linear) is
    differentiable.

    Args:
        in_features  : Input dimension.
        out_features : Output dimension.
        bias         : Add a learnable bias. Default: True.
        gate_init    : Initial value for gate_scores.  gate_init=0 → σ(0)=0.5,
                       meaning gates start half-open.  Default: 0.0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Standard weight, initialised identically to nn.Linear ─────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # ── Learnable gate scores (same shape as weight) ───────────────────
        # These are the raw (pre-sigmoid) gate parameters.  They are learnable
        # parameters, so Adam will update them along with the weights.
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), float(gate_init))
        )

        # ── Optional bias ──────────────────────────────────────────────────
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute: output = (W ⊙ σ(G)) x + b

        Step 1 – gates = sigmoid(gate_scores)   → values in (0, 1)
        Step 2 – pruned_weights = weight * gates → element-wise product
        Step 3 – F.linear(x, pruned_weights, bias)  → standard affine transform

        All three steps are differentiable, so autograd computes:
            ∂L/∂weight      = gates · (∂L/∂output)^T x          (chain rule)
            ∂L/∂gate_scores = weight · σ'(g) · (∂L/∂output)^T x (chain rule)
        Both parameters are updated simultaneously by the optimiser.
        """
        gates         = torch.sigmoid(self.gate_scores)   # (out, in), in (0,1)
        pruned_weight = self.weight * gates                # element-wise
        return F.linear(x, pruned_weight, self.bias)       # standard linear op

    # ── Utility methods ───────────────────────────────────────────────────────
    def get_gates(self) -> torch.Tensor:
        """Return current gate values σ(gate_scores), detached from graph."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def sparsity(self, threshold: float = 0.05) -> float:
        """Fraction of gates below `threshold` (effectively pruned weights)."""
        return (self.get_gates() < threshold).float().mean().item()

    def all_gates_np(self) -> np.ndarray:
        """Flat numpy array of all gate values for this layer."""
        return self.get_gates().cpu().numpy().ravel()

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    """
    CIFAR-10 classifier with a convolutional feature extractor and a
    prunable fully-connected head.

    Architecture:
        Conv Backbone (non-prunable):
            3×32×32 → Conv-BN-ReLU × 2 → MaxPool
                    → Conv-BN-ReLU × 2 → MaxPool
                    → Conv-BN-ReLU × 2 → AdaptiveAvgPool → 512-d features

        Prunable Head (where sparsity is learned):
            512 → PrunableLinear(256) → BN-ReLU-Dropout
                → PrunableLinear(128) → BN-ReLU-Dropout
                → PrunableLinear(10)  → logits
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # ── Convolutional Backbone ─────────────────────────────────────────
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # → 16×16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # → 8×8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                  # → 512-d
        )

        # ── Prunable Classifier Head ───────────────────────────────────────
        self.fc1   = PrunableLinear(512, 256)
        self.bn1   = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout)

        self.fc2   = PrunableLinear(256, 128)
        self.bn2   = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)

        self.fc3   = PrunableLinear(128, num_classes)

        self._prunable: List[PrunableLinear] = [self.fc1, self.fc2, self.fc3]
        self._layer_names = ["FC1 (512→256)", "FC2 (256→128)", "FC3 (128→10)"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).view(x.size(0), -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

    # ── Sparsity utilities ────────────────────────────────────────────────────

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values  =  Σ_layers  Σ_ij  σ(g_ij).

        WHY L1 encourages sparsity:
          - The L1 norm imposes a CONSTANT gradient of ±1 on each gate value,
            regardless of its current magnitude.  Even a tiny gate value near 0
            still receives a consistent "push" toward zero.
          - Compare with L2: gradient ∝ 2·gate  →  shrinks to zero as the gate
            approaches zero, so L2 can never fully zero a value.
          - Because gates are always positive (sigmoid output), L1 = Σ gates,
            making this differentiable and simple to compute.
          - With the λ trade-off, the network must decide per weight: "is this
            connection worth the sparsity penalty?"  Unimportant weights lose
            their gates; important ones keep them.

        Returns: scalar tensor, differentiable w.r.t. gate_scores.
        """
        device = next(self.parameters()).device
        total  = torch.tensor(0.0, device=device)
        for layer in self._prunable:
            total = total + torch.sigmoid(layer.gate_scores).sum()
        return total

    def overall_sparsity(self, threshold: float = 0.05) -> float:
        """Fraction of all prunable weights whose gate < threshold."""
        tp = tw = 0
        for layer in self._prunable:
            g   = layer.get_gates()
            tp += (g < threshold).sum().item()
            tw += g.numel()
        return tp / tw if tw > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        """All gate values concatenated into a 1-D numpy array."""
        return np.concatenate([l.all_gates_np() for l in self._prunable])

    def param_summary(self) -> Dict[str, int]:
        bb   = sum(p.numel() for p in self.backbone.parameters())
        fc_w = sum(l.weight.numel() for l in self._prunable)
        fc_g = sum(l.gate_scores.numel() for l in self._prunable)
        fc_b = sum(l.bias.numel() for l in self._prunable if l.bias is not None)
        bn   = (sum(p.numel() for p in self.bn1.parameters()) +
                sum(p.numel() for p in self.bn2.parameters()))
        return {"backbone": bb, "fc_weights": fc_w, "fc_gates": fc_g,
                "fc_bias+bn": fc_b + bn,
                "total": sum(p.numel() for p in self.parameters())}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (CIFAR-10)
# ══════════════════════════════════════════════════════════════════════════════

def get_cifar10_loaders(
    data_dir: str   = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return (train_loader, test_loader)."""
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        raise ImportError("Run: pip install torchvision")

    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2023, 0.1994, 0.2010)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_set = torchvision.datasets.CIFAR10(
        data_dir, train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_tf)

    return (
        DataLoader(train_set, batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(test_set, batch_size=256, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 + 3 · TRAINING LOOP & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model     : SelfPruningNet,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    lambda_sp : float,
    device    : torch.device,
) -> Tuple[float, float, float, float]:
    """
    One training epoch.

    Total Loss = CrossEntropy(logits, labels)  +  λ × SparsityLoss
                 ─────────────────────────────     ─────────────────
                 Drives correct classification      Drives gates → 0

    Returns (total_loss, ce_loss, sp_loss, accuracy) — per-sample averages.
    """
    model.train()
    total_l = ce_l = sp_l = correct = n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits  = model(imgs)
        ce_loss = F.cross_entropy(logits, labels)
        sp_loss = model.sparsity_loss()              # L1 norm of all gates
        loss    = ce_loss + lambda_sp * sp_loss      # combined objective

        loss.backward()
        # Gradient clipping prevents gate_scores from jumping past sigmoid
        # saturation in one step, which would make training unstable
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs       = imgs.size(0)
        total_l += loss.item()    * bs
        ce_l    += ce_loss.item() * bs
        sp_l    += sp_loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        n       += bs

    return total_l/n, ce_l/n, sp_l/n, correct/n


@torch.no_grad()
def evaluate(
    model  : SelfPruningNet,
    loader : DataLoader,
    device : torch.device,
) -> Tuple[float, float]:
    """Evaluate model, return (avg_ce_loss, accuracy)."""
    model.eval()
    loss_sum = correct = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits    = model(imgs)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct  += (logits.argmax(1) == labels).sum().item()
        n        += imgs.size(0)
    return loss_sum / n, correct / n


def train(
    lambda_sp    : float,
    train_loader : DataLoader,
    test_loader  : DataLoader,
    epochs       : int,
    lr           : float,
    device       : torch.device,
    seed         : int  = 42,
    verbose      : bool = True,
) -> Tuple[SelfPruningNet, dict]:
    """Full training run for a single λ. Returns best-checkpoint model + history."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = SelfPruningNet().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    history: Dict[str, list] = dict(
        train_loss=[], ce_loss=[], sp_loss=[],
        train_acc=[], val_acc=[], sparsity=[], lr=[],
    )
    best_val_acc = 0.0
    best_state   = None

    if verbose:
        ps = model.param_summary()
        print(f"\n{'─'*72}")
        print(f"  Training  λ = {lambda_sp:.5f}  "
              f"| total params: {ps['total']:,}  "
              f"| gate params: {ps['fc_gates']:,}")
        print(f"{'─'*72}")
        print(f"  {'Ep':>4}  {'Total':>8}  {'CE':>8}  {'Sp':>8}  "
              f"{'TrAcc':>8}  {'ValAcc':>8}  {'Sparsity':>9}  {'LR':>8}")
        print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  "
              f"{'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_ce, tr_sp, tr_acc = train_epoch(
            model, train_loader, optimizer, lambda_sp, device
        )
        val_loss, val_acc = evaluate(model, test_loader, device)
        sparsity          = model.overall_sparsity()
        cur_lr            = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["ce_loss"].append(tr_ce)
        history["sp_loss"].append(tr_sp)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["sparsity"].append(sparsity)
        history["lr"].append(cur_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose:
            elapsed = time.time() - t0
            print(
                f"  {epoch:>4}  {tr_loss:>8.4f}  {tr_ce:>8.4f}  {tr_sp:>8.1f}  "
                f"{tr_acc*100:>7.2f}%  {val_acc*100:>7.2f}%  "
                f"{sparsity*100:>8.2f}%  {cur_lr:>8.2e}  [{elapsed:.1f}s]"
            )

    if best_state:
        model.load_state_dict(best_state)

    _, final_acc = evaluate(model, test_loader, device)

    if verbose:
        print(f"\n  [Checkpoint restored — best validation accuracy]")
        print(f"  Final Test Accuracy  : {final_acc*100:.2f}%")
        print(f"  Final Sparsity Level : {model.overall_sparsity()*100:.2f}%")

    return model, history


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = ["#2196F3", "#FF5722", "#4CAF50"]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_gate_distribution(models: dict, best_lam: float, out_path: str):
    """
    Three-panel gate distribution figure.
    A successful result shows a large spike at 0 (pruned) and a smaller
    cluster of active gates away from 0 — a bimodal distribution.
    """
    best_m  = models[best_lam]
    gates   = best_m.all_gate_values()
    sp_pct  = (gates < 0.05).mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(
        f"Gate Value Distribution  |  λ = {best_lam}  "
        f"|  Sparsity = {sp_pct:.1f}%",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # (a) Full histogram — spike at 0 = pruned, cluster away = active
    ax = axes[0]
    _, bins, patches = ax.hist(gates, bins=80, color="#2196F3",
                                alpha=0.85, edgecolor="white", lw=0.2)
    for patch, left in zip(patches, bins[:-1]):
        if left < 0.05:
            patch.set_facecolor("#FF5722")
    ax.axvline(0.05, color="black", lw=1.5, linestyle="--",
               label="Prune threshold (0.05)")
    ax.legend(fontsize=9)
    zero  = (gates < 0.05).mean() * 100
    active = 100 - zero
    ax.text(0.40, 0.85,
            f"Pruned : {zero:.1f}%\nActive : {active:.1f}%\nTotal  : {len(gates):,}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    _style(ax, "All Gate Values (best model)", "Gate Value σ(g_ij)", "Count")

    # (b) Per-layer breakdown
    ax = axes[1]
    for i, (layer, name) in enumerate(zip(best_m._prunable, best_m._layer_names)):
        g = layer.all_gates_np()
        ax.hist(g, bins=60, alpha=0.65, color=PALETTE[i],
                label=f"{name}  ({(g<0.05).mean()*100:.1f}% pruned)")
    ax.axvline(0.05, color="black", lw=1.5, linestyle="--", alpha=0.8)
    ax.legend(fontsize=8)
    _style(ax, "Per-Layer Gate Distribution", "Gate Value", "Count")

    # (c) λ summary bar chart
    ax   = axes[2]
    lams = sorted(models.keys())
    sp_vals  = [models[l].overall_sparsity() * 100 for l in lams]
    acc_vals = [getattr(models[l], "_final_acc", 0.0) * 100 for l in lams]
    x = np.arange(len(lams)); w = 0.35
    b1 = ax.bar(x - w/2, sp_vals,  w, color=PALETTE[:len(lams)], alpha=0.9,
                label="Sparsity %")
    b2 = ax.bar(x + w/2, acc_vals, w, color=PALETTE[:len(lams)], alpha=0.4,
                label="Test Acc %", hatch="///", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={l}" for l in lams], fontsize=8)
    ax.legend(fontsize=9)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8, fontweight="bold")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8)
    _style(ax, "Lambda Trade-off Summary", "Lambda (λ)", "Percentage (%)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_training_dashboard(histories: dict, out_path: str):
    """6-panel training dashboard covering all key metrics across λ values."""
    lams   = sorted(histories.keys())
    colors = PALETTE[:len(lams)]
    epochs = range(1, len(next(iter(histories.values()))["val_acc"]) + 1)

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("Self-Pruning Neural Network — Training Dashboard",
                 fontsize=15, fontweight="bold", y=0.99)
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    panels = [
        ("Validation Accuracy",        "val_acc",    "Accuracy (%)", 100),
        ("Cross-Entropy Loss",          "ce_loss",    "CE Loss",       1),
        ("Sparsity Level",              "sparsity",   "Sparsity (%)", 100),
        ("Raw Sparsity Loss (Σ gates)", "sp_loss",    "Σ σ(g)",        1),
        ("Total Training Loss",         "train_loss", "CE + λ·Sp",     1),
    ]
    for ax, (title, key, ylabel, mul) in zip(axes[:5], panels):
        for lam, col in zip(lams, colors):
            vals = [v * mul for v in histories[lam][key]]
            ax.plot(epochs, vals, color=col, lw=2.2, label=f"λ={lam}")
        ax.legend(fontsize=8)
        _style(ax, title, "Epoch", ylabel)

    # Accuracy–Sparsity scatter
    ax = axes[5]
    for lam, col in zip(lams, colors):
        h   = histories[lam]
        acc = max(h["val_acc"]) * 100
        sp  = h["sparsity"][-1] * 100
        ax.scatter(sp, acc, color=col, s=200, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(f"λ={lam}", (sp, acc),
                    textcoords="offset points", xytext=(6, 5), fontsize=9)
    _style(ax, "Accuracy–Sparsity Trade-off",
           "Final Sparsity (%)", "Best Val Acc (%)")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_weight_heatmaps(models: dict, out_path: str):
    """Visualise effective pruned weight magnitudes of FC1 per λ."""
    lams = sorted(models.keys())
    fig, axes = plt.subplots(1, len(lams), figsize=(5 * len(lams), 4.8))
    if len(lams) == 1:
        axes = [axes]
    fig.suptitle("Effective Pruned Weight Magnitude  |FC1|  per λ",
                 fontsize=13, fontweight="bold")
    for ax, lam in zip(axes, lams):
        m = models[lam]
        with torch.no_grad():
            eff = (m.fc1.weight * torch.sigmoid(m.fc1.gate_scores)
                   ).cpu().abs().numpy()
        rows = min(64, eff.shape[0])
        cols = min(64, eff.shape[1])
        im   = ax.imshow(eff[:rows, :cols], aspect="auto",
                         cmap="viridis", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"λ = {lam}\nFC1 Sparsity = {m.fc1.sparsity()*100:.1f}%",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Input features")
        ax.set_ylabel("Output neurons")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network on CIFAR-10"
    )
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--data-dir",   type=str,   default="./data")
    parser.add_argument("--out-dir",    type=str,   default="./outputs")
    parser.add_argument(
        "--lambdas", type=float, nargs="+",
        default=[1e-4, 1e-3, 1e-2],
        help="Lambda values (low, medium, high)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "═"*70)
    print("   SELF-PRUNING NEURAL NETWORK  ·  CIFAR-10  ·  TREDENCE 2025")
    print("═"*70)
    print(f"  Device    : {device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  LR        : {args.lr}")
    print(f"  Batch     : {args.batch_size}")
    print(f"  Lambda    : {args.lambdas}")
    print(f"  Output    : {args.out_dir}/")
    print("═"*70)

    print("\n  Loading CIFAR-10 ...")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    all_models: Dict[float, SelfPruningNet] = {}
    all_histories: Dict[float, dict]        = {}
    all_results: List[dict]                 = []

    for lam in args.lambdas:
        model, history = train(
            lambda_sp    = lam,
            train_loader = train_loader,
            test_loader  = test_loader,
            epochs       = args.epochs,
            lr           = args.lr,
            device       = device,
            seed         = args.seed,
        )

        _, final_acc = evaluate(model, test_loader, device)
        sparsity     = model.overall_sparsity()
        total_gates  = sum(l.gate_scores.numel() for l in model._prunable)
        pruned_wts   = int(sparsity * total_gates)

        model._final_acc        = final_acc
        all_models[lam]         = model
        all_histories[lam]      = history
        all_results.append(dict(
            lambda_val   = lam,
            test_acc     = final_acc,
            best_val_acc = max(history["val_acc"]),
            sparsity     = sparsity,
            pruned_wts   = pruned_wts,
            total_gates  = total_gates,
        ))

        torch.save(
            model.state_dict(),
            os.path.join(args.out_dir, f"model_lambda_{lam:.5f}.pt"),
        )

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    best_lam = args.lambdas[len(args.lambdas) // 2]

    plot_gate_distribution(
        all_models, best_lam,
        os.path.join(args.out_dir, "gate_distribution.png"),
    )
    plot_training_dashboard(
        all_histories,
        os.path.join(args.out_dir, "training_dashboard.png"),
    )
    plot_weight_heatmaps(
        all_models,
        os.path.join(args.out_dir, "weight_heatmaps.png"),
    )

    # ── Results table ──────────────────────────────────────────────────────
    print("\n" + "═"*75)
    print("  RESULTS SUMMARY")
    print("═"*75)
    print(f"  {'Lambda':>10}  {'Test Acc':>10}  {'Sparsity':>10}  "
          f"{'Pruned/Total':>16}  {'Best Acc':>10}")
    print("─"*75)
    for r in all_results:
        print(
            f"  {r['lambda_val']:>10.5f}  {r['test_acc']*100:>9.2f}%  "
            f"{r['sparsity']*100:>9.2f}%  "
            f"  {r['pruned_wts']:>7,}/{r['total_gates']:,}  "
            f"{r['best_val_acc']*100:>9.2f}%"
        )
    print("═"*75)

    # ── CSV ────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write("lambda,test_accuracy_pct,sparsity_pct,best_val_acc_pct,"
                "pruned_weights,total_gates\n")
        for r in all_results:
            f.write(
                f"{r['lambda_val']},{r['test_acc']*100:.2f},"
                f"{r['sparsity']*100:.2f},{r['best_val_acc']*100:.2f},"
                f"{r['pruned_wts']},{r['total_gates']}\n"
            )
    print(f"\n  Results CSV : {csv_path}")
    print("\n" + "═"*70)
    print("  All experiments complete. Outputs saved to:", args.out_dir)
    print("═"*70 + "\n")


if __name__ == "__main__":
    main()