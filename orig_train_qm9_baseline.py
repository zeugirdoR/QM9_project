#!/usr/bin/env python

import argparse
from dataclasses import dataclass
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool


# =========================
# Config
# =========================

@dataclass
class Config:
    data_root: str = "./data/qm9"
    target_idx: int = 7        # QM9 property index (0–18)
    batch_size: int = 96
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 0       # keep 0 for Colab stability / reproducibility
    hidden_dim: int = 128
    num_layers: int = 6
    experiment_name: str = "qm9_baseline_v1"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Model: simple GCN baseline
# =========================

class GCNNet(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 6, max_z: int = 100):
        super().__init__()
        self.emb = nn.Embedding(max_z, hidden_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        # data: PyG Data with .z, .edge_index, .batch
        x = self.emb(data.z.long())
        edge_index = data.edge_index
        batch = data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # graph-level pooling
        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.view(-1)  # (num_graphs,)


# =========================
# Data: QM9 with raw targets
# =========================

def get_loaders(cfg: Config):
    # Load QM9 with *no* extra target preprocessing.
    # QM9(target=i) makes .y be that column only, in its native units.
    dataset = QM9(root=cfg.data_root, target=cfg.target_idx)

    # Deterministic split matching your earlier counts.
    n_total = len(dataset)
    n_train = 109_898
    n_val = 13_083
    assert n_train + n_val <= n_total, "Split sizes too large for QM9."

    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    print(
        f"Train graphs: {len(train_dataset)}, "
        f"Val graphs: {len(val_dataset)}, "
        f"Test graphs: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader


# =========================
# Eval & training helpers
# =========================

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device):
    model.eval()
    total_abs = 0.0
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)               # (B,)
        y = batch.y.view(-1).to(device)   # raw QM9 target in its native units
        total_abs += (pred - y).abs().sum().item()
        n_graphs += batch.num_graphs
    mae = total_abs / n_graphs
    return mae


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(-1).to(device)
        loss = F.l1_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs

    return total_loss / n_graphs


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Simple, fully reproducible QM9 trainer (no fancy preprocessing)."
    )
    parser.add_argument("--data_root", type=str, default="./data/qm9")
    parser.add_argument("--target_idx", type=int, default=7, help="QM9 target index (0–18).")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--experiment_name", type=str, default="qm9_baseline_v1")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        target_idx=args.target_idx,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=0,
    )

    ensure_dir(cfg.checkpoint_dir)
    ensure_dir(cfg.log_dir)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # 1) Data
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # 2) Model
    model = GCNNet(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # 3) Logging
    log_path = os.path.join(cfg.log_dir, f"{cfg.experiment_name}.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_L1,val_MAE\n")

    best_val = float("inf")
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.experiment_name}_best.pt")

    print("\n🚀 Starting training")
    print("ep  |  train_L1  |  val_MAE  |  best_MAE")
    print("------------------------------------------")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_L1 = train_one_epoch(model, train_loader, optimizer, device)
        val_MAE = evaluate(model, val_loader, device)

        improved = val_MAE <= best_val
        if improved:
            best_val = val_MAE

        # log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_L1:.6f},{val_MAE:.6f}\n")

        # checkpoint if improved
        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "best_val": best_val,
                },
                best_ckpt_path,
            )
            mark = "⭐"
        else:
            mark = " "

        # Print in native units (eV) and meV for convenience
        print(
            f"{epoch:3d} | {train_L1:9.4f} | {val_MAE:8.4f} eV "
            f"({val_MAE*1000:8.1f} meV) | {best_val*1000:8.1f} meV {mark} "
            f"[{time.time() - t0:5.1f}s]"
        )

    print("\n✅ Training done.")
    print(f"Best val MAE: {best_val:.6f} eV ({best_val*1000:.2f} meV)")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
