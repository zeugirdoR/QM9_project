#!/usr/bin/env python3
"""
train_qm9_baseline.py

Very simple, fully reproducible QM9 baseline trainer for V20_AGAA_Motor:

- Uses QM9Config / get_qm9_loaders / extract_target from data.qm9_data.
- Uses V20_AGAA_Motor from models.v20_agaa_micro.
- No atom reference energies, no hidden shifts.
- Loss = L1(pred, target), where target is defined *only* in qm9_data.py.
- Logs MAE in eV and meV.

Run from repo root:

    python train_qm9_baseline.py --epochs 50 --target-idx 7 --per-atom 1

Later we can add a Phase B script that turns on motor regularization,
but this is the “Phase A / data-only” core.
"""

import argparse
import os
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from data.qm9_data import QM9Config, get_qm9_loaders, extract_target
from models.v20_agaa_micro import V20_AGAA_Motor


# ---------------------------------------------------------------------
# 1. Reproducibility helpers
# ---------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# 2. Dense batch helper (from PyG batch → (z, pos, mask))
# ---------------------------------------------------------------------


def batch_to_dense(batch, device):
    """
    Convert a torch_geometric Batch into dense (z, pos, mask).

    - z:    (B, N, 1) or (B, N) depending on your model’s expectations.
    - pos:  (B, N, 3)
    - mask: (B, N) boolean

    We *do not* fix N=29; we let to_dense_batch() pick the max nodes
    in this batch. V20_AGAA_Motor is expected to handle variable N
    via the mask.
    """
    z = batch.z.to(device)        # (total_nodes,)
    pos = batch.pos.to(device)    # (total_nodes, 3)
    batch_idx = batch.batch.to(device)  # (total_nodes,)

    # to_dense_batch: (total_nodes, F) → (B, N, F)
    z_dense, mask = to_dense_batch(z.unsqueeze(-1), batch_idx)  # (B, N, 1), (B, N)
    pos_dense, _ = to_dense_batch(pos, batch_idx)               # (B, N, 3), mask reused

    return z_dense, pos_dense, mask


# ---------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------


def eval_epoch(model, loader, cfg: QM9Config, device, desc: str = "VAL"):
    model.eval()
    total_abs = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            # 1) target from qm9_data (this defines the supervised task)
            y = extract_target(batch, cfg).to(device)  # (B,1)

            # 2) dense inputs
            z_dense, pos_dense, mask = batch_to_dense(batch, device)

            # 3) forward
            out = model(z_dense, pos_dense, mask)
            # Allow for model to return pred or (pred, sig)
            if isinstance(out, tuple):
                pred, _ = out
            else:
                pred = out

            if pred.dim() == 2 and pred.size(1) != 1:
                raise RuntimeError(f"Expected pred shape (B,1), got {pred.shape}")

            diff = (pred - y).abs()  # (B,1)
            total_abs += diff.sum().item()
            total_count += diff.numel()

    mae_eV = total_abs / max(1, total_count)
    mae_meV = mae_eV * 1000.0
    print(f"[{desc}] MAE = {mae_meV:10.2f} meV ({mae_eV:7.4f} eV)")
    return mae_eV, mae_meV


# ---------------------------------------------------------------------
# 4. Training loop (Phase A: data-only)
# ---------------------------------------------------------------------


def train_baseline(
    epochs: int,
    lr: float,
    weight_decay: float,
    cfg: QM9Config,
    device: torch.device,
    save_path: str,
):
    print("Using device:", device)

    # Data
    train_loader, val_loader, _ = get_qm9_loaders(cfg)
    print(f"Train graphs: {len(train_loader.dataset)}, Val graphs: {len(val_loader.dataset)}")

    # Model
    model = V20_AGAA_Motor(
        num_layers=7,
        d_model=192,
        n_heads=16,
        max_z=100,
        n_rbf=20,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: V20-AGAA-Micro | Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Optional resume
    best_mae_meV = None
    if os.path.exists(save_path):
        print(f"ℹ️  Found existing checkpoint at {save_path}, loading for resume...")
        state = torch.load(save_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
            best_mae_meV = state.get("best_mae_meV", None)
        else:
            model.load_state_dict(state)
        print("   Resume complete. Best val MAE (meV):", best_mae_meV)

    # Initial eval
    print("\n🔍 Sanity check: eval before training")
    _, mae_meV0 = eval_epoch(model, val_loader, cfg, device, desc="VAL-before")

    print("\n🚀 Starting baseline training (Phase A, data-only)")
    print("ep  | train_L1(eV) | val_MAE_meV | best_MAE_meV")
    print("----+-------------+------------+-------------")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            y = extract_target(batch, cfg).to(device)  # (B,1)
            z_dense, pos_dense, mask = batch_to_dense(batch, device)

            out = model(z_dense, pos_dense, mask)
            if isinstance(out, tuple):
                pred, _ = out
            else:
                pred = out

            loss = F.l1_loss(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_L1 = total_loss / max(1, n_batches)

        _, val_mae_meV = eval_epoch(
            model, val_loader, cfg, device, desc=f"VAL-ep{ep}"
        )

        mark = ""
        if best_mae_meV is None or val_mae_meV < best_mae_meV:
            best_mae_meV = val_mae_meV
            # save checkpoint
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "best_mae_meV": best_mae_meV,
                    "epoch": ep,
                },
                save_path,
            )
            mark = "⭐"

        print(
            f"{ep:3d} | {train_L1:11.4f} | {val_mae_meV:10.2f} | {best_mae_meV:11.2f} {mark}"
        )

    print("\n✅ Training finished. Best val MAE:", best_mae_meV, "meV")
    print("   Checkpoint saved to:", save_path)


# ---------------------------------------------------------------------
# 5. CLI entrypoint
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="V20-AGAA-Micro QM9 baseline trainer")
    parser.add_argument("--root", type=str, default="./data/qm9", help="QM9 root dir")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="# DataLoader workers")
    parser.add_argument(
        "--target-idx", type=int, default=7, help="QM9 target index (e.g. U0=7)"
    )
    parser.add_argument(
        "--per-atom",
        type=int,
        default=0,
        help="1 = divide target by #atoms (per-atom training), 0 = raw",
    )
    parser.add_argument("--epochs", type=int, default=50, help="# training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="AdamW weight decay"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./checkpoints/v20_agaa_micro_baseline.pt",
        help="checkpoint path",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    cfg = QM9Config(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_idx=args.target_idx,
        per_atom=bool(args.per_atom),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_baseline(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cfg=cfg,
        device=device,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
