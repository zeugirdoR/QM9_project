#!/usr/bin/env python3
"""
data/qm9_data.py

Minimal, reproducible QM9 data loader for the GAHEAD / micro model.

- Uses torch_geometric.datasets.QM9
- Does only the simplest target preprocessing:
    y = data.y[:, target_idx]
    if per_atom: y = y / n_atoms
- Fixed, documented splits so the results are reproducible.

This file is *the* source of truth for how the target is defined.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


@dataclass
class QM9Config:
    root: str = "./data/qm9"
    batch_size: int = 256
    num_workers: int = 2

    # Target index in QM9.y (0..18). By convention, U0 is 7.
    target_idx: int = 7

    # If True, train on per-atom target: y / n_atoms
    per_atom: bool = False

    # Fixed split sizes, matching your “Train graphs: 109898, Val: 13083”
    n_train: int = 109898
    n_val: int = 13083
    # test size will be len(dataset) - n_train - n_val


def load_qm9_dataset(cfg: QM9Config):
    dataset = QM9(cfg.root)
    n = len(dataset)
    n_train = cfg.n_train
    n_val = cfg.n_val
    n_test = n - n_train - n_val

    assert n_train + n_val <= n, f"Requested splits exceed dataset size: {n}"

    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]

    print(f"QM9 loaded from {cfg.root}")
    print(f"  Total graphs: {n}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def _make_loader(dataset, cfg: QM9Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
    )


def get_qm9_loaders(cfg: QM9Config):
    train_ds, val_ds, test_ds = load_qm9_dataset(cfg)
    train_loader = _make_loader(train_ds, cfg, shuffle=True)
    val_loader = _make_loader(val_ds, cfg, shuffle=False)
    test_loader = _make_loader(test_ds, cfg, shuffle=False)
    return train_loader, val_loader, test_loader


def extract_target(batch, cfg: QM9Config) -> torch.Tensor:
    """
    Given a PyG batch, return the training target tensor y of shape (B, 1).

    This is where we encode the *entire* meaning of the supervised task:

        y = y_raw[:, target_idx]
        if per_atom: y = y / n_atoms

    No other shifts or magic happen here.
    """
    y_raw = batch.y  # shape: (B, T)
    if y_raw.dim() != 2:
        raise RuntimeError(f"Expected batch.y to have shape (B,T), got {y_raw.shape}")

    y = y_raw[:, cfg.target_idx:cfg.target_idx + 1]  # (B,1)

    if cfg.per_atom:
        # number of atoms per graph: count of nodes per graph
        # batch.batch: (total_nodes,) with graph indices
        batch_idx = batch.batch  # e.g. 0,0,0,1,1,2,2,2,...
        num_atoms_per_graph = torch.bincount(batch_idx)
        if num_atoms_per_graph.numel() != y.size(0):
            raise RuntimeError(
                f"num_atoms_per_graph has shape {num_atoms_per_graph.shape}, "
                f"but batch size is {y.size(0)}"
            )
        y = y / num_atoms_per_graph.unsqueeze(-1)  # broadcast (B,1)

    return y
