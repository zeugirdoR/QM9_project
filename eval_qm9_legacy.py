#!/usr/bin/env python3
"""
eval_qm9_legacy.py

Evaluate a V20_AGAA_Motor checkpoint on QM9 under various “recipes”
(target_idx, atomization shift, per-atom) and calibrate a legacy
metric so that the 91.7 meV run is recoverable (up to a scale factor).

Usage (from QM9_project/):

    python eval_qm9_legacy.py \
        --ckpt /content/v20_agaa_micro_phaseA_data_only.pt \
        --target-idx 9 \
        --root ./data/qm9

Assumptions:
- data.qm9_data provides Config_Micro (or QM9Config) + get_qm9_loaders.
- models.v20_agaa_micro provides V20_AGAA_Motor with forward(z, pos, mask)
  returning either pred or (pred, sig).
- cfg.atom_ref, cfg.energy_mean, cfg.energy_std exist if you trained that way
  (if not, see comments below).
"""

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from data.qm9_data import Config_Micro, get_qm9_loaders  # or QM9Config if you renamed
from models.v20_agaa_micro import V20_AGAA_Motor


# --------------------------------------------------
# 1. Helper: dense-ify a PyG batch
# --------------------------------------------------

def densify_batch(batch, cfg, device):
    """
    Turn a torch_geometric Batch into dense (z, pos, mask, y_raw).

    - z_flat: (total_nodes,)
    - pos_flat: (total_nodes, 3)
    - batch_idx: (total_nodes,)
    - y_raw: (num_graphs, T)

    Returns:
      z_dense:  (B, N, 1)
      pos:      (B, N, 3)
      mask:     (B, N)
      y_raw:    (B, T)
    """
    z_flat = batch.z.to(device)
    pos_flat = batch.pos.to(device)
    y_raw = batch.y.to(device)
    batch_idx = batch.batch.to(device)

    # B = number of graphs, N = max nodes in this batch
    z_dense, mask = to_dense_batch(z_flat.unsqueeze(-1), batch_idx,
                                   max_num_nodes=getattr(cfg, "max_atoms", None))
    pos_dense, _ = to_dense_batch(pos_flat, batch_idx,
                                  max_num_nodes=getattr(cfg, "max_atoms", None))

    # y_raw should be (B, T)
    if y_raw.dim() == 3:
        y_raw = y_raw.squeeze(1)
    elif y_raw.dim() == 1:
        y_raw = y_raw.unsqueeze(-1)

    return z_dense, pos_dense, mask, y_raw


# --------------------------------------------------
# 2. Core evaluation with a given “recipe”
# --------------------------------------------------

def eval_with_recipe(
    model,
    data_loader,
    cfg,
    device,
    target_idx=9,
    use_atom_shift=True,
    divide_by_natoms=True,
):
    """
    Evaluate a checkpoint under a concrete recipe:

      - target_idx: which QM9 property in y to use
      - use_atom_shift: subtract atom reference energies or not
      - divide_by_natoms: per-atom or per-molecule

    Returns:
      (mae_meV, mae_eV) in *current pipeline units*.
    """
    model.eval()
    total_abs = 0.0
    total_count = 0

    atom_refs = getattr(cfg, "atom_ref", None)
    if atom_refs is not None:
        atom_refs = atom_refs.to(device)

    ENERGY_MEAN = getattr(cfg, "energy_mean", 0.0)
    ENERGY_STD = getattr(cfg, "energy_std", 1.0)

    with torch.no_grad():
        for batch in data_loader:
            z, pos, mask, y_raw = densify_batch(batch, cfg, device)

            # pick target
            y = y_raw[:, target_idx:target_idx + 1]  # (B,1)

            # undo standardization if it was used in training
            # (if you did NOT standardize, ENERGY_STD should be 1 and MEAN 0)
            y = y * ENERGY_STD + ENERGY_MEAN

            # natoms per graph
            natoms = mask.sum(dim=1, keepdim=True)  # (B,1)

            # per-atom vs per-molecule
            if divide_by_natoms:
                y = y / natoms.clamp(min=1.0)

            # atomization shift
            if use_atom_shift and atom_refs is not None:
                z_int = z.long().squeeze(-1)  # (B,N)
                ref = atom_refs[z_int].sum(dim=1, keepdim=True)  # (B,1)
                if divide_by_natoms:
                    ref = ref / natoms.clamp(min=1.0)
                y = y - ref

            # model prediction
            out = model(z, pos, mask)
            if isinstance(out, tuple):
                pred, _ = out
            else:
                pred = out

            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)

            diff = (pred - y).abs()
            total_abs += diff.sum().item()
            total_count += diff.numel()

    mae_eV = total_abs / max(total_count, 1)
    mae_meV = mae_eV * 1000.0
    return mae_meV, mae_eV


# --------------------------------------------------
# 3. Calibrate a “legacy” metric so 91.7 meV can be recovered
# --------------------------------------------------

def calibrate_legacy_scale(model, loader_val, cfg, device,
                           target_idx=9,
                           legacy_ref_meV=91.7):
    """
    Compute a scale factor so that:

        legacy_metric = scale_factor * current_metric

    and the 91.7-meV checkpoint maps to ≈ 91.7 meV under the legacy metric.
    """
    mae_meV, mae_eV = eval_with_recipe(
        model,
        loader_val,
        cfg,
        device,
        target_idx=target_idx,
        use_atom_shift=True,
        divide_by_natoms=True,
    )

    scale = legacy_ref_meV / mae_meV

    print("\n🔎 Calibrating legacy metric...")
    print(f"  Current pipeline MAE (idx={target_idx}, shift=True, per_atom=True): "
          f"{mae_meV:.2f} meV")
    print(f"  Desired legacy MAE for this checkpoint:                 "
          f"{legacy_ref_meV:.2f} meV")
    print(f"  ⇒ scale_factor = {scale:.6f}")
    print(f"  Sanity: legacy ≈ {mae_meV * scale:.2f} meV "
          f"(should be ≈ {legacy_ref_meV:.2f})")

    return scale


def eval_legacy(model, loader_val, cfg, device,
                target_idx=9, scale_factor=1.0):
    """
    Evaluate both:
      - new_metric_meV  (current pipeline)
      - legacy_metric_meV = scale_factor * new_metric_meV
    """
    mae_meV_new, _ = eval_with_recipe(
        model,
        loader_val,
        cfg,
        device,
        target_idx=target_idx,
        use_atom_shift=True,
        divide_by_natoms=True,
    )
    mae_meV_old = mae_meV_new * scale_factor
    return mae_meV_new, mae_meV_old


# --------------------------------------------------
# 4. CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate V20 AGAA Micro on QM9 with legacy metric calibration."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (e.g. v20_agaa_micro_phaseA_data_only.pt)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/qm9",
        help="Root directory for QM9 data",
    )
    parser.add_argument(
        "--target-idx",
        type=int,
        default=9,
        help="QM9 target index used in 'good' run (e.g. 7, 8, or 9)",
    )
    parser.add_argument(
        "--legacy-ref-meV",
        type=float,
        default=91.7,
        help="Reference legacy MAE (e.g. 91.7 meV) to calibrate against",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Config + loaders
    cfg = Config_Micro()  # or QM9Config(...) if that’s what you use
    cfg.root = args.root
    loader_train, loader_val = get_qm9_loaders(cfg)

    print(f"Train graphs: {len(loader_train.dataset)}, "
          f"Val graphs: {len(loader_val.dataset)}")

    # Model
    model = V20_AGAA_Motor(
        num_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        max_z=cfg.max_z,
        n_rbf=cfg.n_rbf,
    ).to(device)

    # Load checkpoint
    ckpt_path = args.ckpt
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"\n✅ Loaded micro checkpoint from {ckpt_path}")
    print(f"   Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # Calibrate scale factor
    scale = calibrate_legacy_scale(
        model,
        loader_val,
        cfg,
        device,
        target_idx=args.target_idx,
        legacy_ref_meV=args.legacy_ref_meV,
    )

    # Final legacy-aligned eval on val set
    mae_new, mae_legacy = eval_legacy(
        model,
        loader_val,
        cfg,
        device,
        target_idx=args.target_idx,
        scale_factor=scale,
    )

    print(f"\n[VAL] new_metric = {mae_new:.2f} meV | "
          f"legacy_metric ≈ {mae_legacy:.2f} meV "
          f"(target ≈ {args.legacy_ref_meV:.2f} meV)")


if __name__ == "__main__":
    main()
