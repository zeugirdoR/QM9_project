#!/usr/bin/env python
"""
Small utility to print out the QM9 environment summary.

You can run:
  python env/print_qm9_env.py

from the root of the repo (QM9_project/).
"""

import importlib
import subprocess
import sys

def safe_import(name):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        return mod, ver
    except Exception as e:
        return None, f"NOT INSTALLED ({e.__class__.__name__})"


def main():
    print("=" * 60)
    print("QM9 PROJECT ENVIRONMENT SUMMARY")
    print("=" * 60)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print()

    # ---- torch ----
    torch, torch_ver = safe_import("torch")
    print(f"torch             : {torch_ver}")
    if torch is not None:
        print(f"  CUDA available  : {torch.cuda.is_available()}")
        print(f"  torch.version.cuda : {torch.version.cuda}")
        try:
            if torch.cuda.is_available():
                print(f"  CUDA device count : {torch.cuda.device_count()}")
                print(f"  CUDA device 0     : {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"  (error querying CUDA devices: {e})")
    print()

    # ---- basic GPU info via nvidia-smi (if present) ----
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        print("nvidia-smi GPU info:")
        print(out)
        print()
    except Exception as e:
        print(f"nvidia-smi        : not available ({e.__class__.__name__})")
        print()

    # ---- torch_geometric + ops ----
    tg, tg_ver = safe_import("torch_geometric")
    print(f"torch_geometric   : {tg_ver}")
    for name in ["torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]:
        _, ver = safe_import(name)
        print(f"{name:16} : {ver}")
    print()

    # ---- other useful libs ----
    for name in ["numpy", "scipy", "tqdm", "psutil", "aiohttp"]:
        _, ver = safe_import(name)
        print(f"{name:16} : {ver}")
    print()

    # ---- Git commit (if repo) ----
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        print(f"Git commit        : {commit}")
    except Exception as e:
        print(f"Git commit        : <unavailable> ({e.__class__.__name__})")

    print("=" * 60)


if __name__ == "__main__":
    main()
