#!/usr/bin/env python
"""
env/setup_pyg_colab.py

Set up a stable PyTorch Geometric stack in Colab:

- Use the *existing* torch version (don’t override Colab’s).
- Install torch-geometric + ops from the matching PyG wheel index.
- Optionally cache wheels in Drive for later reuse.
"""

import os
import sys
import subprocess

def main(cache_wheels: bool = True):
    import torch  # use whatever Colab provides
    torch_ver = torch.__version__
    print(f"Using existing torch: {torch_ver}")

    pyg_index = f"https://data.pyg.org/whl/torch-{torch_ver}.html"
    print(f"Using PyG wheel index: {pyg_index}")

    pkgs = [
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    ]

    # 1) Install directly from PyG index (fast, prebuilt wheels)
    cmd = [
        sys.executable, "-m", "pip", "install",
        *pkgs,
        "-f", pyg_index,
    ]
    print("\n🚀 Installing PyG stack:\n", " ".join(cmd))
    subprocess.check_call(cmd)

    # 2) Optionally cache the wheels for later
    if cache_wheels:
        wheel_dir = os.path.expanduser(
            f"/content/drive/MyDrive/wheelhouse/torch_{torch_ver}"
        )
        os.makedirs(wheel_dir, exist_ok=True)
        print(f"\n📦 Caching wheels into: {wheel_dir}")

        for pkg in pkgs:
            cmd = [
                sys.executable, "-m", "pip", "download",
                "--only-binary", ":all:",
                "-f", pyg_index,
                "-d", wheel_dir,
                pkg,
            ]
            print("Downloading:", " ".join(cmd))
            subprocess.check_call(cmd)

        print("\n✅ Wheel cache contents:")
        print("\n".join(sorted(os.listdir(wheel_dir))))

    print("\n✅ PyG environment ready.")

if __name__ == "__main__":
    main(cache_wheels=True)
