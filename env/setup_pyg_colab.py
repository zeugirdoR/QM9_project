#!/usr/bin/env python3
# Usage from notebook:
# !python env/setup_pyg_colab.py

"""
env/setup_pyg_colab.py

Simple, robust setup script for PyTorch Geometric on Colab (or similar).
- Prints torch / CUDA info.
- Tries to install PyG stack.
- Optionally uses a local wheel dir *if* it already has wheels.
- Does NOT try to build wheels itself (because that often fails
  for bleeding-edge torch versions).
"""

import os
import sys
import subprocess

WHEEL_DIR = "/content/drive/MyDrive/GAHEAD_wheels"  # change if you want


def run(cmd):
    print("\n⚙️  Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def ensure_pyg():
    import torch  # noqa: F401

    torch_ver = torch.__version__
    print(f"Torch version: {torch_ver}")

    # Try to see if torch has a CUDA suffix like '+cu126'
    if "+cu" in torch_ver:
        base_ver, cuda_tag = torch_ver.split("+", 1)
    else:
        base_ver, cuda_tag = torch_ver, "cpu"

    print(f"Detected base torch version: {base_ver}, CUDA tag: {cuda_tag}")

    # PyG wheels are usually hosted at a URL like:
    #   https://data.pyg.org/whl/torch-<base_ver>+<cuda_tag>.html
    # but for future torch versions this may not exist. That's ok:
    # pip will just fall back to PyPI / SDists.
    pyg_index = f"https://data.pyg.org/whl/torch-{base_ver}+{cuda_tag}.html"
    print(f"PyG wheel index (may or may not exist): {pyg_index}")

    pkgs = [
        "pyg-lib",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "torch-geometric",
    ]

    # Prefer installing from your wheel dir if it's already populated.
    use_local_wheels = os.path.isdir(WHEEL_DIR) and any(
        name.endswith(".whl") for name in os.listdir(WHEEL_DIR)
    )
    if use_local_wheels:
        print(f"\n📦 Local wheel dir found with .whl files: {WHEEL_DIR}")
        print("   Trying to install from local wheels first...")
        for pkg in pkgs:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "-f",
                WHEEL_DIR,
                pkg,
            ]
            try:
                run(cmd)
            except subprocess.CalledProcessError:
                print(f"⚠️  Failed to install {pkg} from local wheels, will try online.")
    else:
        print(f"\nℹ️  No usable .whl files yet under {WHEEL_DIR}.")
        print("    Will install directly from online indexes (this can be slow).")

    # Online install as fallback / default
    for pkg in pkgs:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            pkg,
            "-f",
            pyg_index,
        ]
        try:
            run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Online install for {pkg} failed: {e}")
            print("   You may need a different torch/PyG version combo.")
            # We don't re-raise here so that one failure doesn't kill the whole script.

    # Final sanity print
    import torch
    print("\n✅ Final versions:")
    print("  torch:", torch.__version__)
    try:
        import torch_geometric

        print("  torch_geometric:", torch_geometric.__version__)
    except Exception as e:
        print("  torch_geometric: NOT IMPORTABLE:", repr(e))


if __name__ == "__main__":
    print("🔧 Setting up PyG environment...")
    ensure_pyg()
    print("\n✅ setup_pyg_colab.py finished.")
