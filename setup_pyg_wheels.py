import os, sys, textwrap, subprocess, torch

# Where to store wheels (persistent on Drive)
WHEEL_DIR = "/content/drive/MyDrive/GAHEAD_wheels"
os.makedirs(WHEEL_DIR, exist_ok=True)
print("Wheel cache dir:", WHEEL_DIR)

print("Torch version:", torch.__version__)
print("Torch CUDA   :", torch.version.cuda)

# Build the torch+cuda tag for the PyG wheel index
torch_base = torch.__version__.split('+')[0]   # e.g. '2.3.0'
cuda_raw   = torch.version.cuda or "12.1"      # fallback if None
cuda_tag   = "cu" + cuda_raw.replace(".", "")  # e.g. 'cu121'
index_url  = f"https://data.pyg.org/whl/torch-{torch_base}+{cuda_tag}.html"

print("Using PyG wheel index:", index_url)

# Build wheels once into WHEEL_DIR
pkgs = [
    "pyg-lib",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torch-spline-conv",
    "torch-geometric",
]

cmd = [
    sys.executable,
    "-m", "pip", "wheel",
    "-w", WHEEL_DIR,
    "-f", index_url,
] + pkgs

print("\n⚙️ Running:\n", " ".join(cmd))
subprocess.check_call(cmd)
print("\n✅ Done building wheels into", WHEEL_DIR)
