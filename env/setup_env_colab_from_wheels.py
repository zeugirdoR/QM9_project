# env/setup_env_colab_from_wheels.py

import os
from google.colab import drive

drive.mount("/content/drive")

WHEEL_DIR = "/content/drive/MyDrive/wheelhouse/cu126_torch2.8_pyg2.7"
os.environ["WHEEL_DIR"] = WHEEL_DIR

cmd = f"pip install --no-index --find-links={WHEEL_DIR} " \
      "torch torchvision torchaudio torch-geometric " \
      "torch-scatter torch-sparse torch-cluster torch-spline-conv"

print("Running:\n ", cmd)
os.system(cmd)
