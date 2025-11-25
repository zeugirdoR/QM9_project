# Next time you spin up a fresh Colab and want the exact same stack,
#  your environment bootstrap can be:

from google.colab import drive
drive.mount('/content/drive')

WHEEL_DIR = "/content/drive/MyDrive/wheelhouse/cu126_torch2.8_pyg2.7"

# Install EVERYTHING from your Drive wheelhouse
!pip install --no-index --find-links="$WHEEL_DIR" \
    torch torchvision torchaudio torch-geometric \
    torch-scatter torch-sparse torch-cluster torch-spline-conv
