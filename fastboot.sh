%%bash
set -e
WHEEL_DIR="/content/drive/MyDrive/wheelhouse/cu126_torch2.8_pyg2.7"

echo "🚀 Fast boot from wheelhouse: $WHEEL_DIR"

python -m pip install --no-index --find-links="$WHEEL_DIR" \
  torch torchvision torchaudio \
  torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
