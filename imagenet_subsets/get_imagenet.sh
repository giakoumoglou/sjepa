#!/usr/bin/env bash
# =============================================================================
# Downloads and prepares ImageNet ILSVRC 2012 into:
#   ./imagenet/train/<synset>/*.JPEG   (1,281,167 images)
#   ./imagenet/val/<synset>/*.JPEG     (   50,000 images)
#
# Usage: chmod +x imagenet_subsets/get_imagenet.sh && ./imagenet_subsets/get_imagenet.sh
# =============================================================================

set -euo pipefail

TRAIN_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

mkdir -p imagenet/train imagenet/val

# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
echo "==> Downloading train set (~138 GB)..."
wget --continue --show-progress -O imagenet/ILSVRC2012_img_train.tar "$TRAIN_URL"

echo "==> Downloading val set (~6.3 GB)..."
wget --continue --show-progress -O imagenet/ILSVRC2012_img_val.tar "$VAL_URL"

# ---------------------------------------------------------------------------
# 2. Extract training set, then unpack each per-synset tar into its folder
# ---------------------------------------------------------------------------
echo "==> Extracting train set..."
tar -xf imagenet/ILSVRC2012_img_train.tar -C imagenet/train
rm -f imagenet/ILSVRC2012_img_train.tar

echo "==> Extracting per-class tars..."
find imagenet/train -maxdepth 1 -name "*.tar" | while read -r class_tar; do
  synset="$(basename "${class_tar%.tar}")"
  mkdir -p "imagenet/train/$synset"
  tar -xf "$class_tar" -C "imagenet/train/$synset"
  rm -f "$class_tar"
done

# ---------------------------------------------------------------------------
# 3. Extract val set, then sort images into per-class subdirectories
# ---------------------------------------------------------------------------
echo "==> Extracting val set..."
tar -xf imagenet/ILSVRC2012_img_val.tar -C imagenet/val
rm -f imagenet/ILSVRC2012_img_val.tar

echo "==> Organising val images into class subdirectories..."
cd imagenet/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd -

# ---------------------------------------------------------------------------
# 4. Verify
# ---------------------------------------------------------------------------
echo ""
echo "Total files in train should be 1281167, got:"
find imagenet/train -name "*.JPEG" | wc -l

echo "Total files in val should be 50000, got:"
find imagenet/val -name "*.JPEG" | wc -l

echo ""
echo "Done. ImageNet is ready under ./imagenet/"
