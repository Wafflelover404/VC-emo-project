#!/bin/bash

source env/bin/activate

MODEL_PATH=${MODEL_PATH:-wafflelover404_emotion_model.pth}

PRESET=${PRESET:-}
if [ -z "$PRESET" ]; then
  if [ "${FULL:-0}" = "1" ]; then
    PRESET=full
  else
    PRESET=fast
  fi
fi

if [ "$PRESET" = "full" ]; then
  export FAST=0
  export IMG_SIZE=${IMG_SIZE:-224}
  export EPOCHS=${EPOCHS:-8}
  export BATCH_SIZE=${BATCH_SIZE:-32}
  export UNFREEZE=${UNFREEZE:-layer4}
  export LR=${LR:-0.0003}
  export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
elif [ "$PRESET" = "medium" ]; then
  export FAST=0
  export IMG_SIZE=${IMG_SIZE:-160}
  export EPOCHS=${EPOCHS:-4}
  export BATCH_SIZE=${BATCH_SIZE:-64}
  export UNFREEZE=${UNFREEZE:-layer4}
  export LR=${LR:-0.0005}
  export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
else
  export FAST=1
  export IMG_SIZE=${IMG_SIZE:-128}
  export EPOCHS=${EPOCHS:-2}
  export BATCH_SIZE=${BATCH_SIZE:-128}
  export UNFREEZE=${UNFREEZE:-none}
  export LR=${LR:-0.001}
  export WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
fi

export MODEL_PATH

if ! find "train" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -n 1 | grep -q .; then
  echo "No images found under ./train. Check dataset path.";
  exit 1
fi

if ! find "test" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -n 1 | grep -q .; then
  echo "No images found under ./test. Create a test split first (e.g. run src/split_dataset.py).";
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found, starting training..."
    python src/train_model.py || { echo "Training failed. Check dataset preparation."; exit 1; }
    echo "Training completed. Metrics saved to metrics/ folder."
else
    echo "Model found, skipping training."
fi

echo "Starting camera inference..."
python src/camera_inference.py
