#!/bin/bash
set -e

cd "$(dirname "$0")"

PYTHON=${PYTHON:-/home/don/anaconda3/envs/AIGIDetect/bin/python}

TRAIN_PATH="/home/don/dev/Datasets/GenImage"
TRAIN_SPLIT="train"
TEST_PATH="/home/don/dev/Datasets/GenImage/test"

TRAIN_FEATURES="./extracted_features_genimage_train"
TEST_FEATURES="./extracted_features_genimage_test"
CHECKPOINT_DIR="./checkpoints"
CHECKPOINT="${CHECKPOINT_DIR}/classifier_best.pth"
OUTPUT_LOG="./eval_results_genimage.txt"

MODEL_PATH="/home/don/dev/code/Dinov3/dinov3-vit7b16-pretrain-lvd1689m"

RUN_TRAIN=1
RUN_EVAL=1

TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=256
EXTRACT_BATCH_SIZE=32
INTENSITY=0.3
KEEP_LOW=false
MASK_HIGHEST=true
RESPONSE_LAYER=-1
POOLING=mean_max
LR=2e-3
EPOCHS=6
SEED=42
NUM_WORKERS=4
FEATURE_DIM=8192

if [ "${RUN_TRAIN}" = "1" ]; then
    echo "============================================================"
    echo "=> Training  (train_path=${TRAIN_PATH}, split=${TRAIN_SPLIT})"
    echo "============================================================"
    $PYTHON train.py \
        --train_path "${TRAIN_PATH}" \
        --split "${TRAIN_SPLIT}" \
        --features_path "${TRAIN_FEATURES}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --model_path "${MODEL_PATH}" \
        --extract_batch_size ${EXTRACT_BATCH_SIZE} \
        --batch_size ${TRAIN_BATCH_SIZE} \
        --img_resolution 256 \
        --crop_resolution 224 \
        --intensity ${INTENSITY} \
        --response_layer ${RESPONSE_LAYER} \
        --pooling ${POOLING} \
        --keep_low ${KEEP_LOW} \
        --mask_highest ${MASK_HIGHEST} \
        --lr ${LR} \
        --epochs ${EPOCHS} \
        --seed ${SEED} \
        --num_workers ${NUM_WORKERS} \
        --feature_dim ${FEATURE_DIM}
fi

if [ "${RUN_EVAL}" = "1" ]; then
    echo "============================================================"
    echo "=> Evaluation  (test_path=${TEST_PATH})"
    echo "============================================================"
    $PYTHON eval.py \
        --genimage_test_path "${TEST_PATH}" \
        --output_dir "${TEST_FEATURES}" \
        --features_dir "${TEST_FEATURES}" \
        --checkpoint "${CHECKPOINT}" \
        --model_path "${MODEL_PATH}" \
        --extract_batch_size ${EXTRACT_BATCH_SIZE} \
        --batch_size ${EVAL_BATCH_SIZE} \
        --img_resolution 256 \
        --crop_resolution 224 \
        --intensity ${INTENSITY} \
        --response_layer ${RESPONSE_LAYER} \
        --pooling ${POOLING} \
        --keep_low ${KEEP_LOW} \
        --mask_highest ${MASK_HIGHEST} \
        --feature_dim ${FEATURE_DIM} \
        --output_log "${OUTPUT_LOG}"
fi
