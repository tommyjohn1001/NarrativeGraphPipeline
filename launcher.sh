#!/bin/bash

mkdir -p backup

# echo "1. Decompose each document in dataset into paragraphs."
# python -m modules.data_reading.data_reading --num_proc 4

# echo "2. Start training"
CUDA_VISIBLE_DEVICES=0  python -m src.narrativepipeline.NarrativePipeline\
    --batch 5 \
    --num_proc 8 \
    --n_epochs 60 \
    --lr 5e-4 \
    --w_decay 1e-2 \
    --task train \
    # --is_debug True

# echo "3. Start inferring"
CUDA_VISIBLE_DEVICES=0  python -m src.narrativepipeline.NarrativePipeline\
    --batch 5 \
    --num_proc 8 \
    --n_epochs 60 \
    --lr 5e-4 \
    --w_decay 1e-2 \
    --task infer \