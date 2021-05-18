#!/bin/bash

mkdir -p backup

# echo "1. Decompose each document in dataset into paragraphs."
# python -m modules.data_reading.data_reading --num_proc 4

# echo "2. Start training"
CUDA_VISIBLE_DEVICES=0  python -m modules.narrativepipeline.NarrativePipeline\
    --batch 64 \
    --num_proc 8 \
    --n_epochs 60 \
    --lr 5e-4 \
    --w_decay 1e-2 \
    # --is_debug True
